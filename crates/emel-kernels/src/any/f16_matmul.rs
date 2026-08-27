//! Safe scalar F16-by-F16-to-F32 matrix multiplication.
//!
//! The pinned reference contract is `can_run_mul_mat_f16` and
//! `run_mul_mat_f16` in `src/emel/kernel/detail.hpp:3878-4215`. It uses
//! dense F16 operands with explicit, nonzero dense byte strides, source shapes
//! `[k,m]` and `[k,n]`, and a dense F32 destination `[m,n]` whose first
//! dimension is contiguous. This module keeps that operand contract explicit
//! and leaves SIMD-specific implementations to the target lanes.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

use super::tensor_view::{DType, Layout, TensorViewMut};

/// An immutable, validated-on-dispatch F16 tensor view.
#[derive(Clone, Copy, Debug)]
pub struct F16View<'a> {
    data: &'a [u16],
    layout: Layout,
    strides: [u64; 4],
}

impl<'a> F16View<'a> {
    /// Creates an F16 view. Validation is classified by actor guards.
    #[must_use]
    pub const fn new(data: &'a [u16], layout: Layout) -> Self {
        let strides = match effective_f16_strides(layout) {
            Some(value) => value,
            None => [0; 4],
        };
        Self {
            data,
            layout,
            strides,
        }
    }

    /// Returns the view layout.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    pub(crate) fn validate(&self) -> bool {
        let Some(strides) = effective_f16_strides(self.layout) else {
            return false;
        };
        if self.layout.dtype() != DType::F16 || self.layout.element_count().is_none() {
            return false;
        }
        let ne = self.layout.ne();
        let mut dimension = 0;
        while dimension < 4 {
            let extent = ne[dimension];
            if extent == 0 {
                return false;
            }
            // The pinned layout predicate permits arbitrary singleton
            // strides because their coordinate is always zero. Every
            // multi-element dimension still needs a representable aligned
            // F16 byte stride; odd offsets cannot be represented by &[u16].
            if extent > 1 && (strides[dimension] < 2 || !strides[dimension].is_multiple_of(2)) {
                return false;
            }
            dimension += 1;
        }
        let mut max_offset = 0u128;
        dimension = 0;
        while dimension < 4 {
            let extent = u128::from(ne[dimension]);
            let stride = u128::from(strides[dimension]);
            let Some(contribution) = (extent - 1).checked_mul(stride) else {
                return false;
            };
            max_offset = match max_offset.checked_add(contribution) {
                Some(value) => value,
                None => return false,
            };
            dimension += 1;
        }
        let Some(end) = max_offset.checked_add(2) else {
            return false;
        };
        end <= (self.data.len() as u128).saturating_mul(2)
    }

    #[inline]
    pub(crate) fn read(&self, ordinal: usize) -> f32 {
        f16_to_f32(self.read_bits(ordinal))
    }

    #[inline]
    pub(crate) fn read_bits(&self, ordinal: usize) -> u16 {
        self.data[self.element_index(ordinal)]
    }

    #[inline]
    fn element_index(&self, ordinal: usize) -> usize {
        let mut remaining = ordinal as u64;
        let mut byte_offset = 0u64;
        let ne = self.layout.ne();
        let mut dimension = 0;
        while dimension < 4 {
            let extent = ne[dimension];
            let coordinate = remaining % extent;
            let offset = coordinate
                .checked_mul(self.strides[dimension])
                .expect("validated F16 coordinate offset fits");
            byte_offset = byte_offset
                .checked_add(offset)
                .expect("validated F16 byte offset fits");
            remaining /= extent;
            dimension += 1;
        }
        usize::try_from(byte_offset / 2).expect("validated F16 index fits")
    }
}

/// Resolves the pinned implicit-contiguous F16 representation (`nb[0] == 0`)
/// without exposing raw storage or weakening explicit-stride validation.
const fn effective_f16_strides(layout: Layout) -> Option<[u64; 4]> {
    let nb = layout.nb();
    if nb[0] != 0 {
        return Some(nb);
    }

    let ne = layout.ne();
    let mut strides = [0_u64; 4];
    strides[0] = 2;
    let mut dimension = 1;
    while dimension < 4 {
        strides[dimension] = match strides[dimension - 1].checked_mul(ne[dimension - 1]) {
            Some(value) => value,
            None => return None,
        };
        dimension += 1;
    }
    Some(strides)
}

/// Errors returned by F16 matrix multiplication dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F16MatmulError {
    /// One or more operands failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Operands do not satisfy the pinned matrix dimensions or rank contract.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for F16MatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid F16 matrix tensor view"),
            Self::ShapeMismatch => formatter.write_str("F16 matrix tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected F16 matrix event"),
            Self::Internal => formatter.write_str("internal F16 matrix dispatch error"),
        }
    }
}

impl std::error::Error for F16MatmulError {}

/// F16 matrix multiplication operands using the pinned ggml orientation.
#[derive(Debug)]
pub struct OpMulMatF16<'a> {
    /// Left operand with logical shape `[k,m,1,1]`.
    lhs: F16View<'a>,
    /// Right operand with logical shape `[k,n,1,1]`.
    rhs: F16View<'a>,
    /// Dense F32 destination with logical shape `[m,n,1,1]`.
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatF16<'a> {
    /// Creates an F16 matrix event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(lhs: F16View<'a>, rhs: F16View<'a>, destination: TensorViewMut<'a>) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }
}

/// Explicitly reports an event outside the maintained F16 matrix API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedF16Matmul;

/// Result returned after one F16 matrix multiplication dispatch.
pub type F16MatmulResult = Result<(), F16MatmulError>;

#[derive(Clone, Copy)]
struct Runtime<'dispatch, 'data> {
    lhs: F16View<'data>,
    rhs: F16View<'data>,
    destination: &'dispatch RefCell<TensorViewMut<'data>>,
    result: &'dispatch Cell<F16MatmulResult>,
}

#[derive(Clone, Copy)]
struct UnexpectedRuntime<'a> {
    result: &'a Cell<F16MatmulResult>,
}

#[derive(Default)]
struct Context;

sml! {
    F16MatmulMachine<'dispatch, 'data>
    where
        'data: 'dispatch,
    {
        // Keep validation as explicit phase-level choices. The unguarded
        // fallback enters a rejection state, so the ready state has only one
        // event transition and each decision has one positive guard.
        "view_decision"_s <= *"ready"_s + MulMatF16(Runtime<'dispatch, 'data>),
        "views_validated"_s <= "view_decision"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>) [guard_views_valid],
        "views_rejected"_s <= "view_decision"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>),
        "shape_decision"_s <= "views_validated"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>),
        "shape_validated"_s <= "shape_decision"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>) [guard_shape_valid],
        "shape_rejected"_s <= "shape_decision"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>),
        "execution"_s <= "shape_validated"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>),
        "ready"_s <= "views_rejected"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>) / effect_view_reject,
        "ready"_s <= "shape_rejected"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>) / effect_shape_reject,
        "ready"_s <= "execution"_s
            + completion<MulMatF16>(Runtime<'dispatch, 'data>) / effect_execute,

        "unexpected_dispatch"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected_event,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_generic_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "views_validated"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "views_rejected"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_validated"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_rejected"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "execution"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion actor for scalar F16 matrix multiplication.
pub struct F16MatmulKernel {
    machine: F16MatmulMachineStateMachine<Context>,
}

impl fmt::Debug for F16MatmulKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("F16MatmulKernel")
            .finish_non_exhaustive()
    }
}

impl Default for F16MatmulKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl F16MatmulKernel {
    /// Constructs an independent F16 matrix actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: F16MatmulMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated F16 matrix machine does not return to `Ready`
    /// after the run-to-completion dispatch.
    pub fn process_event<E: F16MatmulEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        assert!(
            self.machine.is(&F16MatmulMachineStates::Ready),
            "F16 matrix machine must return to ready after dispatch"
        );
        output
    }

    /// Reports whether the generated F16 matrix machine is ready for another
    /// dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&F16MatmulMachineStates::Ready)
    }

    fn matmul(&mut self, event: OpMulMatF16<'_>) -> F16MatmulResult {
        let destination = RefCell::new(event.destination);
        let result = Cell::new(Err(F16MatmulError::UnexpectedEvent));
        self.machine
            .process_event(F16MatmulMachineEvents::MulMatF16(Runtime {
                lhs: event.lhs,
                rhs: event.rhs,
                destination: &destination,
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedF16Matmul) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::Internal));
        self.machine
            .process_event(F16MatmulMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`F16MatmulKernel`].
pub trait F16MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut F16MatmulKernel) -> Self::Output;
}

impl F16MatmulEvent for OpMulMatF16<'_> {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut F16MatmulKernel) -> Self::Output {
        actor.matmul(self)
    }
}

impl F16MatmulEvent for UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut F16MatmulKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn views_valid(
    lhs: &F16View<'_>,
    rhs: &F16View<'_>,
    destination: &RefCell<TensorViewMut<'_>>,
) -> bool {
    lhs.validate()
        && rhs.validate()
        && destination.borrow().validate().is_ok()
        && lhs.layout().is_dense_contiguous_f16()
        && rhs.layout().is_dense_contiguous_f16()
        && destination.borrow().layout().is_dense_contiguous()
}

fn shape_valid(
    lhs_view: &F16View<'_>,
    rhs_view: &F16View<'_>,
    destination_view: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let lhs = lhs_view.layout().ne();
    let rhs = rhs_view.layout().ne();
    let destination = destination_view.borrow().layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] == lhs[0]
        && rhs[1] > 0
        && destination[0] == lhs[1]
        && destination[1] == rhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

impl F16MatmulMachineStateMachineContext for Context {
    fn guard_views_valid<'dispatch, 'data>(
        &self,
        event: &Runtime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(views_valid(&event.lhs, &event.rhs, event.destination))
    }

    fn guard_shape_valid<'dispatch, 'data>(
        &self,
        event: &Runtime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(shape_valid(&event.lhs, &event.rhs, event.destination))
    }

    #[allow(clippy::cast_possible_truncation, clippy::suboptimal_flops)]
    fn effect_execute<'dispatch, 'data>(
        &mut self,
        event: Runtime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        let lhs_shape = event.lhs.layout().ne();
        let rhs_shape = event.rhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(lhs_shape[1]).expect("guard-proven m fits usize");
        let columns = usize::try_from(rhs_shape[1]).expect("guard-proven n fits usize");
        let mut column = 0;
        while column < columns {
            let mut row = 0;
            while row < rows {
                let mut inner = 0;
                // The pinned scalar `vec_dot_f16_ggml` multiplies converted
                // F32 operands, promotes each F32 product to F64, accumulates
                // in F64, and casts once at the destination boundary.
                let mut accumulator = 0.0_f64;
                while inner < k {
                    let lhs_ordinal = inner
                        .checked_add(
                            k.checked_mul(row)
                                .expect("guard-proven lhs ordinal fits usize"),
                        )
                        .expect("guard-proven lhs ordinal fits usize");
                    let rhs_ordinal = inner
                        .checked_add(
                            k.checked_mul(column)
                                .expect("guard-proven rhs ordinal fits usize"),
                        )
                        .expect("guard-proven rhs ordinal fits usize");
                    let product = event.lhs.read(lhs_ordinal) * event.rhs.read(rhs_ordinal);
                    accumulator += f64::from(product);
                    inner += 1;
                }
                let destination_ordinal = row
                    .checked_add(
                        rows.checked_mul(column)
                            .expect("guard-proven destination ordinal fits usize"),
                    )
                    .expect("guard-proven destination ordinal fits usize");
                event
                    .destination
                    .borrow_mut()
                    .write(destination_ordinal, accumulator as f32);
                row += 1;
            }
            column += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_shape_reject<'dispatch, 'data>(
        &mut self,
        event: Runtime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(F16MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_view_reject<'dispatch, 'data>(
        &mut self,
        event: Runtime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(F16MatmulError::InvalidView));
        Ok(())
    }

    fn effect_unexpected_event(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F16MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut normalized = fraction;
            let mut shift = 0u32;
            while (normalized & 0x0400) == 0 {
                normalized <<= 1;
                shift += 1;
            }
            let mantissa = (normalized & 0x03ff) << 13;
            let adjusted_exponent = 127u32 - (14u32 + shift);
            sign | (adjusted_exponent << 23) | mantissa
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(value)
}
