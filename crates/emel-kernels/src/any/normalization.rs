//! Allocation-free F32 row normalization operations with explicit SML dispatch.
//!
//! The pinned `emel.cpp` source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`:
//! `src/emel/kernel/detail.hpp:45-50` identifies `op_norm` and
//! `op_rms_norm`, `detail.hpp:4546-4564` validates the F32 row contract and
//! epsilon, and `detail.hpp:4568-4648` contains the reference row loops.
//! The x86-64 state-machine rows are
//! `src/emel/kernel/x86_64/sm.hpp:306-323` for `op_norm` and
//! `316-323` for `op_rms_norm`.
//!
//! Events carry caller-owned [`TensorView`](crate::any::tensor_view::TensorView)
//! and [`TensorViewMut`](crate::any::tensor_view::TensorViewMut) values. Guards
//! classify invalid views, shape mismatches, and invalid epsilon separately;
//! actions run only after the selected operation and all invariants are
//! established. The data-plane loops use logical row ordinals, preserve the
//! reference's `f64` accumulation and intermediate F32 casts, and allocate
//! nothing during dispatch.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by a normalization operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizationError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Valid source and destination views have different logical extents.
    ShapeMismatch,
    /// Epsilon is not finite or is negative.
    InvalidParameters,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for NormalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid normalization tensor view"),
            Self::ShapeMismatch => formatter.write_str("normalization tensor shapes differ"),
            Self::InvalidParameters => formatter.write_str("invalid normalization parameters"),
            Self::UnexpectedEvent => formatter.write_str("unexpected normalization event"),
            Self::Internal => formatter.write_str("internal normalization dispatch error"),
        }
    }
}

impl std::error::Error for NormalizationError {}

/// Normalizes each logical row by subtracting its mean and scaling variance.
#[derive(Debug)]
pub struct OpNorm<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    epsilon: f32,
}

impl<'a> OpNorm<'a> {
    /// Creates a mean/variance normalization event.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>, epsilon: f32) -> Self {
        Self {
            input,
            output,
            epsilon,
        }
    }
}

/// Normalizes each logical row by its root-mean-square value.
#[derive(Debug)]
pub struct OpRmsNorm<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    epsilon: f32,
}

impl<'a> OpRmsNorm<'a> {
    /// Creates an RMS normalization event.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>, epsilon: f32) -> Self {
        Self {
            input,
            output,
            epsilon,
        }
    }
}

/// Result returned after normalization dispatch.
pub type NormalizationResult = Result<(), NormalizationError>;

#[derive(Debug)]
pub struct NormRuntime<'a> {
    event: OpNorm<'a>,
    result: &'a Cell<NormalizationResult>,
}

#[derive(Debug)]
pub struct RmsNormRuntime<'a> {
    event: OpRmsNorm<'a>,
    result: &'a Cell<NormalizationResult>,
}

#[derive(Default)]
struct Context;

sml! {
    NormalizationMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Norm(NormRuntime<'dispatch>)
            [guard_norm_valid] / effect_norm_execute,
        "ready"_s <= "ready"_s + Norm(NormRuntime<'dispatch>)
            [guard_norm_shape_mismatch] / effect_norm_shape_reject,
        "ready"_s <= "ready"_s + Norm(NormRuntime<'dispatch>)
            [guard_norm_parameters_invalid] / effect_norm_parameters_reject,
        "ready"_s <= "ready"_s + Norm(NormRuntime<'dispatch>)
            [guard_norm_invalid_view] / effect_norm_view_reject,

        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>)
            [guard_rms_norm_valid] / effect_rms_norm_execute,
        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>)
            [guard_rms_norm_shape_mismatch] / effect_rms_norm_shape_reject,
        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>)
            [guard_rms_norm_parameters_invalid] / effect_rms_norm_parameters_reject,
        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>)
            [guard_rms_norm_invalid_view] / effect_rms_norm_view_reject,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for F32 row normalization.
pub struct NormalizationKernel {
    machine: NormalizationMachineStateMachine<Context>,
}

impl fmt::Debug for NormalizationKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NormalizationKernel")
            .finish_non_exhaustive()
    }
}

impl Default for NormalizationKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalizationKernel {
    /// Constructs an independent normalization actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: NormalizationMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns the event's [`NormalizationError`] when a view, shape, or
    /// epsilon invariant fails, or when dispatch cannot complete.
    pub fn process_event<E: NormalizationEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn norm(&mut self, event: OpNorm<'_>) -> NormalizationResult {
        let result = Cell::new(Err(NormalizationError::UnexpectedEvent));
        self.machine
            .process_event(NormalizationMachineEvents::Norm(NormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| NormalizationError::Internal)?;
        result.get()
    }

    fn rms_norm(&mut self, event: OpRmsNorm<'_>) -> NormalizationResult {
        let result = Cell::new(Err(NormalizationError::UnexpectedEvent));
        self.machine
            .process_event(NormalizationMachineEvents::RmsNorm(RmsNormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| NormalizationError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`NormalizationKernel`].
pub trait NormalizationEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut NormalizationKernel) -> Self::Output;
}

impl NormalizationEvent for OpNorm<'_> {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut NormalizationKernel) -> Self::Output {
        actor.norm(self)
    }
}

impl NormalizationEvent for OpRmsNorm<'_> {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut NormalizationKernel) -> Self::Output {
        actor.rms_norm(self)
    }
}

fn views_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    normalization_view_valid(input.layout(), input.storage_len())
        && normalization_view_valid(output.layout(), output.storage_len())
}

/// Mirrors the pinned `has_valid_tensor_layout` predicate while retaining the
/// safe Rust representation used by [`TensorView`].  In particular, the
/// reference treats `nb[0] == 0` as an implicit contiguous layout.  The common
/// `Layout::validate` contract intentionally rejects that wire shorthand for
/// other kernels, so normalization owns this narrower, source-backed view
/// contract and resolves it before the data-plane actions run.
pub(super) fn normalization_view_valid(
    layout: super::tensor_view::Layout,
    storage_len: usize,
) -> bool {
    let extents = layout.ne();
    if layout.dtype() != super::tensor_view::DType::F32 || extents[0] == 0 {
        return false;
    }
    let explicit_stride = layout.nb()[0] != 0;
    if explicit_stride {
        if layout.nb()[0] < 4 || !layout.nb()[0].is_multiple_of(4) {
            return false;
        }
        for (extent, stride) in extents.iter().zip(layout.nb()) {
            if *extent > 1 && stride == 0 {
                return false;
            }
        }
    }
    // The maintained public kernel rejects zero extents before entering the
    // row loop, matching the observed pinned state-machine boundary.
    if extents.contains(&0) {
        return false;
    }
    let Some(strides) = layout.effective_f32_strides() else {
        return false;
    };
    // A Rust `&[f32]` can safely model only byte-aligned F32 addresses.  The
    // pinned predicate checks this for nb[0]; keeping the same requirement for
    // every dimension is the safe representation boundary of this slice.
    if strides
        .iter()
        .any(|stride| *stride < 4 || !stride.is_multiple_of(4))
    {
        return false;
    }
    let mut max_offset = 0u128;
    for (extent, stride) in extents.iter().zip(strides) {
        let Some(term) = u128::from(*extent - 1).checked_mul(u128::from(stride)) else {
            return false;
        };
        let Some(next) = max_offset.checked_add(term) else {
            return false;
        };
        max_offset = next;
    }
    let Some(end) = max_offset.checked_add(4) else {
        return false;
    };
    end <= (storage_len as u128).saturating_mul(4)
}

fn shapes_match(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    input.layout().ne() == output.layout().ne()
}

fn epsilon_valid(epsilon: f32) -> bool {
    epsilon.is_finite() && epsilon >= 0.0
}

#[allow(clippy::cast_precision_loss)]
const fn f64_from_usize(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
const fn f32_from_f64(value: f64) -> f32 {
    value as f32
}

fn row_shape(input: &TensorView<'_>) -> Option<(usize, usize)> {
    let columns = usize::try_from(input.layout().ne()[0]).ok()?;
    let rows = input.count().checked_div(columns)?;
    Some((columns, rows))
}

// Preserve the reference's separate square and accumulation operations;
// `mul_add` changes floating-point rounding and can change parity bits.
#[allow(clippy::suboptimal_flops)]
fn run_norm(event: &mut NormRuntime<'_>) {
    let (columns, rows) = row_shape(&event.event.input).expect("guard proved row shape");
    let epsilon = event.event.epsilon;

    for row in 0..rows {
        let start = row * columns;
        let end = start + columns;
        let mut sum = 0.0f64;
        for ordinal in start..end {
            sum += f64::from(event.event.input.read(ordinal));
        }
        let mean = f32_from_f64(sum / f64_from_usize(columns));

        let mut variance_sum = 0.0f64;
        for ordinal in start..end {
            let centered = event.event.input.read(ordinal) - mean;
            variance_sum += f64::from(centered) * f64::from(centered);
        }
        let variance = f32_from_f64(variance_sum / f64_from_usize(columns));
        let scale = 1.0f32 / (variance + epsilon).sqrt();
        for ordinal in start..end {
            let normalized = (event.event.input.read(ordinal) - mean) * scale;
            event.event.output.write(ordinal, normalized);
        }
    }
}

fn run_rms_norm(event: &mut RmsNormRuntime<'_>) {
    let (columns, rows) = row_shape(&event.event.input).expect("guard proved row shape");
    let epsilon = event.event.epsilon;

    for row in 0..rows {
        let start = row * columns;
        let end = start + columns;
        let mut sum = 0.0f64;
        for ordinal in start..end {
            let value = event.event.input.read(ordinal);
            sum += f64::from(value * value);
        }
        let mean = f32_from_f64(sum / f64_from_usize(columns));
        let scale = 1.0f32 / (mean + epsilon).sqrt();
        for ordinal in start..end {
            let scaled = event.event.input.read(ordinal) * scale;
            event.event.output.write(ordinal, scaled);
        }
    }
}

fn guard_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>, epsilon: f32) -> bool {
    views_valid(input, output) && shapes_match(input, output) && epsilon_valid(epsilon)
}

fn guard_shape_mismatch(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    views_valid(input, output) && !shapes_match(input, output)
}

fn guard_parameters_invalid(
    input: &TensorView<'_>,
    output: &TensorViewMut<'_>,
    epsilon: f32,
) -> bool {
    views_valid(input, output) && shapes_match(input, output) && !epsilon_valid(epsilon)
}

fn guard_invalid_view(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    !views_valid(input, output)
}

impl NormalizationMachineStateMachineContext for Context {
    fn guard_norm_valid(&self, event: &NormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_valid(
            &event.event.input,
            &event.event.output,
            event.event.epsilon,
        ))
    }

    fn guard_norm_shape_mismatch(&self, event: &NormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_shape_mismatch(
            &event.event.input,
            &event.event.output,
        ))
    }

    fn guard_norm_parameters_invalid(&self, event: &NormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_parameters_invalid(
            &event.event.input,
            &event.event.output,
            event.event.epsilon,
        ))
    }

    fn guard_norm_invalid_view(&self, event: &NormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_invalid_view(&event.event.input, &event.event.output))
    }

    fn effect_norm_execute(&mut self, mut event: NormRuntime<'_>) -> Result<(), ()> {
        run_norm(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_norm_shape_reject(&mut self, event: NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::ShapeMismatch));
        Ok(())
    }

    fn effect_norm_parameters_reject(&mut self, event: NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::InvalidParameters));
        Ok(())
    }

    fn effect_norm_view_reject(&mut self, event: NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::InvalidView));
        Ok(())
    }

    fn guard_rms_norm_valid(&self, event: &RmsNormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_valid(
            &event.event.input,
            &event.event.output,
            event.event.epsilon,
        ))
    }

    fn guard_rms_norm_shape_mismatch(&self, event: &RmsNormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_shape_mismatch(
            &event.event.input,
            &event.event.output,
        ))
    }

    fn guard_rms_norm_parameters_invalid(&self, event: &RmsNormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_parameters_invalid(
            &event.event.input,
            &event.event.output,
            event.event.epsilon,
        ))
    }

    fn guard_rms_norm_invalid_view(&self, event: &RmsNormRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_invalid_view(&event.event.input, &event.event.output))
    }

    fn effect_rms_norm_execute(&mut self, mut event: RmsNormRuntime<'_>) -> Result<(), ()> {
        run_rms_norm(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rms_norm_shape_reject(&mut self, event: RmsNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::ShapeMismatch));
        Ok(())
    }

    fn effect_rms_norm_parameters_reject(&mut self, event: RmsNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::InvalidParameters));
        Ok(())
    }

    fn effect_rms_norm_view_reject(&mut self, event: RmsNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
