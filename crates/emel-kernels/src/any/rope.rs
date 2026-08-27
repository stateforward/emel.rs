//! Safe F32 `RoPE` kernels with explicit mode transitions.
//!
//! The pinned `emel.cpp` source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Its
//! `detail.hpp:4650-4770` contract reads the `RoPE` parameters, validates F32
//! source/destination and I32 positions, then selects norm, neox, or timestep
//! execution through separate x86 transition rows at `sm.hpp:641-658`.
//!
//! This module keeps that mode choice in guards. Actions only execute the
//! already-selected rotation formula and bounded data-plane loops. The
//! maintained safe slice uses positive extents and a one-dimensional I32
//! position tensor (`[src.ne[2], 1, 1, 1]`); malformed or zero-extent metadata
//! is rejected before any data access.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{Layout, TensorView, TensorViewMut};

/// The pinned normal rotary pairing mode.
pub const ROPE_MODE_NORM: i32 = 0;
/// The pinned split-half rotary pairing mode.
pub const ROPE_MODE_NEOX: i32 = 2;
/// The pinned timestep embedding mode.
pub const ROPE_MODE_TIMESTEP: i32 = 4;

/// Typed parameters read from the pinned `RoPE` operation slots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// Number of rotated dimensions.
    pub n_dims: i32,
    /// One of [`ROPE_MODE_NORM`], [`ROPE_MODE_NEOX`], or
    /// [`ROPE_MODE_TIMESTEP`].
    pub mode: i32,
    /// Base period used to derive the rotation frequency.
    pub freq_base: f32,
    /// Position-frequency scale.
    pub freq_scale: f32,
    /// The pinned maintained route requires this extension factor to be zero.
    pub ext_factor: f32,
    /// Output attention scale.
    pub attn_factor: f32,
}

/// A safe I32 view for the position tensor used by `RoPE`.
#[derive(Debug)]
pub struct I32View<'a> {
    data: &'a [i32],
    layout: Layout,
}

impl<'a> I32View<'a> {
    /// Creates an I32 view. Layout and storage validation occurs in guards.
    #[must_use]
    pub const fn new(data: &'a [i32], layout: Layout) -> Self {
        Self { data, layout }
    }

    /// Returns the view layout.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Borrows the backing positions after a dense-layout guard succeeds.
    pub(crate) const fn as_slice(&self) -> &[i32] {
        self.data
    }

    /// Validates I32 dtype, shape, strides, and complete storage span.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or bounds invariant.
    pub fn validate(&self) -> Result<usize, super::tensor_view::ViewError> {
        self.layout.validate_i32(self.data.len())
    }

    fn read_at(&self, i0: usize, i1: usize, i2: usize, i3: usize) -> i32 {
        let coordinates = [i0, i1, i2, i3];
        let strides = self.layout.nb();
        let mut offset = 0u128;
        let mut dimension = 0;
        while dimension < 4 {
            let coordinate = u128::try_from(coordinates[dimension])
                .expect("guard-proven I32 coordinate fits u128");
            offset = offset
                .checked_add(coordinate * u128::from(strides[dimension]))
                .expect("guard-proven I32 offset fits u128");
            dimension += 1;
        }
        let element = usize::try_from(offset / 4).expect("guard-proven I32 offset fits usize");
        self.data[element]
    }
}

/// Errors returned by `RoPE` dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeError {
    /// One or more tensor views failed dtype, shape, stride, or bounds checks.
    InvalidView,
    /// Valid views do not satisfy the pinned shape/layout contract.
    ShapeMismatch,
    /// Parameters are malformed or outside the maintained route.
    InvalidParameters,
    /// The operation mode is not one of the pinned explicit modes.
    InvalidMode,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for RopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid rope tensor view"),
            Self::ShapeMismatch => formatter.write_str("rope tensor shapes differ"),
            Self::InvalidParameters => formatter.write_str("invalid rope parameters"),
            Self::InvalidMode => formatter.write_str("invalid rope mode"),
            Self::UnexpectedEvent => formatter.write_str("unexpected rope event"),
            Self::Internal => formatter.write_str("internal rope dispatch error"),
        }
    }
}

impl std::error::Error for RopeError {}

/// Applies one explicitly guarded `RoPE` mode.
#[derive(Debug)]
pub struct OpRope<'a> {
    source: TensorView<'a>,
    positions: I32View<'a>,
    destination: TensorViewMut<'a>,
    params: RopeParams,
}

impl<'a> OpRope<'a> {
    /// Creates an event. Validation and mode selection occur in machine guards.
    #[must_use]
    pub const fn new(
        source: TensorView<'a>,
        positions: I32View<'a>,
        destination: TensorViewMut<'a>,
        params: RopeParams,
    ) -> Self {
        Self {
            source,
            positions,
            destination,
            params,
        }
    }

    /// Returns the pinned mode used by the owning machine's explicit guard.
    #[must_use]
    pub const fn mode(&self) -> i32 {
        self.params.mode
    }
}

/// Explicitly reports an event outside the maintained `RoPE` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedRope;

type RopeResult = Result<(), RopeError>;

#[derive(Debug)]
struct RopeRuntime<'a> {
    event: OpRope<'a>,
    result: &'a Cell<RopeResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<RopeResult>,
}

#[derive(Default)]
struct Context;

sml! {
    RopeMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_dense_norm] / effect_rope_dense_norm,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_norm] / effect_rope_norm,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_dense_neox] / effect_rope_dense_neox,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_neox] / effect_rope_neox,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_timestep] / effect_rope_timestep,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_mode_invalid] / effect_rope_mode_reject,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_parameters_invalid] / effect_rope_parameters_reject,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_shape_mismatch] / effect_rope_shape_reject,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>)
            [guard_rope_invalid_view] / effect_rope_view_reject,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_rope_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for the maintained `RoPE` routes.
pub struct RopeKernel {
    machine: RopeMachineStateMachine<Context>,
}

impl fmt::Debug for RopeKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RopeKernel").finish_non_exhaustive()
    }
}

impl Default for RopeKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeKernel {
    /// Constructs an independent `RoPE` actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: RopeMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: RopeEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn rope(&mut self, event: OpRope<'_>) -> RopeResult {
        let result = Cell::new(Err(RopeError::UnexpectedEvent));
        self.machine
            .process_event(RopeMachineEvents::Rope(RopeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedRope) -> RopeResult {
        let result = Cell::new(Err(RopeError::Internal));
        self.machine
            .process_event(RopeMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`RopeKernel`].
pub trait RopeEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut RopeKernel) -> Self::Output;
}

impl RopeEvent for OpRope<'_> {
    type Output = RopeResult;

    fn dispatch(self, actor: &mut RopeKernel) -> Self::Output {
        actor.rope(self)
    }
}

impl RopeEvent for UnexpectedRope {
    type Output = RopeResult;

    fn dispatch(self, actor: &mut RopeKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn views_valid(event: &OpRope<'_>) -> bool {
    event.source.validate().is_ok()
        && event.positions.validate().is_ok()
        && event.destination.validate().is_ok()
}

fn shape_valid(event: &OpRope<'_>) -> bool {
    let source = event.source.layout().ne();
    let destination = event.destination.layout().ne();
    let positions = event.positions.layout().ne();
    source == destination && source[0] > 0 && positions == [source[2], 1, 1, 1] && source[2] > 0
}

fn parameters_valid(event: &OpRope<'_>) -> bool {
    let params = event.params;
    let source_cols = event.source.layout().ne()[0];
    params.n_dims > 0
        && params.n_dims % 2 == 0
        && u64::try_from(params.n_dims).is_ok_and(|value| value <= source_cols)
        && params.ext_factor == 0.0
        && params.freq_base.is_finite()
        && params.freq_base > 0.0
        && params.freq_scale.is_finite()
        && params.freq_scale > 0.0
        && params.attn_factor.is_finite()
}

const fn known_mode(mode: i32) -> bool {
    matches!(mode, ROPE_MODE_NORM | ROPE_MODE_NEOX | ROPE_MODE_TIMESTEP)
}

fn timestep_layout_valid(event: &OpRope<'_>) -> bool {
    let source = event.source.layout();
    let positions = event.positions.layout();
    let destination = event.destination.layout();
    u64::try_from(event.params.n_dims).is_ok_and(|n_dims| n_dims == source.ne()[0])
        && source.is_dense_contiguous()
        && positions.is_dense_contiguous_i32()
        && destination.is_dense_contiguous()
}

fn selected_mode_layout_valid(event: &OpRope<'_>) -> bool {
    match event.params.mode {
        ROPE_MODE_NORM | ROPE_MODE_NEOX => true,
        ROPE_MODE_TIMESTEP => timestep_layout_valid(event),
        _ => false,
    }
}

fn dense_rotation_valid(event: &OpRope<'_>, mode: i32) -> bool {
    views_valid(event)
        && parameters_valid(event)
        && shape_valid(event)
        && event.params.mode == mode
        && event.source.layout().is_dense_contiguous()
        && event.positions.layout().is_dense_contiguous_i32()
        && event.destination.layout().is_dense_contiguous()
}

impl RopeMachineStateMachineContext for Context {
    fn guard_rope_dense_norm(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(dense_rotation_valid(&event.event, ROPE_MODE_NORM))
    }

    fn guard_rope_norm(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(&event.event)
            && shape_valid(&event.event)
            && event.event.params.mode == ROPE_MODE_NORM)
    }

    fn guard_rope_neox(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(&event.event)
            && shape_valid(&event.event)
            && event.event.params.mode == ROPE_MODE_NEOX)
    }

    fn guard_rope_dense_neox(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(dense_rotation_valid(&event.event, ROPE_MODE_NEOX))
    }

    fn guard_rope_timestep(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(&event.event)
            && shape_valid(&event.event)
            && event.event.params.mode == ROPE_MODE_TIMESTEP
            && timestep_layout_valid(&event.event))
    }

    fn guard_rope_mode_invalid(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(&event.event)
            && shape_valid(&event.event)
            && !known_mode(event.event.params.mode))
    }

    fn guard_rope_parameters_invalid(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event) && !parameters_valid(&event.event))
    }

    fn guard_rope_shape_mismatch(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(&event.event)
            && (!shape_valid(&event.event) || !selected_mode_layout_valid(&event.event)))
    }

    fn guard_rope_invalid_view(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event))
    }

    fn effect_rope_norm(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        execute_rotation::<false>(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_dense_norm(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        execute_rotation_dense::<false>(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_neox(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        execute_rotation::<true>(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_dense_neox(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        execute_rotation_dense::<true>(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_timestep(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        execute_timestep(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_mode_reject(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RopeError::InvalidMode));
        Ok(())
    }

    fn effect_rope_parameters_reject(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RopeError::InvalidParameters));
        Ok(())
    }

    fn effect_rope_shape_reject(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RopeError::ShapeMismatch));
        Ok(())
    }

    fn effect_rope_view_reject(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RopeError::InvalidView));
        Ok(())
    }

    fn effect_rope_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RopeError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

fn logical_ordinal(layout: [u64; 4], coordinates: [usize; 4]) -> usize {
    let mut ordinal = 0usize;
    let mut dimension = 4usize;
    while dimension > 0 {
        let index = dimension - 1;
        let extent = usize::try_from(layout[index]).expect("guard-proven extent fits usize");
        ordinal = ordinal
            .checked_mul(extent)
            .and_then(|value| value.checked_add(coordinates[index]))
            .expect("guard-proven logical ordinal fits usize");
        dimension -= 1;
    }
    ordinal
}

#[allow(clippy::cast_precision_loss)]
fn execute_rotation_dense<const NEOX: bool>(mut event: OpRope<'_>) {
    let shape = event.source.layout().ne();
    let cols = usize::try_from(shape[0]).expect("guard-proven columns fit usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let positions = usize::try_from(shape[2]).expect("guard-proven positions fit usize");
    let batches = usize::try_from(shape[3]).expect("guard-proven batches fit usize");
    let n_dims = usize::try_from(event.params.n_dims).expect("guard-proven dimensions fit usize");
    let half_dims = n_dims / 2;
    let theta_scale = event.params.freq_base.powf(-2.0 / n_dims as f32);
    let source = event.source.as_slice();
    let position_values = event.positions.as_slice();
    let destination = event.destination.as_mut_slice();
    let mut batch = 0;
    while batch < batches {
        let mut position_index = 0;
        while position_index < positions {
            let position = position_values[position_index] as f32;
            let mut row = 0;
            while row < rows {
                let base = cols * (row + rows * (position_index + positions * batch));
                let mut theta = position;
                let mut pair = 0;
                while pair < half_dims {
                    let rotated = theta * event.params.freq_scale;
                    let cosine = rotated.cos() * event.params.attn_factor;
                    let sine = rotated.sin() * event.params.attn_factor;
                    let (index_a, index_b) = if NEOX {
                        (pair, pair + half_dims)
                    } else {
                        (2 * pair, 2 * pair + 1)
                    };
                    let first = source[base + index_a];
                    let second = source[base + index_b];
                    destination[base + index_a] = first.mul_add(cosine, -(second * sine));
                    destination[base + index_b] = first.mul_add(sine, second * cosine);
                    theta *= theta_scale;
                    pair += 1;
                }
                if n_dims < cols {
                    destination[base + n_dims..base + cols]
                        .copy_from_slice(&source[base + n_dims..base + cols]);
                }
                row += 1;
            }
            position_index += 1;
        }
        batch += 1;
    }
}

#[allow(clippy::cast_precision_loss)]
fn execute_rotation<const NEOX: bool>(mut event: OpRope<'_>) {
    let source_shape = event.source.layout().ne();
    let cols = usize::try_from(source_shape[0]).expect("guard-proven columns fit usize");
    let rows = usize::try_from(source_shape[1]).expect("guard-proven rows fit usize");
    let positions = usize::try_from(source_shape[2]).expect("guard-proven positions fit usize");
    let batches = usize::try_from(source_shape[3]).expect("guard-proven batches fit usize");
    let n_dims = usize::try_from(event.params.n_dims).expect("guard-proven dimensions fit usize");
    let half_dims = n_dims / 2;
    let theta_scale = event
        .params
        .freq_base
        .powf(-2.0 / event.params.n_dims as f32);
    let mut batch = 0;
    while batch < batches {
        let mut position_index = 0;
        while position_index < positions {
            let position = event.positions.read_at(position_index, 0, 0, 0) as f32;
            let mut row = 0;
            while row < rows {
                let mut theta = position;
                let mut pair = 0;
                while pair < half_dims {
                    let rotated = theta * event.params.freq_scale;
                    let cosine = rotated.cos() * event.params.attn_factor;
                    let sine = rotated.sin() * event.params.attn_factor;
                    let (index_a, index_b) = if NEOX {
                        (pair, pair + half_dims)
                    } else {
                        (2 * pair, 2 * pair + 1)
                    };
                    let source_a =
                        logical_ordinal(source_shape, [index_a, row, position_index, batch]);
                    let source_b =
                        logical_ordinal(source_shape, [index_b, row, position_index, batch]);
                    let first = event.source.read(source_a);
                    let second = event.source.read(source_b);
                    let destination_a =
                        logical_ordinal(source_shape, [index_a, row, position_index, batch]);
                    let destination_b =
                        logical_ordinal(source_shape, [index_b, row, position_index, batch]);
                    event
                        .destination
                        // The pinned C++ scalar action lowers each final
                        // rotation expression to one fused multiply-add
                        // (the second product is evaluated as its addend).
                        // `mul_add` is the safe Rust primitive that preserves
                        // that source-visible f32 rounding contract.
                        .write(destination_a, first.mul_add(cosine, -(second * sine)));
                    event
                        .destination
                        .write(destination_b, first.mul_add(sine, second * cosine));
                    theta *= theta_scale;
                    pair += 1;
                }
                let mut column = n_dims;
                while column < cols {
                    let ordinal =
                        logical_ordinal(source_shape, [column, row, position_index, batch]);
                    event.destination.write(ordinal, event.source.read(ordinal));
                    column += 1;
                }
                row += 1;
            }
            position_index += 1;
        }
        batch += 1;
    }
}

#[allow(clippy::cast_precision_loss)]
fn execute_timestep(mut event: OpRope<'_>) {
    let shape = event.source.layout().ne();
    let cols = usize::try_from(shape[0]).expect("guard-proven columns fit usize");
    let half_dims = cols / 2;
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let positions = usize::try_from(shape[2]).expect("guard-proven positions fit usize");
    let batches = usize::try_from(shape[3]).expect("guard-proven batches fit usize");
    let neg_log_period = -event.params.freq_base.ln();
    let mut batch = 0;
    while batch < batches {
        let mut position_index = 0;
        while position_index < positions {
            let position = event.positions.read_at(position_index, 0, 0, 0) as f32;
            let mut row = 0;
            while row < rows {
                let mut pair = 0;
                while pair < half_dims {
                    let real_index = logical_ordinal(shape, [2 * pair, row, position_index, batch]);
                    let imaginary_index =
                        logical_ordinal(shape, [2 * pair + 1, row, position_index, batch]);
                    let real = event.source.read(real_index);
                    let imaginary = event.source.read(imaginary_index);
                    let frequency = (neg_log_period * pair as f32 / half_dims as f32).exp();
                    let argument = position * frequency * event.params.freq_scale;
                    let cosine = argument.cos() * event.params.attn_factor;
                    let sine = argument.sin() * event.params.attn_factor;
                    let real_cosine = std::hint::black_box(real * cosine);
                    let imaginary_sine = std::hint::black_box(imaginary * sine);
                    let real_sine = std::hint::black_box(real * sine);
                    let imaginary_cosine = std::hint::black_box(imaginary * cosine);
                    let real_output = logical_ordinal(shape, [pair, row, position_index, batch]);
                    let imaginary_output =
                        logical_ordinal(shape, [half_dims + pair, row, position_index, batch]);
                    event
                        .destination
                        .write(real_output, real_cosine - imaginary_sine);
                    event
                        .destination
                        .write(imaginary_output, real_sine + imaginary_cosine);
                    pair += 1;
                }
                row += 1;
            }
            position_index += 1;
        }
        batch += 1;
    }
}
