//! Portable cumulative, repetition, and concatenation kernels over validated
//! F32 views.
//!
//! The pinned reference publishes transition rows for `op_cumsum`,
//! `op_repeat`, `op_repeat_back`, and `op_concat`, but its current commit does
//! not contain executable action or guard definitions for these rows. This
//! module therefore exposes a source-contract implementation for the bounded
//! F32 view family. Each event has explicit SML validation and execution rows;
//! no runtime operation router or dispatch allocation is used.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{Layout, TensorView, TensorViewMut};

/// Errors returned by cumulative and repetition dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceError {
    /// A view failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// The output shape is incompatible with the operation.
    ShapeMismatch,
    /// A concatenation axis is outside the four-dimensional view.
    InvalidAxis,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for SequenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid sequence tensor view"),
            Self::ShapeMismatch => formatter.write_str("sequence tensor shapes differ"),
            Self::InvalidAxis => formatter.write_str("sequence concatenation axis is invalid"),
            Self::UnexpectedEvent => formatter.write_str("unexpected sequence event"),
            Self::Internal => formatter.write_str("internal sequence dispatch error"),
        }
    }
}

impl std::error::Error for SequenceError {}

/// Computes a cumulative sum along dimension zero for each logical row.
#[derive(Debug)]
pub struct OpCumsum<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpCumsum<'a> {
    /// Creates a cumulative-sum event. View validation occurs in a guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Repeats a source tensor across every destination dimension.
#[derive(Debug)]
pub struct OpRepeat<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpRepeat<'a> {
    /// Creates a repeat event. Divisibility and view validation occur in a guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Reduces a repeated source tensor into a smaller destination tensor.
///
/// This is the inverse shape relationship of [`OpRepeat`]. Every destination
/// element receives the sum of all source elements whose coordinates are
/// congruent modulo the destination extents.
#[derive(Debug)]
pub struct OpRepeatBack<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpRepeatBack<'a> {
    /// Creates a repeat-back event. Divisibility and view validation occur in
    /// a guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Concatenates two F32 tensors along one logical dimension.
#[derive(Debug)]
pub struct OpConcat<'a> {
    lhs: TensorView<'a>,
    rhs: TensorView<'a>,
    dst: TensorViewMut<'a>,
    axis: usize,
}

impl<'a> OpConcat<'a> {
    /// Creates a concatenation event. Axis, shape, and view validation occur
    /// in explicit guards.
    #[must_use]
    pub const fn new(
        lhs: TensorView<'a>,
        rhs: TensorView<'a>,
        dst: TensorViewMut<'a>,
        axis: usize,
    ) -> Self {
        Self {
            lhs,
            rhs,
            dst,
            axis,
        }
    }
}

/// Explicitly reports an event outside the maintained sequence family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedSequence;

/// Result returned after a sequence event completes its RTC dispatch.
pub type SequenceResult = Result<(), SequenceError>;

type Outcome = SequenceResult;

struct CumsumRuntime<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
    result: &'a Cell<Outcome>,
}

struct RepeatRuntime<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
    result: &'a Cell<Outcome>,
}

struct RepeatBackRuntime<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
    result: &'a Cell<Outcome>,
}

struct ConcatRuntime<'a> {
    lhs: TensorView<'a>,
    rhs: TensorView<'a>,
    dst: TensorViewMut<'a>,
    axis: usize,
    result: &'a Cell<Outcome>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Outcome>,
}

#[derive(Default)]
struct Context;

sml! {
    SequenceMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Cumsum(CumsumRuntime<'dispatch>) [guard_cumsum_valid] / effect_cumsum_execute,
        "ready"_s <= "ready"_s + Cumsum(CumsumRuntime<'dispatch>) [guard_cumsum_shape_invalid] / effect_cumsum_shape_mismatch,
        "ready"_s <= "ready"_s + Cumsum(CumsumRuntime<'dispatch>) [guard_cumsum_view_invalid] / effect_cumsum_invalid_view,

        "ready"_s <= "ready"_s + Repeat(RepeatRuntime<'dispatch>) [guard_repeat_valid] / effect_repeat_execute,
        "ready"_s <= "ready"_s + Repeat(RepeatRuntime<'dispatch>) [guard_repeat_shape_invalid] / effect_repeat_shape_mismatch,
        "ready"_s <= "ready"_s + Repeat(RepeatRuntime<'dispatch>) [guard_repeat_view_invalid] / effect_repeat_invalid_view,

        "ready"_s <= "ready"_s + RepeatBack(RepeatBackRuntime<'dispatch>) [guard_repeat_back_valid] / effect_repeat_back_execute,
        "ready"_s <= "ready"_s + RepeatBack(RepeatBackRuntime<'dispatch>) [guard_repeat_back_shape_invalid] / effect_repeat_back_shape_mismatch,
        "ready"_s <= "ready"_s + RepeatBack(RepeatBackRuntime<'dispatch>) [guard_repeat_back_view_invalid] / effect_repeat_back_invalid_view,

        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_axis0_valid] / effect_concat_axis0_execute,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_axis1_valid] / effect_concat_axis1_execute,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_axis2_valid] / effect_concat_axis2_execute,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_axis3_valid] / effect_concat_axis3_execute,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_axis_invalid] / effect_concat_axis_invalid,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_shape_invalid] / effect_concat_shape_mismatch,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) [guard_concat_view_invalid] / effect_concat_invalid_view,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected_typed,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for sequence kernels.
pub struct SequenceKernel {
    machine: SequenceMachineStateMachine<Context>,
}

impl fmt::Debug for SequenceKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceKernel")
            .finish_non_exhaustive()
    }
}

impl Default for SequenceKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceKernel {
    /// Constructs an independent sequence actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: SequenceMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: SequenceEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn cumsum(&mut self, event: OpCumsum<'_>) -> Outcome {
        let result = Cell::new(Err(SequenceError::UnexpectedEvent));
        self.machine
            .process_event(SequenceMachineEvents::Cumsum(CumsumRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| SequenceError::Internal)?;
        let _ = self.machine.is(&SequenceMachineStates::Ready);
        result.get()
    }

    fn repeat(&mut self, event: OpRepeat<'_>) -> Outcome {
        let result = Cell::new(Err(SequenceError::UnexpectedEvent));
        self.machine
            .process_event(SequenceMachineEvents::Repeat(RepeatRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| SequenceError::Internal)?;
        let _ = self.machine.is(&SequenceMachineStates::Ready);
        result.get()
    }

    fn repeat_back(&mut self, event: OpRepeatBack<'_>) -> Outcome {
        let result = Cell::new(Err(SequenceError::UnexpectedEvent));
        self.machine
            .process_event(SequenceMachineEvents::RepeatBack(RepeatBackRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| SequenceError::Internal)?;
        let _ = self.machine.is(&SequenceMachineStates::Ready);
        result.get()
    }

    fn concat(&mut self, event: OpConcat<'_>) -> Outcome {
        let result = Cell::new(Err(SequenceError::UnexpectedEvent));
        self.machine
            .process_event(SequenceMachineEvents::Concat(ConcatRuntime {
                lhs: event.lhs,
                rhs: event.rhs,
                dst: event.dst,
                axis: event.axis,
                result: &result,
            }))
            .map_err(|_| SequenceError::Internal)?;
        let _ = self.machine.is(&SequenceMachineStates::Ready);
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedSequence) -> Outcome {
        let result = Cell::new(Err(SequenceError::UnexpectedEvent));
        self.machine
            .process_event(SequenceMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| SequenceError::Internal)?;
        let _ = self.machine.is(&SequenceMachineStates::Ready);
        result.get()
    }
}

/// Trait implemented by every public sequence event.
pub trait SequenceEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output;
}

impl SequenceEvent for OpCumsum<'_> {
    type Output = Outcome;

    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output {
        actor.cumsum(self)
    }
}

impl SequenceEvent for OpRepeat<'_> {
    type Output = Outcome;

    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output {
        actor.repeat(self)
    }
}

impl SequenceEvent for OpRepeatBack<'_> {
    type Output = Outcome;

    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output {
        actor.repeat_back(self)
    }
}

impl SequenceEvent for OpConcat<'_> {
    type Output = Outcome;

    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output {
        actor.concat(self)
    }
}

impl SequenceEvent for UnexpectedSequence {
    type Output = Outcome;

    fn dispatch(self, actor: &mut SequenceKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn valid_pair(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.validate().is_ok() && dst.validate().is_ok()
}

fn same_shape(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.layout().ne() == dst.layout().ne()
}

const fn repeat_shape_valid(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    let source = src.layout().ne();
    let target = dst.layout().ne();
    let mut dimension = 0;
    while dimension < 4 {
        if !target[dimension].is_multiple_of(source[dimension]) {
            return false;
        }
        dimension += 1;
    }
    true
}

const fn repeat_back_shape_valid(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    let source = src.layout().ne();
    let target = dst.layout().ne();
    let mut dimension = 0;
    while dimension < 4 {
        if !source[dimension].is_multiple_of(target[dimension]) {
            return false;
        }
        dimension += 1;
    }
    true
}

const fn concat_axis_shape_valid<const AXIS: usize>(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    dst: &TensorViewMut<'_>,
) -> bool {
    let lhs_ne = lhs.layout().ne();
    let rhs_ne = rhs.layout().ne();
    let dst_ne = dst.layout().ne();
    let mut dimension = 0;
    while dimension < 4 {
        if dimension == AXIS {
            match lhs_ne[dimension].checked_add(rhs_ne[dimension]) {
                Some(sum) if sum == dst_ne[dimension] => {}
                _ => return false,
            }
        } else if lhs_ne[dimension] != rhs_ne[dimension] || lhs_ne[dimension] != dst_ne[dimension] {
            return false;
        }
        dimension += 1;
    }
    true
}

#[allow(clippy::cast_possible_truncation)]
const fn decode_ordinal(mut ordinal: usize, layout: Layout) -> [u64; 4] {
    let extents = layout.ne();
    let mut coordinates = [0_u64; 4];
    let mut dimension = 0;
    while dimension < 4 {
        let extent = extents[dimension];
        coordinates[dimension] = (ordinal as u64) % extent;
        ordinal = (ordinal as u64 / extent) as usize;
        dimension += 1;
    }
    coordinates
}

#[allow(clippy::cast_possible_truncation)]
const fn encode_ordinal(coordinates: [u64; 4], layout: Layout) -> usize {
    let extents = layout.ne();
    let mut ordinal = 0_u64;
    let mut stride = 1_u64;
    let mut dimension = 0;
    while dimension < 4 {
        ordinal += coordinates[dimension] * stride;
        stride *= extents[dimension];
        dimension += 1;
    }
    ordinal as usize
}

fn execute_concat<const AXIS: usize>(event: &mut ConcatRuntime<'_>) {
    let lhs_layout = event.lhs.layout();
    let rhs_layout = event.rhs.layout();
    let dst_layout = event.dst.layout();
    let lhs_count = event.lhs.count();
    let rhs_count = event.rhs.count();
    let lhs_ne = lhs_layout.ne();
    let mut ordinal = 0;
    while ordinal < lhs_count {
        let coordinates = decode_ordinal(ordinal, lhs_layout);
        let target = encode_ordinal(coordinates, dst_layout);
        event.dst.write(target, event.lhs.read(ordinal));
        ordinal += 1;
    }
    ordinal = 0;
    while ordinal < rhs_count {
        let mut coordinates = decode_ordinal(ordinal, rhs_layout);
        coordinates[AXIS] += lhs_ne[AXIS];
        let target = encode_ordinal(coordinates, dst_layout);
        event.dst.write(target, event.rhs.read(ordinal));
        ordinal += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
const fn source_ordinal(ordinal: usize, source: Layout, target: Layout) -> usize {
    let mut remaining = ordinal as u64;
    let mut source_ordinal = 0u64;
    let mut source_stride = 1u64;
    let source_ne = source.ne();
    let target_ne = target.ne();
    let mut dimension = 0;
    while dimension < 4 {
        let target_coordinate = remaining % target_ne[dimension];
        remaining /= target_ne[dimension];
        source_ordinal += (target_coordinate % source_ne[dimension]) * source_stride;
        source_stride *= source_ne[dimension];
        dimension += 1;
    }
    source_ordinal as usize
}

impl SequenceMachineStateMachineContext for Context {
    fn guard_cumsum_valid(&self, event: &CumsumRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && same_shape(&event.src, &event.dst))
    }

    fn guard_cumsum_shape_invalid(&self, event: &CumsumRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && !same_shape(&event.src, &event.dst))
    }

    fn guard_cumsum_view_invalid(&self, event: &CumsumRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_pair(&event.src, &event.dst))
    }

    fn guard_repeat_valid(&self, event: &RepeatRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && repeat_shape_valid(&event.src, &event.dst))
    }

    fn guard_repeat_shape_invalid(&self, event: &RepeatRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && !repeat_shape_valid(&event.src, &event.dst))
    }

    fn guard_repeat_view_invalid(&self, event: &RepeatRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_pair(&event.src, &event.dst))
    }

    fn guard_repeat_back_valid(&self, event: &RepeatBackRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && repeat_back_shape_valid(&event.src, &event.dst))
    }

    fn guard_repeat_back_shape_invalid(&self, event: &RepeatBackRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && !repeat_back_shape_valid(&event.src, &event.dst))
    }

    fn guard_repeat_back_view_invalid(&self, event: &RepeatBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_pair(&event.src, &event.dst))
    }

    fn guard_concat_axis0_valid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis == 0
            && valid_pair3(&event.lhs, &event.rhs, &event.dst)
            && concat_axis_shape_valid::<0>(&event.lhs, &event.rhs, &event.dst))
    }

    fn guard_concat_axis1_valid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis == 1
            && valid_pair3(&event.lhs, &event.rhs, &event.dst)
            && concat_axis_shape_valid::<1>(&event.lhs, &event.rhs, &event.dst))
    }

    fn guard_concat_axis2_valid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis == 2
            && valid_pair3(&event.lhs, &event.rhs, &event.dst)
            && concat_axis_shape_valid::<2>(&event.lhs, &event.rhs, &event.dst))
    }

    fn guard_concat_axis3_valid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis == 3
            && valid_pair3(&event.lhs, &event.rhs, &event.dst)
            && concat_axis_shape_valid::<3>(&event.lhs, &event.rhs, &event.dst))
    }

    fn guard_concat_axis_invalid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis >= 4 && valid_pair3(&event.lhs, &event.rhs, &event.dst))
    }

    fn guard_concat_shape_invalid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(event.axis < 4
            && valid_pair3(&event.lhs, &event.rhs, &event.dst)
            && !concat_shape_valid_any_axis(event))
    }

    fn guard_concat_view_invalid(&self, event: &ConcatRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_pair3(&event.lhs, &event.rhs, &event.dst))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_cumsum_execute(&mut self, mut event: CumsumRuntime<'_>) -> Result<(), ()> {
        let width = event.src.layout().ne()[0] as usize;
        let rows = event.src.count() / width;
        let mut row = 0;
        while row < rows {
            let mut column = 0;
            let mut sum = 0.0;
            while column < width {
                sum += event.src.read(row * width + column);
                event.dst.write(row * width + column, sum);
                column += 1;
            }
            row += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_repeat_execute(&mut self, mut event: RepeatRuntime<'_>) -> Result<(), ()> {
        let source_layout = event.src.layout();
        let target_layout = event.dst.layout();
        let mut ordinal = 0;
        while ordinal < event.dst.count() {
            let source = source_ordinal(ordinal, source_layout, target_layout);
            event.dst.write(ordinal, event.src.read(source));
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_cumsum_shape_mismatch(&mut self, event: CumsumRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::ShapeMismatch));
        Ok(())
    }

    fn effect_cumsum_invalid_view(&mut self, event: CumsumRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::InvalidView));
        Ok(())
    }

    fn effect_repeat_shape_mismatch(&mut self, event: RepeatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::ShapeMismatch));
        Ok(())
    }

    fn effect_repeat_invalid_view(&mut self, event: RepeatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::InvalidView));
        Ok(())
    }

    fn effect_repeat_back_execute(&mut self, mut event: RepeatBackRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.dst.count() {
            event.dst.write(ordinal, 0.0);
            ordinal += 1;
        }
        ordinal = 0;
        let source_layout = event.src.layout();
        let target_layout = event.dst.layout();
        while ordinal < event.src.count() {
            let mut coordinates = decode_ordinal(ordinal, source_layout);
            let target_extents = target_layout.ne();
            let mut dimension = 0;
            while dimension < 4 {
                coordinates[dimension] %= target_extents[dimension];
                dimension += 1;
            }
            let target = encode_ordinal(coordinates, target_layout);
            event
                .dst
                .write(target, event.dst.read(target) + event.src.read(ordinal));
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_repeat_back_shape_mismatch(
        &mut self,
        event: RepeatBackRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(SequenceError::ShapeMismatch));
        Ok(())
    }

    fn effect_repeat_back_invalid_view(&mut self, event: RepeatBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::InvalidView));
        Ok(())
    }

    fn effect_concat_axis0_execute(&mut self, mut event: ConcatRuntime<'_>) -> Result<(), ()> {
        execute_concat::<0>(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_concat_axis1_execute(&mut self, mut event: ConcatRuntime<'_>) -> Result<(), ()> {
        execute_concat::<1>(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_concat_axis2_execute(&mut self, mut event: ConcatRuntime<'_>) -> Result<(), ()> {
        execute_concat::<2>(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_concat_axis3_execute(&mut self, mut event: ConcatRuntime<'_>) -> Result<(), ()> {
        execute_concat::<3>(&mut event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_concat_axis_invalid(&mut self, event: ConcatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::InvalidAxis));
        Ok(())
    }

    fn effect_concat_shape_mismatch(&mut self, event: ConcatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::ShapeMismatch));
        Ok(())
    }

    fn effect_concat_invalid_view(&mut self, event: ConcatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::InvalidView));
        Ok(())
    }

    fn effect_unexpected_typed(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SequenceError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

fn valid_pair3(lhs: &TensorView<'_>, rhs: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    lhs.validate().is_ok() && rhs.validate().is_ok() && dst.validate().is_ok()
}

const fn concat_shape_valid_any_axis(event: &ConcatRuntime<'_>) -> bool {
    match event.axis {
        0 => concat_axis_shape_valid::<0>(&event.lhs, &event.rhs, &event.dst),
        1 => concat_axis_shape_valid::<1>(&event.lhs, &event.rhs, &event.dst),
        2 => concat_axis_shape_valid::<2>(&event.lhs, &event.rhs, &event.dst),
        3 => concat_axis_shape_valid::<3>(&event.lhs, &event.rhs, &event.dst),
        _ => false,
    }
}
