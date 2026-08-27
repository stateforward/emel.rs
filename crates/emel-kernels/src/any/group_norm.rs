//! Portable group and L2 normalization operations.
//!
//! The pinned public kernel declares `op_group_norm` and `op_l2_norm` in
//! `src/emel/kernel/detail.hpp` and routes them in the architecture SML, but
//! the pinned tree contains no executable guard or action definitions for
//! either operation.  This module therefore records the portable operator
//! contract without claiming live public-kernel parity.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by group and L2 normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupNormError {
    /// A view failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Input and output have different logical shapes.
    ShapeMismatch,
    /// Groups or epsilon are not valid for the input shape.
    InvalidParameters,
    /// The L2 norm is empty or not positive.
    ZeroNorm,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for GroupNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid group normalization tensor view"),
            Self::ShapeMismatch => formatter.write_str("group normalization tensor shapes differ"),
            Self::InvalidParameters => {
                formatter.write_str("invalid group normalization parameters")
            }
            Self::ZeroNorm => formatter.write_str("L2 norm is empty or non-positive"),
            Self::UnexpectedEvent => formatter.write_str("unexpected group normalization event"),
            Self::Internal => formatter.write_str("internal group normalization dispatch error"),
        }
    }
}

impl std::error::Error for GroupNormError {}

/// Result returned after group or L2 normalization dispatch.
pub type GroupNormResult = Result<(), GroupNormError>;

/// Normalizes each channel group across spatial and channel-group elements.
#[derive(Debug)]
pub struct OpGroupNorm<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    groups: usize,
    epsilon: f32,
}

impl<'a> OpGroupNorm<'a> {
    /// Creates a group-normalization request. Groups partition dimension two.
    #[must_use]
    pub const fn new(
        input: TensorView<'a>,
        output: TensorViewMut<'a>,
        groups: usize,
        epsilon: f32,
    ) -> Self {
        Self {
            input,
            output,
            groups,
            epsilon,
        }
    }
}

/// Normalizes all logical elements by their L2 norm.
#[derive(Debug)]
pub struct OpL2Norm<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
}

impl<'a> OpL2Norm<'a> {
    /// Creates an L2-normalization request.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>) -> Self {
        Self { input, output }
    }
}

#[derive(Debug)]
pub struct GroupNormRuntime<'a> {
    event: OpGroupNorm<'a>,
    result: &'a Cell<GroupNormResult>,
}

#[derive(Debug)]
pub struct L2NormRuntime<'a> {
    event: OpL2Norm<'a>,
    result: &'a Cell<GroupNormResult>,
}

#[derive(Default)]
struct Context;

sml! {
    GroupNormMachine<'dispatch> {
        "ready"_s <= *"ready"_s + GroupNorm(GroupNormRuntime<'dispatch>)
            [guard_group_norm_valid] / effect_group_norm_execute,
        "ready"_s <= "ready"_s + GroupNorm(GroupNormRuntime<'dispatch>)
            [guard_group_norm_shape_mismatch] / effect_group_norm_shape_reject,
        "ready"_s <= "ready"_s + GroupNorm(GroupNormRuntime<'dispatch>)
            [guard_group_norm_parameters_invalid] / effect_group_norm_parameters_reject,
        "ready"_s <= "ready"_s + GroupNorm(GroupNormRuntime<'dispatch>)
            [guard_group_norm_invalid_view] / effect_group_norm_view_reject,

        "ready"_s <= "ready"_s + L2Norm(L2NormRuntime<'dispatch>)
            [guard_l2_norm_valid] / effect_l2_norm_execute,
        "ready"_s <= "ready"_s + L2Norm(L2NormRuntime<'dispatch>)
            [guard_l2_norm_shape_mismatch] / effect_l2_norm_shape_reject,
        "ready"_s <= "ready"_s + L2Norm(L2NormRuntime<'dispatch>)
            [guard_l2_norm_zero] / effect_l2_norm_zero_reject,
        "ready"_s <= "ready"_s + L2Norm(L2NormRuntime<'dispatch>)
            [guard_l2_norm_invalid_view] / effect_l2_norm_view_reject,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for group and L2 normalization.
pub struct GroupNormKernel {
    machine: GroupNormMachineStateMachine<Context>,
}

impl fmt::Debug for GroupNormKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroupNormKernel")
            .finish_non_exhaustive()
    }
}

impl Default for GroupNormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupNormKernel {
    /// Constructs an independent normalization actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: GroupNormMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: GroupNormEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn group_norm(&mut self, event: OpGroupNorm<'_>) -> GroupNormResult {
        let result = Cell::new(Err(GroupNormError::UnexpectedEvent));
        self.machine
            .process_event(GroupNormMachineEvents::GroupNorm(GroupNormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GroupNormError::Internal)?;
        result.get()
    }

    fn l2_norm(&mut self, event: OpL2Norm<'_>) -> GroupNormResult {
        let result = Cell::new(Err(GroupNormError::UnexpectedEvent));
        self.machine
            .process_event(GroupNormMachineEvents::L2Norm(L2NormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GroupNormError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`GroupNormKernel`].
pub trait GroupNormEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut GroupNormKernel) -> Self::Output;
}

impl GroupNormEvent for OpGroupNorm<'_> {
    type Output = GroupNormResult;
    fn dispatch(self, actor: &mut GroupNormKernel) -> Self::Output {
        actor.group_norm(self)
    }
}

impl GroupNormEvent for OpL2Norm<'_> {
    type Output = GroupNormResult;
    fn dispatch(self, actor: &mut GroupNormKernel) -> Self::Output {
        actor.l2_norm(self)
    }
}

fn views_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    super::normalization::normalization_view_valid(input.layout(), input.storage_len())
        && super::normalization::normalization_view_valid(output.layout(), output.storage_len())
}

fn shape_valid(input: &TensorView<'_>, groups: usize) -> bool {
    let shape = input.layout().ne();
    groups != 0
        && shape[2] != 0
        && usize::try_from(shape[2]).is_ok_and(|channels| channels % groups == 0)
}

fn group_norm_valid(event: &OpGroupNorm<'_>) -> bool {
    views_valid(&event.input, &event.output)
        && event.input.layout().ne() == event.output.layout().ne()
        && shape_valid(&event.input, event.groups)
        && event.epsilon.is_finite()
        && event.epsilon >= 0.0
}

fn group_norm_shape_mismatch(event: &OpGroupNorm<'_>) -> bool {
    views_valid(&event.input, &event.output)
        && event.input.layout().ne() != event.output.layout().ne()
}

fn group_norm_parameters_invalid(event: &OpGroupNorm<'_>) -> bool {
    views_valid(&event.input, &event.output)
        && event.input.layout().ne() == event.output.layout().ne()
        && (!shape_valid(&event.input, event.groups)
            || !event.epsilon.is_finite()
            || event.epsilon < 0.0)
}

#[allow(clippy::cast_possible_truncation)]
const fn index(shape: [u64; 4], i0: usize, i1: usize, i2: usize, i3: usize) -> usize {
    i0 + (shape[0] as usize) * (i1 + (shape[1] as usize) * (i2 + (shape[2] as usize) * i3))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[allow(clippy::suboptimal_flops)]
fn run_group_norm(event: &mut OpGroupNorm<'_>) {
    let shape = event.input.layout().ne();
    let width = shape[0] as usize;
    let height = shape[1] as usize;
    let channels = shape[2] as usize;
    let batches = shape[3] as usize;
    let channels_per_group = channels / event.groups;
    let count = width * height * channels_per_group;
    for batch in 0..batches {
        for group in 0..event.groups {
            let first_channel = group * channels_per_group;
            let mut sum = 0.0_f64;
            for channel in first_channel..first_channel + channels_per_group {
                for row in 0..height {
                    for column in 0..width {
                        sum +=
                            f64::from(event.input.read(index(shape, column, row, channel, batch)));
                    }
                }
            }
            let mean = (sum / count as f64) as f32;
            let mut variance_sum = 0.0_f64;
            for channel in first_channel..first_channel + channels_per_group {
                for row in 0..height {
                    for column in 0..width {
                        let centered =
                            event.input.read(index(shape, column, row, channel, batch)) - mean;
                        variance_sum += f64::from(centered) * f64::from(centered);
                    }
                }
            }
            let scale = 1.0 / ((variance_sum / count as f64) as f32 + event.epsilon).sqrt();
            for channel in first_channel..first_channel + channels_per_group {
                for row in 0..height {
                    for column in 0..width {
                        let ordinal = index(shape, column, row, channel, batch);
                        event
                            .output
                            .write(ordinal, (event.input.read(ordinal) - mean) * scale);
                    }
                }
            }
        }
    }
}

fn l2_shape_valid(event: &OpL2Norm<'_>) -> bool {
    views_valid(&event.input, &event.output)
        && event.input.layout().ne() == event.output.layout().ne()
}

#[allow(clippy::suboptimal_flops)]
fn l2_norm_positive(event: &OpL2Norm<'_>) -> bool {
    if !l2_shape_valid(event) {
        return false;
    }
    let mut sum = 0.0_f32;
    for ordinal in 0..event.input.count() {
        sum += event.input.read(ordinal) * event.input.read(ordinal);
    }
    sum > 0.0
}

#[allow(clippy::suboptimal_flops)]
fn run_l2_norm(event: &mut OpL2Norm<'_>) {
    let mut sum = 0.0_f32;
    for ordinal in 0..event.input.count() {
        sum += event.input.read(ordinal) * event.input.read(ordinal);
    }
    let scale = 1.0 / sum.sqrt();
    for ordinal in 0..event.input.count() {
        event
            .output
            .write(ordinal, event.input.read(ordinal) * scale);
    }
}

impl GroupNormMachineStateMachineContext for Context {
    fn guard_group_norm_valid(&self, event: &GroupNormRuntime<'_>) -> Result<bool, ()> {
        Ok(group_norm_valid(&event.event))
    }
    fn guard_group_norm_shape_mismatch(&self, event: &GroupNormRuntime<'_>) -> Result<bool, ()> {
        Ok(group_norm_shape_mismatch(&event.event))
    }
    fn guard_group_norm_parameters_invalid(
        &self,
        event: &GroupNormRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(group_norm_parameters_invalid(&event.event))
    }
    fn guard_group_norm_invalid_view(&self, event: &GroupNormRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.input, &event.event.output))
    }
    fn effect_group_norm_execute(&mut self, mut event: GroupNormRuntime<'_>) -> Result<(), ()> {
        run_group_norm(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_group_norm_shape_reject(&mut self, event: GroupNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::ShapeMismatch));
        Ok(())
    }
    fn effect_group_norm_parameters_reject(
        &mut self,
        event: GroupNormRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::InvalidParameters));
        Ok(())
    }
    fn effect_group_norm_view_reject(&mut self, event: GroupNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::InvalidView));
        Ok(())
    }
    fn guard_l2_norm_valid(&self, event: &L2NormRuntime<'_>) -> Result<bool, ()> {
        Ok(l2_norm_positive(&event.event))
    }
    fn guard_l2_norm_shape_mismatch(&self, event: &L2NormRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && event.event.input.layout().ne() != event.event.output.layout().ne())
    }
    fn guard_l2_norm_zero(&self, event: &L2NormRuntime<'_>) -> Result<bool, ()> {
        Ok(l2_shape_valid(&event.event) && !l2_norm_positive(&event.event))
    }
    fn guard_l2_norm_invalid_view(&self, event: &L2NormRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.input, &event.event.output))
    }
    fn effect_l2_norm_execute(&mut self, mut event: L2NormRuntime<'_>) -> Result<(), ()> {
        run_l2_norm(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_l2_norm_shape_reject(&mut self, event: L2NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::ShapeMismatch));
        Ok(())
    }
    fn effect_l2_norm_zero_reject(&mut self, event: L2NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::ZeroNorm));
        Ok(())
    }
    fn effect_l2_norm_view_reject(&mut self, event: L2NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GroupNormError::InvalidView));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
