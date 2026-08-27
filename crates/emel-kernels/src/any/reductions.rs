//! Portable unary and reduction kernels over validated F32 tensor views.
//!
//! This is the maintained scalar slice for the pinned kernel contract. Each
//! public event has its own SML transition row and actor entry point. Guards
//! classify view and shape errors before actions run; actions contain only the
//! selected, bounded numeric loops. Dispatch is synchronous and allocation
//! free.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by unary and reduction dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// The output shape is incompatible with the selected operation.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ReductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid reduction tensor view"),
            Self::ShapeMismatch => formatter.write_str("reduction tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected reduction event"),
            Self::Internal => formatter.write_str("internal reduction dispatch error"),
        }
    }
}

impl std::error::Error for ReductionError {}

/// Applies `ln` elementwise.
#[derive(Debug)]
pub struct OpLog<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpLog<'a> {
    /// Creates an event; view validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Applies `sin` elementwise.
#[derive(Debug)]
pub struct OpSin<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpSin<'a> {
    /// Creates an event; view validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Applies `cos` elementwise.
#[derive(Debug)]
pub struct OpCos<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpCos<'a> {
    /// Creates an event; view validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Reduces every element to one F32 output value.
#[derive(Debug)]
pub struct OpSum<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpSum<'a> {
    /// Creates an event; scalar output validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Reduces each logical dimension-zero row to one F32 value.
#[derive(Debug)]
pub struct OpSumRows<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpSumRows<'a> {
    /// Creates an event; row-output validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Reduces every element to one F32 mean value.
#[derive(Debug)]
pub struct OpMean<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpMean<'a> {
    /// Creates an event; scalar output validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Writes the first maximal logical index as an F32 scalar.
#[derive(Debug)]
pub struct OpArgmax<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpArgmax<'a> {
    /// Creates an event; scalar output validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Applies numerically stable softmax independently to each row.
#[derive(Debug)]
pub struct OpSoftMax<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpSoftMax<'a> {
    /// Creates an event; row and shape validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Result returned by every composed reduction event.
pub type ReductionResult = Result<(), ReductionError>;

struct UnaryRuntime<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
    result: &'a Cell<ReductionResult>,
}

struct ReduceRuntime<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
    result: &'a Cell<ReductionResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ReductionMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Log(UnaryRuntime<'dispatch>) [guard_unary_valid] / effect_log_execute,
        "ready"_s <= "ready"_s + Log(UnaryRuntime<'dispatch>) [guard_unary_shape_invalid] / effect_unary_shape_invalid,
        "ready"_s <= "ready"_s + Log(UnaryRuntime<'dispatch>) [guard_unary_view_invalid] / effect_unary_view_invalid,

        "ready"_s <= "ready"_s + Sin(UnaryRuntime<'dispatch>) [guard_unary_valid] / effect_sin_execute,
        "ready"_s <= "ready"_s + Sin(UnaryRuntime<'dispatch>) [guard_unary_shape_invalid] / effect_unary_shape_invalid,
        "ready"_s <= "ready"_s + Sin(UnaryRuntime<'dispatch>) [guard_unary_view_invalid] / effect_unary_view_invalid,

        "ready"_s <= "ready"_s + Cos(UnaryRuntime<'dispatch>) [guard_unary_valid] / effect_cos_execute,
        "ready"_s <= "ready"_s + Cos(UnaryRuntime<'dispatch>) [guard_unary_shape_invalid] / effect_unary_shape_invalid,
        "ready"_s <= "ready"_s + Cos(UnaryRuntime<'dispatch>) [guard_unary_view_invalid] / effect_unary_view_invalid,

        "ready"_s <= "ready"_s + Sum(ReduceRuntime<'dispatch>) [guard_scalar_valid] / effect_sum_execute,
        "ready"_s <= "ready"_s + Sum(ReduceRuntime<'dispatch>) [guard_scalar_shape_invalid] / effect_shape_mismatch,
        "ready"_s <= "ready"_s + Sum(ReduceRuntime<'dispatch>) [guard_scalar_view_invalid] / effect_invalid_view,

        "ready"_s <= "ready"_s + SumRows(ReduceRuntime<'dispatch>) [guard_rows_valid] / effect_sum_rows_execute,
        "ready"_s <= "ready"_s + SumRows(ReduceRuntime<'dispatch>) [guard_rows_shape_invalid] / effect_shape_mismatch,
        "ready"_s <= "ready"_s + SumRows(ReduceRuntime<'dispatch>) [guard_rows_view_invalid] / effect_invalid_view,

        "ready"_s <= "ready"_s + Mean(ReduceRuntime<'dispatch>) [guard_scalar_valid] / effect_mean_execute,
        "ready"_s <= "ready"_s + Mean(ReduceRuntime<'dispatch>) [guard_scalar_shape_invalid] / effect_shape_mismatch,
        "ready"_s <= "ready"_s + Mean(ReduceRuntime<'dispatch>) [guard_scalar_view_invalid] / effect_invalid_view,

        "ready"_s <= "ready"_s + Argmax(ReduceRuntime<'dispatch>) [guard_scalar_valid] / effect_argmax_execute,
        "ready"_s <= "ready"_s + Argmax(ReduceRuntime<'dispatch>) [guard_scalar_shape_invalid] / effect_shape_mismatch,
        "ready"_s <= "ready"_s + Argmax(ReduceRuntime<'dispatch>) [guard_scalar_view_invalid] / effect_invalid_view,

        "ready"_s <= "ready"_s + SoftMax(ReduceRuntime<'dispatch>) [guard_soft_max_dense_valid] / effect_soft_max_dense_execute,
        "ready"_s <= "ready"_s + SoftMax(ReduceRuntime<'dispatch>) [guard_soft_max_sparse_valid] / effect_soft_max_sparse_execute,
        "ready"_s <= "ready"_s + SoftMax(ReduceRuntime<'dispatch>) [guard_soft_max_shape_invalid] / effect_shape_mismatch,
        "ready"_s <= "ready"_s + SoftMax(ReduceRuntime<'dispatch>) [guard_soft_max_view_invalid] / effect_invalid_view,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for reductions and unary kernels.
pub struct ReductionKernel {
    machine: ReductionMachineStateMachine<Context>,
}

impl fmt::Debug for ReductionKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReductionKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ReductionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ReductionKernel {
    /// Constructs an independent reduction actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ReductionMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated reduction machine does not return to `Ready`.
    pub fn process_event<E: ReductionEvent>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(
            self.machine.is(&ReductionMachineStates::Ready),
            "reduction machine must return to ready after dispatch"
        );
        result
    }

    /// Reports whether the generated reduction machine is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&ReductionMachineStates::Ready)
    }

    fn log(&mut self, event: OpLog<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Log(UnaryRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn sin(&mut self, event: OpSin<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Sin(UnaryRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn cos(&mut self, event: OpCos<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Cos(UnaryRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn sum(&mut self, event: OpSum<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Sum(ReduceRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn sum_rows(&mut self, event: OpSumRows<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::SumRows(ReduceRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn mean(&mut self, event: OpMean<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Mean(ReduceRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn argmax(&mut self, event: OpArgmax<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::Argmax(ReduceRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }

    fn soft_max(&mut self, event: OpSoftMax<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(ReductionMachineEvents::SoftMax(ReduceRuntime {
                src: event.src,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        assert!(self.is_ready());
        result.get()
    }
}

/// Trait implemented by every public reduction event.
pub trait ReductionEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output;
}

impl ReductionEvent for OpLog<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.log(self)
    }
}

impl ReductionEvent for OpSin<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.sin(self)
    }
}

impl ReductionEvent for OpCos<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.cos(self)
    }
}

impl ReductionEvent for OpSum<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.sum(self)
    }
}

impl ReductionEvent for OpSumRows<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.sum_rows(self)
    }
}

impl ReductionEvent for OpMean<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.mean(self)
    }
}

impl ReductionEvent for OpArgmax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.argmax(self)
    }
}

impl ReductionEvent for OpSoftMax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut ReductionKernel) -> Self::Output {
        actor.soft_max(self)
    }
}

fn unary_views_valid(event: &UnaryRuntime<'_>) -> bool {
    event.src.validate().is_ok() && event.dst.validate().is_ok()
}

fn reduction_views_valid(event: &ReduceRuntime<'_>) -> bool {
    event.src.validate().is_ok() && event.dst.validate().is_ok()
}

fn scalar_output(event: &ReduceRuntime<'_>) -> bool {
    event.dst.layout().ne() == [1, 1, 1, 1]
}

fn rows_output(event: &ReduceRuntime<'_>) -> bool {
    let source = event.src.layout().ne();
    event.dst.layout().ne() == [1, source[1], source[2], source[3]]
}

const fn same_count(event: &UnaryRuntime<'_>) -> bool {
    event.src.count() == event.dst.count()
}

const fn soft_max_same_count(event: &ReduceRuntime<'_>) -> bool {
    event.src.count() == event.dst.count()
}

fn soft_max_shape_valid(event: &ReduceRuntime<'_>) -> bool {
    let source = event.src.layout().ne();
    let Ok(width) = usize::try_from(source[0]) else {
        return false;
    };
    let source_count = event.src.count();
    width > 0
        && source_count > 0
        && source_count.is_multiple_of(width)
        && soft_max_same_count(event)
}

fn soft_max_is_dense(event: &ReduceRuntime<'_>) -> bool {
    soft_max_dense_layout(event.src.layout()) && soft_max_dense_layout(event.dst.layout())
}

fn soft_max_dense_layout(layout: super::tensor_view::Layout) -> bool {
    if layout.nb()[0] == 0 {
        layout.effective_f32_strides().is_some()
    } else {
        layout.is_dense_contiguous()
    }
}

#[allow(clippy::cast_possible_truncation)]
const fn row_width(event: &ReduceRuntime<'_>) -> usize {
    event.src.layout().ne()[0] as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
const fn index_as_f32(index: usize) -> f32 {
    index as f32
}

impl ReductionMachineStateMachineContext for Context {
    fn guard_unary_valid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(unary_views_valid(event) && same_count(event))
    }

    fn guard_unary_shape_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(unary_views_valid(event) && !same_count(event))
    }

    fn guard_unary_view_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!unary_views_valid(event))
    }

    fn guard_scalar_valid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && scalar_output(event))
    }

    fn guard_scalar_shape_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && !scalar_output(event))
    }

    fn guard_scalar_view_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(!reduction_views_valid(event))
    }

    fn guard_rows_valid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && rows_output(event))
    }

    fn guard_rows_shape_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && !rows_output(event))
    }

    fn guard_rows_view_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(!reduction_views_valid(event))
    }

    fn guard_soft_max_dense_valid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && soft_max_shape_valid(event) && soft_max_is_dense(event))
    }

    fn guard_soft_max_sparse_valid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(
            reduction_views_valid(event)
                && soft_max_shape_valid(event)
                && !soft_max_is_dense(event),
        )
    }

    fn guard_soft_max_shape_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(reduction_views_valid(event) && !soft_max_shape_valid(event))
    }

    fn guard_soft_max_view_invalid(&self, event: &ReduceRuntime<'_>) -> Result<bool, ()> {
        Ok(!reduction_views_valid(event))
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn effect_soft_max_dense_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let width = row_width(&event);
        let rows = event.src.count() / width;
        let mut row = 0;
        while row < rows {
            let offset = row * width;
            let mut column = 1;
            let mut max_value = event.src.read(offset);
            while column < width {
                let value = event.src.read(offset + column);
                if value > max_value {
                    max_value = value;
                }
                column += 1;
            }

            column = 0;
            let mut sum = 0.0_f64;
            while column < width {
                let value = (event.src.read(offset + column) - max_value).exp();
                event.dst.write(offset + column, value);
                sum += f64::from(value);
                column += 1;
            }

            let denominator = sum as f32;
            column = 0;
            while column < width {
                let value = event.dst.read(offset + column);
                event.dst.write(offset + column, value / denominator);
                column += 1;
            }
            row += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_soft_max_sparse_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let width = row_width(&event);
        let rows = event.src.count() / width;
        let mut row = 0;
        while row < rows {
            let offset = row * width;
            let mut column = 1;
            let mut max_value = event.src.read(offset);
            while column < width {
                let value = event.src.read(offset + column);
                if value > max_value {
                    max_value = value;
                }
                column += 1;
            }

            column = 0;
            let mut sum = 0.0_f32;
            while column < width {
                let value = (event.src.read(offset + column) - max_value).exp();
                event.dst.write(offset + column, value);
                sum += value;
                column += 1;
            }

            column = 0;
            while column < width {
                let value = event.dst.read(offset + column);
                event.dst.write(offset + column, value / sum);
                column += 1;
            }
            row += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_log_execute(&mut self, mut event: UnaryRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.src.count() {
            event.dst.write(ordinal, event.src.read(ordinal).ln());
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sin_execute(&mut self, mut event: UnaryRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.src.count() {
            event.dst.write(ordinal, event.src.read(ordinal).sin());
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_cos_execute(&mut self, mut event: UnaryRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.src.count() {
            event.dst.write(ordinal, event.src.read(ordinal).cos());
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn effect_sum_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        let mut sum = 0.0_f64;
        while ordinal < event.src.count() {
            sum += f64::from(event.src.read(ordinal));
            ordinal += 1;
        }
        event.dst.write(0, sum as f32);
        event.result.set(Ok(()));
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn effect_sum_rows_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let width = row_width(&event);
        let rows = event.src.count() / width;
        let mut row = 0;
        while row < rows {
            let mut column = 0;
            let mut sum = 0.0_f64;
            while column < width {
                sum += f64::from(event.src.read(row * width + column));
                column += 1;
            }
            event.dst.write(row, sum as f32);
            row += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn effect_mean_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        let mut sum = 0.0_f64;
        while ordinal < event.src.count() {
            sum += f64::from(event.src.read(ordinal));
            ordinal += 1;
        }
        event.dst.write(0, (sum / event.src.count() as f64) as f32);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_argmax_execute(&mut self, mut event: ReduceRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 1;
        let mut best_index = 0;
        let mut best_value = event.src.read(0);
        while ordinal < event.src.count() {
            let value = event.src.read(ordinal);
            if value > best_value {
                best_value = value;
                best_index = ordinal;
            }
            ordinal += 1;
        }
        event.dst.write(0, index_as_f32(best_index));
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_unary_shape_invalid(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReductionError::ShapeMismatch));
        Ok(())
    }

    fn effect_unary_view_invalid(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReductionError::InvalidView));
        Ok(())
    }

    fn effect_invalid_view(&mut self, event: ReduceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReductionError::InvalidView));
        Ok(())
    }

    fn effect_shape_mismatch(&mut self, event: ReduceRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReductionError::ShapeMismatch));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
