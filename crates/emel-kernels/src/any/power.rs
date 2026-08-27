//! Explicit portable F32 square and square-root operations.
//!
//! The pinned scalar reference uses the unary shape predicate for `op_sqr`
//! and `op_sqrt`, then applies `v * v` or `sqrt(v)` over the logical tensor
//! elements. The two formulas remain separate SML actions.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by square and square-root dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowerError {
    /// A source or destination view failed validation.
    InvalidView,
    /// Source and destination logical shapes differ.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for PowerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid power-operation tensor view"),
            Self::ShapeMismatch => formatter.write_str("power-operation tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected power-operation event"),
            Self::Internal => formatter.write_str("internal power-operation dispatch error"),
        }
    }
}

impl std::error::Error for PowerError {}

/// Result returned after a power operation reaches run-to-completion.
pub type PowerResult = Result<(), PowerError>;

/// Squares each F32 element.
#[derive(Debug)]
pub struct OpSqr<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
}

impl<'a> OpSqr<'a> {
    /// Creates a square event; view and shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>) -> Self {
        Self { input, output }
    }
}

/// Computes the non-negative square root element-wise.
#[derive(Debug)]
pub struct OpSqrt<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
}

impl<'a> OpSqrt<'a> {
    /// Creates a square-root event; view and shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>) -> Self {
        Self { input, output }
    }
}

/// Explicitly reports an event outside the maintained power family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedPower;

#[derive(Clone, Copy)]
struct SqrRuntime<'dispatch, 'data> {
    input: TensorView<'data>,
    output: &'dispatch RefCell<TensorViewMut<'data>>,
    result: &'dispatch Cell<PowerResult>,
}

#[derive(Clone, Copy)]
struct SqrtRuntime<'dispatch, 'data> {
    input: TensorView<'data>,
    output: &'dispatch RefCell<TensorViewMut<'data>>,
    result: &'dispatch Cell<PowerResult>,
}

#[derive(Clone, Copy)]
struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<PowerResult>,
}

#[derive(Default)]
struct Context;

sml! {
    PowerMachine<'dispatch, 'data>
    where
        'data: 'dispatch,
    {
        "sqr_view_decision"_s <= *"ready"_s + Sqr(SqrRuntime<'dispatch, 'data>),
        "sqr_shape_decision"_s <= "sqr_view_decision"_s
            + completion<Sqr>(SqrRuntime<'dispatch, 'data>) [guard_sqr_views_valid],
        "ready"_s <= "sqr_view_decision"_s + completion<Sqr>(SqrRuntime<'dispatch, 'data>)
            [guard_sqr_views_invalid] / effect_sqr_view_reject,
        "sqr_execution"_s <= "sqr_shape_decision"_s
            + completion<Sqr>(SqrRuntime<'dispatch, 'data>) [guard_sqr_shape_valid],
        "ready"_s <= "sqr_shape_decision"_s + completion<Sqr>(SqrRuntime<'dispatch, 'data>)
            [guard_sqr_shape_invalid] / effect_sqr_shape_reject,
        "ready"_s <= "sqr_execution"_s + completion<Sqr>(SqrRuntime<'dispatch, 'data>)
            / effect_sqr_execute,

        "sqrt_view_decision"_s <= "ready"_s + Sqrt(SqrtRuntime<'dispatch, 'data>),
        "sqrt_shape_decision"_s <= "sqrt_view_decision"_s
            + completion<Sqrt>(SqrtRuntime<'dispatch, 'data>) [guard_sqrt_views_valid],
        "ready"_s <= "sqrt_view_decision"_s + completion<Sqrt>(SqrtRuntime<'dispatch, 'data>)
            [guard_sqrt_views_invalid] / effect_sqrt_view_reject,
        "sqrt_execution"_s <= "sqrt_shape_decision"_s
            + completion<Sqrt>(SqrtRuntime<'dispatch, 'data>) [guard_sqrt_shape_valid],
        "ready"_s <= "sqrt_shape_decision"_s + completion<Sqrt>(SqrtRuntime<'dispatch, 'data>)
            [guard_sqrt_shape_invalid] / effect_sqrt_shape_reject,
        "ready"_s <= "sqrt_execution"_s + completion<Sqrt>(SqrtRuntime<'dispatch, 'data>)
            / effect_sqrt_execute,

        "unexpected_dispatch"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_generic_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "sqr_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sqr_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sqr_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sqrt_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sqrt_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sqrt_execution"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion actor for square and square root.
pub struct PowerKernel {
    machine: PowerMachineStateMachine<Context>,
}

impl fmt::Debug for PowerKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PowerKernel")
            .finish_non_exhaustive()
    }
}

impl Default for PowerKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerKernel {
    /// Constructs an independent power actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: PowerMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated power machine does not return to `Ready` after
    /// the run-to-completion dispatch.
    pub fn process_event<E: PowerEvent>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(
            self.machine.is(&PowerMachineStates::Ready),
            "power machine must return to ready after dispatch"
        );
        result
    }

    /// Reports whether the power machine is ready for another dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&PowerMachineStates::Ready)
    }

    fn sqr(&mut self, event: OpSqr<'_>) -> PowerResult {
        let output = RefCell::new(event.output);
        let result = Cell::new(Err(PowerError::UnexpectedEvent));
        self.machine
            .process_event(PowerMachineEvents::Sqr(SqrRuntime {
                input: event.input,
                output: &output,
                result: &result,
            }))
            .map_err(|_| PowerError::Internal)?;
        result.get()
    }

    fn sqrt(&mut self, event: OpSqrt<'_>) -> PowerResult {
        let output = RefCell::new(event.output);
        let result = Cell::new(Err(PowerError::UnexpectedEvent));
        self.machine
            .process_event(PowerMachineEvents::Sqrt(SqrtRuntime {
                input: event.input,
                output: &output,
                result: &result,
            }))
            .map_err(|_| PowerError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedPower) -> PowerResult {
        let result = Cell::new(Err(PowerError::Internal));
        self.machine
            .process_event(PowerMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| PowerError::Internal)?;
        result.get()
    }
}

/// Events accepted by [`PowerKernel`].
pub trait PowerEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut PowerKernel) -> Self::Output;
}

impl PowerEvent for OpSqr<'_> {
    type Output = PowerResult;

    fn dispatch(self, actor: &mut PowerKernel) -> Self::Output {
        actor.sqr(self)
    }
}

impl PowerEvent for OpSqrt<'_> {
    type Output = PowerResult;

    fn dispatch(self, actor: &mut PowerKernel) -> Self::Output {
        actor.sqrt(self)
    }
}

impl PowerEvent for UnexpectedPower {
    type Output = PowerResult;

    fn dispatch(self, actor: &mut PowerKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn views_valid(input: &TensorView<'_>, output: &RefCell<TensorViewMut<'_>>) -> bool {
    input.validate().is_ok() && output.borrow().validate().is_ok()
}

fn shape_valid(input: &TensorView<'_>, output: &RefCell<TensorViewMut<'_>>) -> bool {
    input.count() == output.borrow().count()
}

impl PowerMachineStateMachineContext for Context {
    fn guard_sqr_views_valid<'dispatch, 'data>(
        &self,
        event: &SqrRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(views_valid(&event.input, event.output))
    }

    fn guard_sqr_views_invalid<'dispatch, 'data>(
        &self,
        event: &SqrRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(!views_valid(&event.input, event.output))
    }

    fn guard_sqr_shape_valid<'dispatch, 'data>(
        &self,
        event: &SqrRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(shape_valid(&event.input, event.output))
    }

    fn guard_sqr_shape_invalid<'dispatch, 'data>(
        &self,
        event: &SqrRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(!shape_valid(&event.input, event.output))
    }

    fn effect_sqr_execute<'dispatch, 'data>(
        &mut self,
        event: SqrRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        let mut output = event.output.borrow_mut();
        let mut ordinal = 0;
        while ordinal < event.input.count() {
            output.write(
                ordinal,
                event.input.read(ordinal) * event.input.read(ordinal),
            );
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sqr_shape_reject<'dispatch, 'data>(
        &mut self,
        event: SqrRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(PowerError::ShapeMismatch));
        Ok(())
    }

    fn effect_sqr_view_reject<'dispatch, 'data>(
        &mut self,
        event: SqrRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(PowerError::InvalidView));
        Ok(())
    }

    fn guard_sqrt_views_valid<'dispatch, 'data>(
        &self,
        event: &SqrtRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(views_valid(&event.input, event.output))
    }

    fn guard_sqrt_views_invalid<'dispatch, 'data>(
        &self,
        event: &SqrtRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(!views_valid(&event.input, event.output))
    }

    fn guard_sqrt_shape_valid<'dispatch, 'data>(
        &self,
        event: &SqrtRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(shape_valid(&event.input, event.output))
    }

    fn guard_sqrt_shape_invalid<'dispatch, 'data>(
        &self,
        event: &SqrtRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(!shape_valid(&event.input, event.output))
    }

    fn effect_sqrt_execute<'dispatch, 'data>(
        &mut self,
        event: SqrtRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        let mut output = event.output.borrow_mut();
        let mut ordinal = 0;
        while ordinal < event.input.count() {
            output.write(ordinal, event.input.read(ordinal).sqrt());
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sqrt_shape_reject<'dispatch, 'data>(
        &mut self,
        event: SqrtRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(PowerError::ShapeMismatch));
        Ok(())
    }

    fn effect_sqrt_view_reject<'dispatch, 'data>(
        &mut self,
        event: SqrtRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(PowerError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PowerError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
