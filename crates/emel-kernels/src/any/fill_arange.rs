//! Portable scalar `op_fill` and `op_arange` over validated F32 views.
//!
//! The pinned public machines expose both operations, but the pinned C++
//! commit has no executable guard or action definitions for either route.
//! This module therefore provides the maintained source-contract slice with
//! explicit validation transitions and allocation-free scalar kernels.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::TensorViewMut;

/// Errors returned by fill and arange dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillArangeError {
    /// The destination view failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for FillArangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid fill/arange tensor view"),
            Self::UnexpectedEvent => formatter.write_str("unexpected fill/arange event"),
            Self::Internal => formatter.write_str("internal fill/arange dispatch error"),
        }
    }
}

impl std::error::Error for FillArangeError {}

/// Fills every logical element of a destination F32 view with one scalar.
#[derive(Debug)]
pub struct OpFill<'a> {
    dst: TensorViewMut<'a>,
    value: f32,
}

impl<'a> OpFill<'a> {
    /// Creates a fill event. Destination validation occurs in a guard.
    #[must_use]
    pub const fn new(dst: TensorViewMut<'a>, value: f32) -> Self {
        Self { dst, value }
    }
}

/// Writes an arithmetic progression into a destination F32 view.
#[derive(Debug)]
pub struct OpArange<'a> {
    dst: TensorViewMut<'a>,
    start: f32,
    step: f32,
}

impl<'a> OpArange<'a> {
    /// Creates an arange event. Destination validation occurs in a guard.
    #[must_use]
    pub const fn new(dst: TensorViewMut<'a>, start: f32, step: f32) -> Self {
        Self { dst, start, step }
    }
}

/// Explicitly reports an event outside this operation family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedFillArange;

/// Result returned after a fill or arange dispatch completes.
pub type FillArangeResult = Result<(), FillArangeError>;

struct FillRuntime<'a> {
    event: OpFill<'a>,
    result: &'a Cell<FillArangeResult>,
}

struct ArangeRuntime<'a> {
    event: OpArange<'a>,
    result: &'a Cell<FillArangeResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<FillArangeResult>,
}

#[derive(Default)]
struct Context;

sml! {
    FillArangeMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Fill(FillRuntime<'dispatch>) [guard_fill_valid] / effect_fill_execute,
        "ready"_s <= "ready"_s + Fill(FillRuntime<'dispatch>) [guard_fill_invalid] / effect_fill_invalid,

        "ready"_s <= "ready"_s + Arange(ArangeRuntime<'dispatch>) [guard_arange_valid] / effect_arange_execute,
        "ready"_s <= "ready"_s + Arange(ArangeRuntime<'dispatch>) [guard_arange_invalid] / effect_arange_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected_typed,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for fill and arange.
pub struct FillArangeKernel {
    machine: FillArangeMachineStateMachine<Context>,
}

impl fmt::Debug for FillArangeKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FillArangeKernel")
            .finish_non_exhaustive()
    }
}

impl Default for FillArangeKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl FillArangeKernel {
    /// Constructs an independent fill/arange actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: FillArangeMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: FillArangeEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn fill(&mut self, event: OpFill<'_>) -> FillArangeResult {
        let result = Cell::new(Err(FillArangeError::UnexpectedEvent));
        self.machine
            .process_event(FillArangeMachineEvents::Fill(FillRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| FillArangeError::Internal)?;
        let _ = self.machine.is(&FillArangeMachineStates::Ready);
        result.get()
    }

    fn arange(&mut self, event: OpArange<'_>) -> FillArangeResult {
        let result = Cell::new(Err(FillArangeError::UnexpectedEvent));
        self.machine
            .process_event(FillArangeMachineEvents::Arange(ArangeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| FillArangeError::Internal)?;
        let _ = self.machine.is(&FillArangeMachineStates::Ready);
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedFillArange) -> FillArangeResult {
        let result = Cell::new(Err(FillArangeError::UnexpectedEvent));
        self.machine
            .process_event(FillArangeMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| FillArangeError::Internal)?;
        let _ = self.machine.is(&FillArangeMachineStates::Ready);
        result.get()
    }
}

/// Trait implemented by every public fill/arange event.
pub trait FillArangeEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut FillArangeKernel) -> Self::Output;
}

impl FillArangeEvent for OpFill<'_> {
    type Output = FillArangeResult;

    fn dispatch(self, actor: &mut FillArangeKernel) -> Self::Output {
        actor.fill(self)
    }
}

impl FillArangeEvent for OpArange<'_> {
    type Output = FillArangeResult;

    fn dispatch(self, actor: &mut FillArangeKernel) -> Self::Output {
        actor.arange(self)
    }
}

impl FillArangeEvent for UnexpectedFillArange {
    type Output = FillArangeResult;

    fn dispatch(self, actor: &mut FillArangeKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

impl FillArangeMachineStateMachineContext for Context {
    fn guard_fill_valid(&self, event: &FillRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.dst.validate().is_ok())
    }

    fn guard_fill_invalid(&self, event: &FillRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.dst.validate().is_err())
    }

    fn guard_arange_valid(&self, event: &ArangeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.dst.validate().is_ok())
    }

    fn guard_arange_invalid(&self, event: &ArangeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.dst.validate().is_err())
    }

    fn effect_fill_execute(&mut self, mut event: FillRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.event.dst.count() {
            event.event.dst.write(ordinal, event.event.value);
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn effect_arange_execute(&mut self, mut event: ArangeRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.event.dst.count() {
            event.event.dst.write(
                ordinal,
                (ordinal as f32).mul_add(event.event.step, event.event.start),
            );
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_fill_invalid(&mut self, event: FillRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(FillArangeError::InvalidView));
        Ok(())
    }

    fn effect_arange_invalid(&mut self, event: ArangeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(FillArangeError::InvalidView));
        Ok(())
    }

    fn effect_unexpected_typed(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(FillArangeError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
