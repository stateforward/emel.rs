//! Portable count-equal kernel over dense I32 buffers.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by count-equal dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CountEqualError {
    /// The input buffers are empty, differ in length, or the output is not scalar.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for CountEqualError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid count-equal shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected count-equal event"),
            Self::Internal => formatter.write_str("internal count-equal dispatch error"),
        }
    }
}

impl std::error::Error for CountEqualError {}

/// Result returned by count-equal dispatch.
pub type CountEqualResult = Result<(), CountEqualError>;

/// Counts equal I32 elements and writes the count to one scalar output.
#[derive(Debug)]
pub struct OpCountEqual<'a> {
    lhs: &'a [i32],
    rhs: &'a [i32],
    output: &'a mut [i32],
}

impl<'a> OpCountEqual<'a> {
    /// Creates a count-equal request. Shape validation occurs in the actor guard.
    #[must_use]
    pub const fn new(lhs: &'a [i32], rhs: &'a [i32], output: &'a mut [i32]) -> Self {
        Self { lhs, rhs, output }
    }

    /// Returns the left input length without dispatching.
    #[must_use]
    pub const fn lhs_len(&self) -> usize {
        self.lhs.len()
    }

    /// Returns the right input length without dispatching.
    #[must_use]
    pub const fn rhs_len(&self) -> usize {
        self.rhs.len()
    }

    /// Returns the output length without dispatching.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output.len()
    }
}

struct Runtime<'a> {
    event: OpCountEqual<'a>,
    result: &'a Cell<CountEqualResult>,
}

#[derive(Default)]
struct Context;

sml! {
    CountEqualMachine<'dispatch> {
        "ready"_s <= *"ready"_s + CountEqual(Runtime<'dispatch>)
            [guard_count_equal_valid] / effect_count_equal_execute,
        "ready"_s <= "ready"_s + CountEqual(Runtime<'dispatch>)
            [guard_count_equal_invalid] / effect_count_equal_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for count-equal.
pub struct CountEqualKernel {
    machine: CountEqualMachineStateMachine<Context>,
}

impl fmt::Debug for CountEqualKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CountEqualKernel")
            .finish_non_exhaustive()
    }
}

impl Default for CountEqualKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl CountEqualKernel {
    /// Constructs an independent count-equal actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: CountEqualMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns [`CountEqualError::InvalidShape`] when the input lengths or
    /// scalar output contract is invalid.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event(&mut self, event: OpCountEqual<'_>) -> CountEqualResult {
        let result = Cell::new(Err(CountEqualError::UnexpectedEvent));
        self.machine
            .process_event(CountEqualMachineEvents::CountEqual(Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CountEqualError::Internal)?;
        assert!(
            self.machine.is(&CountEqualMachineStates::Ready),
            "count-equal machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&CountEqualMachineStates::Ready)
    }
}

impl CountEqualMachineStateMachineContext for Context {
    fn guard_count_equal_valid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs_len() > 0
            && event.event.lhs_len() == event.event.rhs_len()
            && event.event.output_len() == 1)
    }

    fn guard_count_equal_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_count_equal_valid(event)?)
    }

    fn effect_count_equal_execute(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let mut count = 0_i32;
        for (&lhs, &rhs) in event.event.lhs.iter().zip(event.event.rhs.iter()) {
            count += i32::from(lhs == rhs);
        }
        event.event.output[0] = count;
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_count_equal_reject(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CountEqualError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
