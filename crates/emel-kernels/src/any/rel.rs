//! Dense relative-position kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_get_rel_pos` and `op_add_rel_pos` and names `exec_op_*` routes on the
//! arch machines. Those action/guard types are not defined in the pinned
//! headers, so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::cast_precision_loss)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by relative-position dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelError {
    /// Query/key geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for RelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid rel-pos shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected rel-pos event"),
            Self::Internal => formatter.write_str("internal rel-pos dispatch error"),
        }
    }
}

impl std::error::Error for RelError {}

/// Result returned by relative-position dispatch.
pub type RelResult = Result<(), RelError>;

/// Writes `key - query` offsets for a dense attention grid.
#[derive(Debug)]
pub struct OpGetRelPos<'a> {
    output: &'a mut [f32],
    queries: usize,
    keys: usize,
}

impl<'a> OpGetRelPos<'a> {
    /// Creates a get-rel-pos request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(output: &'a mut [f32], queries: usize, keys: usize) -> Self {
        Self {
            output,
            queries,
            keys,
        }
    }

    const fn valid(&self) -> bool {
        self.queries > 0
            && self.keys > 0
            && self.output.len() == self.queries.saturating_mul(self.keys)
    }
}

/// Adds a dense relative-position bias into attention scores.
#[derive(Debug)]
pub struct OpAddRelPos<'a> {
    scores: &'a [f32],
    bias: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpAddRelPos<'a> {
    /// Creates an add-rel-pos request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(scores: &'a [f32], bias: &'a [f32], output: &'a mut [f32]) -> Self {
        Self {
            scores,
            bias,
            output,
        }
    }

    const fn valid(&self) -> bool {
        !self.scores.is_empty()
            && self.scores.len() == self.bias.len()
            && self.scores.len() == self.output.len()
    }
}

struct GetRelPosRuntime<'a> {
    event: OpGetRelPos<'a>,
    result: &'a Cell<RelResult>,
}

struct AddRelPosRuntime<'a> {
    event: OpAddRelPos<'a>,
    result: &'a Cell<RelResult>,
}

#[derive(Default)]
struct Context;

sml! {
    RelMachine<'dispatch> {
        "ready"_s <= *"ready"_s + GetRelPos(GetRelPosRuntime<'dispatch>) [guard_get_valid] / effect_get,
        "ready"_s <= "ready"_s + GetRelPos(GetRelPosRuntime<'dispatch>) [guard_get_invalid] / effect_get_reject,
        "ready"_s <= "ready"_s + AddRelPos(AddRelPosRuntime<'dispatch>) [guard_add_valid] / effect_add,
        "ready"_s <= "ready"_s + AddRelPos(AddRelPosRuntime<'dispatch>) [guard_add_invalid] / effect_add_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for relative-position events.
pub trait RelEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut RelKernel) -> Self::Output;
}

/// Single-writer relative-position actor.
pub struct RelKernel {
    machine: RelMachineStateMachine<Context>,
}

impl fmt::Debug for RelKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RelKernel").finish_non_exhaustive()
    }
}

impl Default for RelKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RelKernel {
    /// Constructs an independent relative-position actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: RelMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed relative-position event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`RelError::InvalidShape`] when the dense buffers cannot form
    /// the requested relative-position geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: RelEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn get(&mut self, event: OpGetRelPos<'_>) -> RelResult {
        let result = Cell::new(Err(RelError::UnexpectedEvent));
        self.machine
            .process_event(RelMachineEvents::GetRelPos(GetRelPosRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RelError::Internal)?;
        assert!(
            self.machine.is(&RelMachineStates::Ready),
            "rel machine must return to ready after dispatch"
        );
        result.get()
    }

    fn add(&mut self, event: OpAddRelPos<'_>) -> RelResult {
        let result = Cell::new(Err(RelError::UnexpectedEvent));
        self.machine
            .process_event(RelMachineEvents::AddRelPos(AddRelPosRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RelError::Internal)?;
        assert!(
            self.machine.is(&RelMachineStates::Ready),
            "rel machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&RelMachineStates::Ready)
    }
}

impl RelEvent for OpGetRelPos<'_> {
    type Output = RelResult;
    fn dispatch(self, kernel: &mut RelKernel) -> Self::Output {
        kernel.get(self)
    }
}

impl RelEvent for OpAddRelPos<'_> {
    type Output = RelResult;
    fn dispatch(self, kernel: &mut RelKernel) -> Self::Output {
        kernel.add(self)
    }
}

fn get_rel_pos_values(event: &mut OpGetRelPos<'_>) {
    for query in 0..event.queries {
        for key in 0..event.keys {
            event.output[query * event.keys + key] = (key as f32) - (query as f32);
        }
    }
}

fn add_rel_pos_values(event: &mut OpAddRelPos<'_>) {
    for (slot, (score, bias)) in event
        .output
        .iter_mut()
        .zip(event.scores.iter().zip(event.bias.iter()))
    {
        *slot = score.mul_add(1.0, *bias);
    }
}

impl RelMachineStateMachineContext for Context {
    fn guard_get_valid(&self, event: &GetRelPosRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_get_invalid(&self, event: &GetRelPosRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_add_valid(&self, event: &AddRelPosRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_add_invalid(&self, event: &AddRelPosRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_get(&mut self, mut event: GetRelPosRuntime<'_>) -> Result<(), ()> {
        get_rel_pos_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_get_reject(&mut self, event: GetRelPosRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RelError::InvalidShape));
        Ok(())
    }
    fn effect_add(&mut self, mut event: AddRelPosRuntime<'_>) -> Result<(), ()> {
        add_rel_pos_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_add_reject(&mut self, event: AddRelPosRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(RelError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpAddRelPos, OpGetRelPos, RelError, RelKernel};
    use crate::Kernel;

    #[test]
    fn get_rel_pos_writes_key_minus_query() {
        let mut output = [0.0_f32; 4];
        RelKernel::new()
            .process_event(OpGetRelPos::new(&mut output, 2, 2))
            .unwrap();
        assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), (-1.0_f32).to_bits());
    }

    #[test]
    fn add_rel_pos_and_public_kernel_sum() {
        let scores = [1.0_f32, 2.0];
        let bias = [0.5_f32, -0.5];
        let mut output = [0.0_f32; 2];
        Kernel::new()
            .process_event(OpAddRelPos::new(&scores, &bias, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.5_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.5_f32.to_bits());
    }

    #[test]
    fn get_rel_pos_rejects_without_mutation() {
        let mut output = [9.0_f32; 1];
        let mut kernel = RelKernel::new();
        assert_eq!(
            kernel.process_event(OpGetRelPos::new(&mut output, 2, 2)),
            Err(RelError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_get_rel_pos_and_rejects_add() {
        let mut output = [0.0_f32; 4];
        crate::Kernel::new()
            .process_event(OpGetRelPos::new(&mut output, 2, 2))
            .unwrap();
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        let scores = [1.0_f32];
        let bias = [1.0_f32, 2.0];
        let mut dst = [9.0_f32; 1];
        assert_eq!(
            RelKernel::new().process_event(OpAddRelPos::new(&scores, &bias, &mut dst)),
            Err(RelError::InvalidShape)
        );
        assert_eq!(RelError::InvalidShape.to_string(), "invalid rel-pos shape");
        let _ = format!("{:?}", RelKernel::default());
    }
}
