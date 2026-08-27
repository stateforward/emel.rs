//! Dense outer-product and triangular-solve kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_out_prod` and `op_solve_tri` and names `exec_op_*` routes on the arch
//! machines. Those action/guard types are not defined in the pinned headers,
//! so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by dense linalg dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinalgError {
    /// Vector or triangle geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid linalg shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected linalg event"),
            Self::Internal => formatter.write_str("internal linalg dispatch error"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// Result returned by dense linalg dispatch.
pub type LinalgResult = Result<(), LinalgError>;

/// Writes the outer product of two dense F32 vectors.
#[derive(Debug)]
pub struct OpOutProd<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpOutProd<'a> {
    /// Creates an outer-product request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { lhs, rhs, output }
    }

    const fn valid(&self) -> bool {
        !self.lhs.is_empty()
            && !self.rhs.is_empty()
            && self.output.len() == self.lhs.len().saturating_mul(self.rhs.len())
    }
}

/// Solves a lower-triangular dense system `L x = b` in place on `x`.
#[derive(Debug)]
pub struct OpSolveTri<'a> {
    lower: &'a [f32],
    rhs: &'a [f32],
    solution: &'a mut [f32],
}

impl<'a> OpSolveTri<'a> {
    /// Creates a triangular-solve request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(lower: &'a [f32], rhs: &'a [f32], solution: &'a mut [f32]) -> Self {
        Self {
            lower,
            rhs,
            solution,
        }
    }

    const fn side(&self) -> Option<usize> {
        let n = self.rhs.len();
        if n == 0 || self.solution.len() != n || self.lower.len() != n.saturating_mul(n) {
            None
        } else {
            Some(n)
        }
    }
}

struct OutProdRuntime<'a> {
    event: OpOutProd<'a>,
    result: &'a Cell<LinalgResult>,
}

struct SolveTriRuntime<'a> {
    event: OpSolveTri<'a>,
    result: &'a Cell<LinalgResult>,
}

#[derive(Default)]
struct Context;

sml! {
    LinalgMachine<'dispatch> {
        "ready"_s <= *"ready"_s + OutProd(OutProdRuntime<'dispatch>) [guard_out_valid] / effect_out,
        "ready"_s <= "ready"_s + OutProd(OutProdRuntime<'dispatch>) [guard_out_invalid] / effect_out_reject,
        "ready"_s <= "ready"_s + SolveTri(SolveTriRuntime<'dispatch>) [guard_solve_valid] / effect_solve,
        "ready"_s <= "ready"_s + SolveTri(SolveTriRuntime<'dispatch>) [guard_solve_invalid] / effect_solve_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for dense linalg events.
pub trait LinalgEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut LinalgKernel) -> Self::Output;
}

/// Single-writer dense linalg actor.
pub struct LinalgKernel {
    machine: LinalgMachineStateMachine<Context>,
}

impl fmt::Debug for LinalgKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LinalgKernel")
            .finish_non_exhaustive()
    }
}

impl Default for LinalgKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl LinalgKernel {
    /// Constructs an independent linalg actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: LinalgMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed linalg event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::InvalidShape`] when the dense buffers cannot
    /// form the requested outer product or triangular solve.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: LinalgEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn out_prod(&mut self, event: OpOutProd<'_>) -> LinalgResult {
        let result = Cell::new(Err(LinalgError::UnexpectedEvent));
        self.machine
            .process_event(LinalgMachineEvents::OutProd(OutProdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| LinalgError::Internal)?;
        assert!(
            self.machine.is(&LinalgMachineStates::Ready),
            "linalg machine must return to ready after dispatch"
        );
        result.get()
    }

    fn solve_tri(&mut self, event: OpSolveTri<'_>) -> LinalgResult {
        let result = Cell::new(Err(LinalgError::UnexpectedEvent));
        self.machine
            .process_event(LinalgMachineEvents::SolveTri(SolveTriRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| LinalgError::Internal)?;
        assert!(
            self.machine.is(&LinalgMachineStates::Ready),
            "linalg machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&LinalgMachineStates::Ready)
    }
}

impl LinalgEvent for OpOutProd<'_> {
    type Output = LinalgResult;
    fn dispatch(self, kernel: &mut LinalgKernel) -> Self::Output {
        kernel.out_prod(self)
    }
}

impl LinalgEvent for OpSolveTri<'_> {
    type Output = LinalgResult;
    fn dispatch(self, kernel: &mut LinalgKernel) -> Self::Output {
        kernel.solve_tri(self)
    }
}

fn out_prod_values(event: &mut OpOutProd<'_>) {
    let n_rhs = event.rhs.len();
    for (row, lhs) in event.lhs.iter().copied().enumerate() {
        for (col, rhs) in event.rhs.iter().copied().enumerate() {
            event.output[row * n_rhs + col] = lhs * rhs;
        }
    }
}

fn solve_tri_values(event: &mut OpSolveTri<'_>) {
    let Some(side) = event.side() else {
        return;
    };
    for row in 0..side {
        let mut acc = event.rhs[row];
        for col in 0..row {
            acc = event.lower[row * side + col].mul_add(-event.solution[col], acc);
        }
        let diag = event.lower[row * side + row];
        event.solution[row] = if diag == 0.0 { 0.0 } else { acc / diag };
    }
}

impl LinalgMachineStateMachineContext for Context {
    fn guard_out_valid(&self, event: &OutProdRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_out_invalid(&self, event: &OutProdRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_solve_valid(&self, event: &SolveTriRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.side().is_some())
    }

    fn guard_solve_invalid(&self, event: &SolveTriRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.side().is_none())
    }

    fn effect_out(&mut self, mut event: OutProdRuntime<'_>) -> Result<(), ()> {
        out_prod_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_out_reject(&mut self, event: OutProdRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(LinalgError::InvalidShape));
        Ok(())
    }

    fn effect_solve(&mut self, mut event: SolveTriRuntime<'_>) -> Result<(), ()> {
        solve_tri_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_solve_reject(&mut self, event: SolveTriRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(LinalgError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LinalgError, LinalgKernel, OpOutProd, OpSolveTri};
    use crate::Kernel;

    #[test]
    fn out_prod_writes_the_outer_matrix() {
        let lhs = [2.0_f32, 3.0];
        let rhs = [4.0_f32, 5.0];
        let mut output = [0.0_f32; 4];
        LinalgKernel::new()
            .process_event(OpOutProd::new(&lhs, &rhs, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 8.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 15.0_f32.to_bits());
    }

    #[test]
    fn solve_tri_recovers_identity_system() {
        let lower = [1.0_f32, 0.0, 2.0, 1.0];
        let rhs = [1.0_f32, 5.0];
        let mut solution = [0.0_f32; 2];
        LinalgKernel::new()
            .process_event(OpSolveTri::new(&lower, &rhs, &mut solution))
            .unwrap();
        assert_eq!(solution[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(solution[1].to_bits(), 3.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_out_prod() {
        let lhs = [2.0_f32];
        let rhs = [3.0_f32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpOutProd::new(&lhs, &rhs, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 6.0_f32.to_bits());
    }

    #[test]
    fn out_prod_rejects_without_mutation() {
        let lhs = [1.0_f32];
        let rhs = [1.0_f32];
        let mut output = [9.0_f32; 2];
        let mut kernel = LinalgKernel::new();
        assert_eq!(
            kernel.process_event(OpOutProd::new(&lhs, &rhs, &mut output)),
            Err(LinalgError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_solve_tri_and_rejects() {
        let lower = [1.0_f32, 0.0, 2.0, 1.0];
        let rhs = [1.0_f32, 5.0];
        let mut solution = [0.0_f32; 2];
        Kernel::new()
            .process_event(OpSolveTri::new(&lower, &rhs, &mut solution))
            .unwrap();
        assert_eq!(solution[1].to_bits(), 3.0_f32.to_bits());
        let mut bad = [9.0_f32; 1];
        assert_eq!(
            LinalgKernel::new().process_event(OpSolveTri::new(&lower, &rhs, &mut bad)),
            Err(LinalgError::InvalidShape)
        );
        assert_eq!(
            LinalgError::InvalidShape.to_string(),
            "invalid linalg shape"
        );
        let _ = format!("{:?}", LinalgKernel::default());
        assert!(LinalgKernel::new().is_ready());
    }
}
