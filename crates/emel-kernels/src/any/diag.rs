//! Diagonal expand and causal-mask kernels over dense F32 buffers.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_diag`, `op_diag_mask_inf`, and `op_diag_mask_zero` and routes them on
//! the `aarch64`/`x86_64` machines. Those machines name `exec_op_diag*`
//! actions, but the pinned action headers do not define them. This actor is
//! therefore a source-contract implementation.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by diagonal dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagError {
    /// The vector/matrix split or mask geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for DiagError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid diag shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected diag event"),
            Self::Internal => formatter.write_str("internal diag dispatch error"),
        }
    }
}

impl std::error::Error for DiagError {}

/// Result returned by diagonal dispatch.
pub type DiagResult = Result<(), DiagError>;

/// Expands an F32 vector into a dense square diagonal matrix.
#[derive(Debug)]
pub struct OpDiag<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpDiag<'a> {
    /// Creates a diag request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { input, output }
    }

    const fn side(&self) -> Option<usize> {
        let n = self.input.len();
        if n == 0 || self.output.len() != n.saturating_mul(n) {
            None
        } else {
            Some(n)
        }
    }
}

/// Copies a square matrix and writes `-inf` above the causal diagonal.
#[derive(Debug)]
pub struct OpDiagMaskInf<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    n_past: i32,
}

impl<'a> OpDiagMaskInf<'a> {
    /// Creates an inf-mask request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], n_past: i32) -> Self {
        Self {
            input,
            output,
            n_past,
        }
    }
}

/// Copies a square matrix and writes `0` above the causal diagonal.
#[derive(Debug)]
pub struct OpDiagMaskZero<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    n_past: i32,
}

impl<'a> OpDiagMaskZero<'a> {
    /// Creates a zero-mask request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], n_past: i32) -> Self {
        Self {
            input,
            output,
            n_past,
        }
    }
}

const fn square_side(len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let mut side = 1_usize;
    while side.saturating_mul(side) < len {
        side += 1;
    }
    if side.saturating_mul(side) == len {
        Some(side)
    } else {
        None
    }
}

const fn mask_valid(input_len: usize, output_len: usize, n_past: i32) -> bool {
    n_past >= 0 && input_len == output_len && square_side(input_len).is_some()
}

struct DiagRuntime<'a> {
    event: OpDiag<'a>,
    result: &'a Cell<DiagResult>,
}

struct MaskInfRuntime<'a> {
    event: OpDiagMaskInf<'a>,
    result: &'a Cell<DiagResult>,
}

struct MaskZeroRuntime<'a> {
    event: OpDiagMaskZero<'a>,
    result: &'a Cell<DiagResult>,
}

#[derive(Default)]
struct Context;

fn apply_mask(input: &[f32], output: &mut [f32], n_past: i32, masked: f32) {
    let Some(side) = square_side(input.len()) else {
        return;
    };
    let Ok(n_past) = usize::try_from(n_past) else {
        return;
    };
    for row in 0..side {
        for col in 0..side {
            let index = row * side + col;
            output[index] = if col > row + n_past {
                masked
            } else {
                input[index]
            };
        }
    }
}

sml! {
    DiagMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Diag(DiagRuntime<'dispatch>) [guard_diag_valid] / effect_diag,
        "ready"_s <= "ready"_s + Diag(DiagRuntime<'dispatch>) [guard_diag_invalid] / effect_diag_reject,
        "ready"_s <= "ready"_s + MaskInf(MaskInfRuntime<'dispatch>) [guard_mask_inf_valid] / effect_mask_inf,
        "ready"_s <= "ready"_s + MaskInf(MaskInfRuntime<'dispatch>) [guard_mask_inf_invalid] / effect_mask_inf_reject,
        "ready"_s <= "ready"_s + MaskZero(MaskZeroRuntime<'dispatch>) [guard_mask_zero_valid] / effect_mask_zero,
        "ready"_s <= "ready"_s + MaskZero(MaskZeroRuntime<'dispatch>) [guard_mask_zero_invalid] / effect_mask_zero_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for diagonal family events.
pub trait DiagEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut DiagKernel) -> Self::Output;
}

/// Single-writer diagonal actor.
pub struct DiagKernel {
    machine: DiagMachineStateMachine<Context>,
}

impl fmt::Debug for DiagKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("DiagKernel").finish_non_exhaustive()
    }
}

impl Default for DiagKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagKernel {
    /// Constructs an independent diagonal actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: DiagMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed diagonal event run-to-completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: DiagEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn diag(&mut self, event: OpDiag<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(DiagMachineEvents::Diag(DiagRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        assert!(
            self.machine.is(&DiagMachineStates::Ready),
            "diag machine must return to ready after dispatch"
        );
        result.get()
    }

    fn mask_inf(&mut self, event: OpDiagMaskInf<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(DiagMachineEvents::MaskInf(MaskInfRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        assert!(
            self.machine.is(&DiagMachineStates::Ready),
            "diag machine must return to ready after dispatch"
        );
        result.get()
    }

    fn mask_zero(&mut self, event: OpDiagMaskZero<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(DiagMachineEvents::MaskZero(MaskZeroRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        assert!(
            self.machine.is(&DiagMachineStates::Ready),
            "diag machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&DiagMachineStates::Ready)
    }
}

impl DiagEvent for OpDiag<'_> {
    type Output = DiagResult;
    fn dispatch(self, kernel: &mut DiagKernel) -> Self::Output {
        kernel.diag(self)
    }
}

impl DiagEvent for OpDiagMaskInf<'_> {
    type Output = DiagResult;
    fn dispatch(self, kernel: &mut DiagKernel) -> Self::Output {
        kernel.mask_inf(self)
    }
}

impl DiagEvent for OpDiagMaskZero<'_> {
    type Output = DiagResult;
    fn dispatch(self, kernel: &mut DiagKernel) -> Self::Output {
        kernel.mask_zero(self)
    }
}

impl DiagMachineStateMachineContext for Context {
    fn guard_diag_valid(&self, event: &DiagRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.side().is_some())
    }

    fn guard_diag_invalid(&self, event: &DiagRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.side().is_none())
    }

    fn guard_mask_inf_valid(&self, event: &MaskInfRuntime<'_>) -> Result<bool, ()> {
        Ok(mask_valid(
            event.event.input.len(),
            event.event.output.len(),
            event.event.n_past,
        ))
    }

    fn guard_mask_inf_invalid(&self, event: &MaskInfRuntime<'_>) -> Result<bool, ()> {
        Ok(!mask_valid(
            event.event.input.len(),
            event.event.output.len(),
            event.event.n_past,
        ))
    }

    fn guard_mask_zero_valid(&self, event: &MaskZeroRuntime<'_>) -> Result<bool, ()> {
        Ok(mask_valid(
            event.event.input.len(),
            event.event.output.len(),
            event.event.n_past,
        ))
    }

    fn guard_mask_zero_invalid(&self, event: &MaskZeroRuntime<'_>) -> Result<bool, ()> {
        Ok(!mask_valid(
            event.event.input.len(),
            event.event.output.len(),
            event.event.n_past,
        ))
    }

    fn effect_diag(&mut self, event: DiagRuntime<'_>) -> Result<(), ()> {
        let Some(n) = event.event.side() else {
            event.result.set(Err(DiagError::InvalidShape));
            return Ok(());
        };
        event.event.output.fill(0.0);
        for index in 0..n {
            event.event.output[index * n + index] = event.event.input[index];
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_diag_reject(&mut self, event: DiagRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(DiagError::InvalidShape));
        Ok(())
    }

    fn effect_mask_inf(&mut self, event: MaskInfRuntime<'_>) -> Result<(), ()> {
        apply_mask(
            event.event.input,
            event.event.output,
            event.event.n_past,
            f32::NEG_INFINITY,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mask_inf_reject(&mut self, event: MaskInfRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(DiagError::InvalidShape));
        Ok(())
    }

    fn effect_mask_zero(&mut self, event: MaskZeroRuntime<'_>) -> Result<(), ()> {
        apply_mask(
            event.event.input,
            event.event.output,
            event.event.n_past,
            0.0,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mask_zero_reject(&mut self, event: MaskZeroRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(DiagError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DiagError, DiagKernel, OpDiag, OpDiagMaskInf, OpDiagMaskZero};

    #[test]
    fn diag_writes_the_main_diagonal() {
        let input = [1.0_f32, 2.0];
        let mut output = [9.0_f32; 4];
        let mut kernel = DiagKernel::new();
        kernel
            .process_event(OpDiag::new(&input, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 2.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn mask_inf_zeros_the_strict_upper_triangle_when_n_past_is_zero() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        let mut kernel = DiagKernel::new();
        kernel
            .process_event(OpDiagMaskInf::new(&input, &mut output, 0))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert!(output[1].is_infinite() && output[1].is_sign_negative());
        assert_eq!(output[2].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 4.0_f32.to_bits());
    }

    #[test]
    fn invalid_diag_does_not_mutate() {
        let input = [1.0_f32, 2.0];
        let mut output = [7.0_f32; 3];
        let mut kernel = DiagKernel::new();
        assert_eq!(
            kernel.process_event(OpDiag::new(&input, &mut output)),
            Err(DiagError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 7.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn mask_zero_rejects_negative_n_past() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [8.0_f32; 4];
        let mut kernel = DiagKernel::new();
        assert_eq!(
            kernel.process_event(OpDiagMaskZero::new(&input, &mut output, -1)),
            Err(DiagError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 8.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_diag_family() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 4];
        crate::Kernel::new()
            .process_event(OpDiag::new(&input, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        let square = [1.0_f32, 2.0, 3.0, 4.0];
        let mut masked = [0.0_f32; 4];
        crate::Kernel::new()
            .process_event(OpDiagMaskInf::new(&square, &mut masked, 0))
            .unwrap();
        assert!(masked[1].is_infinite());
        crate::Kernel::new()
            .process_event(OpDiagMaskZero::new(&square, &mut masked, 0))
            .unwrap();
        assert_eq!(masked[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(DiagError::InvalidShape.to_string(), "invalid diag shape");
        let _ = format!("{:?}", DiagKernel::default());
    }
}
