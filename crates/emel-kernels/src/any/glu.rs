//! Gated-linear-unit kernels over dense F32 pairs.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_glu` and `glu_subop` in `src/emel/kernel/events.hpp` and routes
//! `dispatch_op_glu` in the `aarch64`/`x86_64` machines. Those machines name
//! `exec_op_glu` / `valid_op_glu`, but the pinned action/guard headers do not
//! define them. This actor is therefore a source-contract implementation with
//! explicit subop guards and one chosen data-plane loop per transition.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]
#![allow(clippy::suboptimal_flops)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

const SWIGLU_OAI_ALPHA: f32 = 1.702;
const SWIGLU_OAI_LIMIT: f32 = 7.0;

/// Pinned `glu_subop` wire codes from `events.hpp`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GluSubOp {
    /// `x * relu(g)`
    ReGlu = 0,
    /// `x * gelu_tanh(g)`
    GeGlu = 1,
    /// `x * silu(g)`
    SwiGlu = 2,
    /// `OpenAI` `SwiGLU` variant with clamped gate.
    SwiGluOai = 3,
    /// `x * gelu_erf(g)`
    GeGluErf = 4,
    /// `x * gelu_quick(g)`
    GeGluQuick = 5,
}

/// Errors returned by GLU dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GluError {
    /// The value/gate split or output length is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for GluError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid glu shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected glu event"),
            Self::Internal => formatter.write_str("internal glu dispatch error"),
        }
    }
}

impl std::error::Error for GluError {}

/// Result returned by GLU dispatch.
pub type GluResult = Result<(), GluError>;

/// Applies one GLU subop to dense F32 value/gate halves.
#[derive(Debug)]
pub struct OpGlu<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    subop: GluSubOp,
}

impl<'a> OpGlu<'a> {
    /// Creates a GLU request. Shape and subop validation occur in guards.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], subop: GluSubOp) -> Self {
        Self {
            input,
            output,
            subop,
        }
    }

    const fn split_len(&self) -> Option<usize> {
        let len = self.input.len();
        if len == 0 || !len.is_multiple_of(2) || self.output.len() != len / 2 {
            None
        } else {
            Some(len / 2)
        }
    }
}

struct Runtime<'a> {
    event: OpGlu<'a>,
    result: &'a Cell<GluResult>,
}

#[derive(Default)]
struct Context;

const fn relu(value: f32) -> f32 {
    value.max(0.0)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn gelu_tanh(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * value * (1.0 + 0.044_715 * value * value)).tanh())
}

fn gelu_quick(value: f32) -> f32 {
    value * (1.0 / (1.0 + (-1.702 * value).exp()))
}

fn gelu_erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / core::f32::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * value * (1.0 + erf)
}

fn apply_split(event: &mut Runtime<'_>, gate: fn(f32) -> f32) {
    let Some(n) = event.event.split_len() else {
        event.result.set(Err(GluError::InvalidShape));
        return;
    };
    let (values, gates) = event.event.input.split_at(n);
    for index in 0..n {
        event.event.output[index] = values[index] * gate(gates[index]);
    }
    event.result.set(Ok(()));
}

sml! {
    GluMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Glu(Runtime<'dispatch>) [guard_reglu] / effect_reglu,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_geglu] / effect_geglu,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_swiglu] / effect_swiglu,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_swiglu_oai] / effect_swiglu_oai,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_geglu_erf] / effect_geglu_erf,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_geglu_quick] / effect_geglu_quick,
        "ready"_s <= "ready"_s + Glu(Runtime<'dispatch>) [guard_invalid] / effect_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer GLU actor.
pub struct GluKernel {
    machine: GluMachineStateMachine<Context>,
}

impl fmt::Debug for GluKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("GluKernel").finish_non_exhaustive()
    }
}

impl Default for GluKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GluKernel {
    /// Constructs an independent GLU actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: GluMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed GLU event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`GluError::InvalidShape`] when the input cannot be split into
    /// equal value/gate halves or the output length does not match.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event(&mut self, event: OpGlu<'_>) -> GluResult {
        let result = Cell::new(Err(GluError::UnexpectedEvent));
        self.machine
            .process_event(GluMachineEvents::Glu(Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GluError::Internal)?;
        assert!(
            self.machine.is(&GluMachineStates::Ready),
            "glu machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&GluMachineStates::Ready)
    }
}

impl GluMachineStateMachineContext for Context {
    fn guard_reglu(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::ReGlu)
    }

    fn guard_geglu(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::GeGlu)
    }

    fn guard_swiglu(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::SwiGlu)
    }

    fn guard_swiglu_oai(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::SwiGluOai)
    }

    fn guard_geglu_erf(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::GeGluErf)
    }

    fn guard_geglu_quick(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_some() && event.event.subop == GluSubOp::GeGluQuick)
    }

    fn guard_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.split_len().is_none())
    }

    fn effect_reglu(&mut self, mut event: Runtime<'_>) -> Result<(), ()> {
        apply_split(&mut event, relu);
        Ok(())
    }

    fn effect_geglu(&mut self, mut event: Runtime<'_>) -> Result<(), ()> {
        apply_split(&mut event, gelu_tanh);
        Ok(())
    }

    fn effect_swiglu(&mut self, mut event: Runtime<'_>) -> Result<(), ()> {
        apply_split(&mut event, silu);
        Ok(())
    }

    fn effect_swiglu_oai(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let Some(n) = event.event.split_len() else {
            event.result.set(Err(GluError::InvalidShape));
            return Ok(());
        };
        let (values, gates) = event.event.input.split_at(n);
        for index in 0..n {
            let gate = gates[index].min(SWIGLU_OAI_LIMIT);
            let value = values[index].min(SWIGLU_OAI_LIMIT);
            event.event.output[index] =
                silu(SWIGLU_OAI_ALPHA * gate) / SWIGLU_OAI_ALPHA * (value + 1.0);
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_geglu_erf(&mut self, mut event: Runtime<'_>) -> Result<(), ()> {
        apply_split(&mut event, gelu_erf);
        Ok(())
    }

    fn effect_geglu_quick(&mut self, mut event: Runtime<'_>) -> Result<(), ()> {
        apply_split(&mut event, gelu_quick);
        Ok(())
    }

    fn effect_reject(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(GluError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GluError, GluKernel, GluSubOp, OpGlu};

    #[test]
    fn reglu_multiplies_relu_gate() {
        let input = [1.0_f32, -2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 2];
        let mut kernel = GluKernel::new();
        kernel
            .process_event(OpGlu::new(&input, &mut output, GluSubOp::ReGlu))
            .unwrap();
        assert_eq!(output[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), (-8.0_f32).to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn odd_length_is_invalid_and_does_not_mutate() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [9.0_f32; 1];
        let mut kernel = GluKernel::new();
        assert_eq!(
            kernel.process_event(OpGlu::new(&input, &mut output, GluSubOp::SwiGlu)),
            Err(GluError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn remaining_glu_subops_write_finite_gated_products() {
        let input = [2.0_f32, 1.0];
        let mut kernel = GluKernel::new();
        for subop in [
            GluSubOp::GeGlu,
            GluSubOp::SwiGlu,
            GluSubOp::SwiGluOai,
            GluSubOp::GeGluErf,
            GluSubOp::GeGluQuick,
        ] {
            let mut output = [9.0_f32; 1];
            kernel
                .process_event(OpGlu::new(&input, &mut output, subop))
                .unwrap();
            assert!(output[0].is_finite());
            assert_ne!(output[0].to_bits(), 9.0_f32.to_bits());
            assert!(kernel.is_ready());
        }
        assert_eq!(GluError::InvalidShape.to_string(), "invalid glu shape");
        assert_eq!(
            GluError::UnexpectedEvent.to_string(),
            "unexpected glu event"
        );
        assert_eq!(
            GluError::Internal.to_string(),
            "internal glu dispatch error"
        );
        let _ = format!("{:?}", GluKernel::default());
    }

    #[test]
    fn public_kernel_dispatches_reglu() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 1];
        crate::Kernel::new()
            .process_event(OpGlu::new(&input, &mut output, GluSubOp::ReGlu))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
    }
}
