//! Dense `RoPE` backward and timestep-embedding kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_rope_back` and `op_timestep_embedding` and names `exec_op_*` routes on
//! the arch machines. Those action/guard types are not defined in the pinned
//! headers, so this actor is source-contract. Forward `RoPE` lives in `rope`.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::cast_precision_loss)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::rope::{ROPE_MODE_NEOX, ROPE_MODE_NORM, RopeParams};

/// Errors returned by positional dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionalError {
    /// Dimension, position, or buffer geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for PositionalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid positional shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected positional event"),
            Self::Internal => formatter.write_str("internal positional dispatch error"),
        }
    }
}

impl std::error::Error for PositionalError {}

/// Result returned by positional dispatch.
pub type PositionalResult = Result<(), PositionalError>;

/// Applies the inverse of a guarded `RoPE` mode to a dense F32 gradient.
#[derive(Debug)]
pub struct OpRopeBack<'a> {
    grad_out: &'a [f32],
    positions: &'a [i32],
    grad_in: &'a mut [f32],
    params: RopeParams,
}

impl<'a> OpRopeBack<'a> {
    /// Creates a RoPE-backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        grad_out: &'a [f32],
        positions: &'a [i32],
        grad_in: &'a mut [f32],
        params: RopeParams,
    ) -> Self {
        Self {
            grad_out,
            positions,
            grad_in,
            params,
        }
    }

    fn dims(&self) -> Option<usize> {
        if self.params.n_dims <= 0 || self.params.n_dims % 2 != 0 {
            None
        } else {
            usize::try_from(self.params.n_dims).ok()
        }
    }

    fn valid_geometry(&self) -> bool {
        self.dims().is_some_and(|dims| {
            !self.positions.is_empty()
                && self.grad_out.len() == self.grad_in.len()
                && self.grad_out.len().is_multiple_of(dims)
                && self.grad_out.len() / dims == self.positions.len()
                && self.params.freq_base > 0.0
                && self.params.freq_base.is_finite()
                && self.params.freq_scale.is_finite()
                && self.params.attn_factor.is_finite()
                && self.params.ext_factor == 0.0
        })
    }
}

/// Writes sine/cosine timestep embeddings into a dense F32 buffer.
#[derive(Debug)]
pub struct OpTimestepEmbedding<'a> {
    timesteps: &'a [f32],
    output: &'a mut [f32],
    n_embd: usize,
    max_period: f32,
}

impl<'a> OpTimestepEmbedding<'a> {
    /// Creates a timestep-embedding request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        timesteps: &'a [f32],
        output: &'a mut [f32],
        n_embd: usize,
        max_period: f32,
    ) -> Self {
        Self {
            timesteps,
            output,
            n_embd,
            max_period,
        }
    }

    const fn valid(&self) -> bool {
        self.n_embd >= 2
            && self.n_embd.is_multiple_of(2)
            && !self.timesteps.is_empty()
            && self.output.len() == self.timesteps.len().saturating_mul(self.n_embd)
            && self.max_period > 0.0
            && self.max_period.is_finite()
    }
}

struct RopeBackRuntime<'a> {
    event: OpRopeBack<'a>,
    result: &'a Cell<PositionalResult>,
}

struct TimestepRuntime<'a> {
    event: OpTimestepEmbedding<'a>,
    result: &'a Cell<PositionalResult>,
}

#[derive(Default)]
struct Context;

sml! {
    PositionalMachine<'dispatch> {
        "ready"_s <= *"ready"_s + RopeBack(RopeBackRuntime<'dispatch>) [guard_norm] / effect_norm,
        "ready"_s <= "ready"_s + RopeBack(RopeBackRuntime<'dispatch>) [guard_neox] / effect_neox,
        "ready"_s <= "ready"_s + RopeBack(RopeBackRuntime<'dispatch>) [guard_rope_invalid] / effect_rope_reject,
        "ready"_s <= "ready"_s + Timestep(TimestepRuntime<'dispatch>) [guard_timestep_valid] / effect_timestep,
        "ready"_s <= "ready"_s + Timestep(TimestepRuntime<'dispatch>) [guard_timestep_invalid] / effect_timestep_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for positional events.
pub trait PositionalEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut PositionalKernel) -> Self::Output;
}

/// Single-writer positional actor.
pub struct PositionalKernel {
    machine: PositionalMachineStateMachine<Context>,
}

impl fmt::Debug for PositionalKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PositionalKernel")
            .finish_non_exhaustive()
    }
}

impl Default for PositionalKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionalKernel {
    /// Constructs an independent positional actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: PositionalMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed positional event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`PositionalError::InvalidShape`] when the dense buffers cannot
    /// form the requested `RoPE` or timestep geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: PositionalEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn rope_back(&mut self, event: OpRopeBack<'_>) -> PositionalResult {
        let result = Cell::new(Err(PositionalError::UnexpectedEvent));
        self.machine
            .process_event(PositionalMachineEvents::RopeBack(RopeBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PositionalError::Internal)?;
        assert!(
            self.machine.is(&PositionalMachineStates::Ready),
            "positional machine must return to ready after dispatch"
        );
        result.get()
    }

    fn timestep(&mut self, event: OpTimestepEmbedding<'_>) -> PositionalResult {
        let result = Cell::new(Err(PositionalError::UnexpectedEvent));
        self.machine
            .process_event(PositionalMachineEvents::Timestep(TimestepRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PositionalError::Internal)?;
        assert!(
            self.machine.is(&PositionalMachineStates::Ready),
            "positional machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&PositionalMachineStates::Ready)
    }
}

impl PositionalEvent for OpRopeBack<'_> {
    type Output = PositionalResult;
    fn dispatch(self, kernel: &mut PositionalKernel) -> Self::Output {
        kernel.rope_back(self)
    }
}

impl PositionalEvent for OpTimestepEmbedding<'_> {
    type Output = PositionalResult;
    fn dispatch(self, kernel: &mut PositionalKernel) -> Self::Output {
        kernel.timestep(self)
    }
}

const fn pair_indices(pair: usize, half_dims: usize, neox: bool) -> (usize, usize) {
    if neox {
        (pair, pair + half_dims)
    } else {
        (2 * pair, 2 * pair + 1)
    }
}

fn rope_back_values(event: &mut OpRopeBack<'_>, neox: bool) {
    let Some(dims) = event.dims() else {
        return;
    };
    let half_dims = dims / 2;
    let theta_scale = event
        .params
        .freq_base
        .powf(-2.0 / event.params.n_dims as f32);
    for (row, position) in event.positions.iter().copied().enumerate() {
        let base = row * dims;
        let mut theta = position as f32;
        for pair in 0..half_dims {
            let rotated = theta * event.params.freq_scale;
            let cosine = rotated.cos() * event.params.attn_factor;
            let sine = rotated.sin() * event.params.attn_factor;
            let (index_a, index_b) = pair_indices(pair, half_dims, neox);
            let grad_a = event.grad_out[base + index_a];
            let grad_b = event.grad_out[base + index_b];
            event.grad_in[base + index_a] = grad_a.mul_add(cosine, grad_b * sine);
            event.grad_in[base + index_b] = grad_b.mul_add(cosine, -(grad_a * sine));
            theta *= theta_scale;
        }
    }
}

fn timestep_values(event: &mut OpTimestepEmbedding<'_>) {
    let half = event.n_embd / 2;
    let denom = if half == 1 { 1.0 } else { (half - 1) as f32 };
    let neg_log = -event.max_period.ln();
    for (step_index, timestep) in event.timesteps.iter().copied().enumerate() {
        let base = step_index * event.n_embd;
        for pair in 0..half {
            let frequency = (neg_log * pair as f32 / denom).exp();
            let argument = timestep * frequency;
            event.output[base + pair] = argument.cos();
            event.output[base + half + pair] = argument.sin();
        }
    }
}

impl PositionalMachineStateMachineContext for Context {
    fn guard_norm(&self, event: &RopeBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid_geometry() && event.event.params.mode == ROPE_MODE_NORM)
    }

    fn guard_neox(&self, event: &RopeBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid_geometry() && event.event.params.mode == ROPE_MODE_NEOX)
    }

    fn guard_rope_invalid(&self, event: &RopeBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid_geometry()
            || (event.event.params.mode != ROPE_MODE_NORM
                && event.event.params.mode != ROPE_MODE_NEOX))
    }

    fn guard_timestep_valid(&self, event: &TimestepRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_timestep_invalid(&self, event: &TimestepRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_norm(&mut self, mut event: RopeBackRuntime<'_>) -> Result<(), ()> {
        rope_back_values(&mut event.event, false);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_neox(&mut self, mut event: RopeBackRuntime<'_>) -> Result<(), ()> {
        rope_back_values(&mut event.event, true);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rope_reject(&mut self, event: RopeBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PositionalError::InvalidShape));
        Ok(())
    }

    fn effect_timestep(&mut self, mut event: TimestepRuntime<'_>) -> Result<(), ()> {
        timestep_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_timestep_reject(&mut self, event: TimestepRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PositionalError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpRopeBack, OpTimestepEmbedding, PositionalError, PositionalKernel};
    use crate::Kernel;
    use crate::any::rope::{ROPE_MODE_NORM, RopeParams};

    fn params() -> RopeParams {
        RopeParams {
            n_dims: 2,
            mode: ROPE_MODE_NORM,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
        }
    }

    #[test]
    fn rope_back_inverts_unit_gradient_at_zero_position() {
        let grad_out = [1.0_f32, 0.0];
        let positions = [0_i32];
        let mut grad_in = [0.0_f32; 2];
        let mut kernel = PositionalKernel::new();
        kernel
            .process_event(OpRopeBack::new(
                &grad_out,
                &positions,
                &mut grad_in,
                params(),
            ))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(grad_in[1].to_bits(), 0.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn timestep_embedding_writes_cos_sin_halves() {
        let timesteps = [0.0_f32];
        let mut output = [9.0_f32; 4];
        PositionalKernel::new()
            .process_event(OpTimestepEmbedding::new(
                &timesteps,
                &mut output,
                4,
                10_000.0,
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_rope_back() {
        let grad_out = [1.0_f32, 0.0];
        let positions = [0_i32];
        let mut grad_in = [0.0_f32; 2];
        Kernel::new()
            .process_event(OpRopeBack::new(
                &grad_out,
                &positions,
                &mut grad_in,
                params(),
            ))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn rope_back_rejects_without_mutation() {
        let grad_out = [1.0_f32];
        let positions = [0_i32];
        let mut grad_in = [5.0_f32; 2];
        let mut kernel = PositionalKernel::new();
        assert_eq!(
            kernel.process_event(OpRopeBack::new(
                &grad_out,
                &positions,
                &mut grad_in,
                params()
            )),
            Err(PositionalError::InvalidShape)
        );
        assert_eq!(grad_in[0].to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_timestep_embedding() {
        let timesteps = [0.0_f32];
        let mut output = [9.0_f32; 4];
        Kernel::new()
            .process_event(OpTimestepEmbedding::new(
                &timesteps,
                &mut output,
                4,
                10_000.0,
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
    }
}
