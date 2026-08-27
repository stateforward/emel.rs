//! Dense gated-linear attention and flash-attention backward kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_gated_linear_attn` and `op_flash_attn_back` and names `exec_op_*`
//! routes on the arch machines. Those action/guard types are not defined in
//! the pinned headers, so this actor is source-contract. Forward flash
//! attention lives in `flash_attn`.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::cast_precision_loss)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by extra attention dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlaError {
    /// Sequence, head, or scratch geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for GlaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid attention shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected attention event"),
            Self::Internal => formatter.write_str("internal attention dispatch error"),
        }
    }
}

impl std::error::Error for GlaError {}

/// Result returned by extra attention dispatch.
pub type GlaResult = Result<(), GlaError>;

/// Sequence/channel geometry for gated linear attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlaParams {
    /// Time steps.
    pub sequence: usize,
    /// Channel count.
    pub channels: usize,
}

/// Linear gated attention: `y = g * q * (h + k v)` then `h = w h + k v`.
#[derive(Debug)]
pub struct OpGatedLinearAttn<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    gate: &'a [f32],
    decay: &'a [f32],
    output: &'a mut [f32],
    params: GlaParams,
}

impl<'a> OpGatedLinearAttn<'a> {
    /// Creates a gated-linear-attention request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        gate: &'a [f32],
        decay: &'a [f32],
        output: &'a mut [f32],
        params: GlaParams,
    ) -> Self {
        Self {
            query,
            key,
            value,
            gate,
            decay,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        let seq_len = self.params.sequence.saturating_mul(self.params.channels);
        self.params.sequence > 0
            && self.params.channels > 0
            && self.query.len() == seq_len
            && self.key.len() == seq_len
            && self.value.len() == seq_len
            && self.gate.len() == seq_len
            && self.decay.len() == seq_len
            && self.output.len() == seq_len
    }
}

/// Dense flash-attention backward geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlashAttnBackParams {
    /// Query tokens.
    pub seq_q: usize,
    /// Key/value tokens.
    pub seq_k: usize,
    /// Head dimension.
    pub dim: usize,
    /// Attention scale.
    pub scale: f32,
}

/// Forward operands for [`OpFlashAttnBack`].
#[derive(Debug)]
pub struct FlashAttnBackSrc<'a> {
    /// Query matrix, `[seq_q, dim]`.
    pub query: &'a [f32],
    /// Key matrix, `[seq_k, dim]`.
    pub key: &'a [f32],
    /// Value matrix, `[seq_k, dim]`.
    pub value: &'a [f32],
    /// Output gradient, `[seq_q, dim]`.
    pub grad_out: &'a [f32],
}

/// Destination buffers for [`OpFlashAttnBack`].
#[derive(Debug)]
pub struct FlashAttnBackDst<'a> {
    /// Query gradient.
    pub grad_query: &'a mut [f32],
    /// Key gradient.
    pub grad_key: &'a mut [f32],
    /// Value gradient.
    pub grad_value: &'a mut [f32],
    /// Caller-owned `[seq_q, seq_k]` attention scratch.
    pub attn: &'a mut [f32],
}

/// Dense flash-attention Jacobian into caller-owned gradient buffers.
#[derive(Debug)]
pub struct OpFlashAttnBack<'a> {
    src: FlashAttnBackSrc<'a>,
    dst: FlashAttnBackDst<'a>,
    params: FlashAttnBackParams,
}

impl<'a> OpFlashAttnBack<'a> {
    /// Creates a flash-attention backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        src: FlashAttnBackSrc<'a>,
        dst: FlashAttnBackDst<'a>,
        params: FlashAttnBackParams,
    ) -> Self {
        Self { src, dst, params }
    }

    const fn valid(&self) -> bool {
        let q_len = self.params.seq_q.saturating_mul(self.params.dim);
        let k_len = self.params.seq_k.saturating_mul(self.params.dim);
        self.params.seq_q > 0
            && self.params.seq_k > 0
            && self.params.dim > 0
            && self.params.scale.is_finite()
            && self.src.query.len() == q_len
            && self.src.grad_out.len() == q_len
            && self.dst.grad_query.len() == q_len
            && self.src.key.len() == k_len
            && self.src.value.len() == k_len
            && self.dst.grad_key.len() == k_len
            && self.dst.grad_value.len() == k_len
            && self.dst.attn.len() == self.params.seq_q.saturating_mul(self.params.seq_k)
    }
}

struct GlaRuntime<'a> {
    event: OpGatedLinearAttn<'a>,
    result: &'a Cell<GlaResult>,
}

struct FlashBackRuntime<'a> {
    event: OpFlashAttnBack<'a>,
    result: &'a Cell<GlaResult>,
}

#[derive(Default)]
struct Context;

sml! {
    GlaMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Gla(GlaRuntime<'dispatch>) [guard_gla_valid] / effect_gla,
        "ready"_s <= "ready"_s + Gla(GlaRuntime<'dispatch>) [guard_gla_invalid] / effect_gla_reject,
        "ready"_s <= "ready"_s + FlashBack(FlashBackRuntime<'dispatch>) [guard_flash_valid] / effect_flash,
        "ready"_s <= "ready"_s + FlashBack(FlashBackRuntime<'dispatch>) [guard_flash_invalid] / effect_flash_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for extra attention events.
pub trait GlaEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut GlaKernel) -> Self::Output;
}

/// Single-writer extra attention actor.
pub struct GlaKernel {
    machine: GlaMachineStateMachine<Context>,
}

impl fmt::Debug for GlaKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("GlaKernel").finish_non_exhaustive()
    }
}

impl Default for GlaKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GlaKernel {
    /// Constructs an independent extra-attention actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: GlaMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed extra-attention event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`GlaError::InvalidShape`] when the dense buffers cannot form
    /// the requested attention geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: GlaEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn gla(&mut self, event: OpGatedLinearAttn<'_>) -> GlaResult {
        let result = Cell::new(Err(GlaError::UnexpectedEvent));
        self.machine
            .process_event(GlaMachineEvents::Gla(GlaRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GlaError::Internal)?;
        assert!(
            self.machine.is(&GlaMachineStates::Ready),
            "gla machine must return to ready after dispatch"
        );
        result.get()
    }

    fn flash_back(&mut self, event: OpFlashAttnBack<'_>) -> GlaResult {
        let result = Cell::new(Err(GlaError::UnexpectedEvent));
        self.machine
            .process_event(GlaMachineEvents::FlashBack(FlashBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GlaError::Internal)?;
        assert!(
            self.machine.is(&GlaMachineStates::Ready),
            "gla machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&GlaMachineStates::Ready)
    }
}

impl GlaEvent for OpGatedLinearAttn<'_> {
    type Output = GlaResult;
    fn dispatch(self, kernel: &mut GlaKernel) -> Self::Output {
        kernel.gla(self)
    }
}

impl GlaEvent for OpFlashAttnBack<'_> {
    type Output = GlaResult;
    fn dispatch(self, kernel: &mut GlaKernel) -> Self::Output {
        kernel.flash_back(self)
    }
}

fn gla_values(event: &mut OpGatedLinearAttn<'_>) {
    let channels = event.params.channels;
    for channel in 0..channels {
        let mut hidden = 0.0_f32;
        for time in 0..event.params.sequence {
            let index = time * channels + channel;
            let kv = event.key[index] * event.value[index];
            let state = hidden + kv;
            event.output[index] = event.gate[index] * event.query[index] * state;
            hidden = event.decay[index].mul_add(hidden, kv);
        }
    }
}

fn softmax_row(scores: &mut [f32]) {
    let mut peak = scores[0];
    for sample in scores.iter().copied().skip(1) {
        if sample > peak {
            peak = sample;
        }
    }
    let mut sum = 0.0_f32;
    for slot in scores.iter_mut() {
        *slot = (*slot - peak).exp();
        sum += *slot;
    }
    let inv = sum.recip();
    for slot in scores.iter_mut() {
        *slot *= inv;
    }
}

#[allow(clippy::needless_range_loop)]
fn flash_attn_back_values(event: &mut OpFlashAttnBack<'_>) {
    let seq_q = event.params.seq_q;
    let seq_k = event.params.seq_k;
    let dim = event.params.dim;
    let scale = event.params.scale;
    event.dst.grad_query.fill(0.0_f32);
    event.dst.grad_key.fill(0.0_f32);
    event.dst.grad_value.fill(0.0_f32);
    for query in 0..seq_q {
        let scores = &mut event.dst.attn[query * seq_k..query * seq_k + seq_k];
        for key in 0..seq_k {
            let mut dot = 0.0_f32;
            let q_off = query * dim;
            let k_off = key * dim;
            for feature in 0..dim {
                dot = event.src.query[q_off + feature].mul_add(event.src.key[k_off + feature], dot);
            }
            scores[key] = dot * scale;
        }
        softmax_row(scores);
        let do_off = query * dim;
        for key in 0..seq_k {
            let v_off = key * dim;
            let weight = event.dst.attn[query * seq_k + key];
            for feature in 0..dim {
                event.dst.grad_value[v_off + feature] = weight.mul_add(
                    event.src.grad_out[do_off + feature],
                    event.dst.grad_value[v_off + feature],
                );
            }
        }
        let mut attn_dot = 0.0_f32;
        for key in 0..seq_k {
            let v_off = key * dim;
            let mut dattn = 0.0_f32;
            for feature in 0..dim {
                dattn = event.src.grad_out[do_off + feature]
                    .mul_add(event.src.value[v_off + feature], dattn);
            }
            attn_dot = event.dst.attn[query * seq_k + key].mul_add(dattn, attn_dot);
        }
        for key in 0..seq_k {
            let v_off = key * dim;
            let mut dattn = 0.0_f32;
            for feature in 0..dim {
                dattn = event.src.grad_out[do_off + feature]
                    .mul_add(event.src.value[v_off + feature], dattn);
            }
            let prob = event.dst.attn[query * seq_k + key];
            let dscore = prob * (dattn - attn_dot);
            let q_off = query * dim;
            let k_off = key * dim;
            for feature in 0..dim {
                event.dst.grad_query[q_off + feature] = (dscore * scale).mul_add(
                    event.src.key[k_off + feature],
                    event.dst.grad_query[q_off + feature],
                );
                event.dst.grad_key[k_off + feature] = (dscore * scale).mul_add(
                    event.src.query[q_off + feature],
                    event.dst.grad_key[k_off + feature],
                );
            }
        }
    }
}

impl GlaMachineStateMachineContext for Context {
    fn guard_gla_valid(&self, event: &GlaRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_gla_invalid(&self, event: &GlaRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_flash_valid(&self, event: &FlashBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_flash_invalid(&self, event: &FlashBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_gla(&mut self, mut event: GlaRuntime<'_>) -> Result<(), ()> {
        gla_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_gla_reject(&mut self, event: GlaRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GlaError::InvalidShape));
        Ok(())
    }
    fn effect_flash(&mut self, mut event: FlashBackRuntime<'_>) -> Result<(), ()> {
        flash_attn_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_flash_reject(&mut self, event: FlashBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(GlaError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FlashAttnBackDst, FlashAttnBackParams, FlashAttnBackSrc, GlaError, GlaKernel, GlaParams,
        OpFlashAttnBack, OpGatedLinearAttn,
    };
    use crate::Kernel;

    #[test]
    fn gla_first_step_is_gate_query_kv() {
        let query = [2.0_f32];
        let key = [3.0_f32];
        let value = [4.0_f32];
        let gate = [1.0_f32];
        let decay = [0.0_f32];
        let mut output = [0.0_f32; 1];
        GlaKernel::new()
            .process_event(OpGatedLinearAttn::new(
                &query,
                &key,
                &value,
                &gate,
                &decay,
                &mut output,
                GlaParams {
                    sequence: 1,
                    channels: 1,
                },
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 24.0_f32.to_bits());
    }

    #[test]
    fn flash_back_one_token_writes_value_grad() {
        let query = [1.0_f32];
        let key = [1.0_f32];
        let value = [2.0_f32];
        let grad_out = [3.0_f32];
        let mut dq = [9.0_f32];
        let mut dk = [9.0_f32];
        let mut dv = [9.0_f32];
        let mut attn = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpFlashAttnBack::new(
                FlashAttnBackSrc {
                    query: &query,
                    key: &key,
                    value: &value,
                    grad_out: &grad_out,
                },
                FlashAttnBackDst {
                    grad_query: &mut dq,
                    grad_key: &mut dk,
                    grad_value: &mut dv,
                    attn: &mut attn,
                },
                FlashAttnBackParams {
                    seq_q: 1,
                    seq_k: 1,
                    dim: 1,
                    scale: 1.0,
                },
            ))
            .unwrap();
        assert_eq!(dv[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(dq[0].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn gla_rejects_without_mutation() {
        let query = [1.0_f32];
        let key = [1.0_f32];
        let value = [1.0_f32];
        let gate = [1.0_f32];
        let decay = [1.0_f32, 1.0];
        let mut output = [7.0_f32; 1];
        let mut kernel = GlaKernel::new();
        assert_eq!(
            kernel.process_event(OpGatedLinearAttn::new(
                &query,
                &key,
                &value,
                &gate,
                &decay,
                &mut output,
                GlaParams {
                    sequence: 1,
                    channels: 1,
                },
            )),
            Err(GlaError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_gated_linear_attn() {
        let query = [2.0_f32];
        let key = [3.0_f32];
        let value = [4.0_f32];
        let gate = [1.0_f32];
        let decay = [0.0_f32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpGatedLinearAttn::new(
                &query,
                &key,
                &value,
                &gate,
                &decay,
                &mut output,
                GlaParams {
                    sequence: 1,
                    channels: 1,
                },
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 24.0_f32.to_bits());
    }
}
