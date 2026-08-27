//! Safe scalar `op_flash_attn_ext` over F32 queries/destinations and F16
//! key/value tensors.
//!
//! This is the bounded portable slice of the pinned
//! `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` contract.  The
//! operation accepts one query token, replicated query heads, active K/V
//! tokens, and an optional scale/total-token pair.  A fixed-capacity workspace
//! is constructed once with the actor; dispatch only reuses its storage.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]
#![allow(clippy::cast_precision_loss)]
use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::f16_matmul::F16View;
use super::tensor_view::{TensorView, TensorViewMut};
use crate::any::quant::fp16_to_f32;

/// Fixed workspace vector capacity from the pinned scalar implementation.
pub const FLASH_ATTN_WORKSPACE_CAPACITY: usize = 4096;

/// Optional operation parameters from the pinned request's parameter bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlashAttnOptions {
    /// Explicit query/key scale. When absent, `1 / sqrt(head_dim)` is used.
    pub scale: Option<f32>,
    /// Total masked token count. When absent, active K/V tokens are used.
    pub masked_total_tokens: Option<u32>,
}

/// Errors returned by flash-attention dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlashAttnError {
    /// The request's dtypes, shapes, layouts, storage, scale, or capacity are
    /// outside the maintained pinned contract.
    InvalidRequest,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for FlashAttnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => formatter.write_str("invalid flash-attention request"),
            Self::UnexpectedEvent => formatter.write_str("unexpected flash-attention event"),
            Self::Internal => formatter.write_str("internal flash-attention dispatch error"),
        }
    }
}

impl std::error::Error for FlashAttnError {}

/// Result returned after one flash-attention dispatch reaches RTC.
pub type FlashAttnResult = Result<(), FlashAttnError>;

/// Extended flash-attention request with F32 Q/destination and F16 K/V.
#[derive(Debug)]
pub struct OpFlashAttnExt<'a> {
    query: TensorView<'a>,
    key: F16View<'a>,
    value: F16View<'a>,
    destination: TensorViewMut<'a>,
    scale: f32,
    masked_total_tokens: u64,
}

impl<'a> OpFlashAttnExt<'a> {
    /// Creates an event; all request validation occurs in actor guards.
    ///
    /// The optional parameter policy is resolved at this public event boundary
    /// to keep the RTC action allocation-free: omitted scale uses the pinned
    /// `1 / sqrt(head_dim)` default and omitted token count uses active K/V
    /// tokens. The event stores those resolved values, so the machine does not
    /// perform an implicit selection in a guard or action. Raw operation
    /// parameter-byte provenance remains outside this bounded API.
    #[must_use]
    pub fn new(
        query: TensorView<'a>,
        key: F16View<'a>,
        value: F16View<'a>,
        destination: TensorViewMut<'a>,
        options: FlashAttnOptions,
    ) -> Self {
        let head_dim = query.layout().ne()[0];
        let scale = options
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        let masked_total_tokens = options
            .masked_total_tokens
            .map_or_else(|| key.layout().ne()[1], u64::from);
        Self {
            query,
            key,
            value,
            destination,
            scale,
            masked_total_tokens,
        }
    }
}

/// Explicitly reports an event outside the maintained attention family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedFlashAttn;

/// Reusable fixed-capacity storage corresponding to the pinned workspace.
#[repr(align(64))]
#[derive(Debug)]
struct FlashAttnWorkspace {
    score_buffer: [f32; FLASH_ATTN_WORKSPACE_CAPACITY],
    value_buffer: [f32; FLASH_ATTN_WORKSPACE_CAPACITY],
    accum_buffer: [f32; FLASH_ATTN_WORKSPACE_CAPACITY],
    q_buffer_f16: [u16; FLASH_ATTN_WORKSPACE_CAPACITY],
    accum_buffer_f16: [u16; FLASH_ATTN_WORKSPACE_CAPACITY],
    prepared_tokens: usize,
    reuse_count: u64,
}

impl FlashAttnWorkspace {
    const fn new() -> Self {
        Self {
            score_buffer: [0.0; FLASH_ATTN_WORKSPACE_CAPACITY],
            value_buffer: [0.0; FLASH_ATTN_WORKSPACE_CAPACITY],
            accum_buffer: [0.0; FLASH_ATTN_WORKSPACE_CAPACITY],
            q_buffer_f16: [0; FLASH_ATTN_WORKSPACE_CAPACITY],
            accum_buffer_f16: [0; FLASH_ATTN_WORKSPACE_CAPACITY],
            prepared_tokens: 0,
            reuse_count: 0,
        }
    }
}

struct Runtime<'a> {
    event: OpFlashAttnExt<'a>,
    workspace: &'a mut FlashAttnWorkspace,
    result: &'a Cell<FlashAttnResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<FlashAttnResult>,
}

#[derive(Default)]
struct Context;

sml! {
    FlashAttnMachine<'dispatch> {
        "ready"_s <= *"ready"_s + FlashAttn(Runtime<'dispatch>)
            [guard_valid] / effect_execute,
        "invalid"_s <= "ready"_s + FlashAttn(Runtime<'dispatch>)
            [guard_invalid] / effect_invalid,
        "ready"_s <= "invalid"_s + completion<_> / effect_recover,

        "unexpected"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "unexpected"_s + completion<_> / effect_recover,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion flash-attention actor.
pub struct FlashAttnKernel {
    machine: FlashAttnMachineStateMachine<Context>,
    workspace: FlashAttnWorkspace,
}

impl fmt::Debug for FlashAttnKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FlashAttnKernel")
            .field("prepared_tokens", &self.workspace.prepared_tokens)
            .field("reuse_count", &self.workspace.reuse_count)
            .finish_non_exhaustive()
    }
}

impl Default for FlashAttnKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl FlashAttnKernel {
    /// Constructs an independent actor and its fixed workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: FlashAttnMachineStateMachine::new(Context),
            workspace: FlashAttnWorkspace::new(),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: FlashAttnEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        let _ = self.machine.is(&FlashAttnMachineStates::Ready);
        output
    }

    /// Returns the active-token count prepared by the last valid dispatch.
    #[must_use]
    pub const fn prepared_tokens(&self) -> usize {
        self.workspace.prepared_tokens
    }

    /// Returns the number of consecutive active-token preparations reused.
    #[must_use]
    pub const fn reuse_count(&self) -> u64 {
        self.workspace.reuse_count
    }

    fn flash_attn(&mut self, event: OpFlashAttnExt<'_>) -> FlashAttnResult {
        let result = Cell::new(Err(FlashAttnError::UnexpectedEvent));
        self.machine
            .process_event(FlashAttnMachineEvents::FlashAttn(Runtime {
                event,
                workspace: &mut self.workspace,
                result: &result,
            }))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedFlashAttn) -> FlashAttnResult {
        let result = Cell::new(Err(FlashAttnError::Internal));
        self.machine
            .process_event(FlashAttnMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }
}

/// Events accepted by [`FlashAttnKernel`].
pub trait FlashAttnEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut FlashAttnKernel) -> Self::Output;
}

impl FlashAttnEvent for OpFlashAttnExt<'_> {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut FlashAttnKernel) -> Self::Output {
        actor.flash_attn(self)
    }
}

impl FlashAttnEvent for UnexpectedFlashAttn {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut FlashAttnKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn valid_request(event: &OpFlashAttnExt<'_>) -> bool {
    let q_shape = event.query.layout().ne();
    let k_shape = event.key.layout().ne();
    let v_shape = event.value.layout().ne();
    let dst_shape = event.destination.layout().ne();
    let head_dim = q_shape[0];
    let query_count = q_shape[1];
    let head_count = q_shape[2];
    let kv_tokens = k_shape[1];
    let kv_head_count = k_shape[2];
    let scale = event.scale;
    let masked_total = event.masked_total_tokens;

    event.query.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.key.validate()
        && event.value.validate()
        && dense_f32_layout(event.query.layout())
        && dense_f32_layout(event.destination.layout())
        && head_dim != 0
        && query_count == 1
        && head_count != 0
        && kv_tokens != 0
        && kv_head_count != 0
        && k_shape[0] == head_dim
        && v_shape[0] == head_dim
        && v_shape[1] == kv_tokens
        && v_shape[2] == kv_head_count
        && dst_shape[0] == head_dim
        && dst_shape[1] == query_count
        && dst_shape[2] == head_count
        && q_shape[3] == 1
        && k_shape[3] == 1
        && v_shape[3] == 1
        && dst_shape[3] == 1
        && head_count.is_multiple_of(kv_head_count)
        && head_dim <= FLASH_ATTN_WORKSPACE_CAPACITY as u64
        && masked_total >= kv_tokens
        && scale.is_finite()
        && scale > 0.0
}

fn dense_f32_layout(layout: super::tensor_view::Layout) -> bool {
    let Some(strides) = layout.effective_f32_strides() else {
        return false;
    };
    if layout.dtype() != super::tensor_view::DType::F32 {
        return false;
    }
    let extents = layout.ne();
    let mut expected = 4_u64;
    let mut dimension = 0;
    while dimension < 4 {
        if strides[dimension] != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(extents[dimension]) else {
            return false;
        };
        expected = next;
        dimension += 1;
    }
    true
}

// Preserve the selected flash-attention scalar recurrence; fused updates can
// change the reference's floating-point rounding.
#[allow(clippy::suboptimal_flops)]
fn execute(mut event: OpFlashAttnExt<'_>, workspace: &mut FlashAttnWorkspace) {
    let q_shape = event.query.layout().ne();
    let k_shape = event.key.layout().ne();
    let head_dim = usize::try_from(q_shape[0]).expect("guard-proven head dimension");
    let head_count = usize::try_from(q_shape[2]).expect("guard-proven head count");
    let kv_tokens = usize::try_from(k_shape[1]).expect("guard-proven token count");
    let kv_head_count = usize::try_from(k_shape[2]).expect("guard-proven KV head count");
    let scale = event.scale;
    if workspace.prepared_tokens == kv_tokens {
        workspace.reuse_count += 1;
    }
    workspace.prepared_tokens = kv_tokens;

    let n_rep = head_count / kv_head_count;
    let mut head = 0;
    while head < head_count {
        let kv_head = head / n_rep;
        let q_base = head * head_dim;
        let dst_base = head * head_dim;
        let mut dim = 0;
        while dim < head_dim {
            workspace.q_buffer_f16[dim] = f32_to_f16(event.query.read(q_base + dim));
            workspace.accum_buffer_f16[dim] = 0;
            dim += 1;
        }

        let mut score_sum = 0.0_f32;
        let mut max_score = f32::NEG_INFINITY;
        let mut token = 0;
        while token < kv_tokens {
            let k_base = (kv_head * kv_tokens + token) * head_dim;
            let score = dot_product_f16_f16_scores(
                &workspace.q_buffer_f16[..head_dim],
                &event.key,
                k_base,
                head_dim,
            ) * scale;
            let old_max = max_score;
            let mut max_scale = 1.0_f32;
            let mut value_scale = 1.0_f32;
            if score > max_score {
                max_score = score;
                max_scale = (old_max - max_score).exp();
                scale_f16_buffer_scalar(&mut workspace.accum_buffer_f16[..head_dim], max_scale);
            } else {
                value_scale = (score - max_score).exp();
            }
            let v_base = (kv_head * kv_tokens + token) * head_dim;
            axpy_f16_buffer_scalar(
                &mut workspace.accum_buffer_f16[..head_dim],
                &event.value,
                v_base,
                value_scale,
                head_dim,
            );
            score_sum = score_sum * max_scale + value_scale;
            token += 1;
        }

        finalize_attention_output(
            &mut event.destination,
            dst_base,
            &workspace.accum_buffer_f16[..head_dim],
            head_dim,
            score_sum,
        );
        head += 1;
    }
}

#[allow(clippy::suboptimal_flops)]
fn finalize_attention_output(
    destination: &mut TensorViewMut<'_>,
    dst_base: usize,
    accumulator: &[u16],
    head_dim: usize,
    score_sum: f32,
) {
    if score_sum == 0.0 {
        // The pinned implementation explicitly overwrites the output when
        // normalization has no mass. This also prevents a NaN in the
        // accumulator from leaking through `NaN * 0.0`.
        let mut dim = 0;
        while dim < head_dim {
            destination.write(dst_base + dim, 0.0);
            dim += 1;
        }
    } else {
        let inverse_score_sum = 1.0 / score_sum;
        let mut dim = 0;
        while dim < head_dim {
            let value = fp16_to_f32(accumulator[dim]);
            destination.write(dst_base + dim, value * inverse_score_sum);
            dim += 1;
        }
    }
}

fn round_fp16_scalar(value: f32) -> f32 {
    fp16_to_f32(f32_to_f16(value))
}

// Preserve the pinned F64 product/accumulation order for parity.
#[allow(clippy::suboptimal_flops)]
fn dot_product_f16_f16_scores(
    lhs: &[u16],
    rhs: &F16View<'_>,
    rhs_base: usize,
    count: usize,
) -> f32 {
    let mut sum = 0.0_f64;
    let mut index = 0;
    while index < count {
        sum += f64::from(fp16_to_f32(lhs[index])) * f64::from(rhs.read(rhs_base + index));
        index += 1;
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        sum as f32
    }
}

fn scale_f16_buffer_scalar(data: &mut [u16], scale: f32) {
    let rounded_scale = round_fp16_scalar(scale);
    let mut index = 0;
    while index < data.len() {
        let rounded_value = fp16_to_f32(data[index]);
        data[index] = f32_to_f16(round_fp16_scalar(rounded_value * rounded_scale));
        index += 1;
    }
}

fn axpy_f16_buffer_scalar(
    dst: &mut [u16],
    src: &F16View<'_>,
    src_base: usize,
    alpha: f32,
    count: usize,
) {
    let rounded_alpha = round_fp16_scalar(alpha);
    let mut index = 0;
    while index < count {
        let rounded_dst = fp16_to_f32(dst[index]);
        let rounded_src = src.read(src_base + index);
        dst[index] = f32_to_f16(round_fp16_scalar(rounded_dst + rounded_src * rounded_alpha));
        index += 1;
    }
}

impl FlashAttnMachineStateMachineContext for Context {
    fn guard_valid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(valid_request(&event.event))
    }

    fn guard_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!valid_request(&event.event))
    }

    fn effect_execute(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        // The guard has proven every shape, stride, span, and capacity
        // invariant; this action only executes the selected numeric algorithm.
        execute(event.event, event.workspace);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(FlashAttnError::InvalidRequest));
        Ok(())
    }

    fn effect_recover(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(FlashAttnError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[allow(clippy::cast_possible_truncation, clippy::missing_const_for_fn)]
fn f32_to_f16(value: f32) -> u16 {
    let word = value.to_bits();
    let sign = ((word >> 16) & 0x8000) as u16;
    let exponent = (word >> 23) & 0xff;
    let fraction = word & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exponent > 142 {
        return sign | 0x7c00;
    }
    if exponent < 103 {
        return sign;
    }
    if exponent < 113 {
        let shift = 113 - exponent;
        let mantissa = 0x80_0000 | fraction;
        let rounding = 1_u32 << (shift + 12);
        return sign | ((mantissa + rounding) >> (shift + 13)) as u16;
    }
    let half_exponent = (exponent - 112) as u16;
    let fraction_bits = (fraction >> 13) as u16;
    let mut half = sign | (half_exponent << 10) | fraction_bits;
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 != 0) {
        half = half.wrapping_add(1);
    }
    half
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod numeric_tests {
    use super::*;

    #[test]
    fn helper_rounds_at_binary16_boundaries() {
        assert_eq!(round_fp16_scalar(1.0003), 1.0);
        assert_eq!(round_fp16_scalar(1.0006), 1.000_976_6);

        let mut data = [f32_to_f16(1.0)];
        scale_f16_buffer_scalar(&mut data, 1.0006);
        assert_eq!(fp16_to_f32(data[0]), 1.000_976_6);

        let source = [f32_to_f16(1.0)];
        let layout = super::super::tensor_view::Layout::new(
            super::super::tensor_view::DType::F16,
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        );
        let view = F16View::new(&source, layout);
        let mut destination = [f32_to_f16(1.0)];
        axpy_f16_buffer_scalar(&mut destination, &view, 0, 1.0006, 1);
        assert_eq!(fp16_to_f32(destination[0]), 2.0);
    }

    #[test]
    fn zero_score_sum_overwrites_nan_accumulator() {
        use crate::any::tensor_view::{DType, Layout};

        let layout = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout fits");
        let mut destination = [7.0_f32; 2];
        let mut destination_view = TensorViewMut::new(&mut destination, layout);
        let accumulator = [0x7e00_u16; 2];

        finalize_attention_output(&mut destination_view, 0, &accumulator, 2, 0.0);

        assert_eq!(destination, [0.0; 2]);
    }
}
