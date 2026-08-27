//! Safe `AArch64` `q4_k` matmul for the pinned kernel contract.
//!
//! The n=1 route is `can_run_neon_mul_mat_q4_vector_f32_rhs_request`.  The
//! `n >= 2` route is the generic quantized `mul_mat` branch.  Both call the
//! pinned DOTPROD `dot_q4_k_q8_k_row_neon` reduction: integer `min_sum` plus
//! one `d * float(low_sum + high_sum)` scaled add per block into the row sum.
//! RHS F32 is quantized to `q8_k` with the pinned `nearest_int` contract.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::aarch64::Arch;
use sml::sml;

use crate::any::quant::fp16_to_f32;

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const K_SCALE_SIZE: usize = 12;
const MAX_Q8_K_BLOCKS: usize = 128;
const KMASK1: u32 = 0x3f3f_3f3f;
const KMASK2: u32 = 0x0f0f_0f0f;
const KMASK3: u32 = 0x0303_0303;

/// Errors returned by the `AArch64` `q4_k` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4KError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q4KError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q4_k NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q4_k shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q4_k event"),
            Self::Internal => formatter.write_str("internal q4_k dispatch error"),
        }
    }
}

impl std::error::Error for Q4KError {}

/// A pinned `AArch64` packed `q4_k` by dense F32 vector request.
#[derive(Debug)]
pub struct OpMulMatQ4KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4KVector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k` by dense F32 GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct OpMulMatQ4K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> OpMulMatQ4K<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// Explicitly reports an event outside the `q4_k` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ4K;

/// Result returned by target `q4_k` events.
pub type Q4KResult = Result<(), Q4KError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:569-577";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1504-1528";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6816-6878";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:442-443";
const PINNED_GEMM_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:738-775";
const PINNED_GEMM_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7347-7394";
const PINNED_GEMM_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:522-525";
const SCOPE_RESIDUAL: &str = "q4_k vector and q4_k GEMM n>=2; F16 neon, remaining q2/q3 generic mul_mat, and x86 remain residuals";

/// Event implemented by the target `q4_k` actor.
pub trait Q4KEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q4KKernel, output: &mut [f32]) -> Self::Output;
}

#[derive(Clone, Copy, Debug)]
struct Q8KScratch {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

impl Q8KScratch {
    const ZERO: Self = Self {
        d: 0.0,
        qs: [0; QK_K],
        bsums: [0; QK_K / 16],
    };
}

struct Q4KVectorRuntime<'a> {
    event: OpMulMatQ4KVector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4KResult>,
}

struct Q4KGemmRuntime<'a> {
    event: OpMulMatQ4K<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4KResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q4KResult>,
}

struct Context {
    backend_available: bool,
    q8_scratch: Box<[Q8KScratch]>,
}

sml! {
    Q4KMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q4KVector(Q4KVectorRuntime<'dispatch>) [guard_vector_ready] / effect_vector_execute,
        "ready"_s <= "ready"_s + Q4KVector(Q4KVectorRuntime<'dispatch>) [guard_vector_invalid] / effect_vector_invalid,
        "ready"_s <= "ready"_s + Q4KGemm(Q4KGemmRuntime<'dispatch>) [guard_gemm_ready] / effect_gemm_execute,
        "ready"_s <= "ready"_s + Q4KGemm(Q4KGemmRuntime<'dispatch>) [guard_gemm_invalid] / effect_gemm_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q4_k` actor.
pub struct Q4KKernel {
    machine: Q4KMachineStateMachine<Context>,
}

impl fmt::Debug for Q4KKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Q4KKernel").finish_non_exhaustive()
    }
}

impl Q4KKernel {
    /// Resolves NEON and one-time `q8_k` scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(_backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q4KMachineStateMachine::new(Context {
                backend_available: true,
                q8_scratch: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q4KEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q4KMachineStates::Ready)
    }

    fn vector(&mut self, event: OpMulMatQ4KVector<'_>, output: &mut [f32]) -> Q4KResult {
        let result = Cell::new(Err(Q4KError::UnexpectedEvent));
        self.machine
            .process_event(Q4KMachineEvents::Q4KVector(Q4KVectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q4KError::Internal)?;
        result.get()
    }

    fn gemm(&mut self, event: OpMulMatQ4K<'_>, output: &mut [f32]) -> Q4KResult {
        let result = Cell::new(Err(Q4KError::UnexpectedEvent));
        self.machine
            .process_event(Q4KMachineEvents::Q4KGemm(Q4KGemmRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q4KError::Internal)?;
        result.get()
    }
}

impl Q4KEvent for OpMulMatQ4KVector<'_> {
    type Output = Q4KResult;

    fn dispatch(self, actor: &mut Q4KKernel, output: &mut [f32]) -> Self::Output {
        actor.vector(self, output)
    }
}

impl Q4KEvent for OpMulMatQ4K<'_> {
    type Output = Q4KResult;

    fn dispatch(self, actor: &mut Q4KKernel, output: &mut [f32]) -> Self::Output {
        actor.gemm(self, output)
    }
}

impl Q4KEvent for UnexpectedQ4K {
    type Output = Q4KResult;

    fn dispatch(self, actor: &mut Q4KKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4KError::Internal));
        actor
            .machine
            .process_event(Q4KMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q4KError::Internal)?;
        result.get()
    }
}

impl Q4KMachineStateMachineContext for Context {
    fn guard_vector_ready(&self, event: &Q4KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && vector_request_valid(event))
    }

    fn guard_vector_invalid(&self, event: &Q4KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!vector_request_valid(event))
    }

    fn effect_vector_execute(&mut self, event: Q4KVectorRuntime<'_>) -> Result<(), ()> {
        execute_q4_k_vector(
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
            &mut self.q8_scratch,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_vector_invalid(&mut self, event: Q4KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4KError::InvalidShape));
        Ok(())
    }

    fn guard_gemm_ready(&self, event: &Q4KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && gemm_request_valid(event))
    }

    fn guard_gemm_invalid(&self, event: &Q4KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(!gemm_request_valid(event))
    }

    fn effect_gemm_execute(&mut self, event: Q4KGemmRuntime<'_>) -> Result<(), ()> {
        execute_q4_k_gemm(
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
            event.event.n,
            &mut self.q8_scratch,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_gemm_invalid(&mut self, event: Q4KGemmRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4KError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4KError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn vector_request_valid(event: &Q4KVectorRuntime<'_>) -> bool {
    packed_request_valid(
        event.event.lhs,
        event.event.rhs.len(),
        event.output.len(),
        event.event.m,
        event.event.k,
        1,
    )
}

const fn gemm_request_valid(event: &Q4KGemmRuntime<'_>) -> bool {
    if event.event.n < 2 {
        return false;
    }
    packed_request_valid(
        event.event.lhs,
        event.event.rhs.len(),
        event.output.len(),
        event.event.m,
        event.event.k,
        event.event.n,
    )
}

const fn packed_request_valid(
    lhs: &[u8],
    rhs_len: usize,
    out_len: usize,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if m == 0 || k == 0 || n == 0 || !k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(row_bytes) = blocks.checked_mul(Q4_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = m.checked_mul(row_bytes) else {
        return false;
    };
    let Some(expected_rhs) = k.checked_mul(n) else {
        return false;
    };
    let Some(expected_out) = m.checked_mul(n) else {
        return false;
    };
    lhs.len() == lhs_len && rhs_len == expected_rhs && out_len == expected_out
}

fn execute_q4_k_vector(
    lhs: &[u8],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    scratch: &mut [Q8KScratch],
) {
    let blocks = k / QK_K;
    quantize_q8_k_dense(rhs, scratch, blocks);
    let row_bytes = blocks * Q4_K_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        output[row] = dot_q4_k_q8_k_row(
            &lhs[row * row_bytes..(row + 1) * row_bytes],
            &scratch[..blocks],
        );
        row += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_q4_k_gemm(
    lhs: &[u8],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut [Q8KScratch],
) {
    let blocks = k / QK_K;
    let row_bytes = blocks * Q4_K_BLOCK_BYTES;
    let mut col = 0;
    while col < n {
        quantize_q8_k_strided(rhs, col, n, scratch, blocks);
        let mut row = 0;
        while row < m {
            output[row * n + col] = dot_q4_k_q8_k_row(
                &lhs[row * row_bytes..(row + 1) * row_bytes],
                &scratch[..blocks],
            );
            row += 1;
        }
        col += 1;
    }
}

fn nearest_int(value: f32) -> i32 {
    let biased = value + 12_582_912.0_f32;
    let bits = biased.to_bits() & 0x007f_ffff;
    i32::try_from(bits).expect("nearest_int mantissa fits i32") - 0x0040_0000
}

#[allow(clippy::cast_possible_truncation)]
fn quantize_q8_k_dense(rhs: &[f32], scratch: &mut [Q8KScratch], blocks: usize) {
    quantize_q8_k_strided(rhs, 0, 1, scratch, blocks);
}

#[allow(clippy::cast_possible_truncation)]
fn quantize_q8_k_strided(
    rhs: &[f32],
    column: usize,
    stride: usize,
    scratch: &mut [Q8KScratch],
    blocks: usize,
) {
    let mut block = 0;
    while block < blocks {
        let mut maximum = 0.0_f32;
        let mut absolute = 0.0_f32;
        let mut index = 0;
        while index < QK_K {
            let value = rhs[(block * QK_K + index) * stride + column];
            if value.abs() > absolute {
                absolute = value.abs();
                maximum = value;
            }
            index += 1;
        }
        if absolute == 0.0 {
            scratch[block] = Q8KScratch::ZERO;
            block += 1;
            continue;
        }
        let inverse = -127.0 / maximum;
        scratch[block].d = 1.0 / inverse;
        index = 0;
        while index < QK_K {
            let quantized =
                nearest_int(inverse * rhs[(block * QK_K + index) * stride + column]).min(127);
            scratch[block].qs[index] = quantized as i8;
            index += 1;
        }
        let mut group = 0;
        while group < QK_K / 16 {
            let mut sum = 0_i16;
            index = 0;
            while index < 16 {
                sum += i16::from(scratch[block].qs[group * 16 + index]);
                index += 1;
            }
            scratch[block].bsums[group] = sum;
            group += 1;
        }
        block += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q4_k_q8_k_row(row: &[u8], rhs: &[Q8KScratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let offset = block * Q4_K_BLOCK_BYTES;
        accumulate_q4_k_q8_k_block(
            &row[offset..offset + Q4_K_BLOCK_BYTES],
            &rhs[block],
            &mut sum,
        );
        block += 1;
    }
    sum
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_q4_k_q8_k_block(block: &[u8], rhs: &Q8KScratch, sum: &mut f32) {
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut auxs = [0_u32; 4];
    auxs[0] = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    auxs[1] = u32::from_le_bytes([block[8], block[9], block[10], block[11]]);
    auxs[2] = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
    auxs[3] = ((auxs[2] >> 4) & KMASK2) | (((auxs[1] >> 6) & KMASK3) << 4);
    let aux = auxs[1] & KMASK1;
    auxs[1] = (auxs[2] & KMASK2) | (((auxs[0] >> 6) & KMASK3) << 4);
    auxs[2] = aux;
    auxs[0] &= KMASK1;
    let scale_bytes = [
        auxs[0].to_le_bytes(),
        auxs[1].to_le_bytes(),
        auxs[2].to_le_bytes(),
        auxs[3].to_le_bytes(),
    ];
    let scales = [
        scale_bytes[0][0],
        scale_bytes[0][1],
        scale_bytes[0][2],
        scale_bytes[0][3],
        scale_bytes[1][0],
        scale_bytes[1][1],
        scale_bytes[1][2],
        scale_bytes[1][3],
    ];
    let mins = [
        scale_bytes[2][0],
        scale_bytes[2][1],
        scale_bytes[2][2],
        scale_bytes[2][3],
        scale_bytes[3][0],
        scale_bytes[3][1],
        scale_bytes[3][2],
        scale_bytes[3][3],
    ];
    let mut decoded = [0_i8; QK_K];
    let qs = &block[16..Q4_K_BLOCK_BYTES];
    let mut chunk = 0;
    while chunk < QK_K / 64 {
        let src = chunk * 32;
        let dst = chunk * 64;
        let mut lane = 0;
        while lane < 32 {
            decoded[dst + lane] = i8::try_from(qs[src + lane] & 0x0f).unwrap_or(0);
            decoded[dst + 32 + lane] = i8::try_from(qs[src + lane] >> 4).unwrap_or(0);
            lane += 1;
        }
        chunk += 1;
    }
    let mut min_sum = 0_i32;
    let mut group = 0;
    while group < QK_K / 16 {
        min_sum += i32::from(rhs.bsums[group]) * i32::from(mins[group / 2]);
        group += 1;
    }
    let mut low_sum = 0_i32;
    let mut high_sum = 0_i32;
    chunk = 0;
    while chunk < QK_K / 64 {
        let base = chunk * 64;
        let mut low_dot = 0_i32;
        let mut high_dot = 0_i32;
        let mut index = 0;
        while index < 32 {
            low_dot += i32::from(rhs.qs[base + index]) * i32::from(decoded[base + index]);
            high_dot +=
                i32::from(rhs.qs[base + 32 + index]) * i32::from(decoded[base + 32 + index]);
            index += 1;
        }
        low_sum += low_dot * i32::from(scales[chunk * 2]);
        high_sum += high_dot * i32::from(scales[chunk * 2 + 1]);
        chunk += 1;
    }
    let d = d * rhs.d;
    let dmin = dmin * rhs.d;
    let min_term = dmin * min_sum as f32;
    *sum -= min_term;
    let dot_term = d * (low_sum + high_sum) as f32;
    *sum += dot_term;
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::float_cmp,
        clippy::unreadable_literal
    )]

    use super::{
        MAX_Q8_K_BLOCKS, OpMulMatQ4K, OpMulMatQ4KVector, PINNED_EXECUTION_SPAN,
        PINNED_GEMM_ACTION_SPAN, PINNED_GEMM_GUARD_SPAN, PINNED_GEMM_TRANSITION_SPAN,
        PINNED_GUARD_SPAN, Q4_K_BLOCK_BYTES, Q4KError, Q4KEvent, Q4KKernel, QK_K, SCOPE_RESIDUAL,
        UnexpectedQ4K,
    };

    fn pack_q4_k_row(d: u16, dmin: u16, scale: u8, qs: u8) -> [u8; Q4_K_BLOCK_BYTES] {
        let mut block = [0_u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        let mut index = 0;
        while index < 12 {
            block[4 + index] = scale;
            index += 1;
        }
        index = 0;
        while index < 128 {
            block[16 + index] = qs;
            index += 1;
        }
        block
    }

    #[test]
    fn q4_k_constructs_on_neon() {
        assert!(Q4KKernel::try_new().is_some());
    }

    #[test]
    fn q4_k_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:569-577"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6816-6878"
        );
        assert_eq!(
            PINNED_GEMM_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:738-775"
        );
        assert_eq!(
            PINNED_GEMM_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7347-7394"
        );
        assert_eq!(
            PINNED_GEMM_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:522-525"
        );
        assert!(SCOPE_RESIDUAL.contains("q4_k vector and q4_k GEMM"));
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
    }

    #[test]
    fn q4_k_vector_zero_qs_is_finite() {
        let Some(mut kernel) = Q4KKernel::try_new() else {
            return;
        };
        let lhs = pack_q4_k_row(0x3c00, 0x3800, 1, 0);
        let rhs = [0.25_f32; QK_K];
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ4KVector::new(&lhs, &rhs, 1, QK_K), &mut output),
            Ok(())
        );
        assert!(output[0].is_finite());
        assert!(kernel.is_ready());
    }

    #[test]
    fn q4_k_gemm_matches_independent_column_dots() {
        let Some(mut kernel) = Q4KKernel::try_new() else {
            return;
        };
        let row0 = pack_q4_k_row(0x3c00, 0x3400, 2, 0x13);
        let row1 = pack_q4_k_row(0x4000, 0x3800, 3, 0x24);
        let mut lhs = [0_u8; 2 * Q4_K_BLOCK_BYTES];
        lhs[..Q4_K_BLOCK_BYTES].copy_from_slice(&row0);
        lhs[Q4_K_BLOCK_BYTES..].copy_from_slice(&row1);
        let mut rhs = [0.0_f32; QK_K * 2];
        let mut index = 0;
        while index < QK_K {
            rhs[index * 2] = (index as f32 - 128.0) * 0.03125;
            rhs[index * 2 + 1] = (64.0 - index as f32) * 0.015_625;
            index += 1;
        }
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ4K::new(&lhs, &rhs, 2, QK_K, 2), &mut output),
            Ok(())
        );
        let mut col0 = [0.0_f32; QK_K];
        let mut col1 = [0.0_f32; QK_K];
        index = 0;
        while index < QK_K {
            col0[index] = rhs[index * 2];
            col1[index] = rhs[index * 2 + 1];
            index += 1;
        }
        let mut expected0 = [0.0_f32; 2];
        let mut expected1 = [0.0_f32; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ4KVector::new(&lhs, &col0, 2, QK_K), &mut expected0),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ4KVector::new(&lhs, &col1, 2, QK_K), &mut expected1),
            Ok(())
        );
        assert_eq!(output[0].to_bits(), expected0[0].to_bits());
        assert_eq!(output[2].to_bits(), expected0[1].to_bits());
        assert_eq!(output[1].to_bits(), expected1[0].to_bits());
        assert_eq!(output[3].to_bits(), expected1[1].to_bits());
    }

    #[test]
    fn q4_k_rejects_invalid_and_unexpected() {
        let Some(mut kernel) = Q4KKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 2];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4K::new(&[0; 2], &[0.0; 2], 2, QK_K, 1),
                &mut output
            ),
            Err(Q4KError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(UnexpectedQ4K, &mut []),
            Err(Q4KError::UnexpectedEvent)
        );
        let _ = Q4KEvent::dispatch(UnexpectedQ4K, &mut kernel, &mut []);
    }
}
