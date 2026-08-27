//! Safe `AArch64` packed `q2_k` matmul for the pinned kernel contract.
//!
//! C++ has no specialized `q2` vector row.  Both n=1 and n>=2 use the generic
//! quantized `mul_mat` branch, whose `q2_k` body is `dot_q2_k_q8_k_row_neon`,
//! which returns the DOTPROD `dot_q2_k_q8_k_row_scalar` neon vdot sequence.
//! Capability detection happens during construction.  The dispatch action
//! receives a typed `pulp::aarch64::Neon` backend and therefore never performs
//! runtime backend selection, dequantizes into an F32 GEMV, or falls back to
//! the portable actor.
//!
//! Integer products use the pinned `vdotq_s32` widening contract expressed
//! through pulp's stable `i16` multiply after an explicit 16-lane widen.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{
    aarch64::{Arch, Neon},
    cast, i8x16, i16x8, u8x16,
};
use sml::sml;

use crate::any::quant::fp16_to_f32;

const QK_K: usize = 256;
const Q2_K_BLOCK_BYTES: usize = 84;
const MAX_Q8_K_BLOCKS: usize = 128;

/// Errors returned by the `AArch64` `q2_k` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q2KError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q2KError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q2_k NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q2_k shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q2_k event"),
            Self::Internal => formatter.write_str("internal q2_k dispatch error"),
        }
    }
}

impl std::error::Error for Q2KError {}

/// A pinned `AArch64` packed `q2_k` by dense F32 vector request.
#[derive(Debug)]
pub struct OpMulMatQ2KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ2KVector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q2_k` by dense F32 GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct OpMulMatQ2K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> OpMulMatQ2K<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// Explicitly reports an event outside the `q2_k` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ2K;

/// Result returned by target `q2_k` events.
pub type Q2KResult = Result<(), Q2KError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_DOT_SPAN: &str = "src/emel/kernel/detail.hpp:2459-2555";
const PINNED_ROW_NEON_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:2520-2523";
const PINNED_GEMM_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:738-775";
const PINNED_GEMM_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7347-7394";
const PINNED_GEMM_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:522-525";
const SCOPE_RESIDUAL: &str = "q2_k vector n=1 and q2_k GEMM n>=2 share the generic neon vdot formula; F16 neon, remaining q3 generic mul_mat, and x86 remain residuals";

/// Event implemented by the target `q2_k` actor.
pub trait Q2KEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q2KKernel, output: &mut [f32]) -> Self::Output;
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

struct Q2KVectorRuntime<'a> {
    event: OpMulMatQ2KVector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q2KResult>,
}

struct Q2KGemmRuntime<'a> {
    event: OpMulMatQ2K<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q2KResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q2KResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[Q8KScratch]>,
}

sml! {
    Q2KMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q2KVector(Q2KVectorRuntime<'dispatch>) [guard_vector_ready] / effect_vector_execute,
        "ready"_s <= "ready"_s + Q2KVector(Q2KVectorRuntime<'dispatch>) [guard_vector_invalid] / effect_vector_invalid,
        "ready"_s <= "ready"_s + Q2KGemm(Q2KGemmRuntime<'dispatch>) [guard_gemm_ready] / effect_gemm_execute,
        "ready"_s <= "ready"_s + Q2KGemm(Q2KGemmRuntime<'dispatch>) [guard_gemm_invalid] / effect_gemm_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q2_k` actor.
pub struct Q2KKernel {
    machine: Q2KMachineStateMachine<Context>,
}

impl fmt::Debug for Q2KKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Q2KKernel").finish_non_exhaustive()
    }
}

impl Q2KKernel {
    /// Resolves NEON and one-time `q8_k` scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q2KMachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q2KEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q2KMachineStates::Ready)
    }

    fn vector(&mut self, event: OpMulMatQ2KVector<'_>, output: &mut [f32]) -> Q2KResult {
        let result = Cell::new(Err(Q2KError::UnexpectedEvent));
        self.machine
            .process_event(Q2KMachineEvents::Q2KVector(Q2KVectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q2KError::Internal)?;
        result.get()
    }

    fn gemm(&mut self, event: OpMulMatQ2K<'_>, output: &mut [f32]) -> Q2KResult {
        let result = Cell::new(Err(Q2KError::UnexpectedEvent));
        self.machine
            .process_event(Q2KMachineEvents::Q2KGemm(Q2KGemmRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q2KError::Internal)?;
        result.get()
    }
}

impl Q2KEvent for OpMulMatQ2KVector<'_> {
    type Output = Q2KResult;

    fn dispatch(self, actor: &mut Q2KKernel, output: &mut [f32]) -> Self::Output {
        actor.vector(self, output)
    }
}

impl Q2KEvent for OpMulMatQ2K<'_> {
    type Output = Q2KResult;

    fn dispatch(self, actor: &mut Q2KKernel, output: &mut [f32]) -> Self::Output {
        actor.gemm(self, output)
    }
}

impl Q2KEvent for UnexpectedQ2K {
    type Output = Q2KResult;

    fn dispatch(self, actor: &mut Q2KKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q2KError::Internal));
        actor
            .machine
            .process_event(Q2KMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q2KError::Internal)?;
        result.get()
    }
}

impl Q2KMachineStateMachineContext for Context {
    fn guard_vector_ready(&self, event: &Q2KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && vector_request_valid(event))
    }

    fn guard_vector_invalid(&self, event: &Q2KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!vector_request_valid(event))
    }

    fn effect_vector_execute(&mut self, event: Q2KVectorRuntime<'_>) -> Result<(), ()> {
        execute_q2_k_vector(
            self.backend,
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

    fn effect_vector_invalid(&mut self, event: Q2KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q2KError::InvalidShape));
        Ok(())
    }

    fn guard_gemm_ready(&self, event: &Q2KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && gemm_request_valid(event))
    }

    fn guard_gemm_invalid(&self, event: &Q2KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(!gemm_request_valid(event))
    }

    fn effect_gemm_execute(&mut self, event: Q2KGemmRuntime<'_>) -> Result<(), ()> {
        execute_q2_k_gemm(
            self.backend,
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

    fn effect_gemm_invalid(&mut self, event: Q2KGemmRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q2KError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q2KError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn vector_request_valid(event: &Q2KVectorRuntime<'_>) -> bool {
    packed_request_valid(
        event.event.lhs,
        event.event.rhs.len(),
        event.output.len(),
        event.event.m,
        event.event.k,
        1,
    )
}

const fn gemm_request_valid(event: &Q2KGemmRuntime<'_>) -> bool {
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
    let Some(row_bytes) = blocks.checked_mul(Q2_K_BLOCK_BYTES) else {
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

fn execute_q2_k_vector(
    backend: Neon,
    lhs: &[u8],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    scratch: &mut [Q8KScratch],
) {
    let blocks = k / QK_K;
    quantize_q8_k_dense(rhs, scratch, blocks);
    let row_bytes = blocks * Q2_K_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        output[row] = dot_q2_k_q8_k_neon(
            backend,
            &lhs[row * row_bytes..(row + 1) * row_bytes],
            &scratch[..blocks],
        );
        row += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_q2_k_gemm(
    backend: Neon,
    lhs: &[u8],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut [Q8KScratch],
) {
    let blocks = k / QK_K;
    let row_bytes = blocks * Q2_K_BLOCK_BYTES;
    let mut col = 0;
    while col < n {
        quantize_q8_k_strided(rhs, col, n, scratch, blocks);
        let mut row = 0;
        while row < m {
            output[row * n + col] = dot_q2_k_q8_k_neon(
                backend,
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
fn dot_q2_k_q8_k_neon(backend: Neon, row: &[u8], rhs: &[Q8KScratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let packed = &row[block * Q2_K_BLOCK_BYTES..(block + 1) * Q2_K_BLOCK_BYTES];
        let d = rhs[block].d * fp16_to_f32(u16::from_le_bytes([packed[80], packed[81]]));
        let dmin = -rhs[block].d * fp16_to_f32(u16::from_le_bytes([packed[82], packed[83]]));
        let mins_and_scales = load_u8x16(&packed[..16]);
        let scale_mask = backend.splat_u8x16(0x0f);
        let scales: [u8; 16] = cast(backend.and_u8x16(mins_and_scales, scale_mask));
        let mins: [u8; 16] = cast(backend.shr_const_u8x16::<4>(mins_and_scales));
        let mut min_sum = 0_i32;
        let mut group = 0;
        while group < 16 {
            min_sum += i32::from(mins[group]) * i32::from(rhs[block].bsums[group]);
            group += 1;
        }
        sum += dmin * min_sum as f32;
        sum += d * q2_integer_sum(backend, &packed[16..80], &scales, &rhs[block].qs) as f32;
        block += 1;
    }
    sum
}

fn q2_integer_sum(backend: Neon, qs: &[u8], scales: &[u8; 16], q8: &[i8; QK_K]) -> i32 {
    let m3 = backend.splat_u8x16(0x03);
    let mut isum = 0_i32;
    let mut tile = 0;
    while tile < 2 {
        let q2bits0 = load_u8x16(&qs[tile * 32..tile * 32 + 16]);
        let q2bits1 = load_u8x16(&qs[tile * 32 + 16..tile * 32 + 32]);
        let q8_base = tile * 128;
        let scale_base = tile * 8;
        let q80 = load_i8x16(&q8[q8_base..q8_base + 16]);
        let q81 = load_i8x16(&q8[q8_base + 16..q8_base + 32]);
        isum += i32::from(scales[scale_base])
            * widening_i8_dot(backend, cast(backend.and_u8x16(q2bits0, m3)), q80);
        isum += i32::from(scales[scale_base + 1])
            * widening_i8_dot(backend, cast(backend.and_u8x16(q2bits1, m3)), q81);
        let q82 = load_i8x16(&q8[q8_base + 32..q8_base + 48]);
        let q83 = load_i8x16(&q8[q8_base + 48..q8_base + 64]);
        isum += i32::from(scales[scale_base + 2])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<2>(q2bits0), m3)),
                q82,
            );
        isum += i32::from(scales[scale_base + 3])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<2>(q2bits1), m3)),
                q83,
            );
        let q84 = load_i8x16(&q8[q8_base + 64..q8_base + 80]);
        let q85 = load_i8x16(&q8[q8_base + 80..q8_base + 96]);
        isum += i32::from(scales[scale_base + 4])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<4>(q2bits0), m3)),
                q84,
            );
        isum += i32::from(scales[scale_base + 5])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<4>(q2bits1), m3)),
                q85,
            );
        let q86 = load_i8x16(&q8[q8_base + 96..q8_base + 112]);
        let q87 = load_i8x16(&q8[q8_base + 112..q8_base + 128]);
        isum += i32::from(scales[scale_base + 6])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<6>(q2bits0), m3)),
                q86,
            );
        isum += i32::from(scales[scale_base + 7])
            * widening_i8_dot(
                backend,
                cast(backend.and_u8x16(backend.shr_const_u8x16::<6>(q2bits1), m3)),
                q87,
            );
        tile += 1;
    }
    isum
}

fn load_u8x16(bytes: &[u8]) -> u8x16 {
    let mut lanes = [0_u8; 16];
    lanes.copy_from_slice(&bytes[..16]);
    cast(lanes)
}

fn load_i8x16(bytes: &[i8]) -> i8x16 {
    let mut lanes = [0_i8; 16];
    lanes.copy_from_slice(&bytes[..16]);
    cast(lanes)
}

fn widening_i8_dot(backend: Neon, lhs: i8x16, rhs: i8x16) -> i32 {
    let (lhs_lo, lhs_hi) = widen_i8x16(lhs);
    let (rhs_lo, rhs_hi) = widen_i8x16(rhs);
    let products = backend.wrapping_add_i16x8(
        backend.mul_i16x8(lhs_lo, rhs_lo),
        backend.mul_i16x8(lhs_hi, rhs_hi),
    );
    reduce_sum_i16x8(products)
}

fn widen_i8x16(values: i8x16) -> (i16x8, i16x8) {
    let lanes: [i8; 16] = cast(values);
    (
        i16x8(
            i16::from(lanes[0]),
            i16::from(lanes[1]),
            i16::from(lanes[2]),
            i16::from(lanes[3]),
            i16::from(lanes[4]),
            i16::from(lanes[5]),
            i16::from(lanes[6]),
            i16::from(lanes[7]),
        ),
        i16x8(
            i16::from(lanes[8]),
            i16::from(lanes[9]),
            i16::from(lanes[10]),
            i16::from(lanes[11]),
            i16::from(lanes[12]),
            i16::from(lanes[13]),
            i16::from(lanes[14]),
            i16::from(lanes[15]),
        ),
    )
}

fn reduce_sum_i16x8(values: i16x8) -> i32 {
    let lanes: [i16; 8] = cast(values);
    let mut sum = 0_i32;
    let mut index = 0;
    while index < lanes.len() {
        sum += i32::from(lanes[index]);
        index += 1;
    }
    sum
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
        MAX_Q8_K_BLOCKS, OpMulMatQ2K, OpMulMatQ2KVector, PINNED_DOT_SPAN, PINNED_EMEL_CPP_COMMIT,
        PINNED_GEMM_ACTION_SPAN, PINNED_GEMM_GUARD_SPAN, PINNED_GEMM_TRANSITION_SPAN,
        PINNED_ROW_NEON_SPAN, Q2_K_BLOCK_BYTES, Q2KError, Q2KEvent, Q2KKernel, QK_K,
        SCOPE_RESIDUAL, UnexpectedQ2K,
    };

    fn pack_q2_k_row(d: u16, dmin: u16, scale: u8, qs: u8) -> [u8; Q2_K_BLOCK_BYTES] {
        let mut block = [0_u8; Q2_K_BLOCK_BYTES];
        let mut index = 0;
        while index < 16 {
            block[index] = scale;
            index += 1;
        }
        index = 0;
        while index < 64 {
            block[16 + index] = qs;
            index += 1;
        }
        block[80..82].copy_from_slice(&d.to_le_bytes());
        block[82..84].copy_from_slice(&dmin.to_le_bytes());
        block
    }

    #[test]
    fn q2_k_constructs_on_neon() {
        assert!(Q2KKernel::try_new().is_some());
    }

    #[test]
    fn q2_k_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(PINNED_DOT_SPAN, "src/emel/kernel/detail.hpp:2459-2555");
        assert_eq!(
            PINNED_ROW_NEON_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:2520-2523"
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
        assert!(SCOPE_RESIDUAL.contains("q2_k vector n=1 and q2_k GEMM"));
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(Q2_K_BLOCK_BYTES, 84);
    }

    #[test]
    fn q2_k_vector_zero_qs_is_finite() {
        let Some(mut kernel) = Q2KKernel::try_new() else {
            return;
        };
        let lhs = pack_q2_k_row(0x3c00, 0x3800, 1, 0);
        let rhs = [0.25_f32; QK_K];
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ2KVector::new(&lhs, &rhs, 1, QK_K), &mut output),
            Ok(())
        );
        assert!(output[0].is_finite());
        assert!(kernel.is_ready());
    }

    #[test]
    fn q2_k_gemm_matches_independent_column_dots() {
        let Some(mut kernel) = Q2KKernel::try_new() else {
            return;
        };
        let row0 = pack_q2_k_row(0x3c00, 0x3400, 0x12, 0xe4);
        let row1 = pack_q2_k_row(0x4000, 0x3800, 0x34, 0x1b);
        let mut lhs = [0_u8; 2 * Q2_K_BLOCK_BYTES];
        lhs[..Q2_K_BLOCK_BYTES].copy_from_slice(&row0);
        lhs[Q2_K_BLOCK_BYTES..].copy_from_slice(&row1);
        let mut rhs = [0.0_f32; QK_K * 2];
        let mut index = 0;
        while index < QK_K {
            rhs[index * 2] = (index as f32 - 128.0) * 0.03125;
            rhs[index * 2 + 1] = (64.0 - index as f32) * 0.015_625;
            index += 1;
        }
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ2K::new(&lhs, &rhs, 2, QK_K, 2), &mut output),
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
            kernel.process_event(OpMulMatQ2KVector::new(&lhs, &col0, 2, QK_K), &mut expected0),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ2KVector::new(&lhs, &col1, 2, QK_K), &mut expected1),
            Ok(())
        );
        assert_eq!(output[0].to_bits(), expected0[0].to_bits());
        assert_eq!(output[2].to_bits(), expected0[1].to_bits());
        assert_eq!(output[1].to_bits(), expected1[0].to_bits());
        assert_eq!(output[3].to_bits(), expected1[1].to_bits());
    }

    #[test]
    fn q2_k_rejects_invalid_and_unexpected() {
        let Some(mut kernel) = Q2KKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 2];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ2K::new(&[0; 2], &[0.0; 2], 2, QK_K, 1),
                &mut output
            ),
            Err(Q2KError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ2K::new(&[], &[], 0, 0, 0), &mut output),
            Err(Q2KError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(UnexpectedQ2K, &mut []),
            Err(Q2KError::UnexpectedEvent)
        );
        let _ = Q2KEvent::dispatch(UnexpectedQ2K, &mut kernel, &mut []);
    }
}
