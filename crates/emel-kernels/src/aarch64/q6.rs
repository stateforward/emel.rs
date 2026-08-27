//! Safe `AArch64` NEON `q6_k` vector matmul for the pinned kernel contract.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_run_neon_mul_mat_q6_vector_request`: packed `q6_k` `src0` with shape
//! `[k, m]`, dense F32 `src1` with shape `[1, k]`, and dense F32 `dst` with
//! shape `[1, m]`.  Capability detection happens during construction.  The
//! dispatch action receives a typed `pulp::aarch64::Neon` backend and therefore
//! never performs runtime backend selection, dequantizes into an F32 GEMV, or
//! falls back to the portable actor.
//!
//! Integer products use the pinned `vdot`/`vmull` widening contract expressed
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
const Q6_K_BLOCK_BYTES: usize = 210;
const MAX_Q8_K_BLOCKS: usize = 128;

/// Errors returned by the `AArch64` `q6_k` vector actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q6VectorError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q6VectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q6_k vector NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q6_k vector shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q6_k vector event"),
            Self::Internal => formatter.write_str("internal q6_k vector dispatch error"),
        }
    }
}

impl std::error::Error for Q6VectorError {}

/// A pinned `AArch64` packed `q6_k` by dense F32 vector request.
#[derive(Debug)]
pub struct OpMulMatQ6Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6Vector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q6_k` by dense F32 GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct OpMulMatQ6<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> OpMulMatQ6<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// Explicitly reports an event outside the `q6_k` vector API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ6Vector;

/// Result returned by target `q6_k` vector events.
pub type Q6VectorResult = Result<(), Q6VectorError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:331-340";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:933-961";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6317-6367";
const PINNED_DOT_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:3624-3727";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:510-513";
const PINNED_Q6_GEMM_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:738-775";
const PINNED_Q6_GEMM_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7306-7406";
const PINNED_Q6_GEMM_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:522-525";
const SCOPE_RESIDUAL: &str = "q6_k vector and q6_k GEMM n>=2 (packed src0[k,m], dense F32 src1, dense F32 dst); F16 neon, remaining quantized generic mul_mat, and x86 remain residuals";

/// Event implemented by the target `q6_k` vector actor.
pub trait Q6VectorEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q6VectorKernel, output: &mut [f32]) -> Self::Output;
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

struct Q6VectorRuntime<'a> {
    event: OpMulMatQ6Vector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6VectorResult>,
}

struct Q6GemmRuntime<'a> {
    event: OpMulMatQ6<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6VectorResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q6VectorResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[Q8KScratch]>,
}

sml! {
    Q6VectorMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q6Vector(Q6VectorRuntime<'dispatch>) [guard_q6_ready] / effect_q6_execute,
        "ready"_s <= "ready"_s + Q6Vector(Q6VectorRuntime<'dispatch>) [guard_q6_invalid] / effect_q6_invalid,
        "ready"_s <= "ready"_s + Q6Gemm(Q6GemmRuntime<'dispatch>) [guard_q6_gemm_ready] / effect_q6_gemm_execute,
        "ready"_s <= "ready"_s + Q6Gemm(Q6GemmRuntime<'dispatch>) [guard_q6_gemm_invalid] / effect_q6_gemm_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q6_k` vector actor.
pub struct Q6VectorKernel {
    machine: Q6VectorMachineStateMachine<Context>,
}

impl fmt::Debug for Q6VectorKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q6VectorKernel")
            .finish_non_exhaustive()
    }
}

impl Q6VectorKernel {
    /// Resolves NEON and one-time `q8_k` scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q6VectorMachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q6VectorEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q6VectorMachineStates::Ready)
    }

    fn q6_vector(&mut self, event: OpMulMatQ6Vector<'_>, output: &mut [f32]) -> Q6VectorResult {
        let result = Cell::new(Err(Q6VectorError::UnexpectedEvent));
        self.machine
            .process_event(Q6VectorMachineEvents::Q6Vector(Q6VectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q6VectorError::Internal)?;
        result.get()
    }

    fn q6_gemm(&mut self, event: OpMulMatQ6<'_>, output: &mut [f32]) -> Q6VectorResult {
        let result = Cell::new(Err(Q6VectorError::UnexpectedEvent));
        self.machine
            .process_event(Q6VectorMachineEvents::Q6Gemm(Q6GemmRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q6VectorError::Internal)?;
        result.get()
    }
}

impl Q6VectorEvent for OpMulMatQ6Vector<'_> {
    type Output = Q6VectorResult;

    fn dispatch(self, actor: &mut Q6VectorKernel, output: &mut [f32]) -> Self::Output {
        actor.q6_vector(self, output)
    }
}

impl Q6VectorEvent for OpMulMatQ6<'_> {
    type Output = Q6VectorResult;

    fn dispatch(self, actor: &mut Q6VectorKernel, output: &mut [f32]) -> Self::Output {
        actor.q6_gemm(self, output)
    }
}

impl Q6VectorEvent for UnexpectedQ6Vector {
    type Output = Q6VectorResult;

    fn dispatch(self, actor: &mut Q6VectorKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6VectorError::Internal));
        actor
            .machine
            .process_event(Q6VectorMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q6VectorError::Internal)?;
        result.get()
    }
}

impl Q6VectorMachineStateMachineContext for Context {
    fn guard_q6_ready(&self, event: &Q6VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && q6_request_valid(event))
    }

    fn guard_q6_invalid(&self, event: &Q6VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!q6_request_valid(event))
    }

    fn effect_q6_execute(&mut self, event: Q6VectorRuntime<'_>) -> Result<(), ()> {
        execute_q6_vector(
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

    fn effect_q6_invalid(&mut self, event: Q6VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6VectorError::InvalidShape));
        Ok(())
    }

    fn guard_q6_gemm_ready(&self, event: &Q6GemmRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && q6_gemm_request_valid(event))
    }

    fn guard_q6_gemm_invalid(&self, event: &Q6GemmRuntime<'_>) -> Result<bool, ()> {
        Ok(!q6_gemm_request_valid(event))
    }

    fn effect_q6_gemm_execute(&mut self, event: Q6GemmRuntime<'_>) -> Result<(), ()> {
        execute_q6_gemm(
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

    fn effect_q6_gemm_invalid(&mut self, event: Q6GemmRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6VectorError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6VectorError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn q6_request_valid(event: &Q6VectorRuntime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0 || request.k == 0 || !request.k.is_multiple_of(QK_K) {
        return false;
    }
    let Some(blocks) = request.k.checked_div(QK_K) else {
        return false;
    };
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(row_bytes) = blocks.checked_mul(Q6_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = request.m.checked_mul(row_bytes) else {
        return false;
    };
    request.lhs.len() == lhs_len
        && request.rhs.len() == request.k
        && event.output.len() == request.m
}

const fn q6_gemm_request_valid(event: &Q6GemmRuntime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0 || request.k == 0 || request.n < 2 || !request.k.is_multiple_of(QK_K) {
        return false;
    }
    let Some(blocks) = request.k.checked_div(QK_K) else {
        return false;
    };
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(row_bytes) = blocks.checked_mul(Q6_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = request.m.checked_mul(row_bytes) else {
        return false;
    };
    let Some(rhs_len) = request.k.checked_mul(request.n) else {
        return false;
    };
    let Some(out_len) = request.m.checked_mul(request.n) else {
        return false;
    };
    request.lhs.len() == lhs_len && request.rhs.len() == rhs_len && event.output.len() == out_len
}

fn execute_q6_vector(
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
    let row_bytes = blocks * Q6_K_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        let packed = &lhs[row * row_bytes..(row + 1) * row_bytes];
        output[row] = dot_q6_k_q8_k_neon(backend, packed, &scratch[..blocks]);
        row += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_q6_gemm(
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
    let row_bytes = blocks * Q6_K_BLOCK_BYTES;
    let mut col = 0;
    while col < n {
        quantize_q8_k_strided(rhs, col, n, scratch, blocks);
        let mut row = 0;
        while row < m {
            let packed = &lhs[row * row_bytes..(row + 1) * row_bytes];
            output[row * n + col] = dot_q6_k_q8_k_neon(backend, packed, &scratch[..blocks]);
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
    let mut block = 0;
    while block < blocks {
        let base = block * QK_K;
        let mut maximum = 0.0_f32;
        let mut absolute = 0.0_f32;
        let mut index = 0;
        while index < QK_K {
            let value = rhs[base + index];
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
            let quantized = nearest_int(inverse * rhs[base + index]).min(127);
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

// Keep the reference's integer isum minus 32*sum_mins, then one scale product.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q6_k_q8_k_neon(backend: Neon, row: &[u8], rhs: &[Q8KScratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let offset = block * Q6_K_BLOCK_BYTES;
        let ql = &row[offset..offset + 128];
        let qh = &row[offset + 128..offset + 192];
        let mut scales = [0_i8; 16];
        let mut scale_index = 0;
        while scale_index < 16 {
            scales[scale_index] = i8::from_ne_bytes([row[offset + 192 + scale_index]]);
            scale_index += 1;
        }
        let d = fp16_to_f32(u16::from_le_bytes([row[offset + 208], row[offset + 209]]));
        let isum = q6_integer_sum(backend, ql, qh, &scales, &rhs[block].qs);
        let mut sum_mins = 0_i32;
        let mut group = 0;
        while group < 16 {
            sum_mins += i32::from(rhs[block].bsums[group]) * i32::from(scales[group]);
            group += 1;
        }
        sum += (d * rhs[block].d) * (isum - 32 * sum_mins) as f32;
        block += 1;
    }
    sum
}

fn q6_integer_sum(backend: Neon, ql: &[u8], qh: &[u8], scales: &[i8; 16], q8: &[i8; QK_K]) -> i32 {
    let m4b = backend.splat_u8x16(0x0f);
    let mone = backend.splat_u8x16(3);
    let mut isum = 0_i32;
    let mut chunk = 0;
    while chunk < 2 {
        let qh0 = load_u8x16(&qh[chunk * 32..chunk * 32 + 16]);
        let qh1 = load_u8x16(&qh[chunk * 32 + 16..chunk * 32 + 32]);
        let ql0 = load_u8x16(&ql[chunk * 64..chunk * 64 + 16]);
        let ql1 = load_u8x16(&ql[chunk * 64 + 16..chunk * 64 + 32]);
        let ql2 = load_u8x16(&ql[chunk * 64 + 32..chunk * 64 + 48]);
        let ql3 = load_u8x16(&ql[chunk * 64 + 48..chunk * 64 + 64]);
        let q80 = load_i8x16(&q8[chunk * 128..chunk * 128 + 16]);
        let q81 = load_i8x16(&q8[chunk * 128 + 16..chunk * 128 + 32]);
        let q82 = load_i8x16(&q8[chunk * 128 + 32..chunk * 128 + 48]);
        let q83 = load_i8x16(&q8[chunk * 128 + 48..chunk * 128 + 64]);
        let q84 = load_i8x16(&q8[chunk * 128 + 64..chunk * 128 + 80]);
        let q85 = load_i8x16(&q8[chunk * 128 + 80..chunk * 128 + 96]);
        let q86 = load_i8x16(&q8[chunk * 128 + 96..chunk * 128 + 112]);
        let q87 = load_i8x16(&q8[chunk * 128 + 112..chunk * 128 + 128]);
        let scale_base = chunk * 8;
        let h0 = backend.shl_const_u8x16::<4>(backend.and_u8x16(qh0, mone));
        let h1 = backend.shl_const_u8x16::<4>(backend.and_u8x16(qh1, mone));
        let h2 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<2>(qh0), mone));
        let h3 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<2>(qh1), mone));
        isum += i32::from(scales[scale_base])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.and_u8x16(ql0, m4b), h0)),
                q80,
            );
        isum += i32::from(scales[scale_base + 1])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.and_u8x16(ql1, m4b), h1)),
                q81,
            );
        isum += i32::from(scales[scale_base + 2])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.and_u8x16(ql2, m4b), h2)),
                q82,
            );
        isum += i32::from(scales[scale_base + 3])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.and_u8x16(ql3, m4b), h3)),
                q83,
            );
        let h4 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<4>(qh0), mone));
        let h5 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<4>(qh1), mone));
        let h6 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<6>(qh0), mone));
        let h7 = backend
            .shl_const_u8x16::<4>(backend.and_u8x16(backend.shr_const_u8x16::<6>(qh1), mone));
        isum += i32::from(scales[scale_base + 4])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.shr_const_u8x16::<4>(ql0), h4)),
                q84,
            );
        isum += i32::from(scales[scale_base + 5])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.shr_const_u8x16::<4>(ql1), h5)),
                q85,
            );
        isum += i32::from(scales[scale_base + 6])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.shr_const_u8x16::<4>(ql2), h6)),
                q86,
            );
        isum += i32::from(scales[scale_base + 7])
            * widening_i8_dot(
                backend,
                cast(backend.or_u8x16(backend.shr_const_u8x16::<4>(ql3), h7)),
                q87,
            );
        chunk += 1;
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
    horizontal_sum_i16x8(products)
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

fn horizontal_sum_i16x8(values: i16x8) -> i32 {
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
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::unreadable_literal
    )]

    use super::{
        MAX_Q8_K_BLOCKS, OpMulMatQ6, OpMulMatQ6Vector, PINNED_Q6_GEMM_ACTION_SPAN,
        PINNED_Q6_GEMM_GUARD_SPAN, PINNED_Q6_GEMM_TRANSITION_SPAN, Q6_K_BLOCK_BYTES, Q6VectorError,
        Q6VectorEvent, Q6VectorKernel, QK_K, UnexpectedQ6Vector, quantize_q8_k_dense,
    };
    use crate::any::quant::fp16_to_f32;
    use allocation_counter::measure;

    fn pack_q6_row(scale: u16, ql: u8, qh: u8, group_scale: i8) -> [u8; Q6_K_BLOCK_BYTES] {
        let mut block = [0_u8; Q6_K_BLOCK_BYTES];
        let mut index = 0;
        while index < 128 {
            block[index] = ql;
            index += 1;
        }
        index = 0;
        while index < 64 {
            block[128 + index] = qh;
            index += 1;
        }
        index = 0;
        while index < 16 {
            block[192 + index] = group_scale.to_ne_bytes()[0];
            index += 1;
        }
        block[208..210].copy_from_slice(&scale.to_le_bytes());
        block
    }

    fn scalar_q6_dot(row: &[u8], qs: [i8; QK_K], bsums: [i16; 16], rhs_d: f32) -> f32 {
        let ql = &row[..128];
        let qh = &row[128..192];
        let mut scales = [0_i8; 16];
        let mut index = 0;
        while index < 16 {
            scales[index] = i8::from_ne_bytes([row[192 + index]]);
            index += 1;
        }
        let d = fp16_to_f32(u16::from_le_bytes([row[208], row[209]]));
        let mut isum = 0_i32;
        let mut chunk = 0;
        while chunk < 2 {
            let mut group = 0;
            while group < 4 {
                let mut lane = 0;
                while lane < 16 {
                    let ql_byte = ql[chunk * 64 + group * 16 + lane];
                    let qh_byte = qh[chunk * 32 + (group % 2) * 16 + lane];
                    let shift = if group < 2 { 0 } else { 2 };
                    let q6 = i32::from((ql_byte & 0x0f) | (((qh_byte >> shift) & 3) << 4));
                    isum += i32::from(scales[chunk * 8 + group])
                        * q6
                        * i32::from(qs[chunk * 128 + group * 16 + lane]);
                    lane += 1;
                }
                group += 1;
            }
            group = 0;
            while group < 4 {
                let mut lane = 0;
                while lane < 16 {
                    let ql_byte = ql[chunk * 64 + group * 16 + lane];
                    let qh_byte = qh[chunk * 32 + (group % 2) * 16 + lane];
                    let shift = if group < 2 { 4 } else { 6 };
                    let q6 = i32::from((ql_byte >> 4) | (((qh_byte >> shift) & 3) << 4));
                    isum += i32::from(scales[chunk * 8 + 4 + group])
                        * q6
                        * i32::from(qs[chunk * 128 + 64 + group * 16 + lane]);
                    lane += 1;
                }
                group += 1;
            }
            chunk += 1;
        }
        let mut sum_mins = 0_i32;
        index = 0;
        while index < 16 {
            sum_mins += i32::from(bsums[index]) * i32::from(scales[index]);
            index += 1;
        }
        (d * rhs_d) * (isum - 32 * sum_mins) as f32
    }

    #[test]
    fn target_capability_is_resolved_before_dispatch() {
        assert!(Q6VectorKernel::try_new().is_some());
    }

    #[test]
    fn pinned_source_identity_is_explicit() {
        assert_eq!(
            super::PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            super::PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:331-340"
        );
        assert_eq!(
            super::PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:933-961"
        );
        assert_eq!(
            super::PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6317-6367"
        );
        assert_eq!(
            super::PINNED_DOT_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:3624-3727"
        );
        assert_eq!(
            super::PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:510-513"
        );
        assert!(super::SCOPE_RESIDUAL.contains("q6_k vector and q6_k GEMM"));
        assert_eq!(
            PINNED_Q6_GEMM_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:738-775"
        );
        assert_eq!(
            PINNED_Q6_GEMM_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7306-7406"
        );
        assert_eq!(
            PINNED_Q6_GEMM_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:522-525"
        );
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
        assert_eq!(QK_K, 256);
    }

    #[test]
    fn neon_integer_dot_matches_packed_scalar_oracle() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q6_row(0x3c00, 0x5a, 0xa5, 3);
        let mut rhs = [0.0_f32; QK_K];
        let mut index = 0;
        while index < QK_K {
            rhs[index] = (index as i32 % 17 - 8) as f32 * 0.125;
            index += 1;
        }
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Vector::new(&lhs, &rhs, 1, QK_K), &mut output),
            Ok(())
        );
        assert!(kernel.is_ready());
        let mut scratch = vec![super::Q8KScratch::ZERO; 1];
        quantize_q8_k_dense(&rhs, &mut scratch, 1);
        let expected = scalar_q6_dot(&lhs, scratch[0].qs, scratch[0].bsums, scratch[0].d);
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn four_plus_remainder_rows_match_scalar_oracle() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let mut lhs = [0_u8; 5 * Q6_K_BLOCK_BYTES];
        let mut row = 0;
        while row < 5 {
            let packed = pack_q6_row(
                0x3c00,
                u8::try_from(0x11 * (row + 1)).unwrap(),
                0x3c,
                row as i8 + 1,
            );
            lhs[row * Q6_K_BLOCK_BYTES..(row + 1) * Q6_K_BLOCK_BYTES].copy_from_slice(&packed);
            row += 1;
        }
        let mut rhs = [0.0_f32; QK_K];
        let mut index = 0;
        while index < QK_K {
            rhs[index] = (index as i32 % 13 - 6) as f32 * 0.25;
            index += 1;
        }
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Vector::new(&lhs, &rhs, 5, QK_K), &mut output),
            Ok(())
        );
        let mut scratch = vec![super::Q8KScratch::ZERO; 1];
        quantize_q8_k_dense(&rhs, &mut scratch, 1);
        row = 0;
        while row < 5 {
            let packed = &lhs[row * Q6_K_BLOCK_BYTES..(row + 1) * Q6_K_BLOCK_BYTES];
            let expected = scalar_q6_dot(packed, scratch[0].qs, scratch[0].bsums, scratch[0].d);
            assert_eq!(output[row].to_bits(), expected.to_bits());
            row += 1;
        }
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Vector::new(&[1, 2], &[3.0], 2, 2), &mut output),
            Err(Q6VectorError::InvalidShape)
        );
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ6Vector, &mut []),
            Err(Q6VectorError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_trait_remains_dispatch_surface() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q6_row(0x3c00, 0, 0, 0);
        let rhs = [1.0_f32; QK_K];
        let mut output = [7.0];
        assert_eq!(
            Q6VectorEvent::dispatch(
                OpMulMatQ6Vector::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut output
            ),
            Ok(())
        );
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q6_row(0x3c00, 0, 0, 0);
        let rhs = [1.0_f32; QK_K];
        let mut output = [0.0];
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(OpMulMatQ6Vector::new(&lhs, &rhs, 1, QK_K), &mut output),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn neon_q6_gemm_matches_independent_column_dots() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let row0 = pack_q6_row(0x3c00, 0x12, 0x34, 2);
        let row1 = pack_q6_row(0x4000, 0x56, 0x78, -1);
        let mut lhs = [0_u8; 2 * Q6_K_BLOCK_BYTES];
        lhs[..Q6_K_BLOCK_BYTES].copy_from_slice(&row0);
        lhs[Q6_K_BLOCK_BYTES..].copy_from_slice(&row1);
        let mut rhs = [0.0_f32; QK_K * 2];
        let mut index = 0;
        while index < QK_K {
            rhs[index * 2] = (index as f32 - 128.0) * 0.03125;
            rhs[index * 2 + 1] = (64.0 - index as f32) * 0.015_625;
            index += 1;
        }
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ6::new(&lhs, &rhs, 2, QK_K, 2), &mut output),
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
        let mut expected = [0.0_f32; 4];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6Vector::new(&lhs, &col0, 2, QK_K),
                &mut expected[..2]
            ),
            Ok(())
        );
        let mut expected_col1 = [0.0_f32; 2];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6Vector::new(&lhs, &col1, 2, QK_K),
                &mut expected_col1
            ),
            Ok(())
        );
        assert_eq!(output[0].to_bits(), expected[0].to_bits());
        assert_eq!(output[2].to_bits(), expected[1].to_bits());
        assert_eq!(output[1].to_bits(), expected_col1[0].to_bits());
        assert_eq!(output[3].to_bits(), expected_col1[1].to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn q6_gemm_rejects_vector_n() {
        let Some(mut kernel) = Q6VectorKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ6::new(&[0; 2], &[0.0; 2], 2, QK_K, 1), &mut output),
            Err(Q6VectorError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 2]);
    }
}
