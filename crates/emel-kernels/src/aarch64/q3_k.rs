//! Safe `AArch64` packed `q3_k` matmul for the pinned kernel contract.
//!
//! C++ has no specialized `q3` vector row.  Both n=1 and n>=2 use the generic
//! quantized `mul_mat` branch, whose `q3_k` body is `dot_q3_k_q8_k_row_neon`.
//! On a DOTPROD host that function is a real neon vdot kernel, not the scalar
//! fallback.  Capability detection happens during construction.  The dispatch
//! action receives a typed `pulp::aarch64::Neon` backend and therefore never
//! performs runtime backend selection, dequantizes into an F32 GEMV, or falls
//! back to the portable actor.
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
const Q3_K_BLOCK_BYTES: usize = 110;
const MAX_Q8_K_BLOCKS: usize = 128;
const KMASK1: u32 = 0x0303_0303;
const KMASK2: u32 = 0x0f0f_0f0f;

/// Errors returned by the `AArch64` `q3_k` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q3KError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q3KError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q3_k NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q3_k shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q3_k event"),
            Self::Internal => formatter.write_str("internal q3_k dispatch error"),
        }
    }
}

impl std::error::Error for Q3KError {}

/// A pinned `AArch64` packed `q3_k` by dense F32 vector request.
#[derive(Debug)]
pub struct OpMulMatQ3KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ3KVector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q3_k` by dense F32 GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct OpMulMatQ3K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> OpMulMatQ3K<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// Explicitly reports an event outside the `q3_k` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ3K;

/// Result returned by target `q3_k` events.
pub type Q3KResult = Result<(), Q3KError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_ROW_NEON_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:2640-2775";
const PINNED_GEMM_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:738-775";
const PINNED_GEMM_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7347-7398";
const PINNED_GEMM_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:522-525";
const SCOPE_RESIDUAL: &str = "q3_k vector n=1 and q3_k GEMM n>=2 share the generic neon vdot formula; F16 neon, remaining quantized generic mul_mat, and x86 remain residuals";

/// Event implemented by the target `q3_k` actor.
pub trait Q3KEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q3KKernel, output: &mut [f32]) -> Self::Output;
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

struct Q3KVectorRuntime<'a> {
    event: OpMulMatQ3KVector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q3KResult>,
}

struct Q3KGemmRuntime<'a> {
    event: OpMulMatQ3K<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q3KResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q3KResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[Q8KScratch]>,
}

sml! {
    Q3KMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q3KVector(Q3KVectorRuntime<'dispatch>) [guard_vector_ready] / effect_vector_execute,
        "ready"_s <= "ready"_s + Q3KVector(Q3KVectorRuntime<'dispatch>) [guard_vector_invalid] / effect_vector_invalid,
        "ready"_s <= "ready"_s + Q3KGemm(Q3KGemmRuntime<'dispatch>) [guard_gemm_ready] / effect_gemm_execute,
        "ready"_s <= "ready"_s + Q3KGemm(Q3KGemmRuntime<'dispatch>) [guard_gemm_invalid] / effect_gemm_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q3_k` actor.
pub struct Q3KKernel {
    machine: Q3KMachineStateMachine<Context>,
}

impl fmt::Debug for Q3KKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Q3KKernel").finish_non_exhaustive()
    }
}

impl Q3KKernel {
    /// Resolves NEON and one-time `q8_k` scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q3KMachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q3KEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q3KMachineStates::Ready)
    }

    fn vector(&mut self, event: OpMulMatQ3KVector<'_>, output: &mut [f32]) -> Q3KResult {
        let result = Cell::new(Err(Q3KError::UnexpectedEvent));
        self.machine
            .process_event(Q3KMachineEvents::Q3KVector(Q3KVectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q3KError::Internal)?;
        result.get()
    }

    fn gemm(&mut self, event: OpMulMatQ3K<'_>, output: &mut [f32]) -> Q3KResult {
        let result = Cell::new(Err(Q3KError::UnexpectedEvent));
        self.machine
            .process_event(Q3KMachineEvents::Q3KGemm(Q3KGemmRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q3KError::Internal)?;
        result.get()
    }
}

impl Q3KEvent for OpMulMatQ3KVector<'_> {
    type Output = Q3KResult;

    fn dispatch(self, actor: &mut Q3KKernel, output: &mut [f32]) -> Self::Output {
        actor.vector(self, output)
    }
}

impl Q3KEvent for OpMulMatQ3K<'_> {
    type Output = Q3KResult;

    fn dispatch(self, actor: &mut Q3KKernel, output: &mut [f32]) -> Self::Output {
        actor.gemm(self, output)
    }
}

impl Q3KEvent for UnexpectedQ3K {
    type Output = Q3KResult;

    fn dispatch(self, actor: &mut Q3KKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q3KError::Internal));
        actor
            .machine
            .process_event(Q3KMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q3KError::Internal)?;
        result.get()
    }
}

impl Q3KMachineStateMachineContext for Context {
    fn guard_vector_ready(&self, event: &Q3KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && vector_request_valid(event))
    }

    fn guard_vector_invalid(&self, event: &Q3KVectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!vector_request_valid(event))
    }

    fn effect_vector_execute(&mut self, event: Q3KVectorRuntime<'_>) -> Result<(), ()> {
        execute_q3_k_vector(
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

    fn effect_vector_invalid(&mut self, event: Q3KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q3KError::InvalidShape));
        Ok(())
    }

    fn guard_gemm_ready(&self, event: &Q3KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && gemm_request_valid(event))
    }

    fn guard_gemm_invalid(&self, event: &Q3KGemmRuntime<'_>) -> Result<bool, ()> {
        Ok(!gemm_request_valid(event))
    }

    fn effect_gemm_execute(&mut self, event: Q3KGemmRuntime<'_>) -> Result<(), ()> {
        execute_q3_k_gemm(
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

    fn effect_gemm_invalid(&mut self, event: Q3KGemmRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q3KError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q3KError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn vector_request_valid(event: &Q3KVectorRuntime<'_>) -> bool {
    packed_request_valid(
        event.event.lhs,
        event.event.rhs.len(),
        event.output.len(),
        event.event.m,
        event.event.k,
        1,
    )
}

const fn gemm_request_valid(event: &Q3KGemmRuntime<'_>) -> bool {
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
    let Some(row_bytes) = blocks.checked_mul(Q3_K_BLOCK_BYTES) else {
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

fn execute_q3_k_vector(
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
    let row_bytes = blocks * Q3_K_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        output[row] = dot_q3_k_q8_k_neon(
            backend,
            &lhs[row * row_bytes..(row + 1) * row_bytes],
            &scratch[..blocks],
        );
        row += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_q3_k_gemm(
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
    let row_bytes = blocks * Q3_K_BLOCK_BYTES;
    let mut col = 0;
    while col < n {
        quantize_q8_k_strided(rhs, col, n, scratch, blocks);
        let mut row = 0;
        while row < m {
            output[row * n + col] = dot_q3_k_q8_k_neon(
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

const fn unpack_q3_scales(scales: &[u8]) -> [i8; 16] {
    let aux0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let aux1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let aux2 = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let utmp0 = (aux0 & KMASK2) | ((aux2 & KMASK1) << 4);
    let utmp1 = (aux1 & KMASK2) | (((aux2 >> 2) & KMASK1) << 4);
    let utmp2 = ((aux0 >> 4) & KMASK2) | (((aux2 >> 4) & KMASK1) << 4);
    let utmp3 = ((aux1 >> 4) & KMASK2) | (((aux2 >> 6) & KMASK1) << 4);
    let bytes = [
        utmp0.to_le_bytes(),
        utmp1.to_le_bytes(),
        utmp2.to_le_bytes(),
        utmp3.to_le_bytes(),
    ];
    [
        i8::from_ne_bytes([bytes[0][0]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[0][1]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[0][2]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[0][3]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[1][0]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[1][1]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[1][2]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[1][3]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[2][0]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[2][1]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[2][2]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[2][3]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[3][0]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[3][1]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[3][2]]).wrapping_sub(32),
        i8::from_ne_bytes([bytes[3][3]]).wrapping_sub(32),
    ]
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q3_k_q8_k_neon(backend: Neon, row: &[u8], rhs: &[Q8KScratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let packed = &row[block * Q3_K_BLOCK_BYTES..(block + 1) * Q3_K_BLOCK_BYTES];
        let d = rhs[block].d * fp16_to_f32(u16::from_le_bytes([packed[108], packed[109]]));
        let scales = unpack_q3_scales(&packed[96..108]);
        let isum = q3_integer_sum(
            backend,
            &packed[32..96],
            &packed[..32],
            &scales,
            &rhs[block].qs,
        );
        sum += d * isum as f32;
        block += 1;
    }
    sum
}

fn q3_integer_sum(
    backend: Neon,
    qs: &[u8],
    hmask: &[u8],
    scales: &[i8; 16],
    q8: &[i8; QK_K],
) -> i32 {
    let m3b = backend.splat_u8x16(0x03);
    let m0 = backend.splat_u8x16(1);
    let m1 = backend.shl_const_u8x16::<1>(m0);
    let m2 = backend.shl_const_u8x16::<2>(m0);
    let m3 = backend.shl_const_u8x16::<3>(m0);
    let mut hmask0 = load_u8x16(&hmask[..16]);
    let mut hmask1 = load_u8x16(&hmask[16..32]);
    let mut isum = 0_i32;
    let mut tile = 0;
    while tile < 2 {
        let packed0 = load_u8x16(&qs[tile * 32..tile * 32 + 16]);
        let packed1 = load_u8x16(&qs[tile * 32 + 16..tile * 32 + 32]);
        let q8_base = tile * 128;
        let scale_base = tile * 8;
        let high0 = backend.shl_const_u8x16::<2>(backend.and_u8x16(m0, backend.not_u8x16(hmask0)));
        let high1 = backend.shl_const_u8x16::<2>(backend.and_u8x16(m0, backend.not_u8x16(hmask1)));
        let high2 = backend.shl_const_u8x16::<1>(backend.and_u8x16(m1, backend.not_u8x16(hmask0)));
        let high3 = backend.shl_const_u8x16::<1>(backend.and_u8x16(m1, backend.not_u8x16(hmask1)));
        let high4 = backend.and_u8x16(m2, backend.not_u8x16(hmask0));
        let high5 = backend.and_u8x16(m2, backend.not_u8x16(hmask1));
        let high6 = backend.shr_const_u8x16::<1>(backend.and_u8x16(m3, backend.not_u8x16(hmask0)));
        let high7 = backend.shr_const_u8x16::<1>(backend.and_u8x16(m3, backend.not_u8x16(hmask1)));
        isum += i32::from(scales[scale_base])
            * widening_i8_dot(
                backend,
                q3_sub(backend, backend.and_u8x16(packed0, m3b), high0),
                load_i8x16(&q8[q8_base..q8_base + 16]),
            );
        isum += i32::from(scales[scale_base + 1])
            * widening_i8_dot(
                backend,
                q3_sub(backend, backend.and_u8x16(packed1, m3b), high1),
                load_i8x16(&q8[q8_base + 16..q8_base + 32]),
            );
        isum += i32::from(scales[scale_base + 2])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<2>(packed0), m3b),
                    high2,
                ),
                load_i8x16(&q8[q8_base + 32..q8_base + 48]),
            );
        isum += i32::from(scales[scale_base + 3])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<2>(packed1), m3b),
                    high3,
                ),
                load_i8x16(&q8[q8_base + 48..q8_base + 64]),
            );
        isum += i32::from(scales[scale_base + 4])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<4>(packed0), m3b),
                    high4,
                ),
                load_i8x16(&q8[q8_base + 64..q8_base + 80]),
            );
        isum += i32::from(scales[scale_base + 5])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<4>(packed1), m3b),
                    high5,
                ),
                load_i8x16(&q8[q8_base + 80..q8_base + 96]),
            );
        isum += i32::from(scales[scale_base + 6])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<6>(packed0), m3b),
                    high6,
                ),
                load_i8x16(&q8[q8_base + 96..q8_base + 112]),
            );
        isum += i32::from(scales[scale_base + 7])
            * widening_i8_dot(
                backend,
                q3_sub(
                    backend,
                    backend.and_u8x16(backend.shr_const_u8x16::<6>(packed1), m3b),
                    high7,
                ),
                load_i8x16(&q8[q8_base + 112..q8_base + 128]),
            );
        hmask0 = backend.shr_const_u8x16::<4>(hmask0);
        hmask1 = backend.shr_const_u8x16::<4>(hmask1);
        tile += 1;
    }
    isum
}

fn q3_sub(backend: Neon, q3_low: u8x16, q3h: u8x16) -> i8x16 {
    backend.wrapping_sub_i8x16(cast(q3_low), cast(q3h))
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
        MAX_Q8_K_BLOCKS, OpMulMatQ3K, OpMulMatQ3KVector, PINNED_EMEL_CPP_COMMIT,
        PINNED_GEMM_ACTION_SPAN, PINNED_GEMM_GUARD_SPAN, PINNED_GEMM_TRANSITION_SPAN,
        PINNED_ROW_NEON_SPAN, Q3_K_BLOCK_BYTES, Q3KError, Q3KEvent, Q3KKernel, QK_K,
        SCOPE_RESIDUAL, UnexpectedQ3K,
    };

    fn pack_q3_k_row(d: u16, hmask: u8, qs: u8, scale: u8) -> [u8; Q3_K_BLOCK_BYTES] {
        let mut block = [0_u8; Q3_K_BLOCK_BYTES];
        let mut index = 0;
        while index < 32 {
            block[index] = hmask;
            index += 1;
        }
        index = 0;
        while index < 64 {
            block[32 + index] = qs;
            index += 1;
        }
        index = 0;
        while index < 12 {
            block[96 + index] = scale;
            index += 1;
        }
        block[108..110].copy_from_slice(&d.to_le_bytes());
        block
    }

    #[test]
    fn q3_k_constructs_on_neon() {
        assert!(Q3KKernel::try_new().is_some());
    }

    #[test]
    fn q3_k_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            PINNED_ROW_NEON_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:2640-2775"
        );
        assert_eq!(
            PINNED_GEMM_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:738-775"
        );
        assert_eq!(
            PINNED_GEMM_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7347-7398"
        );
        assert_eq!(
            PINNED_GEMM_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:522-525"
        );
        assert!(SCOPE_RESIDUAL.contains("q3_k vector n=1 and q3_k GEMM"));
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(Q3_K_BLOCK_BYTES, 110);
    }

    #[test]
    fn q3_k_vector_zero_qs_is_finite() {
        let Some(mut kernel) = Q3KKernel::try_new() else {
            return;
        };
        let lhs = pack_q3_k_row(0x3c00, 0xa5, 0, 0x12);
        let rhs = [0.25_f32; QK_K];
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ3KVector::new(&lhs, &rhs, 1, QK_K), &mut output),
            Ok(())
        );
        assert!(output[0].is_finite());
        assert!(kernel.is_ready());
    }

    #[test]
    fn q3_k_gemm_matches_independent_column_dots() {
        let Some(mut kernel) = Q3KKernel::try_new() else {
            return;
        };
        let row0 = pack_q3_k_row(0x3c00, 0xa5, 0xe4, 0x12);
        let row1 = pack_q3_k_row(0x4000, 0x5a, 0x1b, 0x34);
        let mut lhs = [0_u8; 2 * Q3_K_BLOCK_BYTES];
        lhs[..Q3_K_BLOCK_BYTES].copy_from_slice(&row0);
        lhs[Q3_K_BLOCK_BYTES..].copy_from_slice(&row1);
        let mut rhs = [0.0_f32; QK_K * 2];
        let mut index = 0;
        while index < QK_K {
            rhs[index * 2] = (index as f32 - 128.0) * 0.03125;
            rhs[index * 2 + 1] = (64.0 - index as f32) * 0.015_625;
            index += 1;
        }
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ3K::new(&lhs, &rhs, 2, QK_K, 2), &mut output),
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
            kernel.process_event(OpMulMatQ3KVector::new(&lhs, &col0, 2, QK_K), &mut expected0),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ3KVector::new(&lhs, &col1, 2, QK_K), &mut expected1),
            Ok(())
        );
        assert_eq!(output[0].to_bits(), expected0[0].to_bits());
        assert_eq!(output[2].to_bits(), expected0[1].to_bits());
        assert_eq!(output[1].to_bits(), expected1[0].to_bits());
        assert_eq!(output[3].to_bits(), expected1[1].to_bits());
    }

    #[test]
    fn q3_k_rejects_invalid_and_unexpected() {
        let Some(mut kernel) = Q3KKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 2];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ3K::new(&[0; 2], &[0.0; 2], 2, QK_K, 1),
                &mut output
            ),
            Err(Q3KError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ3K::new(&[], &[], 0, 0, 0), &mut output),
            Err(Q3KError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(UnexpectedQ3K, &mut []),
            Err(Q3KError::UnexpectedEvent)
        );
        let _ = Q3KEvent::dispatch(UnexpectedQ3K, &mut kernel, &mut []);
    }
}
