//! Safe `AArch64` NEON `q4_1` vector matmul for the pinned kernel contract.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_run_neon_mul_mat_q4_1_vector_request`: packed `q4_1` `src0` with shape
//! `[k, m]`, dense F32 `src1` with shape `[1, k]`, and dense F32 `dst` with
//! shape `[1, m]`.  Capability detection happens during construction.  The
//! dispatch action receives a typed `pulp::aarch64::Neon` backend and therefore
//! never performs runtime backend selection, dequantizes into an F32 GEMV, or
//! falls back to the portable actor.
//!
//! Integer products use the pinned `vmull` widening contract expressed through
//! pulp's stable `i16` multiply after an explicit 16-lane widen.  That integer
//! sum matches `vdotq_s32` on DOTPROD hosts, which is the live C++ path on
//! this machine.

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

use crate::any::quant::{BLOCK_VALUES, fp16_to_f32, fp32_to_fp16};

const Q4_1_BLOCK_BYTES: usize = 20;

/// Errors returned by the `AArch64` `q4_1` vector actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4_1VectorError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q4_1VectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q4_1 vector NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q4_1 vector shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q4_1 vector event"),
            Self::Internal => formatter.write_str("internal q4_1 vector dispatch error"),
        }
    }
}

impl std::error::Error for Q4_1VectorError {}

/// A pinned `AArch64` packed `q4_1` by dense F32 vector request.
///
/// `lhs` is row-major packed `q4_1` with `m` rows of `k / 32` blocks.
/// `rhs` is one dense F32 vector of length `k`.
#[derive(Debug)]
pub struct OpMulMatQ4_1Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4_1Vector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the `q4_1` vector API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ4_1Vector;

/// Result returned by target `q4_1` vector events.
pub type Q4_1VectorResult = Result<(), Q4_1VectorError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:364-373";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1065-1096";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7245-7304";
const PINNED_DOT_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:4107-4155";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:365-368";
const SCOPE_RESIDUAL: &str = "q4_1 vector only (packed src0[k,m], dense F32 src1[1,k], dense F32 dst[1,m]); q5/q6/q8 SIMD, F16 neon, and x86 remain residuals";

const MAX_Q8_0_BLOCKS: usize = 1024;

/// Event implemented by the target `q4_1` vector actor.
pub trait Q4_1VectorEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q4_1VectorKernel, output: &mut [f32]) -> Self::Output;
}

#[derive(Clone, Copy, Debug)]
struct Q8Scratch {
    d: u16,
    qs: [i8; BLOCK_VALUES],
}

impl Q8Scratch {
    const ZERO: Self = Self {
        d: 0,
        qs: [0; BLOCK_VALUES],
    };
}

struct Q4_1VectorRuntime<'a> {
    event: OpMulMatQ4_1Vector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4_1VectorResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q4_1VectorResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[Q8Scratch]>,
}

sml! {
    Q4_1VectorMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q4_1Vector(Q4_1VectorRuntime<'dispatch>) [guard_q4_1_ready] / effect_q4_1_execute,
        "ready"_s <= "ready"_s + Q4_1Vector(Q4_1VectorRuntime<'dispatch>) [guard_q4_1_invalid] / effect_q4_1_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q4_1` vector actor.
pub struct Q4_1VectorKernel {
    machine: Q4_1VectorMachineStateMachine<Context>,
}

impl fmt::Debug for Q4_1VectorKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q4_1VectorKernel")
            .finish_non_exhaustive()
    }
}

impl Q4_1VectorKernel {
    /// Resolves NEON and one-time q8 scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q4_1VectorMachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![Q8Scratch::ZERO; MAX_Q8_0_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q4_1VectorEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q4_1VectorMachineStates::Ready)
    }

    fn q4_1_vector(
        &mut self,
        event: OpMulMatQ4_1Vector<'_>,
        output: &mut [f32],
    ) -> Q4_1VectorResult {
        let result = Cell::new(Err(Q4_1VectorError::UnexpectedEvent));
        self.machine
            .process_event(Q4_1VectorMachineEvents::Q4_1Vector(Q4_1VectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q4_1VectorError::Internal)?;
        result.get()
    }
}

impl Q4_1VectorEvent for OpMulMatQ4_1Vector<'_> {
    type Output = Q4_1VectorResult;

    fn dispatch(self, actor: &mut Q4_1VectorKernel, output: &mut [f32]) -> Self::Output {
        actor.q4_1_vector(self, output)
    }
}

impl Q4_1VectorEvent for UnexpectedQ4_1Vector {
    type Output = Q4_1VectorResult;

    fn dispatch(self, actor: &mut Q4_1VectorKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4_1VectorError::Internal));
        actor
            .machine
            .process_event(Q4_1VectorMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q4_1VectorError::Internal)?;
        result.get()
    }
}

impl Q4_1VectorMachineStateMachineContext for Context {
    fn guard_q4_1_ready(&self, event: &Q4_1VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && q4_1_request_valid(event))
    }

    fn guard_q4_1_invalid(&self, event: &Q4_1VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!q4_1_request_valid(event))
    }

    fn effect_q4_1_execute(&mut self, event: Q4_1VectorRuntime<'_>) -> Result<(), ()> {
        execute_q4_1_vector(
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

    fn effect_q4_1_invalid(&mut self, event: Q4_1VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4_1VectorError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4_1VectorError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn q4_1_request_valid(event: &Q4_1VectorRuntime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0 || request.k == 0 || !request.k.is_multiple_of(BLOCK_VALUES) {
        return false;
    }
    let Some(blocks) = request.k.checked_div(BLOCK_VALUES) else {
        return false;
    };
    if blocks == 0 || blocks > MAX_Q8_0_BLOCKS {
        return false;
    }
    let Some(row_bytes) = blocks.checked_mul(Q4_1_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = request.m.checked_mul(row_bytes) else {
        return false;
    };
    request.lhs.len() == lhs_len
        && request.rhs.len() == request.k
        && event.output.len() == request.m
}

fn execute_q4_1_vector(
    backend: Neon,
    lhs: &[u8],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    scratch: &mut [Q8Scratch],
) {
    let blocks = k / BLOCK_VALUES;
    quantize_q8_0_dense(rhs, scratch, blocks);
    let row_bytes = blocks * Q4_1_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        let packed = &lhs[row * row_bytes..(row + 1) * row_bytes];
        output[row] = dot_q4_1_q8_0_neon(backend, packed, &scratch[..blocks]);
        row += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
fn quantize_q8_0_dense(rhs: &[f32], scratch: &mut [Q8Scratch], blocks: usize) {
    let mut block = 0;
    while block < blocks {
        let base = block * BLOCK_VALUES;
        let mut amax = 0.0_f32;
        let mut index = 0;
        while index < BLOCK_VALUES {
            amax = amax.max(rhs[base + index].abs());
            index += 1;
        }
        let scale = amax / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        scratch[block].d = fp32_to_fp16(scale);
        index = 0;
        while index < BLOCK_VALUES {
            let quantized = (rhs[base + index] * inverse).round().clamp(-127.0, 127.0);
            scratch[block].qs[index] = quantized as i8;
            index += 1;
        }
        block += 1;
    }
}

// Keep the reference's separate scale product and affine minimum term.
// Fusing those multiplies changes intermediate rounding and parity bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q4_1_q8_0_neon(backend: Neon, row: &[u8], rhs: &[Q8Scratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let offset = block * Q4_1_BLOCK_BYTES;
        let mut packed = [0_u8; 16];
        packed.copy_from_slice(&row[offset + 4..offset + Q4_1_BLOCK_BYTES]);
        let qs: u8x16 = cast(packed);
        let low_mask = backend.splat_u8x16(0x0f);
        let lhs_lo: i8x16 = cast(backend.and_u8x16(qs, low_mask));
        let lhs_hi: i8x16 = cast(backend.shr_const_u8x16::<4>(qs));
        let mut rhs_lo_bytes = [0_i8; 16];
        let mut rhs_hi_bytes = [0_i8; 16];
        rhs_lo_bytes.copy_from_slice(&rhs[block].qs[..16]);
        rhs_hi_bytes.copy_from_slice(&rhs[block].qs[16..]);
        let rhs_lo: i8x16 = cast(rhs_lo_bytes);
        let rhs_hi: i8x16 = cast(rhs_hi_bytes);
        let integer =
            widening_i8_dot(backend, lhs_lo, rhs_lo) + widening_i8_dot(backend, lhs_hi, rhs_hi);
        let rhs_sum = horizontal_sum_i8x16(rhs_lo) + horizontal_sum_i8x16(rhs_hi);
        let rhs_scale = fp16_to_f32(rhs[block].d);
        sum += rhs_scale
            * (fp16_to_f32(u16::from_le_bytes([row[offset], row[offset + 1]])) * integer as f32
                + fp16_to_f32(u16::from_le_bytes([row[offset + 2], row[offset + 3]]))
                    * rhs_sum as f32);
        block += 1;
    }
    sum
}

fn horizontal_sum_i8x16(values: i8x16) -> i32 {
    let (lo, hi) = widen_i8x16(values);
    horizontal_sum_i16x8(lo) + horizontal_sum_i16x8(hi)
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
        clippy::suboptimal_flops
    )]

    use super::{
        BLOCK_VALUES, MAX_Q8_0_BLOCKS, OpMulMatQ4_1Vector, Q4_1_BLOCK_BYTES, Q4_1VectorError,
        Q4_1VectorEvent, Q4_1VectorKernel, UnexpectedQ4_1Vector, quantize_q8_0_dense,
    };
    use crate::any::quant::{fp16_to_f32, fp32_to_fp16};
    use allocation_counter::measure;

    fn pack_q4_row(
        scale: u16,
        min: u16,
        lows: [u8; 16],
        highs: [u8; 16],
    ) -> [u8; Q4_1_BLOCK_BYTES] {
        let mut block = [0_u8; Q4_1_BLOCK_BYTES];
        block[..2].copy_from_slice(&scale.to_le_bytes());
        block[2..4].copy_from_slice(&min.to_le_bytes());
        let mut index = 0;
        while index < 16 {
            block[4 + index] = (lows[index] & 0x0f) | ((highs[index] & 0x0f) << 4);
            index += 1;
        }
        block
    }

    fn scalar_q4_1_dot(row: &[u8], qs: [i8; BLOCK_VALUES], d: u16) -> f32 {
        let mut integer = 0_i32;
        let mut rhs_sum = 0_i32;
        let mut index = 0;
        while index < 16 {
            let packed = row[4 + index];
            let rhs_low = i32::from(qs[index]);
            let rhs_high = i32::from(qs[index + 16]);
            integer += i32::from(packed & 0x0f) * rhs_low;
            integer += i32::from(packed >> 4) * rhs_high;
            rhs_sum += rhs_low + rhs_high;
            index += 1;
        }
        let rhs_scale = fp16_to_f32(d);
        rhs_scale
            * (fp16_to_f32(u16::from_le_bytes([row[0], row[1]])) * integer as f32
                + fp16_to_f32(u16::from_le_bytes([row[2], row[3]])) * rhs_sum as f32)
    }

    #[test]
    fn target_capability_is_resolved_before_dispatch() {
        assert!(Q4_1VectorKernel::try_new().is_some());
    }

    #[test]
    fn pinned_source_identity_is_explicit() {
        assert_eq!(
            super::PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            super::PINNED_ACTION_BLOB,
            "267d4f74e6e7498155c8535920322ffef2c02fb6"
        );
        assert_eq!(
            super::PINNED_GUARD_BLOB,
            "c25714566ec9a02679daef85089544575123408e"
        );
        assert_eq!(
            super::PINNED_SM_BLOB,
            "865a9cc6ba6115382ed043c464f3d62bcd851357"
        );
        assert_eq!(
            super::PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:364-373"
        );
        assert_eq!(
            super::PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1065-1096"
        );
        assert_eq!(
            super::PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7245-7304"
        );
        assert_eq!(
            super::PINNED_DOT_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:4107-4155"
        );
        assert_eq!(
            super::PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:365-368"
        );
        assert!(super::SCOPE_RESIDUAL.contains("q4_1 vector only"));
        assert_eq!(MAX_Q8_0_BLOCKS, 1024);
        assert_eq!(Q4_1_BLOCK_BYTES, 20);
    }

    #[test]
    fn neon_integer_dot_matches_packed_scalar_oracle() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        let mut lows = [0_u8; 16];
        let mut highs = [0_u8; 16];
        let mut index = 0;
        while index < 16 {
            lows[index] = u8::try_from(index).unwrap();
            highs[index] = u8::try_from(15 - index).unwrap();
            index += 1;
        }
        let lhs = pack_q4_row(0x3c00, 0x3800, lows, highs);
        let mut rhs = [0.0_f32; BLOCK_VALUES];
        index = 0;
        while index < BLOCK_VALUES {
            rhs[index] = (index as i32 - 16) as f32 * 0.25;
            index += 1;
        }
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4_1Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        assert!(kernel.is_ready());

        let mut scratch = vec![super::Q8Scratch::ZERO; 1];
        quantize_q8_0_dense(&rhs, &mut scratch, 1);
        let expected = scalar_q4_1_dot(&lhs, scratch[0].qs, scratch[0].d);
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn four_plus_remainder_rows_match_scalar_oracle() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        let mut lhs = [0_u8; 5 * Q4_1_BLOCK_BYTES];
        let mut row = 0;
        while row < 5 {
            let mut lows = [0_u8; 16];
            let mut highs = [0_u8; 16];
            let mut index = 0;
            while index < 16 {
                lows[index] = u8::try_from((row + index) % 16).unwrap();
                highs[index] = u8::try_from((row * 3 + index) % 16).unwrap();
                index += 1;
            }
            let packed = pack_q4_row(
                fp32_to_fp16(1.0 + row as f32 * 0.25),
                fp32_to_fp16(0.5 + row as f32 * 0.125),
                lows,
                highs,
            );
            lhs[row * Q4_1_BLOCK_BYTES..(row + 1) * Q4_1_BLOCK_BYTES].copy_from_slice(&packed);
            row += 1;
        }
        let mut rhs = [0.0_f32; BLOCK_VALUES];
        let mut index = 0;
        while index < BLOCK_VALUES {
            rhs[index] = (index as i32 - 8) as f32 * 0.125;
            index += 1;
        }
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4_1Vector::new(&lhs, &rhs, 5, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );

        let mut scratch = vec![super::Q8Scratch::ZERO; 1];
        quantize_q8_0_dense(&rhs, &mut scratch, 1);
        row = 0;
        while row < 5 {
            let packed = &lhs[row * Q4_1_BLOCK_BYTES..(row + 1) * Q4_1_BLOCK_BYTES];
            let expected = scalar_q4_1_dot(packed, scratch[0].qs, scratch[0].d);
            assert_eq!(output[row].to_bits(), expected.to_bits());
            row += 1;
        }
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ4_1Vector::new(&[1, 2], &[3.0], 2, 2), &mut output),
            Err(Q4_1VectorError::InvalidShape)
        );
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ4_1Vector, &mut []),
            Err(Q4_1VectorError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_trait_remains_dispatch_surface() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q4_row(0x3c00, 0x0000, [0; 16], [0; 16]);
        let rhs = [1.0_f32; BLOCK_VALUES];
        let mut output = [7.0];
        assert_eq!(
            Q4_1VectorEvent::dispatch(
                OpMulMatQ4_1Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut kernel,
                &mut output
            ),
            Ok(())
        );
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut kernel) = Q4_1VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q4_row(0x3c00, 0x0000, [0; 16], [0; 16]);
        let rhs = [1.0_f32; BLOCK_VALUES];
        let mut output = [0.0];
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ4_1Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                    &mut output
                ),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(output, [0.0]);
    }
}
