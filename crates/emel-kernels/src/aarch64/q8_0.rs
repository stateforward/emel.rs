//! Safe `AArch64` NEON `q8_0` vector matmul for the pinned kernel contract.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_run_neon_mul_mat_q8_0_vector_request`: packed `q8_0` `src0` with shape
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
    cast, i8x16, i16x8,
};
use sml::sml;

use crate::any::quant::{BLOCK_VALUES, Q8_0_BLOCK_BYTES, fp16_to_f32, fp32_to_fp16};

/// Errors returned by the `AArch64` `q8_0` vector actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8_0VectorError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q8_0VectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q8_0 vector NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q8_0 vector shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q8_0 vector event"),
            Self::Internal => formatter.write_str("internal q8_0 vector dispatch error"),
        }
    }
}

impl std::error::Error for Q8_0VectorError {}

/// A pinned `AArch64` packed `q8_0` by dense F32 vector request.
///
/// `lhs` is row-major packed `q8_0` with `m` rows of `k / 32` blocks.
/// `rhs` is one dense F32 vector of length `k`.
#[derive(Debug)]
pub struct OpMulMatQ8_0Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ8_0Vector<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q8_0` by packed `q8_0` vector request.
///
/// `lhs` is row-major packed `q8_0` with `m` rows of `k / 32` blocks.
/// `rhs` is one packed `q8_0` row of `k / 32` blocks.
#[derive(Debug)]
pub struct OpMulMatQ8_0VectorQ8Rhs<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ8_0VectorQ8Rhs<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the `q8_0` vector API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ8_0Vector;

/// Result returned by target `q8_0` vector events.
pub type Q8_0VectorResult = Result<(), Q8_0VectorError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:375-384";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1133-1163";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6669-6713";
const PINNED_DOT_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:3750-3776";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:395-398";
const PINNED_Q8_RHS_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:466-475";
const PINNED_Q8_RHS_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1166-1203";
const PINNED_Q8_RHS_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6715-6752";
const PINNED_Q8_RHS_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:390-393";
const SCOPE_RESIDUAL: &str = "q8_0 vector and q8_0 q8-rhs vector (packed src0[k,m], F32 or packed q8_0 src1[1,k], dense F32 dst[1,m]); packed q8 x4, F16 neon, generic quantized mul_mat, and x86 remain residuals";

const MAX_Q8_0_BLOCKS: usize = 1024;

/// Event implemented by the target `q8_0` vector actor.
pub trait Q8_0VectorEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q8_0VectorKernel, output: &mut [f32]) -> Self::Output;
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

struct Q8_0VectorRuntime<'a> {
    event: OpMulMatQ8_0Vector<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q8_0VectorResult>,
}

struct Q8_0VectorQ8RhsRuntime<'a> {
    event: OpMulMatQ8_0VectorQ8Rhs<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q8_0VectorResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q8_0VectorResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[Q8Scratch]>,
}

sml! {
    Q8_0VectorMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Q8_0Vector(Q8_0VectorRuntime<'dispatch>) [guard_q8_0_ready] / effect_q8_0_execute,
        "ready"_s <= "ready"_s + Q8_0Vector(Q8_0VectorRuntime<'dispatch>) [guard_q8_0_invalid] / effect_q8_0_invalid,
        "ready"_s <= "ready"_s + Q8_0VectorQ8Rhs(Q8_0VectorQ8RhsRuntime<'dispatch>) [guard_q8_rhs_ready] / effect_q8_rhs_execute,
        "ready"_s <= "ready"_s + Q8_0VectorQ8Rhs(Q8_0VectorQ8RhsRuntime<'dispatch>) [guard_q8_rhs_invalid] / effect_q8_rhs_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` `q8_0` vector actor.
pub struct Q8_0VectorKernel {
    machine: Q8_0VectorMachineStateMachine<Context>,
}

impl fmt::Debug for Q8_0VectorKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q8_0VectorKernel")
            .finish_non_exhaustive()
    }
}

impl Q8_0VectorKernel {
    /// Resolves NEON and one-time q8 scratch before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q8_0VectorMachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![Q8Scratch::ZERO; MAX_Q8_0_BLOCKS].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q8_0VectorEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q8_0VectorMachineStates::Ready)
    }

    fn q8_0_vector(
        &mut self,
        event: OpMulMatQ8_0Vector<'_>,
        output: &mut [f32],
    ) -> Q8_0VectorResult {
        let result = Cell::new(Err(Q8_0VectorError::UnexpectedEvent));
        self.machine
            .process_event(Q8_0VectorMachineEvents::Q8_0Vector(Q8_0VectorRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q8_0VectorError::Internal)?;
        result.get()
    }

    fn q8_0_vector_q8_rhs(
        &mut self,
        event: OpMulMatQ8_0VectorQ8Rhs<'_>,
        output: &mut [f32],
    ) -> Q8_0VectorResult {
        let result = Cell::new(Err(Q8_0VectorError::UnexpectedEvent));
        self.machine
            .process_event(Q8_0VectorMachineEvents::Q8_0VectorQ8Rhs(
                Q8_0VectorQ8RhsRuntime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0VectorError::Internal)?;
        result.get()
    }
}

impl Q8_0VectorEvent for OpMulMatQ8_0Vector<'_> {
    type Output = Q8_0VectorResult;

    fn dispatch(self, actor: &mut Q8_0VectorKernel, output: &mut [f32]) -> Self::Output {
        actor.q8_0_vector(self, output)
    }
}

impl Q8_0VectorEvent for OpMulMatQ8_0VectorQ8Rhs<'_> {
    type Output = Q8_0VectorResult;

    fn dispatch(self, actor: &mut Q8_0VectorKernel, output: &mut [f32]) -> Self::Output {
        actor.q8_0_vector_q8_rhs(self, output)
    }
}

impl Q8_0VectorEvent for UnexpectedQ8_0Vector {
    type Output = Q8_0VectorResult;

    fn dispatch(self, actor: &mut Q8_0VectorKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0VectorError::Internal));
        actor
            .machine
            .process_event(Q8_0VectorMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q8_0VectorError::Internal)?;
        result.get()
    }
}

impl Q8_0VectorMachineStateMachineContext for Context {
    fn guard_q8_0_ready(&self, event: &Q8_0VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && q8_0_request_valid(event))
    }

    fn guard_q8_0_invalid(&self, event: &Q8_0VectorRuntime<'_>) -> Result<bool, ()> {
        Ok(!q8_0_request_valid(event))
    }

    fn effect_q8_0_execute(&mut self, event: Q8_0VectorRuntime<'_>) -> Result<(), ()> {
        execute_q8_0_vector(
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

    fn effect_q8_0_invalid(&mut self, event: Q8_0VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0VectorError::InvalidShape));
        Ok(())
    }

    fn guard_q8_rhs_ready(&self, event: &Q8_0VectorQ8RhsRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && q8_rhs_request_valid(event))
    }

    fn guard_q8_rhs_invalid(&self, event: &Q8_0VectorQ8RhsRuntime<'_>) -> Result<bool, ()> {
        Ok(!q8_rhs_request_valid(event))
    }

    fn effect_q8_rhs_execute(&mut self, event: Q8_0VectorQ8RhsRuntime<'_>) -> Result<(), ()> {
        execute_q8_0_vector_q8_rhs(
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

    fn effect_q8_rhs_invalid(&mut self, event: Q8_0VectorQ8RhsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0VectorError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0VectorError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn q8_0_request_valid(event: &Q8_0VectorRuntime<'_>) -> bool {
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
    let Some(row_bytes) = blocks.checked_mul(Q8_0_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = request.m.checked_mul(row_bytes) else {
        return false;
    };
    request.lhs.len() == lhs_len
        && request.rhs.len() == request.k
        && event.output.len() == request.m
}

const fn q8_rhs_request_valid(event: &Q8_0VectorQ8RhsRuntime<'_>) -> bool {
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
    let Some(row_bytes) = blocks.checked_mul(Q8_0_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_len) = request.m.checked_mul(row_bytes) else {
        return false;
    };
    request.lhs.len() == lhs_len
        && request.rhs.len() == row_bytes
        && event.output.len() == request.m
}

fn execute_q8_0_vector(
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
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        let packed = &lhs[row * row_bytes..(row + 1) * row_bytes];
        output[row] = dot_q8_0_q8_0_neon(backend, packed, &scratch[..blocks]);
        row += 1;
    }
}

fn execute_q8_0_vector_q8_rhs(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
    scratch: &mut [Q8Scratch],
) {
    let blocks = k / BLOCK_VALUES;
    unpack_q8_0_row(rhs, scratch, blocks);
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    let mut row = 0;
    while row < m {
        let packed = &lhs[row * row_bytes..(row + 1) * row_bytes];
        output[row] = dot_q8_0_q8_0_neon(backend, packed, &scratch[..blocks]);
        row += 1;
    }
}

const fn unpack_q8_0_row(bytes: &[u8], scratch: &mut [Q8Scratch], blocks: usize) {
    let mut block = 0;
    while block < blocks {
        let offset = block * Q8_0_BLOCK_BYTES;
        scratch[block].d = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        let mut index = 0;
        while index < BLOCK_VALUES {
            scratch[block].qs[index] = i8::from_ne_bytes([bytes[offset + 2 + index]]);
            index += 1;
        }
        block += 1;
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

// Keep the reference's separate scale product and accumulation.  Fusing the
// final multiply changes intermediate rounding and can change parity bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q8_0_q8_0_neon(backend: Neon, row: &[u8], rhs: &[Q8Scratch]) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < rhs.len() {
        let offset = block * Q8_0_BLOCK_BYTES;
        let mut lhs_lo_bytes = [0_u8; 16];
        let mut lhs_hi_bytes = [0_u8; 16];
        lhs_lo_bytes.copy_from_slice(&row[offset + 2..offset + 18]);
        lhs_hi_bytes.copy_from_slice(&row[offset + 18..offset + Q8_0_BLOCK_BYTES]);
        let lhs_lo: i8x16 = cast(lhs_lo_bytes);
        let lhs_hi: i8x16 = cast(lhs_hi_bytes);
        let mut rhs_lo_bytes = [0_i8; 16];
        let mut rhs_hi_bytes = [0_i8; 16];
        rhs_lo_bytes.copy_from_slice(&rhs[block].qs[..16]);
        rhs_hi_bytes.copy_from_slice(&rhs[block].qs[16..]);
        let rhs_lo: i8x16 = cast(rhs_lo_bytes);
        let rhs_hi: i8x16 = cast(rhs_hi_bytes);
        let integer =
            widening_i8_dot(backend, lhs_lo, rhs_lo) + widening_i8_dot(backend, lhs_hi, rhs_hi);
        sum += integer as f32
            * (fp16_to_f32(u16::from_le_bytes([row[offset], row[offset + 1]]))
                * fp16_to_f32(rhs[block].d));
        block += 1;
    }
    sum
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
        BLOCK_VALUES, MAX_Q8_0_BLOCKS, OpMulMatQ8_0Vector, OpMulMatQ8_0VectorQ8Rhs,
        Q8_0VectorError, Q8_0VectorEvent, Q8_0VectorKernel, UnexpectedQ8_0Vector,
        quantize_q8_0_dense,
    };
    use crate::any::quant::{Q8_0_BLOCK_BYTES, Q8_0Row, dot_q8_0_q8_0, fp32_to_fp16};
    use allocation_counter::measure;

    fn pack_q8_row(scale: u16, values: [i8; BLOCK_VALUES]) -> [u8; Q8_0_BLOCK_BYTES] {
        let mut block = [0_u8; Q8_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&scale.to_le_bytes());
        let mut index = 0;
        while index < BLOCK_VALUES {
            block[2 + index] = values[index].to_ne_bytes()[0];
            index += 1;
        }
        block
    }

    fn pack_q8_from_scratch(d: u16, qs: [i8; BLOCK_VALUES]) -> [u8; Q8_0_BLOCK_BYTES] {
        pack_q8_row(d, qs)
    }

    fn ramp_i8() -> [i8; BLOCK_VALUES] {
        let mut values = [0_i8; BLOCK_VALUES];
        let mut index = 0;
        while index < BLOCK_VALUES {
            values[index] = i8::try_from(index as i32 - 16).unwrap();
            index += 1;
        }
        values
    }

    #[test]
    fn target_capability_is_resolved_before_dispatch() {
        assert!(Q8_0VectorKernel::try_new().is_some());
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
            "src/emel/kernel/aarch64/guards.hpp:375-384"
        );
        assert_eq!(
            super::PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1133-1163"
        );
        assert_eq!(
            super::PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6669-6713"
        );
        assert_eq!(
            super::PINNED_DOT_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:3750-3776"
        );
        assert_eq!(
            super::PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:395-398"
        );
        assert!(super::SCOPE_RESIDUAL.contains("q8_0 vector and q8_0 q8-rhs vector"));
        assert_eq!(MAX_Q8_0_BLOCKS, 1024);
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
    }

    #[test]
    fn neon_integer_dot_matches_packed_scalar_oracle() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q8_row(0x3c00, ramp_i8());
        let mut rhs = [0.0_f32; BLOCK_VALUES];
        let mut index = 0;
        while index < BLOCK_VALUES {
            rhs[index] = (index as i32 - 16) as f32 * 0.25;
            index += 1;
        }
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        assert!(kernel.is_ready());

        let mut scratch = vec![super::Q8Scratch::ZERO; 1];
        quantize_q8_0_dense(&rhs, &mut scratch, 1);
        let q8 = pack_q8_from_scratch(scratch[0].d, scratch[0].qs);
        let expected = dot_q8_0_q8_0(
            Q8_0Row::from_bytes(&lhs).unwrap(),
            Q8_0Row::from_bytes(&q8).unwrap(),
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn four_plus_remainder_rows_match_scalar_oracle() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let mut lhs = [0_u8; 5 * Q8_0_BLOCK_BYTES];
        let mut row = 0;
        while row < 5 {
            let mut values = ramp_i8();
            let mut index = 0;
            while index < BLOCK_VALUES {
                values[index] = values[index].saturating_add(i8::try_from(row).unwrap());
                index += 1;
            }
            let packed = pack_q8_row(fp32_to_fp16(1.0 + row as f32 * 0.25), values);
            lhs[row * Q8_0_BLOCK_BYTES..(row + 1) * Q8_0_BLOCK_BYTES].copy_from_slice(&packed);
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
                OpMulMatQ8_0Vector::new(&lhs, &rhs, 5, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );

        let mut scratch = vec![super::Q8Scratch::ZERO; 1];
        quantize_q8_0_dense(&rhs, &mut scratch, 1);
        let q8 = pack_q8_from_scratch(scratch[0].d, scratch[0].qs);
        row = 0;
        while row < 5 {
            let packed = &lhs[row * Q8_0_BLOCK_BYTES..(row + 1) * Q8_0_BLOCK_BYTES];
            let expected = dot_q8_0_q8_0(
                Q8_0Row::from_bytes(packed).unwrap(),
                Q8_0Row::from_bytes(&q8).unwrap(),
            )
            .unwrap();
            assert_eq!(output[row].to_bits(), expected.to_bits());
            row += 1;
        }
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ8_0Vector::new(&[1, 2], &[3.0], 2, 2), &mut output),
            Err(Q8_0VectorError::InvalidShape)
        );
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ8_0Vector, &mut []),
            Err(Q8_0VectorError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_trait_remains_dispatch_surface() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q8_row(0x3c00, [0; BLOCK_VALUES]);
        let rhs = [1.0_f32; BLOCK_VALUES];
        let mut output = [7.0];
        assert_eq!(
            Q8_0VectorEvent::dispatch(
                OpMulMatQ8_0Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut kernel,
                &mut output
            ),
            Ok(())
        );
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q8_row(0x3c00, [0; BLOCK_VALUES]);
        let rhs = [1.0_f32; BLOCK_VALUES];
        let mut output = [0.0];
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ8_0Vector::new(&lhs, &rhs, 1, BLOCK_VALUES),
                    &mut output
                ),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn q8_rhs_vector_matches_packed_scalar_oracle() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let lhs = pack_q8_row(0x3c00, ramp_i8());
        let rhs = pack_q8_row(0x3800, ramp_i8());
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0VectorQ8Rhs::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        let expected = dot_q8_0_q8_0(
            Q8_0Row::from_bytes(&lhs).unwrap(),
            Q8_0Row::from_bytes(&rhs).unwrap(),
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn q8_rhs_invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = Q8_0VectorKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0VectorQ8Rhs::new(&[1, 2], &[3], 2, 2),
                &mut output
            ),
            Err(Q8_0VectorError::InvalidShape)
        );
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn q8_rhs_pinned_source_identity_is_explicit() {
        assert_eq!(
            super::PINNED_Q8_RHS_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:466-475"
        );
        assert_eq!(
            super::PINNED_Q8_RHS_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1166-1203"
        );
        assert_eq!(
            super::PINNED_Q8_RHS_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6715-6752"
        );
        assert_eq!(
            super::PINNED_Q8_RHS_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:390-393"
        );
    }
}
