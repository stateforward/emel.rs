//! Safe `AArch64` NEON packed `q6_k_x8` by packed `q8_k` vector n=1, `matrix_x4`,
//! and n=1 argmax.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_use_neon_mul_mat_q6_vector_packed_q8_rhs`, the four-RHS route
//! selected by `can_use_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4`, and
//! the argmax route selected by
//! `can_use_neon_mul_mat_argmax_q6_vector_packed_q8_rhs`: interleaved
//! `q6_k_x8` `src0` with packed groups of 8 rows, packed `q8_k` `src1` with
//! shape `[1, k]` or four packed `q8_k` rows, dense F32 `dst` with shape
//! `[1, m]` or batch-major `[4, m]`, and argmax dest `[1, 1]` plus an `i32`
//! first-tie index.  Capability detection happens during construction.
//! Dispatch never dequantizes into an F32 GEMV, never uses `pulp::cast!` or
//! `core::arch`, and never falls back to the portable actor.
//!
//! Packing is lossless.  Each packed group is deinterleaved to eight native
//! `q6_k` rows, then the live generic q6 integer decode and float association
//! run with pulp widening i8 dots.  C++ `matrix_x4` loops that same group neon
//! kernel once per RHS row and stores batch-major results.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{
    aarch64::{Arch, Neon},
    i8x16, i16x8,
};
use sml::sml;

use crate::any::quant::fp16_to_f32;

/// Errors returned by the `AArch64` packed `q6_k_x8` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q6PackedError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q6PackedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q6 packed NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q6 packed shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q6 packed event"),
            Self::Internal => formatter.write_str("internal q6 packed dispatch error"),
        }
    }
}

impl std::error::Error for Q6PackedError {}

/// A pinned `AArch64` packed `q6_k_x8` by packed `q8_k` vector request.
///
/// `lhs` is group-major `block_q6_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is one packed `q8_k` row of `k / 256`
/// blocks.
#[derive(Debug)]
pub struct OpMulMatQ6Packed<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6Packed<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q6_k_x8` by 4 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q6_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is four packed `q8_k` rows of
/// `k / 256` blocks.  `dst` is batch-major F32 with shape `[4, m]`.
#[derive(Debug)]
pub struct OpMulMatQ6PackedMatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6PackedMatrixX4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q6_k_x8` by packed `q8_k` vector argmax request.
///
/// `lhs` is group-major `block_q6_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is one packed `q8_k` row of `k / 256`
/// blocks.  Destination is dense F32 with shape `[1, 1]` and the returned
/// index is the first-tie row.
#[derive(Debug)]
pub struct OpMulMatArgmaxQ6Packed<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatArgmaxQ6Packed<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the packed `q6_k_x8` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ6Packed;

/// Result returned by target packed `q6_k_x8` events.
pub type Q6PackedResult = Result<(), Q6PackedError>;

/// Result returned by the packed `q6_k_x8` argmax event.
pub type Q6PackedArgmaxResult = Result<i32, Q6PackedError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:609-615";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1646-1649";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6074-6108";
const PINNED_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:4669-4855";
const PINNED_PACK_SPAN: &str = "src/emel/kernel/detail.hpp:1034-1067";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:496-498";
const PINNED_MATRIX_X4_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:109-143";
const PINNED_MATRIX_X4_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:161-171";
const PINNED_MATRIX_X4_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6111-6150";
const PINNED_MATRIX_X4_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:4669-4855";
const PINNED_MATRIX_X4_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:490-493";
const PINNED_ARGMAX_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:673-681";
const PINNED_ARGMAX_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1736-1805";
const PINNED_ARGMAX_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5546-5590";
const PINNED_ARGMAX_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:4669-4855";
const PINNED_ARGMAX_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:473-478";
const SCOPE_RESIDUAL: &str = "q6_k_x8 lhs packed q8_k rhs n=1, packed q8_k matrix_x4, and packed q8_k n=1 argmax; F16 neon and x86 remain residuals";

const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q6_K_X8_ROWS: usize = 8;
const MATRIX_X4_ROWS: usize = 4;
const Q6_K_X8_BLOCK_BYTES: usize = 1680;
const Q8_K_BLOCK_BYTES: usize = 292;
const INTERLEAVE_BYTES: usize = 8;
const QL_OFFSET: usize = 144;
const QH_OFFSET: usize = 1168;
const SCALE_OFFSET: usize = 16;
const QL_INTERLEAVE_GROUPS: usize = 128;
const QH_INTERLEAVE_GROUPS: usize = 64;
const MAX_Q8_K_BLOCKS: usize = 128;

/// Event implemented by the target packed `q6_k_x8` actor.
pub trait Q6PackedEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q6PackedKernel, output: &mut [f32]) -> Self::Output;
}

struct PackedRuntime<'a> {
    event: OpMulMatQ6Packed<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PackedResult>,
}

struct PackedMatrixX4Runtime<'a> {
    event: OpMulMatQ6PackedMatrixX4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PackedResult>,
}

struct PackedArgmaxRuntime<'a> {
    event: OpMulMatArgmaxQ6Packed<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PackedArgmaxResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q6PackedResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
}

sml! {
    Q6PackedMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Packed(PackedRuntime<'dispatch>) [guard_ready] / effect_execute,
        "ready"_s <= "ready"_s + Packed(PackedRuntime<'dispatch>) [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + PackedMatrixX4(PackedMatrixX4Runtime<'dispatch>) [guard_matrix_ready] / effect_matrix_execute,
        "ready"_s <= "ready"_s + PackedMatrixX4(PackedMatrixX4Runtime<'dispatch>) [guard_matrix_invalid] / effect_matrix_invalid,
        "ready"_s <= "ready"_s + PackedArgmax(PackedArgmaxRuntime<'dispatch>) [guard_argmax_ready] / effect_argmax_execute,
        "ready"_s <= "ready"_s + PackedArgmax(PackedArgmaxRuntime<'dispatch>) [guard_argmax_invalid] / effect_argmax_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` packed `q6_k_x8` actor.
pub struct Q6PackedKernel {
    machine: Q6PackedMachineStateMachine<Context>,
}

impl fmt::Debug for Q6PackedKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q6PackedKernel")
            .finish_non_exhaustive()
    }
}

impl Q6PackedKernel {
    /// Resolves NEON before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q6PackedMachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q6PackedEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q6PackedMachineStates::Ready)
    }

    fn packed(&mut self, event: OpMulMatQ6Packed<'_>, output: &mut [f32]) -> Q6PackedResult {
        let result = Cell::new(Err(Q6PackedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PackedMachineEvents::Packed(PackedRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }

    fn packed_matrix_x4(
        &mut self,
        event: OpMulMatQ6PackedMatrixX4<'_>,
        output: &mut [f32],
    ) -> Q6PackedResult {
        let result = Cell::new(Err(Q6PackedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PackedMachineEvents::PackedMatrixX4(
                PackedMatrixX4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }

    fn packed_argmax(
        &mut self,
        event: OpMulMatArgmaxQ6Packed<'_>,
        output: &mut [f32],
    ) -> Q6PackedArgmaxResult {
        let result = Cell::new(Err(Q6PackedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PackedMachineEvents::PackedArgmax(PackedArgmaxRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }
}

impl Q6PackedEvent for OpMulMatQ6Packed<'_> {
    type Output = Q6PackedResult;

    fn dispatch(self, actor: &mut Q6PackedKernel, output: &mut [f32]) -> Self::Output {
        actor.packed(self, output)
    }
}

impl Q6PackedEvent for OpMulMatQ6PackedMatrixX4<'_> {
    type Output = Q6PackedResult;

    fn dispatch(self, actor: &mut Q6PackedKernel, output: &mut [f32]) -> Self::Output {
        actor.packed_matrix_x4(self, output)
    }
}

impl Q6PackedEvent for OpMulMatArgmaxQ6Packed<'_> {
    type Output = Q6PackedArgmaxResult;

    fn dispatch(self, actor: &mut Q6PackedKernel, output: &mut [f32]) -> Self::Output {
        actor.packed_argmax(self, output)
    }
}

impl Q6PackedEvent for UnexpectedQ6Packed {
    type Output = Q6PackedResult;

    fn dispatch(self, actor: &mut Q6PackedKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PackedError::Internal));
        actor
            .machine
            .process_event(Q6PackedMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }
}

impl Q6PackedMachineStateMachineContext for Context {
    fn guard_ready(&self, event: &PackedRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_request_valid(event))
    }

    fn guard_invalid(&self, event: &PackedRuntime<'_>) -> Result<bool, ()> {
        Ok(!packed_request_valid(event))
    }

    fn guard_matrix_ready(&self, event: &PackedMatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_matrix_x4_request_valid(event))
    }

    fn guard_matrix_invalid(&self, event: &PackedMatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_matrix_x4_request_valid(event))
    }

    fn guard_argmax_ready(&self, event: &PackedArgmaxRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_argmax_request_valid(event))
    }

    fn guard_argmax_invalid(&self, event: &PackedArgmaxRuntime<'_>) -> Result<bool, ()> {
        Ok(!packed_argmax_request_valid(event))
    }

    fn effect_execute(&mut self, event: PackedRuntime<'_>) -> Result<(), ()> {
        execute_packed(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: PackedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PackedError::InvalidShape));
        Ok(())
    }

    fn effect_matrix_execute(&mut self, event: PackedMatrixX4Runtime<'_>) -> Result<(), ()> {
        execute_packed_matrix_x4(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_matrix_invalid(&mut self, event: PackedMatrixX4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PackedError::InvalidShape));
        Ok(())
    }

    fn effect_argmax_execute(&mut self, event: PackedArgmaxRuntime<'_>) -> Result<(), ()> {
        let index = execute_packed_argmax(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(index));
        Ok(())
    }

    fn effect_argmax_invalid(&mut self, event: PackedArgmaxRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PackedError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PackedError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn packed_request_valid(event: &PackedRuntime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0 || request.k == 0 || !request.k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = request.k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = request.m.checked_add(Q6_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q6_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    request.lhs.len() == lhs_bytes
        && request.rhs.len() == rhs_bytes
        && event.output.len() == request.m
}

const fn packed_matrix_x4_request_valid(event: &PackedMatrixX4Runtime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0 || request.k == 0 || !request.k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = request.k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = request.m.checked_add(Q6_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q6_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_row_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(rhs_bytes) = rhs_row_bytes.checked_mul(MATRIX_X4_ROWS) else {
        return false;
    };
    let Some(output_len) = request.m.checked_mul(MATRIX_X4_ROWS) else {
        return false;
    };
    request.lhs.len() == lhs_bytes
        && request.rhs.len() == rhs_bytes
        && event.output.len() == output_len
}

const fn packed_argmax_request_valid(event: &PackedArgmaxRuntime<'_>) -> bool {
    let request = &event.event;
    if request.m == 0
        || request.m > i32::MAX as usize
        || request.k == 0
        || !request.k.is_multiple_of(QK_K)
    {
        return false;
    }
    let blocks = request.k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = request.m.checked_add(Q6_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q6_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    request.lhs.len() == lhs_bytes && request.rhs.len() == rhs_bytes && event.output.len() == 1
}

fn execute_packed(backend: Neon, lhs: &[u8], rhs: &[u8], output: &mut [f32], m: usize, k: usize) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let group_bytes = blocks * Q6_K_X8_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q6_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q6_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block(
                backend,
                &lhs[lhs_offset..lhs_offset + Q6_K_X8_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        store_q6_k_x8_results(output, group * Q6_K_X8_ROWS, m, acc);
        group += 1;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn execute_packed_argmax(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) -> i32 {
    // Guard `packed_argmax_request_valid` already proved `m <= i32::MAX`.
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let group_bytes = blocks * Q6_K_X8_BLOCK_BYTES;
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut have_best = false;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q6_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q6_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block(
                backend,
                &lhs[lhs_offset..lhs_offset + Q6_K_X8_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        let row_base = group * Q6_K_X8_ROWS;
        let remaining = m - row_base;
        let store_count = remaining.min(Q6_K_X8_ROWS);
        let mut lane = 0;
        while lane < store_count {
            let value = acc[lane];
            if !have_best || value > best_value {
                have_best = true;
                best_value = value;
                best_index = (row_base + lane) as i32;
            }
            lane += 1;
        }
        group += 1;
    }
    output[0] = best_value;
    best_index
}

fn execute_packed_matrix_x4(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let group_bytes = blocks * Q6_K_X8_BLOCK_BYTES;
    let rhs_row_bytes = blocks * Q8_K_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [[0.0_f32; Q6_K_X8_ROWS]; MATRIX_X4_ROWS];
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X4_ROWS {
            let mut block = 0;
            while block < blocks {
                let lhs_offset = group * group_bytes + block * Q6_K_X8_BLOCK_BYTES;
                let rhs_offset = rhs_row * rhs_row_bytes + block * Q8_K_BLOCK_BYTES;
                accumulate_group_block(
                    backend,
                    &lhs[lhs_offset..lhs_offset + Q6_K_X8_BLOCK_BYTES],
                    &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                    &mut acc[rhs_row],
                );
                block += 1;
            }
            rhs_row += 1;
        }
        store_batch_major_group_results(output, group * Q6_K_X8_ROWS, m, acc);
        group += 1;
    }
}

fn accumulate_group_block(backend: Neon, lhs: &[u8], rhs: &[u8], acc: &mut [f32; Q6_K_X8_ROWS]) {
    let rows = deinterleave_q6_k_x8(lhs);
    let mut q8_qs = [0_i8; QK_K];
    let mut index = 0;
    while index < QK_K {
        q8_qs[index] = i8::from_ne_bytes([rhs[4 + index]]);
        index += 1;
    }
    let mut bsums = [0_i16; QK_K / 16];
    index = 0;
    while index < QK_K / 16 {
        let offset = 260 + index * 2;
        bsums[index] = i16::from_le_bytes([rhs[offset], rhs[offset + 1]]);
        index += 1;
    }
    let mut q8_d_bytes = [0_u8; 4];
    q8_d_bytes.copy_from_slice(&rhs[..4]);
    let q8_d = f32::from_le_bytes(q8_d_bytes);
    let mut row = 0;
    while row < Q6_K_X8_ROWS {
        acc[row] += dot_q6_k_q8_k_block(backend, &rows[row], &q8_qs, &bsums, q8_d);
        row += 1;
    }
}

fn store_q6_k_x8_results(
    output: &mut [f32],
    row_base: usize,
    total_rows: usize,
    values: [f32; Q6_K_X8_ROWS],
) {
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q6_K_X8_ROWS);
    let mut lane = 0;
    while lane < store_count {
        output[row_base + lane] = values[lane];
        lane += 1;
    }
}

fn store_batch_major_group_results(
    output: &mut [f32],
    row_base: usize,
    total_rows: usize,
    values: [[f32; Q6_K_X8_ROWS]; MATRIX_X4_ROWS],
) {
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q6_K_X8_ROWS);
    let mut rhs_row = 0;
    while rhs_row < MATRIX_X4_ROWS {
        let dst_row = rhs_row * total_rows + row_base;
        let mut lane = 0;
        while lane < store_count {
            output[dst_row + lane] = values[rhs_row][lane];
            lane += 1;
        }
        rhs_row += 1;
    }
}

const fn deinterleave_q6_k_x8(packed: &[u8]) -> [[u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS] {
    let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
    let mut row = 0;
    while row < Q6_K_X8_ROWS {
        rows[row][208] = packed[row * 2];
        rows[row][209] = packed[row * 2 + 1];
        row += 1;
    }
    let mut index = 0;
    while index < QL_INTERLEAVE_GROUPS {
        let src_row = index % Q6_K_X8_ROWS;
        let src_offset = (index / Q6_K_X8_ROWS) * INTERLEAVE_BYTES;
        let dst_offset = index * INTERLEAVE_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BYTES {
            rows[src_row][src_offset + lane] = packed[QL_OFFSET + dst_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    index = 0;
    while index < QH_INTERLEAVE_GROUPS {
        let src_row = index % Q6_K_X8_ROWS;
        let src_offset = (index / Q6_K_X8_ROWS) * INTERLEAVE_BYTES;
        let dst_offset = index * INTERLEAVE_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BYTES {
            rows[src_row][128 + src_offset + lane] = packed[QH_OFFSET + dst_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    row = 0;
    while row < Q6_K_X8_ROWS {
        let mut scale = 0;
        while scale < QK_K / 16 {
            rows[row][192 + scale] = packed[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row];
            scale += 1;
        }
        row += 1;
    }
    rows
}

const fn make_block_q6_k_x8(
    rows: &[[u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS],
) -> [u8; Q6_K_X8_BLOCK_BYTES] {
    let mut out = [0_u8; Q6_K_X8_BLOCK_BYTES];
    let mut row = 0;
    while row < Q6_K_X8_ROWS {
        out[row * 2] = rows[row][208];
        out[row * 2 + 1] = rows[row][209];
        row += 1;
    }
    let mut index = 0;
    while index < QL_INTERLEAVE_GROUPS {
        let src_row = index % Q6_K_X8_ROWS;
        let src_offset = (index / Q6_K_X8_ROWS) * INTERLEAVE_BYTES;
        let dst_offset = index * INTERLEAVE_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BYTES {
            out[QL_OFFSET + dst_offset + lane] = rows[src_row][src_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    index = 0;
    while index < QH_INTERLEAVE_GROUPS {
        let src_row = index % Q6_K_X8_ROWS;
        let src_offset = (index / Q6_K_X8_ROWS) * INTERLEAVE_BYTES;
        let dst_offset = index * INTERLEAVE_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BYTES {
            out[QH_OFFSET + dst_offset + lane] = rows[src_row][128 + src_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    row = 0;
    while row < Q6_K_X8_ROWS {
        let mut scale = 0;
        while scale < QK_K / 16 {
            out[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row] = rows[row][192 + scale];
            scale += 1;
        }
        row += 1;
    }
    out
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q6_k_q8_k_block(
    backend: Neon,
    row: &[u8; Q6_K_BLOCK_BYTES],
    q8: &[i8; QK_K],
    bsums: &[i16; QK_K / 16],
    q8_d: f32,
) -> f32 {
    let ql = &row[..128];
    let qh = &row[128..192];
    let mut scales = [0_i8; 16];
    let mut scale_index = 0;
    while scale_index < 16 {
        scales[scale_index] = i8::from_ne_bytes([row[192 + scale_index]]);
        scale_index += 1;
    }
    let d = fp16_to_f32(u16::from_le_bytes([row[208], row[209]]));
    let isum = q6_integer_sum(backend, ql, qh, &scales, q8);
    let mut sum_mins = 0_i32;
    let mut group = 0;
    while group < 16 {
        sum_mins += i32::from(bsums[group]) * i32::from(scales[group]);
        group += 1;
    }
    (d * q8_d) * (isum - 32 * sum_mins) as f32
}

fn q6_integer_sum(backend: Neon, ql: &[u8], qh: &[u8], scales: &[i8; 16], q8: &[i8; QK_K]) -> i32 {
    let mut isum = 0_i32;
    let mut chunk = 0;
    while chunk < 2 {
        let ql0 = &ql[chunk * 64..chunk * 64 + 16];
        let ql1 = &ql[chunk * 64 + 16..chunk * 64 + 32];
        let ql2 = &ql[chunk * 64 + 32..chunk * 64 + 48];
        let ql3 = &ql[chunk * 64 + 48..chunk * 64 + 64];
        let qh0 = &qh[chunk * 32..chunk * 32 + 16];
        let qh1 = &qh[chunk * 32 + 16..chunk * 32 + 32];
        let q8_base = chunk * 128;
        let scale_base = chunk * 8;
        isum += i32::from(scales[scale_base])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<0, 0>(ql0, qh0)),
                load_i8x16(&q8[q8_base..q8_base + 16]),
            );
        isum += i32::from(scales[scale_base + 1])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<0, 0>(ql1, qh1)),
                load_i8x16(&q8[q8_base + 16..q8_base + 32]),
            );
        isum += i32::from(scales[scale_base + 2])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<2, 0>(ql2, qh0)),
                load_i8x16(&q8[q8_base + 32..q8_base + 48]),
            );
        isum += i32::from(scales[scale_base + 3])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<2, 0>(ql3, qh1)),
                load_i8x16(&q8[q8_base + 48..q8_base + 64]),
            );
        isum += i32::from(scales[scale_base + 4])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<4, 4>(ql0, qh0)),
                load_i8x16(&q8[q8_base + 64..q8_base + 80]),
            );
        isum += i32::from(scales[scale_base + 5])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<4, 4>(ql1, qh1)),
                load_i8x16(&q8[q8_base + 80..q8_base + 96]),
            );
        isum += i32::from(scales[scale_base + 6])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<6, 4>(ql2, qh0)),
                load_i8x16(&q8[q8_base + 96..q8_base + 112]),
            );
        isum += i32::from(scales[scale_base + 7])
            * widening_i8_dot(
                backend,
                load_i8x16(&decode_q6_lanes::<6, 4>(ql3, qh1)),
                load_i8x16(&q8[q8_base + 112..q8_base + 128]),
            );
        chunk += 1;
    }
    isum
}

fn decode_q6_lanes<const QH_SHIFT: u32, const QL_SHIFT: u32>(ql: &[u8], qh: &[u8]) -> [i8; 16] {
    let mut out = [0_i8; 16];
    let mut lane = 0;
    while lane < 16 {
        let q6 = ((ql[lane] >> QL_SHIFT) & 0x0f) | (((qh[lane] >> QH_SHIFT) & 3) << 4);
        out[lane] = i8::from_ne_bytes([q6]);
        lane += 1;
    }
    out
}

const fn load_i8x16(src: &[i8]) -> i8x16 {
    i8x16(
        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10],
        src[11], src[12], src[13], src[14], src[15],
    )
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
    (
        i16x8(
            i16::from(values.0),
            i16::from(values.1),
            i16::from(values.2),
            i16::from(values.3),
            i16::from(values.4),
            i16::from(values.5),
            i16::from(values.6),
            i16::from(values.7),
        ),
        i16x8(
            i16::from(values.8),
            i16::from(values.9),
            i16::from(values.10),
            i16::from(values.11),
            i16::from(values.12),
            i16::from(values.13),
            i16::from(values.14),
            i16::from(values.15),
        ),
    )
}

fn horizontal_sum_i16x8(values: i16x8) -> i32 {
    i32::from(values.0)
        + i32::from(values.1)
        + i32::from(values.2)
        + i32::from(values.3)
        + i32::from(values.4)
        + i32::from(values.5)
        + i32::from(values.6)
        + i32::from(values.7)
}

#[cfg(test)]
fn pack_q6_k_rows_x8(src: &[[u8; Q6_K_BLOCK_BYTES]], m: usize, blocks: usize) -> Vec<u8> {
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let mut packed = vec![0_u8; group_count * blocks * Q6_K_X8_BLOCK_BYTES];
    let mut group = 0;
    while group < group_count {
        let row_base = group * Q6_K_X8_ROWS;
        let mut block = 0;
        while block < blocks {
            let mut group_rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
            let mut row = 0;
            while row < Q6_K_X8_ROWS {
                let logical = row_base + row;
                if logical < m {
                    group_rows[row] = src[logical * blocks + block];
                }
                row += 1;
            }
            let offset = (group * blocks + block) * Q6_K_X8_BLOCK_BYTES;
            packed[offset..offset + Q6_K_X8_BLOCK_BYTES]
                .copy_from_slice(&make_block_q6_k_x8(&group_rows));
            block += 1;
        }
        group += 1;
    }
    packed
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::float_cmp,
        clippy::similar_names,
        clippy::unreadable_literal
    )]

    use super::{
        MATRIX_X4_ROWS, MAX_Q8_K_BLOCKS, OpMulMatArgmaxQ6Packed, OpMulMatQ6Packed,
        OpMulMatQ6PackedMatrixX4, PINNED_ACTION_BLOB, PINNED_ACTION_SPAN,
        PINNED_ARGMAX_ACTION_SPAN, PINNED_ARGMAX_EXECUTION_SPAN, PINNED_ARGMAX_GUARD_SPAN,
        PINNED_ARGMAX_KERNEL_SPAN, PINNED_ARGMAX_TRANSITION_SPAN, PINNED_EMEL_CPP_COMMIT,
        PINNED_EXECUTION_SPAN, PINNED_GUARD_BLOB, PINNED_GUARD_SPAN, PINNED_KERNEL_SPAN,
        PINNED_MATRIX_X4_ACTION_SPAN, PINNED_MATRIX_X4_EXECUTION_SPAN, PINNED_MATRIX_X4_GUARD_SPAN,
        PINNED_MATRIX_X4_KERNEL_SPAN, PINNED_MATRIX_X4_TRANSITION_SPAN, PINNED_PACK_SPAN,
        PINNED_SM_BLOB, PINNED_TRANSITION_SPAN, Q6_K_BLOCK_BYTES, Q6_K_X8_BLOCK_BYTES,
        Q6_K_X8_ROWS, Q6PackedError, Q6PackedEvent, Q6PackedKernel, Q8_K_BLOCK_BYTES, QK_K,
        SCOPE_RESIDUAL, UnexpectedQ6Packed, deinterleave_q6_k_x8, make_block_q6_k_x8,
        pack_q6_k_rows_x8,
    };
    use allocation_counter::measure;

    fn native_q6_k_row(row: usize) -> [u8; Q6_K_BLOCK_BYTES] {
        let mut out = [0_u8; Q6_K_BLOCK_BYTES];
        let d = (0x3c00 + row * 17) as u16;
        let mut index = 0;
        while index < 128 {
            out[index] = ((row * 29 + index * 5 + 3) & 0xff) as u8;
            index += 1;
        }
        index = 0;
        while index < 64 {
            out[128 + index] = ((row * 19 + index * 7 + 11) & 0xff) as u8;
            index += 1;
        }
        index = 0;
        while index < 16 {
            let scale = (row as i32 * 3 + index as i32).rem_euclid(15) - 7;
            out[192 + index] = (scale as i8).to_ne_bytes()[0];
            index += 1;
        }
        out[208..210].copy_from_slice(&d.to_le_bytes());
        out
    }

    fn pack_q8_k_row() -> [u8; Q8_K_BLOCK_BYTES] {
        pack_q8_k_row_at(0)
    }

    fn pack_q8_k_row_at(row: usize) -> [u8; Q8_K_BLOCK_BYTES] {
        let mut out = [0_u8; Q8_K_BLOCK_BYTES];
        let scale = 0.0625_f32 * (row as f32 + 1.0);
        out[0..4].copy_from_slice(&scale.to_le_bytes());
        let mut qs = [0_i8; QK_K];
        let mut index = 0;
        while index < QK_K {
            qs[index] = ((index as i32 * 7 + row as i32 * 11) % 31 - 15) as i8;
            out[4 + index] = qs[index].to_ne_bytes()[0];
            index += 1;
        }
        let mut group = 0;
        while group < QK_K / 16 {
            let mut sum = 0_i32;
            let mut lane = 0;
            while lane < 16 {
                sum += i32::from(qs[group * 16 + lane]);
                lane += 1;
            }
            let offset = 260 + group * 2;
            out[offset..offset + 2].copy_from_slice(&(sum as i16).to_le_bytes());
            group += 1;
        }
        out
    }

    fn pack_q8_k_rows_x4() -> [u8; MATRIX_X4_ROWS * Q8_K_BLOCK_BYTES] {
        let mut out = [0_u8; MATRIX_X4_ROWS * Q8_K_BLOCK_BYTES];
        let mut row = 0;
        while row < MATRIX_X4_ROWS {
            let packed = pack_q8_k_row_at(row);
            let offset = row * Q8_K_BLOCK_BYTES;
            out[offset..offset + Q8_K_BLOCK_BYTES].copy_from_slice(&packed);
            row += 1;
        }
        out
    }

    #[test]
    fn packed_kernel_constructs_on_neon() {
        assert!(Q6PackedKernel::try_new().is_some());
    }

    #[test]
    fn packed_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            PINNED_GUARD_BLOB,
            "c25714566ec9a02679daef85089544575123408e"
        );
        assert_eq!(
            PINNED_ACTION_BLOB,
            "267d4f74e6e7498155c8535920322ffef2c02fb6"
        );
        assert_eq!(PINNED_SM_BLOB, "865a9cc6ba6115382ed043c464f3d62bcd851357");
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:609-615"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1646-1649"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6074-6108"
        );
        assert_eq!(
            PINNED_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:4669-4855"
        );
        assert_eq!(PINNED_PACK_SPAN, "src/emel/kernel/detail.hpp:1034-1067");
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:496-498"
        );
        assert!(SCOPE_RESIDUAL.contains("q6_k_x8 lhs packed q8_k rhs n=1"));
        assert!(SCOPE_RESIDUAL.contains("packed q8_k matrix_x4"));
        assert!(SCOPE_RESIDUAL.contains("packed q8_k n=1 argmax"));
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
        assert_eq!(Q6_K_X8_BLOCK_BYTES, 1680);
        assert_eq!(Q6_K_X8_ROWS, 8);
        assert_eq!(Q8_K_BLOCK_BYTES, 292);
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(QK_K, 256);
    }

    #[test]
    fn packing_is_lossless_for_native_rows() {
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let packed = make_block_q6_k_x8(&rows);
        let restored = deinterleave_q6_k_x8(&packed);
        assert_eq!(restored, rows);
    }

    #[test]
    fn neon_packed_full_group_m8_dispatches() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 8, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 8, QK_K), &mut output),
            Ok(())
        );
        assert!(kernel.is_ready());
        let mut lane = 0;
        while lane < 8 {
            assert!(output[lane].is_finite());
            lane += 1;
        }
    }

    #[test]
    fn neon_packed_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 5, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 5, QK_K), &mut output),
            Ok(())
        );
        let mut lane = 0;
        while lane < 5 {
            assert!(output[lane].is_finite());
            lane += 1;
        }
    }

    #[test]
    fn packed_rejects_zero_dims() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [0.0];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&[], &[], 0, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
    }

    #[test]
    fn packed_rejects_non_block_k() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [0.0; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&[1, 2], &[3], 5, 255), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&[], &[], 5, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
    }

    #[test]
    fn packed_reports_unexpected_event() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ6Packed, &mut []),
            Err(Q6PackedError::UnexpectedEvent)
        );
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_event_trait_dispatches() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 1];
        assert_eq!(
            Q6PackedEvent::dispatch(
                OpMulMatQ6Packed::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut output,
            ),
            Ok(())
        );
        assert!(output[0].is_finite());
    }

    #[test]
    fn packed_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 1, QK_K), &mut second),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn packed_matrix_x4_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_MATRIX_X4_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:109-143"
        );
        assert_eq!(
            PINNED_MATRIX_X4_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:161-171"
        );
        assert_eq!(
            PINNED_MATRIX_X4_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6111-6150"
        );
        assert_eq!(
            PINNED_MATRIX_X4_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:4669-4855"
        );
        assert_eq!(
            PINNED_MATRIX_X4_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:490-493"
        );
        assert_eq!(MATRIX_X4_ROWS, 4);
    }

    #[test]
    fn neon_packed_matrix_x4_full_group_m8_dispatches() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut output = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Ok(())
        );
        assert!(kernel.is_ready());
        row = 0;
        while row < output.len() {
            assert!(output[row].is_finite());
            row += 1;
        }
    }

    #[test]
    fn neon_packed_matrix_x4_matches_independent_packed_dots() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut matrix = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut matrix
            ),
            Ok(())
        );
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X4_ROWS {
            let packed = pack_q8_k_row_at(rhs_row);
            let mut vector = [f32::NAN; 8];
            assert_eq!(
                kernel.process_event(OpMulMatQ6Packed::new(&lhs, &packed, 8, QK_K), &mut vector),
                Ok(())
            );
            let mut lane = 0;
            while lane < 8 {
                assert_eq!(matrix[rhs_row * 8 + lane].to_bits(), vector[lane].to_bits());
                lane += 1;
            }
            rhs_row += 1;
        }
    }

    #[test]
    fn packed_matrix_x4_rejects_zero_dims() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PackedMatrixX4::new(&[], &[], 0, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 4]);
    }

    #[test]
    fn packed_matrix_x4_rejects_k_zero() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PackedMatrixX4::new(&[], &[], 8, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_matrix_x4_rejects_rhs_rows_not_four() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 8, 1);
        let rhs = pack_q8_k_row_at(0);
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_matrix_x4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut via_process = [f32::NAN; MATRIX_X4_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q6PackedEvent::dispatch(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn packed_matrix_x4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut first = [f32::NAN; MATRIX_X4_ROWS];
        let mut second = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ6PackedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                    &mut second
                ),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }

    #[test]
    fn packed_argmax_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_ARGMAX_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:673-681"
        );
        assert_eq!(
            PINNED_ARGMAX_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1736-1805"
        );
        assert_eq!(
            PINNED_ARGMAX_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5546-5590"
        );
        assert_eq!(
            PINNED_ARGMAX_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:4669-4855"
        );
        assert_eq!(
            PINNED_ARGMAX_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:473-478"
        );
    }

    #[test]
    fn neon_packed_argmax_m5_matches_independent_packed_dots() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8(&rows, 5, 1);
        let rhs = pack_q8_k_row();
        let mut packed = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 5, QK_K), &mut packed),
            Ok(())
        );
        let mut expected_index = 0_i32;
        let mut expected_value = packed[0];
        row = 1;
        while row < 5 {
            if packed[row] > expected_value {
                expected_value = packed[row];
                expected_index = row as i32;
            }
            row += 1;
        }
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 5, QK_K),
                &mut output
            ),
            Ok(expected_index)
        );
        assert_eq!(output[0].to_bits(), expected_value.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_argmax_first_tie_keeps_earlier_index() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; 2];
        rows[0] = native_q6_k_row(0);
        rows[1] = native_q6_k_row(0);
        let lhs = pack_q6_k_rows_x8(&rows, 2, 1);
        let rhs = pack_q8_k_row();
        let mut packed = [f32::NAN; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Packed::new(&lhs, &rhs, 2, QK_K), &mut packed),
            Ok(())
        );
        assert_eq!(packed[0].to_bits(), packed[1].to_bits());
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 2, QK_K),
                &mut output
            ),
            Ok(0)
        );
        assert_eq!(output[0].to_bits(), packed[0].to_bits());
    }

    #[test]
    fn packed_argmax_rejects_zero_dims() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatArgmaxQ6Packed::new(&[], &[], 0, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_argmax_rejects_k_zero() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatArgmaxQ6Packed::new(&[], &[], 5, 0), &mut output),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_argmax_rejects_non_block_k() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ6Packed::new(&[1, 2], &[3], 5, 255),
                &mut output
            ),
            Err(Q6PackedError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_argmax_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut via_process = [f32::NAN; 1];
        let mut via_trait = [f32::NAN; 1];
        let process_index = kernel
            .process_event(
                OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 1, QK_K),
                &mut via_process,
            )
            .expect("argmax process_event");
        let trait_index = Q6PackedEvent::dispatch(
            OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 1, QK_K),
            &mut kernel,
            &mut via_trait,
        )
        .expect("argmax trait dispatch");
        assert_eq!(process_index, trait_index);
        assert_eq!(via_process[0].to_bits(), via_trait[0].to_bits());
    }

    #[test]
    fn packed_argmax_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PackedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        let first_index = kernel
            .process_event(OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 1, QK_K), &mut first)
            .expect("first argmax");
        let mut second_index = -1;
        let info = measure(|| {
            second_index = kernel
                .process_event(
                    OpMulMatArgmaxQ6Packed::new(&lhs, &rhs, 1, QK_K),
                    &mut second,
                )
                .expect("second argmax");
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first_index, second_index);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }
}
