//! Safe `AArch64` prepared `q6_k_x8_q8` by packed `q8_k` vector n=1, `matrix_x4`,
//! and `matrix_x8`.
//!
//! The source-backed boundary is the live host route selected by
//! `can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm`,
//! `can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4`, and
//! `can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8`.  On this
//! host `FEAT_I8MM=1`, so C++ compile-disables
//! `can_use_neon_mul_mat_q6_vector_prepared_q8_rhs` and executes
//! `execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_unchecked`,
//! `execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4_unchecked`, or
//! `execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8_unchecked`.
//! The I8MM `matrix_x4` and `matrix_x8` kernels pair RHS rows for mmla
//! throughput while keeping integer dots exact and the per-block `vfma` fold
//! unchanged, so independent n=1 prepared dots stay bit-identical.  Prepared
//! LHS uses the I8MM qs layout from `make_block_q6_k_x8_q8_prepared`.  Integer
//! dots are reconstructed with pulp widening i8 products.  Dispatch never uses
//! `pulp::cast!`, `core::arch`, EMEL unsafe, or a dequantize-to-F32 GEMV.

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

/// Errors returned by the `AArch64` prepared `q6_k_x8_q8` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q6PreparedError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q6PreparedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("q6 prepared NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid q6 prepared shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q6 prepared event"),
            Self::Internal => formatter.write_str("internal q6 prepared dispatch error"),
        }
    }
}

impl std::error::Error for Q6PreparedError {}

/// A pinned `AArch64` prepared `q6_k_x8_q8` by packed `q8_k` vector request.
///
/// `lhs` is group-major `block_q6_kx8_q8_prepared` storage with `ceil(m / 8)`
/// groups of `k / 256` I8MM-prepared blocks.  `rhs` is one packed `q8_k` row of
/// `k / 256` blocks.
#[derive(Debug)]
pub struct OpMulMatQ6Prepared<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6Prepared<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` prepared `q6_k_x8_q8` by 4 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q6_kx8_q8_prepared` storage with `ceil(m / 8)`
/// groups of `k / 256` I8MM-prepared blocks.  `rhs` is four packed `q8_k` rows
/// of `k / 256` blocks.  `dst` is batch-major F32 with shape `[4, m]`.
#[derive(Debug)]
pub struct OpMulMatQ6PreparedMatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6PreparedMatrixX4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` prepared `q6_k_x8_q8` by 8 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q6_kx8_q8_prepared` storage with `ceil(m / 8)`
/// groups of `k / 256` I8MM-prepared blocks.  `rhs` is eight packed `q8_k` rows
/// of `k / 256` blocks.  `dst` is batch-major F32 with shape `[8, m]`.
#[derive(Debug)]
pub struct OpMulMatQ6PreparedMatrixX8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ6PreparedMatrixX8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the prepared `q6_k_x8_q8` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ6Prepared;

/// Result returned by target prepared `q6_k_x8_q8` events.
pub type Q6PreparedResult = Result<(), Q6PreparedError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:637-647";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1664-1667";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5422-5456";
const PINNED_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:4943-5026";
const PINNED_PACK_SPAN: &str = "src/emel/kernel/detail.hpp:1103-1163";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:455-458";
const PINNED_MATRIX_X4_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:109-143";
const PINNED_MATRIX_X4_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:172-182";
const PINNED_MATRIX_X4_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6191-6251";
const PINNED_MATRIX_X4_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5028-5175";
const PINNED_MATRIX_X4_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:450-453";
const PINNED_MATRIX_X8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:221-254";
const PINNED_MATRIX_X8_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:265-273";
const PINNED_MATRIX_X8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6254-6315";
const PINNED_MATRIX_X8_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5177-5325";
const PINNED_MATRIX_X8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:445-448";
const SCOPE_RESIDUAL: &str = "q6_k_x8_q8 prepared lhs packed q8_k rhs n=1, packed q8_k matrix_x4, and packed q8_k matrix_x8 i8mm live path; argmax, generic neon prepared, F16 neon, and x86 remain residuals";

const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q6_K_X8_ROWS: usize = 8;
const MATRIX_X4_ROWS: usize = 4;
const MATRIX_X8_ROWS: usize = 8;
const Q6_K_X8_PREPARED_BLOCK_BYTES: usize = 2192;
const Q8_K_BLOCK_BYTES: usize = 292;
const SCALE_OFFSET: usize = 16;
const QS_OFFSET: usize = 144;
const SCALE_COUNT: usize = QK_K / 16;
const PAIR_COUNT: usize = Q6_K_X8_ROWS / 2;
const PAIR_BYTES: usize = 32;
const SCALE_QS_BYTES: usize = PAIR_COUNT * PAIR_BYTES;
const MAX_Q8_K_BLOCKS: usize = 128;

/// Event implemented by the target prepared `q6_k_x8_q8` actor.
pub trait Q6PreparedEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q6PreparedKernel, output: &mut [f32]) -> Self::Output;
}

struct PreparedRuntime<'a> {
    event: OpMulMatQ6Prepared<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PreparedResult>,
}

struct PreparedMatrixX4Runtime<'a> {
    event: OpMulMatQ6PreparedMatrixX4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PreparedResult>,
}

struct PreparedMatrixX8Runtime<'a> {
    event: OpMulMatQ6PreparedMatrixX8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q6PreparedResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q6PreparedResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
}

sml! {
    Q6PreparedMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Prepared(PreparedRuntime<'dispatch>) [guard_ready] / effect_execute,
        "ready"_s <= "ready"_s + Prepared(PreparedRuntime<'dispatch>) [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + PreparedMatrixX4(PreparedMatrixX4Runtime<'dispatch>) [guard_matrix_ready] / effect_matrix_execute,
        "ready"_s <= "ready"_s + PreparedMatrixX4(PreparedMatrixX4Runtime<'dispatch>) [guard_matrix_invalid] / effect_matrix_invalid,
        "ready"_s <= "ready"_s + PreparedMatrixX8(PreparedMatrixX8Runtime<'dispatch>) [guard_matrix_x8_ready] / effect_matrix_x8_execute,
        "ready"_s <= "ready"_s + PreparedMatrixX8(PreparedMatrixX8Runtime<'dispatch>) [guard_matrix_x8_invalid] / effect_matrix_x8_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` prepared `q6_k_x8_q8` actor.
pub struct Q6PreparedKernel {
    machine: Q6PreparedMachineStateMachine<Context>,
}

impl fmt::Debug for Q6PreparedKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q6PreparedKernel")
            .finish_non_exhaustive()
    }
}

impl Q6PreparedKernel {
    /// Resolves NEON before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q6PreparedMachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q6PreparedEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q6PreparedMachineStates::Ready)
    }

    fn prepared(&mut self, event: OpMulMatQ6Prepared<'_>, output: &mut [f32]) -> Q6PreparedResult {
        let result = Cell::new(Err(Q6PreparedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PreparedMachineEvents::Prepared(PreparedRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }

    fn prepared_matrix_x4(
        &mut self,
        event: OpMulMatQ6PreparedMatrixX4<'_>,
        output: &mut [f32],
    ) -> Q6PreparedResult {
        let result = Cell::new(Err(Q6PreparedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PreparedMachineEvents::PreparedMatrixX4(
                PreparedMatrixX4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }

    fn prepared_matrix_x8(
        &mut self,
        event: OpMulMatQ6PreparedMatrixX8<'_>,
        output: &mut [f32],
    ) -> Q6PreparedResult {
        let result = Cell::new(Err(Q6PreparedError::UnexpectedEvent));
        self.machine
            .process_event(Q6PreparedMachineEvents::PreparedMatrixX8(
                PreparedMatrixX8Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }
}

impl Q6PreparedEvent for OpMulMatQ6Prepared<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Q6PreparedKernel, output: &mut [f32]) -> Self::Output {
        actor.prepared(self, output)
    }
}

impl Q6PreparedEvent for OpMulMatQ6PreparedMatrixX4<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Q6PreparedKernel, output: &mut [f32]) -> Self::Output {
        actor.prepared_matrix_x4(self, output)
    }
}

impl Q6PreparedEvent for OpMulMatQ6PreparedMatrixX8<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Q6PreparedKernel, output: &mut [f32]) -> Self::Output {
        actor.prepared_matrix_x8(self, output)
    }
}

impl Q6PreparedEvent for UnexpectedQ6Prepared {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Q6PreparedKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PreparedError::Internal));
        actor
            .machine
            .process_event(Q6PreparedMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }
}

impl Q6PreparedMachineStateMachineContext for Context {
    fn guard_ready(&self, event: &PreparedRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && prepared_request_valid(event))
    }

    fn guard_invalid(&self, event: &PreparedRuntime<'_>) -> Result<bool, ()> {
        Ok(!prepared_request_valid(event))
    }

    fn guard_matrix_ready(&self, event: &PreparedMatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && prepared_matrix_x4_request_valid(event))
    }

    fn guard_matrix_invalid(&self, event: &PreparedMatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(!prepared_matrix_x4_request_valid(event))
    }

    fn guard_matrix_x8_ready(&self, event: &PreparedMatrixX8Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && prepared_matrix_x8_request_valid(event))
    }

    fn guard_matrix_x8_invalid(&self, event: &PreparedMatrixX8Runtime<'_>) -> Result<bool, ()> {
        Ok(!prepared_matrix_x8_request_valid(event))
    }

    fn effect_execute(&mut self, event: PreparedRuntime<'_>) -> Result<(), ()> {
        execute_prepared(
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

    fn effect_invalid(&mut self, event: PreparedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PreparedError::InvalidShape));
        Ok(())
    }

    fn effect_matrix_execute(&mut self, event: PreparedMatrixX4Runtime<'_>) -> Result<(), ()> {
        execute_prepared_matrix::<MATRIX_X4_ROWS>(
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

    fn effect_matrix_invalid(&mut self, event: PreparedMatrixX4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PreparedError::InvalidShape));
        Ok(())
    }

    fn effect_matrix_x8_execute(&mut self, event: PreparedMatrixX8Runtime<'_>) -> Result<(), ()> {
        execute_prepared_matrix::<MATRIX_X8_ROWS>(
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

    fn effect_matrix_x8_invalid(&mut self, event: PreparedMatrixX8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PreparedError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q6PreparedError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn prepared_request_valid(event: &PreparedRuntime<'_>) -> bool {
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
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_PREPARED_BLOCK_BYTES) else {
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

const fn prepared_matrix_x4_request_valid(event: &PreparedMatrixX4Runtime<'_>) -> bool {
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
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_PREPARED_BLOCK_BYTES) else {
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

const fn prepared_matrix_x8_request_valid(event: &PreparedMatrixX8Runtime<'_>) -> bool {
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
    let Some(group_bytes) = blocks.checked_mul(Q6_K_X8_PREPARED_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_row_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(rhs_bytes) = rhs_row_bytes.checked_mul(MATRIX_X8_ROWS) else {
        return false;
    };
    let Some(output_len) = request.m.checked_mul(MATRIX_X8_ROWS) else {
        return false;
    };
    request.lhs.len() == lhs_bytes
        && request.rhs.len() == rhs_bytes
        && event.output.len() == output_len
}

fn execute_prepared(backend: Neon, lhs: &[u8], rhs: &[u8], output: &mut [f32], m: usize, k: usize) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let group_bytes = blocks * Q6_K_X8_PREPARED_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q6_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q6_K_X8_PREPARED_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block(
                backend,
                &lhs[lhs_offset..lhs_offset + Q6_K_X8_PREPARED_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        store_q6_k_x8_results(output, group * Q6_K_X8_ROWS, m, acc);
        group += 1;
    }
}

fn execute_prepared_matrix<const RHS_ROWS: usize>(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let group_bytes = blocks * Q6_K_X8_PREPARED_BLOCK_BYTES;
    let rhs_row_bytes = blocks * Q8_K_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [[0.0_f32; Q6_K_X8_ROWS]; RHS_ROWS];
        let mut rhs_row = 0;
        while rhs_row < RHS_ROWS {
            let mut block = 0;
            while block < blocks {
                let lhs_offset = group * group_bytes + block * Q6_K_X8_PREPARED_BLOCK_BYTES;
                let rhs_offset = rhs_row * rhs_row_bytes + block * Q8_K_BLOCK_BYTES;
                accumulate_group_block(
                    backend,
                    &lhs[lhs_offset..lhs_offset + Q6_K_X8_PREPARED_BLOCK_BYTES],
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

#[allow(clippy::cast_precision_loss)]
fn accumulate_group_block(backend: Neon, lhs: &[u8], rhs: &[u8], acc: &mut [f32; Q6_K_X8_ROWS]) {
    let mut q8_d_bytes = [0_u8; 4];
    q8_d_bytes.copy_from_slice(&rhs[..4]);
    let q8_d = f32::from_le_bytes(q8_d_bytes);
    let mut isum = [0_i32; Q6_K_X8_ROWS];
    let mut scale = 0;
    while scale < SCALE_COUNT {
        let q8_offset = 4 + scale * 16;
        let q8_values = load_i8_slice(&rhs[q8_offset..q8_offset + 16]);
        let mut pair = 0;
        while pair < PAIR_COUNT {
            let pair_base = QS_OFFSET + scale * SCALE_QS_BYTES + pair * PAIR_BYTES;
            let row0 = pair * 2;
            let row1 = row0 + 1;
            let lhs_row0 = load_prepared_row(lhs, pair_base, 0);
            let lhs_row1 = load_prepared_row(lhs, pair_base, 1);
            let scale_row0 = i8::from_ne_bytes([lhs[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row0]]);
            let scale_row1 = i8::from_ne_bytes([lhs[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row1]]);
            isum[row0] += i32::from(scale_row0) * widening_i8_dot(backend, lhs_row0, q8_values);
            isum[row1] += i32::from(scale_row1) * widening_i8_dot(backend, lhs_row1, q8_values);
            pair += 1;
        }
        scale += 1;
    }
    let mut row = 0;
    while row < Q6_K_X8_ROWS {
        let d = fp16_to_f32(u16::from_le_bytes([lhs[row * 2], lhs[row * 2 + 1]]));
        let block_scale = d * q8_d;
        acc[row] = (isum[row] as f32).mul_add(block_scale, acc[row]);
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

fn store_batch_major_group_results<const RHS_ROWS: usize>(
    output: &mut [f32],
    row_base: usize,
    total_rows: usize,
    values: [[f32; Q6_K_X8_ROWS]; RHS_ROWS],
) {
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q6_K_X8_ROWS);
    let mut rhs_row = 0;
    while rhs_row < RHS_ROWS {
        let dst_row = rhs_row * total_rows + row_base;
        let mut lane = 0;
        while lane < store_count {
            output[dst_row + lane] = values[rhs_row][lane];
            lane += 1;
        }
        rhs_row += 1;
    }
}

const fn load_prepared_row(lhs: &[u8], pair_base: usize, row_in_pair: usize) -> i8x16 {
    let half = row_in_pair * 8;
    let lo = pair_base + half;
    let hi = pair_base + 16 + half;
    i8x16(
        i8::from_ne_bytes([lhs[lo]]),
        i8::from_ne_bytes([lhs[lo + 1]]),
        i8::from_ne_bytes([lhs[lo + 2]]),
        i8::from_ne_bytes([lhs[lo + 3]]),
        i8::from_ne_bytes([lhs[lo + 4]]),
        i8::from_ne_bytes([lhs[lo + 5]]),
        i8::from_ne_bytes([lhs[lo + 6]]),
        i8::from_ne_bytes([lhs[lo + 7]]),
        i8::from_ne_bytes([lhs[hi]]),
        i8::from_ne_bytes([lhs[hi + 1]]),
        i8::from_ne_bytes([lhs[hi + 2]]),
        i8::from_ne_bytes([lhs[hi + 3]]),
        i8::from_ne_bytes([lhs[hi + 4]]),
        i8::from_ne_bytes([lhs[hi + 5]]),
        i8::from_ne_bytes([lhs[hi + 6]]),
        i8::from_ne_bytes([lhs[hi + 7]]),
    )
}

const fn load_i8_slice(src: &[u8]) -> i8x16 {
    i8x16(
        i8::from_ne_bytes([src[0]]),
        i8::from_ne_bytes([src[1]]),
        i8::from_ne_bytes([src[2]]),
        i8::from_ne_bytes([src[3]]),
        i8::from_ne_bytes([src[4]]),
        i8::from_ne_bytes([src[5]]),
        i8::from_ne_bytes([src[6]]),
        i8::from_ne_bytes([src[7]]),
        i8::from_ne_bytes([src[8]]),
        i8::from_ne_bytes([src[9]]),
        i8::from_ne_bytes([src[10]]),
        i8::from_ne_bytes([src[11]]),
        i8::from_ne_bytes([src[12]]),
        i8::from_ne_bytes([src[13]]),
        i8::from_ne_bytes([src[14]]),
        i8::from_ne_bytes([src[15]]),
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

const fn decode_q6_nibble(ql_nibble: u8, qh_bits: u8) -> i8 {
    let q6 = (ql_nibble & 0x0f) | ((qh_bits & 0x03) << 4);
    q6.cast_signed().wrapping_sub(32)
}

const fn decode_q6_k_row(row: &[u8; Q6_K_BLOCK_BYTES]) -> [i8; QK_K] {
    let mut decoded = [0_i8; QK_K];
    let mut half = 0;
    while half < 2 {
        let half_value_base = half * 128;
        let low_base = half * 64;
        let high_base = half * 32;
        let mut lane = 0;
        while lane < 32 {
            let qh_byte = row[128 + high_base + lane];
            let ql_low = row[low_base + lane];
            let ql_high = row[low_base + 32 + lane];
            decoded[half_value_base + lane] = decode_q6_nibble(ql_low, qh_byte);
            decoded[half_value_base + lane + 32] = decode_q6_nibble(ql_high, qh_byte >> 2);
            decoded[half_value_base + lane + 64] = decode_q6_nibble(ql_low >> 4, qh_byte >> 4);
            decoded[half_value_base + lane + 96] = decode_q6_nibble(ql_high >> 4, qh_byte >> 6);
            lane += 1;
        }
        half += 1;
    }
    decoded
}

const fn make_block_q6_k_x8_q8_prepared(
    rows: &[[u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS],
) -> [u8; Q6_K_X8_PREPARED_BLOCK_BYTES] {
    let mut out = [0_u8; Q6_K_X8_PREPARED_BLOCK_BYTES];
    let mut row = 0;
    while row < Q6_K_X8_ROWS {
        out[row * 2] = rows[row][208];
        out[row * 2 + 1] = rows[row][209];
        let decoded = decode_q6_k_row(&rows[row]);
        let mut scale = 0;
        while scale < SCALE_COUNT {
            out[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row] = rows[row][192 + scale];
            let pair_index = row / 2;
            let pair_base = QS_OFFSET + scale * SCALE_QS_BYTES + pair_index * PAIR_BYTES;
            let half_base = pair_base + (row % 2) * 8;
            let mut lane = 0;
            while lane < 8 {
                out[half_base + lane] = decoded[scale * 16 + lane].to_ne_bytes()[0];
                out[half_base + 16 + lane] = decoded[scale * 16 + 8 + lane].to_ne_bytes()[0];
                lane += 1;
            }
            scale += 1;
        }
        row += 1;
    }
    out
}

#[cfg(test)]
fn pack_q6_k_rows_x8_q8_prepared(
    src: &[[u8; Q6_K_BLOCK_BYTES]],
    m: usize,
    blocks: usize,
) -> Vec<u8> {
    let group_count = m.div_ceil(Q6_K_X8_ROWS);
    let mut packed = vec![0_u8; group_count * blocks * Q6_K_X8_PREPARED_BLOCK_BYTES];
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
            let offset = (group * blocks + block) * Q6_K_X8_PREPARED_BLOCK_BYTES;
            packed[offset..offset + Q6_K_X8_PREPARED_BLOCK_BYTES]
                .copy_from_slice(&make_block_q6_k_x8_q8_prepared(&group_rows));
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
        MATRIX_X4_ROWS, MATRIX_X8_ROWS, MAX_Q8_K_BLOCKS, OpMulMatQ6Prepared,
        OpMulMatQ6PreparedMatrixX4, OpMulMatQ6PreparedMatrixX8, PINNED_ACTION_BLOB,
        PINNED_ACTION_SPAN, PINNED_EMEL_CPP_COMMIT, PINNED_EXECUTION_SPAN, PINNED_GUARD_BLOB,
        PINNED_GUARD_SPAN, PINNED_KERNEL_SPAN, PINNED_MATRIX_X4_ACTION_SPAN,
        PINNED_MATRIX_X4_EXECUTION_SPAN, PINNED_MATRIX_X4_GUARD_SPAN, PINNED_MATRIX_X4_KERNEL_SPAN,
        PINNED_MATRIX_X4_TRANSITION_SPAN, PINNED_MATRIX_X8_ACTION_SPAN,
        PINNED_MATRIX_X8_EXECUTION_SPAN, PINNED_MATRIX_X8_GUARD_SPAN, PINNED_MATRIX_X8_KERNEL_SPAN,
        PINNED_MATRIX_X8_TRANSITION_SPAN, PINNED_PACK_SPAN, PINNED_SM_BLOB, PINNED_TRANSITION_SPAN,
        Q6_K_BLOCK_BYTES, Q6_K_X8_PREPARED_BLOCK_BYTES, Q6_K_X8_ROWS, Q6PreparedError,
        Q6PreparedEvent, Q6PreparedKernel, Q8_K_BLOCK_BYTES, QK_K, QS_OFFSET, SCALE_OFFSET,
        SCOPE_RESIDUAL, UnexpectedQ6Prepared, decode_q6_k_row, make_block_q6_k_x8_q8_prepared,
        pack_q6_k_rows_x8_q8_prepared,
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

    fn pack_q8_k_rows_x8() -> [u8; MATRIX_X8_ROWS * Q8_K_BLOCK_BYTES] {
        let mut out = [0_u8; MATRIX_X8_ROWS * Q8_K_BLOCK_BYTES];
        let mut row = 0;
        while row < MATRIX_X8_ROWS {
            let packed = pack_q8_k_row_at(row);
            let offset = row * Q8_K_BLOCK_BYTES;
            out[offset..offset + Q8_K_BLOCK_BYTES].copy_from_slice(&packed);
            row += 1;
        }
        out
    }

    #[test]
    fn prepared_kernel_constructs_on_neon() {
        assert!(Q6PreparedKernel::try_new().is_some());
    }

    #[test]
    fn prepared_pin_constants_match_pinned_source() {
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
            "src/emel/kernel/aarch64/guards.hpp:637-647"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1664-1667"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5422-5456"
        );
        assert_eq!(
            PINNED_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:4943-5026"
        );
        assert_eq!(PINNED_PACK_SPAN, "src/emel/kernel/detail.hpp:1103-1163");
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:455-458"
        );
        assert!(SCOPE_RESIDUAL.contains("prepared lhs packed q8_k rhs n=1"));
        assert!(SCOPE_RESIDUAL.contains("packed q8_k matrix_x4"));
        assert!(SCOPE_RESIDUAL.contains("packed q8_k matrix_x8"));
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
        assert_eq!(Q6_K_X8_PREPARED_BLOCK_BYTES, 2192);
        assert_eq!(Q6_K_X8_ROWS, 8);
        assert_eq!(Q8_K_BLOCK_BYTES, 292);
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(QK_K, 256);
        assert_eq!(SCALE_OFFSET, 16);
        assert_eq!(QS_OFFSET, 144);
    }

    #[test]
    fn i8mm_prepared_layout_places_decoded_pairs() {
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let packed = make_block_q6_k_x8_q8_prepared(&rows);
        let decoded0 = decode_q6_k_row(&rows[0]);
        let decoded1 = decode_q6_k_row(&rows[1]);
        let mut lane = 0;
        while lane < 8 {
            assert_eq!(
                i8::from_ne_bytes([packed[QS_OFFSET + lane]]),
                decoded0[lane]
            );
            assert_eq!(
                i8::from_ne_bytes([packed[QS_OFFSET + 16 + lane]]),
                decoded0[8 + lane]
            );
            assert_eq!(
                i8::from_ne_bytes([packed[QS_OFFSET + 8 + lane]]),
                decoded1[lane]
            );
            assert_eq!(
                i8::from_ne_bytes([packed[QS_OFFSET + 24 + lane]]),
                decoded1[8 + lane]
            );
            lane += 1;
        }
        assert_eq!(packed[SCALE_OFFSET], rows[0][192]);
        assert_eq!(packed[SCALE_OFFSET + 1], rows[1][192]);
    }

    #[test]
    fn neon_prepared_full_group_m8_dispatches() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &rhs, 8, QK_K), &mut output),
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
    fn neon_prepared_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 5, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &rhs, 5, QK_K), &mut output),
            Ok(())
        );
        let mut lane = 0;
        while lane < 5 {
            assert!(output[lane].is_finite());
            lane += 1;
        }
    }

    #[test]
    fn prepared_rejects_zero_dims() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&[], &[], 0, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn prepared_rejects_non_block_k_and_k_zero() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&[1, 2], &[3], 5, 255), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&[], &[], 5, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn prepared_reports_unexpected_event() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ6Prepared, &mut []),
            Err(Q6PreparedError::UnexpectedEvent)
        );
        assert!(kernel.is_ready());
    }

    #[test]
    fn prepared_event_trait_dispatches() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 1];
        assert_eq!(
            Q6PreparedEvent::dispatch(
                OpMulMatQ6Prepared::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut output,
            ),
            Ok(())
        );
        assert!(output[0].is_finite());
    }

    #[test]
    fn prepared_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &rhs, 1, QK_K), &mut second),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn prepared_matrix_x4_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_MATRIX_X4_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:109-143"
        );
        assert_eq!(
            PINNED_MATRIX_X4_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:172-182"
        );
        assert_eq!(
            PINNED_MATRIX_X4_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6191-6251"
        );
        assert_eq!(
            PINNED_MATRIX_X4_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5028-5175"
        );
        assert_eq!(
            PINNED_MATRIX_X4_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:450-453"
        );
        assert_eq!(MATRIX_X4_ROWS, 4);
    }

    #[test]
    fn neon_prepared_matrix_x4_full_group_m8_dispatches() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut output = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 8, QK_K),
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
    fn neon_prepared_matrix_x4_matches_independent_prepared_dots() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut matrix = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut matrix
            ),
            Ok(())
        );
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X4_ROWS {
            let packed = pack_q8_k_row_at(rhs_row);
            let mut vector = [f32::NAN; 8];
            assert_eq!(
                kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &packed, 8, QK_K), &mut vector),
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
    fn prepared_matrix_x4_rejects_zero_dims() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PreparedMatrixX4::new(&[], &[], 0, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 4]);
    }

    #[test]
    fn prepared_matrix_x4_rejects_k_zero() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PreparedMatrixX4::new(&[], &[], 8, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn prepared_matrix_x4_rejects_rhs_rows_not_four() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_row_at(0);
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn prepared_matrix_x4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut via_process = [f32::NAN; MATRIX_X4_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q6PreparedEvent::dispatch(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn prepared_matrix_x4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut first = [f32::NAN; MATRIX_X4_ROWS];
        let mut second = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 1, QK_K),
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
    fn prepared_matrix_x8_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_MATRIX_X8_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:221-254"
        );
        assert_eq!(
            PINNED_MATRIX_X8_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:265-273"
        );
        assert_eq!(
            PINNED_MATRIX_X8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6254-6315"
        );
        assert_eq!(
            PINNED_MATRIX_X8_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5177-5325"
        );
        assert_eq!(
            PINNED_MATRIX_X8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:445-448"
        );
        assert_eq!(MATRIX_X8_ROWS, 8);
    }

    #[test]
    fn neon_prepared_matrix_x8_full_group_m8_dispatches() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut output = [f32::NAN; MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 8, QK_K),
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
    fn neon_prepared_matrix_x8_matches_independent_prepared_dots() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut matrix = [f32::NAN; MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 8, QK_K),
                &mut matrix
            ),
            Ok(())
        );
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X8_ROWS {
            let packed = pack_q8_k_row_at(rhs_row);
            let mut vector = [f32::NAN; 8];
            assert_eq!(
                kernel.process_event(OpMulMatQ6Prepared::new(&lhs, &packed, 8, QK_K), &mut vector),
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
    fn prepared_matrix_x8_rejects_zero_dims() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PreparedMatrixX8::new(&[], &[], 0, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 8]);
    }

    #[test]
    fn prepared_matrix_x8_rejects_k_zero() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ6PreparedMatrixX8::new(&[], &[], 8, 0), &mut output),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X8_ROWS * 8]);
    }

    #[test]
    fn prepared_matrix_x8_rejects_rhs_rows_not_eight() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
        let mut row = 0;
        while row < Q6_K_X8_ROWS {
            rows[row] = native_q6_k_row(row);
            row += 1;
        }
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 8, 1);
        let rhs = pack_q8_k_row_at(0);
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q6PreparedError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X8_ROWS * 8]);
    }

    #[test]
    fn prepared_matrix_x8_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut via_process = [f32::NAN; MATRIX_X8_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X8_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q6PreparedEvent::dispatch(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn prepared_matrix_x8_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q6PreparedKernel::try_new() else {
            return;
        };
        let rows = [native_q6_k_row(0)];
        let lhs = pack_q6_k_rows_x8_q8_prepared(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut first = [f32::NAN; MATRIX_X8_ROWS];
        let mut second = [f32::NAN; MATRIX_X8_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 1, QK_K),
                    &mut second
                ),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }
}
