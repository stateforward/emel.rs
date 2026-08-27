//! Safe `AArch64` NEON packed `q8_0` x4/bl4 and x4/bl8 matmul for the pinned kernel contract.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_use_neon_mul_mat_q8_0_packed_bl4` / `bl8`: interleaved `q8_0_x4_bl4` or `q8_0_x4_bl8` `src0`
//! with shape `[k, m]`, packed `q8_0` `src1` with shape `[1, k]`, and dense
//! F32 `dst` with shape `[1, m]`.  Capability detection happens during
//! construction.  The dispatch action receives a typed `pulp::aarch64::Neon`
//! backend and therefore never performs runtime backend selection, dequantizes
//! into an F32 GEMV, or falls back to the portable actor.
//!
//! Integer products use the pinned `vdotq_laneq_s32` / `vdotq_s32` contract expressed through
//! pulp's stable `i16` multiply after reconstructing each of the four
//! interleaved rows and widening.  That integer sum matches the live C++ DOTPROD
//! path on this machine.

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

use crate::any::quant::{BLOCK_VALUES, Q8_0_BLOCK_BYTES, fp16_to_f32};

/// Errors returned by the `AArch64` packed `q8_0` x4/bl4 actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8_0PackedBl4Error {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q8_0PackedBl4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => {
                formatter.write_str("q8_0 packed bl4 NEON backend unavailable")
            }
            Self::InvalidShape => formatter.write_str("invalid q8_0 packed bl4 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q8_0 packed bl4 event"),
            Self::Internal => formatter.write_str("internal q8_0 packed bl4 dispatch error"),
        }
    }
}

impl std::error::Error for Q8_0PackedBl4Error {}

/// A pinned `AArch64` packed `q8_0_x4_bl4` by packed `q8_0` vector request.
///
/// `lhs` is group-major `block_q8_0x4` storage with `ceil(m / 4)` groups of
/// `k / 32` interleaved blocks.  `rhs` is one packed `q8_0` row of `k / 32`
/// blocks.
#[derive(Debug)]
pub struct OpMulMatQ8_0PackedBl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ8_0PackedBl4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q8_0_x4_bl8` by packed `q8_0` vector request.
///
/// Layout matches `block_q8_0x4` with 8-byte interleave.  `rhs` is one packed
/// `q8_0` row of `k / 32` blocks.
#[derive(Debug)]
pub struct OpMulMatQ8_0PackedBl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ8_0PackedBl8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q8_0_x4_bl8` matrix-by-4 request.
///
/// `lhs` and `rhs` are both `block_q8_0x4` with 8-byte interleave.  `rhs` holds
/// exactly four packed rows.  `dst` is dense F32 with shape `[4, m]`.
#[derive(Debug)]
pub struct OpMulMatQ8_0PackedBl8MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ8_0PackedBl8MatrixX4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the packed `q8_0` x4/bl4 API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ8_0PackedBl4;

/// Result returned by target packed `q8_0` x4/bl4 events.
pub type Q8_0PackedBl4Result = Result<(), Q8_0PackedBl4Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:455-466";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1282-1322";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6416-6466";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:387-388";
const PINNED_BL8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:420-428";
const PINNED_BL8_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1324-1348";
const PINNED_BL8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6468-6522";
const PINNED_BL8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:382-383";
const PINNED_BL8_MATRIX_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:35-72";
const PINNED_BL8_MATRIX_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:444-452";
const PINNED_BL8_MATRIX_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:6584-6667";
const PINNED_BL8_MATRIX_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:372-373";
const SCOPE_RESIDUAL: &str = "q8_0 packed x4/bl4, x4/bl8, and bl8 matrix_x4 (interleaved src0, packed q8_0 or x4 src1, dense F32 dst); F16 neon, generic quantized mul_mat, and x86 remain residuals";

const MAX_Q8_0_BLOCKS: usize = 1024;
const Q8_0_X4_ROWS: usize = 4;
const Q8_0_X4_BLOCK_BYTES: usize = 8 + BLOCK_VALUES * Q8_0_X4_ROWS;
const INTERLEAVE_BLOCK_BYTES: usize = 4;
const INTERLEAVE_BL8_BYTES: usize = 8;

/// Event implemented by the target packed `q8_0` x4/bl4 actor.
pub trait Q8_0PackedBl4Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q8_0PackedBl4Kernel, output: &mut [f32]) -> Self::Output;
}

struct PackedBl4Runtime<'a> {
    event: OpMulMatQ8_0PackedBl4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q8_0PackedBl4Result>,
}

struct PackedBl8Runtime<'a> {
    event: OpMulMatQ8_0PackedBl8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q8_0PackedBl4Result>,
}

struct PackedBl8MatrixX4Runtime<'a> {
    event: OpMulMatQ8_0PackedBl8MatrixX4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q8_0PackedBl4Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q8_0PackedBl4Result>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
}

sml! {
    Q8_0PackedBl4Machine<'dispatch> {
        "ready"_s <= *"ready"_s + PackedBl4(PackedBl4Runtime<'dispatch>) [guard_ready] / effect_execute,
        "ready"_s <= "ready"_s + PackedBl4(PackedBl4Runtime<'dispatch>) [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + PackedBl8(PackedBl8Runtime<'dispatch>) [guard_bl8_ready] / effect_bl8_execute,
        "ready"_s <= "ready"_s + PackedBl8(PackedBl8Runtime<'dispatch>) [guard_bl8_invalid] / effect_bl8_invalid,
        "ready"_s <= "ready"_s + PackedBl8MatrixX4(PackedBl8MatrixX4Runtime<'dispatch>) [guard_bl8_matrix_ready] / effect_bl8_matrix_execute,
        "ready"_s <= "ready"_s + PackedBl8MatrixX4(PackedBl8MatrixX4Runtime<'dispatch>) [guard_bl8_matrix_invalid] / effect_bl8_matrix_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` packed `q8_0` x4/bl4 actor.
pub struct Q8_0PackedBl4Kernel {
    machine: Q8_0PackedBl4MachineStateMachine<Context>,
}

impl fmt::Debug for Q8_0PackedBl4Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q8_0PackedBl4Kernel")
            .finish_non_exhaustive()
    }
}

impl Q8_0PackedBl4Kernel {
    /// Resolves NEON before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q8_0PackedBl4MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q8_0PackedBl4Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q8_0PackedBl4MachineStates::Ready)
    }

    fn packed_bl4(
        &mut self,
        event: OpMulMatQ8_0PackedBl4<'_>,
        output: &mut [f32],
    ) -> Q8_0PackedBl4Result {
        let result = Cell::new(Err(Q8_0PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q8_0PackedBl4MachineEvents::PackedBl4(PackedBl4Runtime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl8(
        &mut self,
        event: OpMulMatQ8_0PackedBl8<'_>,
        output: &mut [f32],
    ) -> Q8_0PackedBl4Result {
        let result = Cell::new(Err(Q8_0PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q8_0PackedBl4MachineEvents::PackedBl8(PackedBl8Runtime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl8_matrix_x4(
        &mut self,
        event: OpMulMatQ8_0PackedBl8MatrixX4<'_>,
        output: &mut [f32],
    ) -> Q8_0PackedBl4Result {
        let result = Cell::new(Err(Q8_0PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q8_0PackedBl4MachineEvents::PackedBl8MatrixX4(
                PackedBl8MatrixX4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }
}

impl Q8_0PackedBl4Event for OpMulMatQ8_0PackedBl4<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Q8_0PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl4(self, output)
    }
}

impl Q8_0PackedBl4Event for OpMulMatQ8_0PackedBl8<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Q8_0PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl8(self, output)
    }
}

impl Q8_0PackedBl4Event for OpMulMatQ8_0PackedBl8MatrixX4<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Q8_0PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl8_matrix_x4(self, output)
    }
}

impl Q8_0PackedBl4Event for UnexpectedQ8_0PackedBl4 {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Q8_0PackedBl4Kernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0PackedBl4Error::Internal));
        actor
            .machine
            .process_event(Q8_0PackedBl4MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }
}

impl Q8_0PackedBl4MachineStateMachineContext for Context {
    fn guard_ready(&self, event: &PackedBl4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl4_request_valid(event))
    }

    fn guard_invalid(&self, event: &PackedBl4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl4_request_valid(event))
    }

    fn effect_execute(&mut self, event: PackedBl4Runtime<'_>) -> Result<(), ()> {
        execute_packed(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
            INTERLEAVE_BLOCK_BYTES,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: PackedBl4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl8_ready(&self, event: &PackedBl8Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl8_request_valid(event))
    }

    fn guard_bl8_invalid(&self, event: &PackedBl8Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl8_request_valid(event))
    }

    fn effect_bl8_execute(&mut self, event: PackedBl8Runtime<'_>) -> Result<(), ()> {
        execute_packed(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
            INTERLEAVE_BL8_BYTES,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_bl8_invalid(&mut self, event: PackedBl8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl8_matrix_ready(&self, event: &PackedBl8MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl8_matrix_request_valid(event))
    }

    fn guard_bl8_matrix_invalid(&self, event: &PackedBl8MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl8_matrix_request_valid(event))
    }

    fn effect_bl8_matrix_execute(&mut self, event: PackedBl8MatrixX4Runtime<'_>) -> Result<(), ()> {
        execute_packed_bl8_matrix_x4(
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

    fn effect_bl8_matrix_invalid(&mut self, event: PackedBl8MatrixX4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q8_0PackedBl4Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn packed_bl4_request_valid(event: &PackedBl4Runtime<'_>) -> bool {
    packed_request_valid(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_bl8_request_valid(event: &PackedBl8Runtime<'_>) -> bool {
    packed_request_valid(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_bl8_matrix_request_valid(event: &PackedBl8MatrixX4Runtime<'_>) -> bool {
    let m = event.event.m;
    let k = event.event.k;
    if m == 0 || k == 0 || !m.is_multiple_of(Q8_0_X4_ROWS) || !k.is_multiple_of(BLOCK_VALUES) {
        return false;
    }
    let blocks = k / BLOCK_VALUES;
    if blocks == 0 || blocks > MAX_Q8_0_BLOCKS {
        return false;
    }
    let group_count = m / Q8_0_X4_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q8_0_X4_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(output_len) = Q8_0_X4_ROWS.checked_mul(m) else {
        return false;
    };
    event.event.lhs.len() == lhs_bytes
        && event.event.rhs.len() == group_bytes
        && event.output.len() == output_len
}

const fn packed_request_valid(lhs: &[u8], rhs: &[u8], output: &[f32], m: usize, k: usize) -> bool {
    if m == 0 || k == 0 || !k.is_multiple_of(BLOCK_VALUES) {
        return false;
    }
    let blocks = k / BLOCK_VALUES;
    if blocks == 0 || blocks > MAX_Q8_0_BLOCKS {
        return false;
    }
    let Some(group_count) = m.checked_add(Q8_0_X4_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q8_0_X4_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q8_0_X4_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_bytes) = blocks.checked_mul(Q8_0_BLOCK_BYTES) else {
        return false;
    };
    lhs.len() == lhs_bytes && rhs.len() == rhs_bytes && output.len() == m
}

fn execute_packed(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
    interleave: usize,
) {
    let blocks = k / BLOCK_VALUES;
    let group_count = m.div_ceil(Q8_0_X4_ROWS);
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q8_0_X4_ROWS];
        let group_offset = group * blocks * Q8_0_X4_BLOCK_BYTES;
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group_offset + block * Q8_0_X4_BLOCK_BYTES;
            let rhs_offset = block * Q8_0_BLOCK_BYTES;
            accumulate_packed_block(
                backend,
                &lhs[lhs_offset..lhs_offset + Q8_0_X4_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_0_BLOCK_BYTES],
                &mut acc,
                interleave,
            );
            block += 1;
        }
        store_q8_0_x4_results(output, group * Q8_0_X4_ROWS, m, acc);
        group += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn execute_packed_bl8_matrix_x4(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / BLOCK_VALUES;
    let group_count = m / Q8_0_X4_ROWS;
    let mut group = 0;
    while group < group_count {
        let mut acc = [[0.0_f32; Q8_0_X4_ROWS]; Q8_0_X4_ROWS];
        let lhs_group = group * blocks * Q8_0_X4_BLOCK_BYTES;
        let mut block = 0;
        while block < blocks {
            let lhs_offset = lhs_group + block * Q8_0_X4_BLOCK_BYTES;
            let rhs_offset = block * Q8_0_X4_BLOCK_BYTES;
            accumulate_packed_bl8_matrix_block(
                backend,
                &lhs[lhs_offset..lhs_offset + Q8_0_X4_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_0_X4_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        let mut rhs_row = 0;
        while rhs_row < Q8_0_X4_ROWS {
            let dst_row = rhs_row * m + group * Q8_0_X4_ROWS;
            output[dst_row..dst_row + Q8_0_X4_ROWS].copy_from_slice(&acc[rhs_row]);
            rhs_row += 1;
        }
        group += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_packed_bl8_matrix_block(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    acc: &mut [[f32; Q8_0_X4_ROWS]; Q8_0_X4_ROWS],
) {
    let lhs_scale = load_x4_scales(lhs);
    let rhs_scale = load_x4_scales(rhs);
    let lhs_rows = deinterleave_x4_qs(&lhs[8..], INTERLEAVE_BL8_BYTES);
    let rhs_rows = deinterleave_x4_qs(&rhs[8..], INTERLEAVE_BL8_BYTES);
    let mut rhs_row = 0;
    while rhs_row < Q8_0_X4_ROWS {
        let mut lhs_col = 0;
        while lhs_col < Q8_0_X4_ROWS {
            let integer = packed_row_integer_dot(backend, lhs_rows[lhs_col], rhs_rows[rhs_row]);
            let scale = lhs_scale[lhs_col] * rhs_scale[rhs_row];
            acc[rhs_row][lhs_col] += integer as f32 * scale;
            lhs_col += 1;
        }
        rhs_row += 1;
    }
}

fn load_x4_scales(block: &[u8]) -> [f32; Q8_0_X4_ROWS] {
    let mut scales = [0.0_f32; Q8_0_X4_ROWS];
    let mut row = 0;
    while row < Q8_0_X4_ROWS {
        scales[row] = fp16_to_f32(u16::from_le_bytes([block[row * 2], block[row * 2 + 1]]));
        row += 1;
    }
    scales
}

fn packed_row_integer_dot(backend: Neon, lhs: [i8; BLOCK_VALUES], rhs: [i8; BLOCK_VALUES]) -> i32 {
    let mut lhs_lo = [0_i8; 16];
    let mut lhs_hi = [0_i8; 16];
    let mut rhs_lo = [0_i8; 16];
    let mut rhs_hi = [0_i8; 16];
    lhs_lo.copy_from_slice(&lhs[..16]);
    lhs_hi.copy_from_slice(&lhs[16..]);
    rhs_lo.copy_from_slice(&rhs[..16]);
    rhs_hi.copy_from_slice(&rhs[16..]);
    widening_i8_dot(backend, cast(lhs_lo), cast(rhs_lo))
        + widening_i8_dot(backend, cast(lhs_hi), cast(rhs_hi))
}

// Keep the reference's separate scale product and accumulation.  Fusing the
// final multiply changes intermediate rounding and can change parity bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_packed_block(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    acc: &mut [f32; 4],
    interleave: usize,
) {
    let mut scales = [0.0_f32; Q8_0_X4_ROWS];
    let mut row = 0;
    while row < Q8_0_X4_ROWS {
        scales[row] = fp16_to_f32(u16::from_le_bytes([lhs[row * 2], lhs[row * 2 + 1]]));
        row += 1;
    }
    let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs[0], rhs[1]]));
    let rows = deinterleave_x4_qs(&lhs[8..], interleave);
    let mut rhs_lo_bytes = [0_i8; 16];
    let mut rhs_hi_bytes = [0_i8; 16];
    copy_i8_bytes(&mut rhs_lo_bytes, &rhs[2..18]);
    copy_i8_bytes(&mut rhs_hi_bytes, &rhs[18..Q8_0_BLOCK_BYTES]);
    let rhs_lo: i8x16 = cast(rhs_lo_bytes);
    let rhs_hi: i8x16 = cast(rhs_hi_bytes);
    row = 0;
    while row < Q8_0_X4_ROWS {
        let mut lhs_lo_bytes = [0_i8; 16];
        let mut lhs_hi_bytes = [0_i8; 16];
        lhs_lo_bytes.copy_from_slice(&rows[row][..16]);
        lhs_hi_bytes.copy_from_slice(&rows[row][16..]);
        let lhs_lo: i8x16 = cast(lhs_lo_bytes);
        let lhs_hi: i8x16 = cast(lhs_hi_bytes);
        let integer =
            widening_i8_dot(backend, lhs_lo, rhs_lo) + widening_i8_dot(backend, lhs_hi, rhs_hi);
        let scale = scales[row] * rhs_scale;
        acc[row] += integer as f32 * scale;
        row += 1;
    }
}

fn store_q8_0_x4_results(output: &mut [f32], row_base: usize, total_rows: usize, values: [f32; 4]) {
    if row_base + Q8_0_X4_ROWS <= total_rows {
        output[row_base..row_base + Q8_0_X4_ROWS].copy_from_slice(&values);
        return;
    }
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q8_0_X4_ROWS);
    let mut lane = 0;
    while lane < store_count {
        output[row_base + lane] = values[lane];
        lane += 1;
    }
}

const fn deinterleave_x4_qs(qs: &[u8], interleave: usize) -> [[i8; BLOCK_VALUES]; Q8_0_X4_ROWS] {
    let mut rows = [[0_i8; BLOCK_VALUES]; Q8_0_X4_ROWS];
    if interleave == 0 || !BLOCK_VALUES.is_multiple_of(interleave) {
        return rows;
    }
    let group_count = (BLOCK_VALUES * Q8_0_X4_ROWS) / interleave;
    let mut group = 0;
    while group < group_count {
        let src_row = group % Q8_0_X4_ROWS;
        let src_offset = (group / Q8_0_X4_ROWS) * interleave;
        let dst_offset = group * interleave;
        let mut lane = 0;
        while lane < interleave {
            rows[src_row][src_offset + lane] = i8::from_ne_bytes([qs[dst_offset + lane]]);
            lane += 1;
        }
        group += 1;
    }
    rows
}

const fn copy_i8_bytes(dst: &mut [i8], src: &[u8]) {
    let mut index = 0;
    while index < dst.len() {
        dst[index] = i8::from_ne_bytes([src[index]]);
        index += 1;
    }
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

#[cfg(test)]
fn pack_q8_0_x4_bl4(rows: &[[u8; Q8_0_BLOCK_BYTES]], m: usize, blocks: usize) -> Vec<u8> {
    pack_q8_0_x4(rows, m, blocks, INTERLEAVE_BLOCK_BYTES)
}

#[cfg(test)]
fn pack_q8_0_x4_bl8(rows: &[[u8; Q8_0_BLOCK_BYTES]], m: usize, blocks: usize) -> Vec<u8> {
    pack_q8_0_x4(rows, m, blocks, INTERLEAVE_BL8_BYTES)
}

#[cfg(test)]
fn pack_q8_0_x4(
    rows: &[[u8; Q8_0_BLOCK_BYTES]],
    m: usize,
    blocks: usize,
    interleave: usize,
) -> Vec<u8> {
    let group_count = m.div_ceil(Q8_0_X4_ROWS);
    let mut packed = vec![0_u8; group_count * blocks * Q8_0_X4_BLOCK_BYTES];
    let mut group = 0;
    while group < group_count {
        let row_base = group * Q8_0_X4_ROWS;
        let mut block = 0;
        while block < blocks {
            let mut group_rows = [[0_u8; Q8_0_BLOCK_BYTES]; Q8_0_X4_ROWS];
            let mut row = 0;
            while row < Q8_0_X4_ROWS {
                let logical_row = row_base + row;
                if logical_row < m {
                    group_rows[row] = rows[logical_row * blocks + block];
                }
                row += 1;
            }
            let offset = (group * blocks + block) * Q8_0_X4_BLOCK_BYTES;
            packed[offset..offset + Q8_0_X4_BLOCK_BYTES]
                .copy_from_slice(&make_block_q8_0_x4(&group_rows, interleave));
            block += 1;
        }
        group += 1;
    }
    packed
}

#[cfg(test)]
fn make_block_q8_0_x4(
    rows: &[[u8; Q8_0_BLOCK_BYTES]; Q8_0_X4_ROWS],
    interleave: usize,
) -> [u8; Q8_0_X4_BLOCK_BYTES] {
    let mut out = [0_u8; Q8_0_X4_BLOCK_BYTES];
    let mut row = 0;
    while row < Q8_0_X4_ROWS {
        out[row * 2..row * 2 + 2].copy_from_slice(&rows[row][..2]);
        row += 1;
    }
    if interleave == 0 || !BLOCK_VALUES.is_multiple_of(interleave) {
        return out;
    }
    let end = (BLOCK_VALUES * Q8_0_X4_ROWS) / interleave;
    let mut index = 0;
    while index < end {
        let src_row = index % Q8_0_X4_ROWS;
        let src_offset = (index / Q8_0_X4_ROWS) * interleave;
        let dst_offset = index * interleave;
        out[8 + dst_offset..8 + dst_offset + interleave]
            .copy_from_slice(&rows[src_row][2 + src_offset..2 + src_offset + interleave]);
        index += 1;
    }
    out
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
        BLOCK_VALUES, INTERLEAVE_BL8_BYTES, INTERLEAVE_BLOCK_BYTES, MAX_Q8_0_BLOCKS,
        OpMulMatQ8_0PackedBl4, OpMulMatQ8_0PackedBl8, OpMulMatQ8_0PackedBl8MatrixX4,
        PINNED_ACTION_BLOB, PINNED_ACTION_SPAN, PINNED_BL8_ACTION_SPAN, PINNED_BL8_EXECUTION_SPAN,
        PINNED_BL8_GUARD_SPAN, PINNED_BL8_MATRIX_ACTION_SPAN, PINNED_BL8_MATRIX_EXECUTION_SPAN,
        PINNED_BL8_MATRIX_GUARD_SPAN, PINNED_BL8_MATRIX_TRANSITION_SPAN,
        PINNED_BL8_TRANSITION_SPAN, PINNED_EMEL_CPP_COMMIT, PINNED_EXECUTION_SPAN,
        PINNED_GUARD_BLOB, PINNED_GUARD_SPAN, PINNED_SM_BLOB, PINNED_TRANSITION_SPAN,
        Q8_0_X4_BLOCK_BYTES, Q8_0_X4_ROWS, Q8_0PackedBl4Error, Q8_0PackedBl4Event,
        Q8_0PackedBl4Kernel, UnexpectedQ8_0PackedBl4, pack_q8_0_x4_bl4, pack_q8_0_x4_bl8,
        pack_q8_row,
    };
    use crate::any::quant::{Q8_0_BLOCK_BYTES, Q8_0Row, dot_q8_0_q8_0};

    fn ramp_i8(row: usize) -> [i8; BLOCK_VALUES] {
        let mut values = [0_i8; BLOCK_VALUES];
        let mut index = 0;
        while index < BLOCK_VALUES {
            values[index] = i8::try_from(index as i32 - 16 + i32::try_from(row).unwrap()).unwrap();
            index += 1;
        }
        values
    }

    #[test]
    fn packed_bl4_kernel_constructs_on_neon() {
        assert!(Q8_0PackedBl4Kernel::try_new().is_some());
    }

    #[test]
    fn packed_bl4_pin_constants_match_pinned_source() {
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
            "src/emel/kernel/aarch64/guards.hpp:455-466"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1282-1322"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6416-6466"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:387-388"
        );
        assert!(super::SCOPE_RESIDUAL.contains("q8_0 packed x4/bl4, x4/bl8, and bl8 matrix_x4"));
        assert_eq!(
            PINNED_BL8_MATRIX_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:35-72"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:444-452"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6584-6667"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:372-373"
        );
        assert_eq!(
            PINNED_BL8_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:420-428"
        );
        assert_eq!(
            PINNED_BL8_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1324-1348"
        );
        assert_eq!(
            PINNED_BL8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:6468-6522"
        );
        assert_eq!(
            PINNED_BL8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:382-383"
        );
        assert_eq!(MAX_Q8_0_BLOCKS, 1024);
        assert_eq!(Q8_0_X4_ROWS, 4);
        assert_eq!(Q8_0_X4_BLOCK_BYTES, 136);
        assert_eq!(INTERLEAVE_BLOCK_BYTES, 4);
        assert_eq!(INTERLEAVE_BL8_BYTES, 8);
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
    }

    #[test]
    fn neon_packed_bl4_matches_independent_q8_dots() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q8_0_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = pack_q8_row(0x3c00, ramp_i8(row));
            row += 1;
        }
        let lhs = pack_q8_0_x4_bl4(&rows, 5, 1);
        let rhs = pack_q8_row(0x3800, ramp_i8(0));
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl4::new(&lhs, &rhs, 5, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < 5 {
            let expected = dot_q8_0_q8_0(
                Q8_0Row::from_bytes(&rows[row]).unwrap(),
                Q8_0Row::from_bytes(&rhs).unwrap(),
            )
            .unwrap();
            assert_eq!(output[row].to_bits(), expected.to_bits());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl4_rejects_non_block_k() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl4::new(&[1, 2], &[3], 5, 31),
                &mut output
            ),
            Err(Q8_0PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn packed_bl4_rejects_unexpected_event() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ8_0PackedBl4, &mut []),
            Err(Q8_0PackedBl4Error::UnexpectedEvent)
        );
    }

    #[test]
    fn packed_bl4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [pack_q8_row(0x3c00, ramp_i8(0))];
        let lhs = pack_q8_0_x4_bl4(&rows, 1, 1);
        let rhs = pack_q8_row(0x3800, ramp_i8(0));
        let mut output = [f32::NAN; 1];
        assert_eq!(
            Q8_0PackedBl4Event::dispatch(
                OpMulMatQ8_0PackedBl4::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut kernel,
                &mut output,
            ),
            Ok(())
        );
        let expected = dot_q8_0_q8_0(
            Q8_0Row::from_bytes(&rows[0]).unwrap(),
            Q8_0Row::from_bytes(&rhs).unwrap(),
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn packed_bl4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [pack_q8_row(0x3c00, ramp_i8(0))];
        let lhs = pack_q8_0_x4_bl4(&rows, 1, 1);
        let rhs = pack_q8_row(0x3800, ramp_i8(0));
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl4::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut first
            ),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl4::new(&lhs, &rhs, 1, BLOCK_VALUES),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first, second);
    }
    #[test]
    fn neon_packed_bl8_matches_independent_q8_dots() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q8_0_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = pack_q8_row(0x3c00, ramp_i8(row));
            row += 1;
        }
        let lhs = pack_q8_0_x4_bl8(&rows, 5, 1);
        let rhs = pack_q8_row(0x3800, ramp_i8(0));
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl8::new(&lhs, &rhs, 5, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < 5 {
            let expected = dot_q8_0_q8_0(
                Q8_0Row::from_bytes(&rows[row]).unwrap(),
                Q8_0Row::from_bytes(&rhs).unwrap(),
            )
            .unwrap();
            assert_eq!(output[row].to_bits(), expected.to_bits());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl8_rejects_non_block_k() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl8::new(&[1, 2], &[3], 5, 31),
                &mut output
            ),
            Err(Q8_0PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }
    #[test]
    fn neon_packed_bl8_matrix_x4_matches_independent_q8_dots() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut lhs_rows = [[0_u8; Q8_0_BLOCK_BYTES]; 4];
        let mut rhs_rows = [[0_u8; Q8_0_BLOCK_BYTES]; 4];
        let mut row = 0;
        while row < 4 {
            lhs_rows[row] = pack_q8_row(0x3c00, ramp_i8(row));
            rhs_rows[row] = pack_q8_row(0x3800, ramp_i8(row + 1));
            row += 1;
        }
        let lhs = pack_q8_0_x4_bl8(&lhs_rows, 4, 1);
        let rhs = pack_q8_0_x4_bl8(&rhs_rows, 4, 1);
        let mut output = [f32::NAN; 16];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl8MatrixX4::new(&lhs, &rhs, 4, BLOCK_VALUES),
                &mut output
            ),
            Ok(())
        );
        let mut rhs_row = 0;
        while rhs_row < 4 {
            let mut lhs_col = 0;
            while lhs_col < 4 {
                let expected = dot_q8_0_q8_0(
                    Q8_0Row::from_bytes(&lhs_rows[lhs_col]).unwrap(),
                    Q8_0Row::from_bytes(&rhs_rows[rhs_row]).unwrap(),
                )
                .unwrap();
                assert_eq!(output[rhs_row * 4 + lhs_col].to_bits(), expected.to_bits());
                lhs_col += 1;
            }
            rhs_row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl8_matrix_x4_rejects_non_group_m() {
        let Some(mut kernel) = Q8_0PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 20];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ8_0PackedBl8MatrixX4::new(&[1, 2], &[3], 5, BLOCK_VALUES),
                &mut output
            ),
            Err(Q8_0PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 20]);
    }
}
