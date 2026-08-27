//! Safe `AArch64` NEON packed `q4_k_x8_bl4`/`bl8` by packed `q8_k` or dense F32.
//!
//! The source-backed boundary is the single-RHS route selected by
//! `can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4` or `_bl8`, the dense
//! F32-rhs bl4/bl8 routes, and
//! `can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4`,
//! `can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4` and
//! `can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8`: interleaved
//! `q4_k_x8_bl4`/`q4_k_x8_bl8` `src0` with packed groups of 8 rows, packed
//! `q8_k` or dense F32 `src1` with shape `[1, k]`, or four or eight packed
//! `q8_k` RHS rows with batch-major F32 `dst` of shape `[4, m]` or `[8, m]`.
//! The F32-rhs routes
//! quantize `src1` with the pinned `nearest_int` `q8_k` contract into
//! construction-time scratch, then reuse the matching bl4 or bl8 integer/float
//! kernel.  Capability detection happens during construction.  Dispatch never
//! dequantizes into an F32 GEMV, never uses `pulp::cast!` or `core::arch`, and
//! never falls back to the portable actor.
//!
//! BL4 integer products reconstruct the pinned `vdotq_laneq_s32` pairs as
//! widening i8 dots after 4-byte deinterleave.  Superblock scaling uses
//! `mul_add` for the C++ `vfmaq_f32` association and a later `mul_add` of the
//! negated min bias for `vmlsq_f32`.
//!
//! BL8 integer products reconstruct `vdotq_s32` as pulp widening i8 dots after
//! 8-byte deinterleave, then scale and accumulate low/high/bias in integer.
//! Float uses per-row `accumulate_q4_k_scaled_sums`.
//!
//! BL4 `matrix_x4` has no dedicated x4 neon kernel in C++.  It loops four
//! packed `q8_k` RHS rows through `dot_q4_k_x8_q8_k_group_bl4_neon` and
//! stores batch-major F32.  The Rust path reuses the live bl4 group kernel
//! with that same per-RHS association.
//!
//! BL8 `matrix_x4` and `matrix_x8` reconstruct the dedicated multi-RHS neon
//! tiles, then `d.mul_add` of the scaled lo sum and the scaled hi sum per
//! superblock, with an unfused min subtract after the block's superblocks.
//! Eight independent bl8 dots would not preserve that vfma association.

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

/// Errors returned by the `AArch64` packed `q4_k_x8_bl4` actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4PackedBl4Error {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Q4PackedBl4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => {
                formatter.write_str("q4 packed bl4 NEON backend unavailable")
            }
            Self::InvalidShape => formatter.write_str("invalid q4 packed bl4 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected q4 packed bl4 event"),
            Self::Internal => formatter.write_str("internal q4 packed bl4 dispatch error"),
        }
    }
}

impl std::error::Error for Q4PackedBl4Error {}

/// A pinned `AArch64` packed `q4_k_x8_bl4` by packed `q8_k` vector request.
///
/// `lhs` is group-major `block_q4_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is one packed `q8_k` row of `k / 256`
/// blocks.
#[derive(Debug)]
pub struct OpMulMatQ4PackedBl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedBl4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl4` by 4 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q4_kx8` storage with 4-byte qs interleave.
/// `rhs` is four packed `q8_k` rows of `k / 256` blocks each.  Destination
/// storage is batch-major F32 with shape `[4, m]`.
#[derive(Debug)]
pub struct OpMulMatQ4PackedBl4MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedBl4MatrixX4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl8` by packed `q8_k` vector request.
///
/// Layout matches `block_q4_kx8` with 8-byte qs interleave.  `rhs` is one
/// packed `q8_k` row of `k / 256` blocks.
#[derive(Debug)]
pub struct OpMulMatQ4PackedBl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedBl8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl4` by dense F32 vector request.
///
/// `lhs` is group-major `block_q4_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is one dense F32 row of length `k`.
#[derive(Debug)]
pub struct OpMulMatQ4PackedF32Bl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedF32Bl4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl4` by dense F32 vector argmax request.
///
/// `lhs` is group-major `block_q4_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is one dense F32 row of length `k`.
/// Destination is dense F32 with shape `[1, 1]`; the returned index is the
/// first row attaining the maximum.
#[derive(Debug)]
pub struct OpMulMatArgmaxQ4PackedF32Bl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatArgmaxQ4PackedF32Bl4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl8` by dense F32 vector argmax request.
///
/// `lhs` is group-major `block_q4_kx8` storage with 8-byte qs interleave.
/// `rhs` is one dense F32 row of length `k`.  Destination is dense F32 with
/// shape `[1, 1]`; the returned index is the first row attaining the maximum.
#[derive(Debug)]
pub struct OpMulMatArgmaxQ4PackedF32Bl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatArgmaxQ4PackedF32Bl8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl8` by dense F32 vector request.
///
/// `lhs` is group-major `block_q4_kx8` storage with 8-byte qs interleave.
/// `rhs` is one dense F32 row of length `k`.
#[derive(Debug)]
pub struct OpMulMatQ4PackedF32Bl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedF32Bl8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl8` by 4 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q4_kx8` storage with 8-byte qs interleave.
/// `rhs` is four packed `q8_k` rows of `k / 256` blocks each.  Destination
/// storage is batch-major F32 with shape `[4, m]`.
#[derive(Debug)]
pub struct OpMulMatQ4PackedBl8MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedBl8MatrixX4<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A pinned `AArch64` packed `q4_k_x8_bl8` by 8 packed `q8_k` rows.
///
/// `lhs` is group-major `block_q4_kx8` storage with `ceil(m / 8)` groups of
/// `k / 256` interleaved blocks.  `rhs` is eight packed `q8_k` rows.  `dst`
/// is batch-major F32 with shape `[8, m]`.
#[derive(Debug)]
pub struct OpMulMatQ4PackedBl8MatrixX8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> OpMulMatQ4PackedBl8MatrixX8<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the packed `q4_k_x8_bl4` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedQ4PackedBl4;

/// Result returned by target packed `q4_k_x8_bl4` events.
pub type Q4PackedBl4Result = Result<(), Q4PackedBl4Error>;

/// Result returned by the packed `q4_k_x8_bl4` F32-rhs argmax event.
pub type Q4PackedF32Bl4ArgmaxResult = Result<i32, Q4PackedBl4Error>;

/// Result returned by the packed `q4_k_x8_bl8` F32-rhs argmax event.
pub type Q4PackedF32Bl8ArgmaxResult = Result<i32, Q4PackedBl4Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:477-487";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1409-1441";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5695-5734";
const PINNED_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:2937-3100";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:427-428";
const PINNED_BL8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:512-522";
const PINNED_BL8_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1611-1616";
const PINNED_BL8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5827-5862";
const PINNED_BL8_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:3102-3249";
const PINNED_BL8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:412-413";
const PINNED_F32_BL4_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:500-511";
const PINNED_F32_BL4_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1442-1469";
const PINNED_F32_BL4_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5732-5765";
const PINNED_F32_BL4_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:430-433";
const PINNED_ARGMAX_F32_BL4_GUARD_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1769-1788";
const PINNED_ARGMAX_F32_BL4_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5593-5640";
const PINNED_ARGMAX_F32_BL4_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:471-474";
const PINNED_ARGMAX_F32_BL8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1769-1798";
const PINNED_ARGMAX_F32_BL8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5644-5693";
const PINNED_ARGMAX_F32_BL8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:467";
const PINNED_F32_BL8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:524-534";
const PINNED_F32_BL8_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1625-1630";
const PINNED_F32_BL8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5864-5903";
const PINNED_F32_BL8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:415-418";
const PINNED_BL8_MATRIX_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:73-108";
const PINNED_BL8_MATRIX_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:536-545";
const PINNED_BL8_MATRIX_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5906-5966";
const PINNED_BL8_MATRIX_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:3251-3435";
const PINNED_BL8_MATRIX_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:406-408";
const PINNED_BL8_MATRIX_X8_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:184-218";
const PINNED_BL8_MATRIX_X8_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:547-556";
const PINNED_BL8_MATRIX_X8_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5969-6030";
const PINNED_BL8_MATRIX_X8_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:3436-3622";
const PINNED_BL8_MATRIX_X8_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:400-403";
const PINNED_BL4_MATRIX_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:143-151";
const PINNED_BL4_MATRIX_ACTION_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:489-497";
const PINNED_BL4_MATRIX_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:5784-5825";
const PINNED_BL4_MATRIX_KERNEL_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:2937-3100";
const PINNED_BL4_MATRIX_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:420-423";
const SCOPE_RESIDUAL: &str = "q4_k_x8_bl4/bl8 lhs packed q8_k rhs n=1, dense F32 rhs n=1, bl4 packed q8_k matrix_x4, and bl8 packed q8_k matrix_x4/matrix_x8; F16 neon and x86 remain residuals";

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_X8_ROWS: usize = 8;
const MATRIX_X4_ROWS: usize = 4;
const MATRIX_X8_ROWS: usize = 8;
const Q4_K_X8_BLOCK_BYTES: usize = 1152;
const Q8_K_BLOCK_BYTES: usize = 292;
const K_SCALE_SIZE: usize = 12;
const INTERLEAVE_BLOCK_BYTES: usize = 4;
const INTERLEAVE_BL8_BYTES: usize = 8;
const MAX_Q8_K_BLOCKS: usize = 128;
const KMASK1: u32 = 0x3f3f_3f3f;
const KMASK2: u32 = 0x0f0f_0f0f;
const KMASK3: u32 = 0x0303_0303;

/// Event implemented by the target packed `q4_k_x8_bl4` actor.
pub trait Q4PackedBl4Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output;
}

struct PackedBl4Runtime<'a> {
    event: OpMulMatQ4PackedBl4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedBl4MatrixX4Runtime<'a> {
    event: OpMulMatQ4PackedBl4MatrixX4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedBl8Runtime<'a> {
    event: OpMulMatQ4PackedBl8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedF32Bl4Runtime<'a> {
    event: OpMulMatQ4PackedF32Bl4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedF32Bl4ArgmaxRuntime<'a> {
    event: OpMulMatArgmaxQ4PackedF32Bl4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedF32Bl4ArgmaxResult>,
}

struct PackedF32Bl8ArgmaxRuntime<'a> {
    event: OpMulMatArgmaxQ4PackedF32Bl8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedF32Bl8ArgmaxResult>,
}

struct PackedF32Bl8Runtime<'a> {
    event: OpMulMatQ4PackedF32Bl8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedBl8MatrixX4Runtime<'a> {
    event: OpMulMatQ4PackedBl8MatrixX4<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct PackedBl8MatrixX8Runtime<'a> {
    event: OpMulMatQ4PackedBl8MatrixX8<'a>,
    output: &'a mut [f32],
    result: &'a Cell<Q4PackedBl4Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<Q4PackedBl4Result>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
    q8_scratch: Box<[u8]>,
}

sml! {
    Q4PackedBl4Machine<'dispatch> {
        "ready"_s <= *"ready"_s + PackedBl4(PackedBl4Runtime<'dispatch>) [guard_ready] / effect_execute,
        "ready"_s <= "ready"_s + PackedBl4(PackedBl4Runtime<'dispatch>) [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + PackedBl4MatrixX4(PackedBl4MatrixX4Runtime<'dispatch>) [guard_bl4_matrix_ready] / effect_bl4_matrix_execute,
        "ready"_s <= "ready"_s + PackedBl4MatrixX4(PackedBl4MatrixX4Runtime<'dispatch>) [guard_bl4_matrix_invalid] / effect_bl4_matrix_invalid,
        "ready"_s <= "ready"_s + PackedBl8(PackedBl8Runtime<'dispatch>) [guard_bl8_ready] / effect_bl8_execute,
        "ready"_s <= "ready"_s + PackedBl8(PackedBl8Runtime<'dispatch>) [guard_bl8_invalid] / effect_bl8_invalid,
        "ready"_s <= "ready"_s + PackedF32Bl4(PackedF32Bl4Runtime<'dispatch>) [guard_f32_bl4_ready] / effect_f32_bl4_execute,
        "ready"_s <= "ready"_s + PackedF32Bl4(PackedF32Bl4Runtime<'dispatch>) [guard_f32_bl4_invalid] / effect_f32_bl4_invalid,
        "ready"_s <= "ready"_s + PackedF32Bl4Argmax(PackedF32Bl4ArgmaxRuntime<'dispatch>) [guard_f32_bl4_argmax_ready] / effect_f32_bl4_argmax_execute,
        "ready"_s <= "ready"_s + PackedF32Bl4Argmax(PackedF32Bl4ArgmaxRuntime<'dispatch>) [guard_f32_bl4_argmax_invalid] / effect_f32_bl4_argmax_invalid,
        "ready"_s <= "ready"_s + PackedF32Bl8Argmax(PackedF32Bl8ArgmaxRuntime<'dispatch>) [guard_f32_bl8_argmax_ready] / effect_f32_bl8_argmax_execute,
        "ready"_s <= "ready"_s + PackedF32Bl8Argmax(PackedF32Bl8ArgmaxRuntime<'dispatch>) [guard_f32_bl8_argmax_invalid] / effect_f32_bl8_argmax_invalid,
        "ready"_s <= "ready"_s + PackedF32Bl8(PackedF32Bl8Runtime<'dispatch>) [guard_f32_bl8_ready] / effect_f32_bl8_execute,
        "ready"_s <= "ready"_s + PackedF32Bl8(PackedF32Bl8Runtime<'dispatch>) [guard_f32_bl8_invalid] / effect_f32_bl8_invalid,
        "ready"_s <= "ready"_s + PackedBl8MatrixX4(PackedBl8MatrixX4Runtime<'dispatch>) [guard_bl8_matrix_ready] / effect_bl8_matrix_execute,
        "ready"_s <= "ready"_s + PackedBl8MatrixX4(PackedBl8MatrixX4Runtime<'dispatch>) [guard_bl8_matrix_invalid] / effect_bl8_matrix_invalid,
        "ready"_s <= "ready"_s + PackedBl8MatrixX8(PackedBl8MatrixX8Runtime<'dispatch>) [guard_bl8_matrix_x8_ready] / effect_bl8_matrix_x8_execute,
        "ready"_s <= "ready"_s + PackedBl8MatrixX8(PackedBl8MatrixX8Runtime<'dispatch>) [guard_bl8_matrix_x8_invalid] / effect_bl8_matrix_x8_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` packed `q4_k_x8_bl4` actor.
pub struct Q4PackedBl4Kernel {
    machine: Q4PackedBl4MachineStateMachine<Context>,
}

impl fmt::Debug for Q4PackedBl4Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Q4PackedBl4Kernel")
            .finish_non_exhaustive()
    }
}

impl Q4PackedBl4Kernel {
    /// Resolves NEON before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: Q4PackedBl4MachineStateMachine::new(Context {
                backend_available: true,
                backend,
                q8_scratch: vec![0_u8; MAX_Q8_K_BLOCKS * Q8_K_BLOCK_BYTES].into_boxed_slice(),
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: Q4PackedBl4Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Q4PackedBl4MachineStates::Ready)
    }

    fn packed_bl4(
        &mut self,
        event: OpMulMatQ4PackedBl4<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedBl4(PackedBl4Runtime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl4_matrix_x4(
        &mut self,
        event: OpMulMatQ4PackedBl4MatrixX4<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedBl4MatrixX4(
                PackedBl4MatrixX4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl8(
        &mut self,
        event: OpMulMatQ4PackedBl8<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedBl8(PackedBl8Runtime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_f32_bl4(
        &mut self,
        event: OpMulMatQ4PackedF32Bl4<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedF32Bl4(
                PackedF32Bl4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_f32_bl4_argmax(
        &mut self,
        event: OpMulMatArgmaxQ4PackedF32Bl4<'_>,
        output: &mut [f32],
    ) -> Q4PackedF32Bl4ArgmaxResult {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedF32Bl4Argmax(
                PackedF32Bl4ArgmaxRuntime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_f32_bl8(
        &mut self,
        event: OpMulMatQ4PackedF32Bl8<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedF32Bl8(
                PackedF32Bl8Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl8_matrix_x4(
        &mut self,
        event: OpMulMatQ4PackedBl8MatrixX4<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedBl8MatrixX4(
                PackedBl8MatrixX4Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_bl8_matrix_x8(
        &mut self,
        event: OpMulMatQ4PackedBl8MatrixX8<'_>,
        output: &mut [f32],
    ) -> Q4PackedBl4Result {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedBl8MatrixX8(
                PackedBl8MatrixX8Runtime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }

    fn packed_f32_bl8_argmax(
        &mut self,
        event: OpMulMatArgmaxQ4PackedF32Bl8<'_>,
        output: &mut [f32],
    ) -> Q4PackedF32Bl8ArgmaxResult {
        let result = Cell::new(Err(Q4PackedBl4Error::UnexpectedEvent));
        self.machine
            .process_event(Q4PackedBl4MachineEvents::PackedF32Bl8Argmax(
                PackedF32Bl8ArgmaxRuntime {
                    event,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedBl4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl4(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedBl4MatrixX4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl4_matrix_x4(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedBl8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl8(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedF32Bl4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_f32_bl4(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatArgmaxQ4PackedF32Bl4<'_> {
    type Output = Q4PackedF32Bl4ArgmaxResult;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_f32_bl4_argmax(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedF32Bl8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_f32_bl8(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatArgmaxQ4PackedF32Bl8<'_> {
    type Output = Q4PackedF32Bl8ArgmaxResult;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_f32_bl8_argmax(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedBl8MatrixX4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl8_matrix_x4(self, output)
    }
}

impl Q4PackedBl4Event for OpMulMatQ4PackedBl8MatrixX8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, output: &mut [f32]) -> Self::Output {
        actor.packed_bl8_matrix_x8(self, output)
    }
}

impl Q4PackedBl4Event for UnexpectedQ4PackedBl4 {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Q4PackedBl4Kernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(Q4PackedBl4MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl Q4PackedBl4MachineStateMachineContext for Context {
    fn guard_ready(&self, event: &PackedBl4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl4_request_valid(event))
    }

    fn guard_invalid(&self, event: &PackedBl4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl4_request_valid(event))
    }

    fn effect_execute(&mut self, event: PackedBl4Runtime<'_>) -> Result<(), ()> {
        execute_packed(
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: PackedBl4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl4_matrix_ready(&self, event: &PackedBl4MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl4_matrix_x4_request_valid(event))
    }

    fn guard_bl4_matrix_invalid(&self, event: &PackedBl4MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl4_matrix_x4_request_valid(event))
    }

    fn effect_bl4_matrix_execute(&mut self, event: PackedBl4MatrixX4Runtime<'_>) -> Result<(), ()> {
        execute_packed_bl4_matrix_x4(
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_bl4_matrix_invalid(&mut self, event: PackedBl4MatrixX4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl8_ready(&self, event: &PackedBl8Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl8_request_valid(event))
    }

    fn guard_bl8_invalid(&self, event: &PackedBl8Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl8_request_valid(event))
    }

    fn effect_bl8_execute(&mut self, event: PackedBl8Runtime<'_>) -> Result<(), ()> {
        execute_packed_bl8(
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

    fn effect_bl8_invalid(&mut self, event: PackedBl8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_f32_bl4_ready(&self, event: &PackedF32Bl4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_f32_bl4_request_valid(event))
    }

    fn guard_f32_bl4_invalid(&self, event: &PackedF32Bl4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_f32_bl4_request_valid(event))
    }

    fn effect_f32_bl4_execute(&mut self, event: PackedF32Bl4Runtime<'_>) -> Result<(), ()> {
        execute_packed_f32(
            event.event.lhs,
            event.event.rhs,
            &mut self.q8_scratch,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_f32_bl4_invalid(&mut self, event: PackedF32Bl4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_f32_bl4_argmax_ready(
        &self,
        event: &PackedF32Bl4ArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(self.backend_available && packed_f32_bl4_argmax_request_valid(event))
    }

    fn guard_f32_bl4_argmax_invalid(
        &self,
        event: &PackedF32Bl4ArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!packed_f32_bl4_argmax_request_valid(event))
    }

    fn effect_f32_bl4_argmax_execute(
        &mut self,
        event: PackedF32Bl4ArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        let index = execute_packed_f32_argmax(
            event.event.lhs,
            event.event.rhs,
            &mut self.q8_scratch,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(index));
        Ok(())
    }

    fn effect_f32_bl4_argmax_invalid(
        &mut self,
        event: PackedF32Bl4ArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_f32_bl8_ready(&self, event: &PackedF32Bl8Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_f32_bl8_request_valid(event))
    }

    fn guard_f32_bl8_invalid(&self, event: &PackedF32Bl8Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_f32_bl8_request_valid(event))
    }

    fn effect_f32_bl8_execute(&mut self, event: PackedF32Bl8Runtime<'_>) -> Result<(), ()> {
        execute_packed_f32_bl8(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            &mut self.q8_scratch,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_f32_bl8_invalid(&mut self, event: PackedF32Bl8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_f32_bl8_argmax_ready(
        &self,
        event: &PackedF32Bl8ArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(self.backend_available && packed_f32_bl8_argmax_request_valid(event))
    }

    fn guard_f32_bl8_argmax_invalid(
        &self,
        event: &PackedF32Bl8ArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!packed_f32_bl8_argmax_request_valid(event))
    }

    fn effect_f32_bl8_argmax_execute(
        &mut self,
        event: PackedF32Bl8ArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        let index = execute_packed_f32_bl8_argmax(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            &mut self.q8_scratch,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(index));
        Ok(())
    }

    fn effect_f32_bl8_argmax_invalid(
        &mut self,
        event: PackedF32Bl8ArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl8_matrix_ready(&self, event: &PackedBl8MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl8_matrix_x4_request_valid(event))
    }

    fn guard_bl8_matrix_invalid(&self, event: &PackedBl8MatrixX4Runtime<'_>) -> Result<bool, ()> {
        Ok(!packed_bl8_matrix_x4_request_valid(event))
    }

    fn effect_bl8_matrix_execute(&mut self, event: PackedBl8MatrixX4Runtime<'_>) -> Result<(), ()> {
        execute_packed_bl8_matrix::<MATRIX_X4_ROWS>(
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
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn guard_bl8_matrix_x8_ready(&self, event: &PackedBl8MatrixX8Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && packed_bl8_matrix_x8_request_valid(event))
    }

    fn guard_bl8_matrix_x8_invalid(
        &self,
        event: &PackedBl8MatrixX8Runtime<'_>,
    ) -> Result<bool, ()> {
        Ok(!packed_bl8_matrix_x8_request_valid(event))
    }

    fn effect_bl8_matrix_x8_execute(
        &mut self,
        event: PackedBl8MatrixX8Runtime<'_>,
    ) -> Result<(), ()> {
        execute_packed_bl8_matrix::<MATRIX_X8_ROWS>(
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

    fn effect_bl8_matrix_x8_invalid(
        &mut self,
        event: PackedBl8MatrixX8Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Q4PackedBl4Error::UnexpectedEvent));
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

const fn packed_bl4_matrix_x4_request_valid(event: &PackedBl4MatrixX4Runtime<'_>) -> bool {
    packed_matrix_request_valid::<MATRIX_X4_ROWS>(
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

const fn packed_f32_bl4_request_valid(event: &PackedF32Bl4Runtime<'_>) -> bool {
    packed_f32_request_valid(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_f32_bl4_argmax_request_valid(event: &PackedF32Bl4ArgmaxRuntime<'_>) -> bool {
    if event.event.m == 0
        || event.event.m > i32::MAX as usize
        || event.event.k == 0
        || !event.event.k.is_multiple_of(QK_K)
    {
        return false;
    }
    let blocks = event.event.k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = event.event.m.checked_add(Q4_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q4_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q4_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    event.event.lhs.len() == lhs_bytes
        && event.event.rhs.len() == event.event.k
        && event.output.len() == 1
}

const fn packed_f32_bl8_request_valid(event: &PackedF32Bl8Runtime<'_>) -> bool {
    packed_f32_request_valid(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_f32_bl8_argmax_request_valid(event: &PackedF32Bl8ArgmaxRuntime<'_>) -> bool {
    if event.event.m == 0
        || event.event.m > i32::MAX as usize
        || event.event.k == 0
        || !event.event.k.is_multiple_of(QK_K)
    {
        return false;
    }
    let blocks = event.event.k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = event.event.m.checked_add(Q4_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q4_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q4_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    event.event.lhs.len() == lhs_bytes
        && event.event.rhs.len() == event.event.k
        && event.output.len() == 1
}

const fn packed_bl8_matrix_x4_request_valid(event: &PackedBl8MatrixX4Runtime<'_>) -> bool {
    packed_matrix_request_valid::<MATRIX_X4_ROWS>(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_bl8_matrix_x8_request_valid(event: &PackedBl8MatrixX8Runtime<'_>) -> bool {
    packed_matrix_request_valid::<MATRIX_X8_ROWS>(
        event.event.lhs,
        event.event.rhs,
        event.output,
        event.event.m,
        event.event.k,
    )
}

const fn packed_request_valid(lhs: &[u8], rhs: &[u8], output: &[f32], m: usize, k: usize) -> bool {
    if m == 0 || k == 0 || !k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = m.checked_add(Q4_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q4_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q4_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    lhs.len() == lhs_bytes && rhs.len() == rhs_bytes && output.len() == m
}

const fn packed_f32_request_valid(
    lhs: &[u8],
    rhs: &[f32],
    output: &[f32],
    m: usize,
    k: usize,
) -> bool {
    if m == 0 || k == 0 || !k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = m.checked_add(Q4_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q4_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q4_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    lhs.len() == lhs_bytes && rhs.len() == k && output.len() == m
}

const fn packed_matrix_request_valid<const RHS_ROWS: usize>(
    lhs: &[u8],
    rhs: &[u8],
    output: &[f32],
    m: usize,
    k: usize,
) -> bool {
    if m == 0 || k == 0 || !k.is_multiple_of(QK_K) {
        return false;
    }
    let blocks = k / QK_K;
    if blocks == 0 || blocks > MAX_Q8_K_BLOCKS {
        return false;
    }
    let Some(group_count) = m.checked_add(Q4_K_X8_ROWS - 1) else {
        return false;
    };
    let group_count = group_count / Q4_K_X8_ROWS;
    let Some(group_bytes) = blocks.checked_mul(Q4_K_X8_BLOCK_BYTES) else {
        return false;
    };
    let Some(lhs_bytes) = group_count.checked_mul(group_bytes) else {
        return false;
    };
    let Some(rhs_row_bytes) = blocks.checked_mul(Q8_K_BLOCK_BYTES) else {
        return false;
    };
    let Some(rhs_bytes) = rhs_row_bytes.checked_mul(RHS_ROWS) else {
        return false;
    };
    let Some(output_len) = m.checked_mul(RHS_ROWS) else {
        return false;
    };
    lhs.len() == lhs_bytes && rhs.len() == rhs_bytes && output.len() == output_len
}

fn execute_packed_f32(
    lhs: &[u8],
    rhs: &[f32],
    scratch: &mut [u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    quantize_row_q8_k_strided(rhs, scratch, blocks);
    execute_packed(lhs, &scratch[..blocks * Q8_K_BLOCK_BYTES], output, m, k);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn execute_packed_f32_argmax(
    lhs: &[u8],
    rhs: &[f32],
    scratch: &mut [u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) -> i32 {
    // Guard `packed_f32_bl4_argmax_request_valid` already proved `m <= i32::MAX`.
    let blocks = k / QK_K;
    quantize_row_q8_k_strided(rhs, scratch, blocks);
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut have_best = false;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q4_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block(
                &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                &scratch[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        let row_base = group * Q4_K_X8_ROWS;
        let store_count = (m - row_base).min(Q4_K_X8_ROWS);
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

fn execute_packed_f32_bl8(
    backend: Neon,
    lhs: &[u8],
    rhs: &[f32],
    scratch: &mut [u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    quantize_row_q8_k_strided(rhs, scratch, blocks);
    execute_packed_bl8(
        backend,
        lhs,
        &scratch[..blocks * Q8_K_BLOCK_BYTES],
        output,
        m,
        k,
    );
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn execute_packed_f32_bl8_argmax(
    backend: Neon,
    lhs: &[u8],
    rhs: &[f32],
    scratch: &mut [u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) -> i32 {
    // Guard `packed_f32_bl8_argmax_request_valid` already proved `m <= i32::MAX`.
    let blocks = k / QK_K;
    quantize_row_q8_k_strided(rhs, scratch, blocks);
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut have_best = false;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q4_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block_bl8(
                backend,
                &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                &scratch[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        let row_base = group * Q4_K_X8_ROWS;
        let store_count = (m - row_base).min(Q4_K_X8_ROWS);
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

fn nearest_int(value: f32) -> i32 {
    let biased = value + 12_582_912.0_f32;
    let bits = biased.to_bits() & 0x007f_ffff;
    i32::try_from(bits).expect("nearest_int mantissa fits i32") - 0x0040_0000
}

#[allow(clippy::cast_possible_truncation)]
fn quantize_row_q8_k_strided(rhs: &[f32], scratch: &mut [u8], blocks: usize) {
    let mut block = 0;
    while block < blocks {
        let block_offset = block * Q8_K_BLOCK_BYTES;
        let mut maximum = 0.0_f32;
        let mut absolute = 0.0_f32;
        let mut index = 0;
        while index < QK_K {
            let value = rhs[block * QK_K + index];
            if value.abs() > absolute {
                absolute = value.abs();
                maximum = value;
            }
            index += 1;
        }
        if absolute == 0.0 {
            scratch[block_offset..block_offset + Q8_K_BLOCK_BYTES].fill(0);
            block += 1;
            continue;
        }
        let inverse = -127.0 / maximum;
        index = 0;
        while index < QK_K {
            let quantized = nearest_int(inverse * rhs[block * QK_K + index]).min(127);
            scratch[block_offset + 4 + index] = (quantized as i8).to_ne_bytes()[0];
            index += 1;
        }
        let mut group = 0;
        while group < QK_K / 16 {
            let mut sum = 0_i32;
            index = 0;
            while index < 16 {
                sum += i32::from(i8::from_ne_bytes([
                    scratch[block_offset + 4 + group * 16 + index]
                ]));
                index += 1;
            }
            let offset = block_offset + 260 + group * 2;
            scratch[offset..offset + 2].copy_from_slice(&(sum as i16).to_le_bytes());
            group += 1;
        }
        scratch[block_offset..block_offset + 4].copy_from_slice(&(1.0 / inverse).to_le_bytes());
        block += 1;
    }
}

fn execute_packed(lhs: &[u8], rhs: &[u8], output: &mut [f32], m: usize, k: usize) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q4_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block(
                &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        store_q4_k_x8_results(output, group * Q4_K_X8_ROWS, m, acc);
        group += 1;
    }
}

fn execute_packed_bl4_matrix_x4(lhs: &[u8], rhs: &[u8], output: &mut [f32], m: usize, k: usize) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let rhs_row_bytes = blocks * Q8_K_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [[0.0_f32; Q4_K_X8_ROWS]; MATRIX_X4_ROWS];
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X4_ROWS {
            let mut block = 0;
            while block < blocks {
                let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
                let rhs_offset = rhs_row * rhs_row_bytes + block * Q8_K_BLOCK_BYTES;
                accumulate_group_block(
                    &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                    &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                    &mut acc[rhs_row],
                );
                block += 1;
            }
            rhs_row += 1;
        }
        store_batch_major_group_results(output, group * Q4_K_X8_ROWS, m, acc);
        group += 1;
    }
}

#[allow(clippy::cast_precision_loss)]
fn accumulate_group_block(lhs: &[u8], rhs: &[u8], acc: &mut [f32; Q4_K_X8_ROWS]) {
    let mut sb_scale = [0.0_f32; Q4_K_X8_ROWS];
    let mut sb_min = [0.0_f32; Q4_K_X8_ROWS];
    let mut q8_d_bytes = [0_u8; 4];
    q8_d_bytes.copy_from_slice(&rhs[..4]);
    let q8_d = f32::from_le_bytes(q8_d_bytes);
    let mut row = 0;
    while row < Q4_K_X8_ROWS {
        let d = fp16_to_f32(u16::from_le_bytes([lhs[row * 2], lhs[row * 2 + 1]]));
        let dmin = fp16_to_f32(u16::from_le_bytes([
            lhs[16 + row * 2],
            lhs[16 + row * 2 + 1],
        ]));
        sb_scale[row] = d * q8_d;
        sb_min[row] = dmin * q8_d;
        row += 1;
    }

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
    let mut bsums_pair = [0_i16; Q4_K_X8_ROWS];
    index = 0;
    while index < Q4_K_X8_ROWS {
        bsums_pair[index] = bsums[2 * index].wrapping_add(bsums[2 * index + 1]);
        index += 1;
    }

    let mut bias = [0_i32; Q4_K_X8_ROWS];
    let mut sb = 0;
    while sb < QK_K / 64 {
        let scale_base = 32 + sb * 24;
        let (mins_lo, scales_lo) = decode_q4_k_x8_6bit_scales(&lhs[scale_base..scale_base + 12]);
        let (mins_hi, scales_hi) =
            decode_q4_k_x8_6bit_scales(&lhs[scale_base + 12..scale_base + 24]);
        let rows = deinterleave_superblock_bl4(&lhs[128..], sb);
        let q8_base = sb * 64;
        row = 0;
        while row < Q4_K_X8_ROWS {
            let (acc_lo, acc_hi) = vdot_lane_superblock_dots(rows[row], &q8_qs, q8_base);
            let scaled = i32::from(scales_lo[row]) * acc_lo + i32::from(scales_hi[row]) * acc_hi;
            acc[row] = sb_scale[row].mul_add(scaled as f32, acc[row]);
            bias[row] += i32::from(bsums_pair[2 * sb]) * i32::from(mins_lo[row])
                + i32::from(bsums_pair[2 * sb + 1]) * i32::from(mins_hi[row]);
            row += 1;
        }
        sb += 1;
    }
    row = 0;
    while row < Q4_K_X8_ROWS {
        acc[row] = sb_min[row].mul_add(-(bias[row] as f32), acc[row]);
        row += 1;
    }
}

fn vdot_lane_superblock_dots(row_qs: [u8; 32], q8_qs: &[i8; QK_K], q8_base: usize) -> (i32, i32) {
    let mut acc_lo = 0_i32;
    let mut acc_hi = 0_i32;
    let mut chunk = 0;
    while chunk < 8 {
        let q4_offset = chunk * 4;
        let q8_offset = q8_base + chunk * 4;
        let mut lane = 0;
        while lane < 4 {
            let q4 = row_qs[q4_offset + lane];
            acc_lo += i32::from(q4 & 0x0f) * i32::from(q8_qs[q8_offset + lane]);
            acc_hi += i32::from(q4 >> 4) * i32::from(q8_qs[q8_offset + 32 + lane]);
            lane += 1;
        }
        chunk += 1;
    }
    (acc_lo, acc_hi)
}

fn decode_q4_k_x8_6bit_scales(scales_in: &[u8]) -> ([i16; Q4_K_X8_ROWS], [i16; Q4_K_X8_ROWS]) {
    let sm0 = u32::from_le_bytes([scales_in[0], scales_in[1], scales_in[2], scales_in[3]]);
    let sm1 = u32::from_le_bytes([scales_in[4], scales_in[5], scales_in[6], scales_in[7]]);
    let sm2 = u32::from_le_bytes([scales_in[8], scales_in[9], scales_in[10], scales_in[11]]);
    let mins_0_3 = (sm1 & KMASK1).to_le_bytes();
    let mins_4_7 = (((sm2 >> 4) & KMASK2) | (((sm1 >> 6) & KMASK3) << 4)).to_le_bytes();
    let scales_0_3 = (sm0 & KMASK1).to_le_bytes();
    let scales_4_7 = ((sm2 & KMASK2) | (((sm0 >> 6) & KMASK3) << 4)).to_le_bytes();
    let mins = [
        i16::from(mins_0_3[0]),
        i16::from(mins_0_3[1]),
        i16::from(mins_0_3[2]),
        i16::from(mins_0_3[3]),
        i16::from(mins_4_7[0]),
        i16::from(mins_4_7[1]),
        i16::from(mins_4_7[2]),
        i16::from(mins_4_7[3]),
    ];
    let scales = [
        i16::from(i8::from_ne_bytes([scales_0_3[0]])),
        i16::from(i8::from_ne_bytes([scales_0_3[1]])),
        i16::from(i8::from_ne_bytes([scales_0_3[2]])),
        i16::from(i8::from_ne_bytes([scales_0_3[3]])),
        i16::from(i8::from_ne_bytes([scales_4_7[0]])),
        i16::from(i8::from_ne_bytes([scales_4_7[1]])),
        i16::from(i8::from_ne_bytes([scales_4_7[2]])),
        i16::from(i8::from_ne_bytes([scales_4_7[3]])),
    ];
    (mins, scales)
}

const fn deinterleave_superblock_bl4(qs: &[u8], sb: usize) -> [[u8; 32]; Q4_K_X8_ROWS] {
    let mut rows = [[0_u8; 32]; Q4_K_X8_ROWS];
    let mut local = 0;
    while local < QK_K / INTERLEAVE_BLOCK_BYTES {
        let src_row = local % Q4_K_X8_ROWS;
        let src_offset = (local / Q4_K_X8_ROWS) * INTERLEAVE_BLOCK_BYTES;
        let dst_offset = sb * QK_K + local * INTERLEAVE_BLOCK_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BLOCK_BYTES {
            rows[src_row][src_offset + lane] = qs[dst_offset + lane];
            lane += 1;
        }
        local += 1;
    }
    rows
}

fn store_q4_k_x8_results(
    output: &mut [f32],
    row_base: usize,
    total_rows: usize,
    values: [f32; Q4_K_X8_ROWS],
) {
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q4_K_X8_ROWS);
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
    values: [[f32; Q4_K_X8_ROWS]; RHS_ROWS],
) {
    let remaining = total_rows - row_base;
    let store_count = remaining.min(Q4_K_X8_ROWS);
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

fn execute_packed_bl8(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [0.0_f32; Q4_K_X8_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
            let rhs_offset = block * Q8_K_BLOCK_BYTES;
            accumulate_group_block_bl8(
                backend,
                &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES],
                &mut acc,
            );
            block += 1;
        }
        store_q4_k_x8_results(output, group * Q4_K_X8_ROWS, m, acc);
        group += 1;
    }
}

fn execute_packed_bl8_matrix<const RHS_ROWS: usize>(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    let blocks = k / QK_K;
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let group_bytes = blocks * Q4_K_X8_BLOCK_BYTES;
    let rhs_row_bytes = blocks * Q8_K_BLOCK_BYTES;
    let mut group = 0;
    while group < group_count {
        let mut acc = [[0.0_f32; Q4_K_X8_ROWS]; RHS_ROWS];
        let mut block = 0;
        while block < blocks {
            let lhs_offset = group * group_bytes + block * Q4_K_X8_BLOCK_BYTES;
            accumulate_group_block_bl8_matrix(
                backend,
                &lhs[lhs_offset..lhs_offset + Q4_K_X8_BLOCK_BYTES],
                rhs,
                rhs_row_bytes,
                block,
                &mut acc,
            );
            block += 1;
        }
        store_batch_major_group_results(output, group * Q4_K_X8_ROWS, m, acc);
        group += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_group_block_bl8(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    acc: &mut [f32; Q4_K_X8_ROWS],
) {
    let mut q8_d_bytes = [0_u8; 4];
    q8_d_bytes.copy_from_slice(&rhs[..4]);
    let q8_d = f32::from_le_bytes(q8_d_bytes);
    let mut sb_d = [0.0_f32; Q4_K_X8_ROWS];
    let mut sb_dmin = [0.0_f32; Q4_K_X8_ROWS];
    let mut row = 0;
    while row < Q4_K_X8_ROWS {
        sb_d[row] = fp16_to_f32(u16::from_le_bytes([lhs[row * 2], lhs[row * 2 + 1]]));
        sb_dmin[row] = fp16_to_f32(u16::from_le_bytes([
            lhs[16 + row * 2],
            lhs[16 + row * 2 + 1],
        ]));
        row += 1;
    }

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
    let mut bsums_pair = [0_i16; Q4_K_X8_ROWS];
    index = 0;
    while index < Q4_K_X8_ROWS {
        bsums_pair[index] = bsums[2 * index].wrapping_add(bsums[2 * index + 1]);
        index += 1;
    }

    let mut low_acc = [0_i32; Q4_K_X8_ROWS];
    let mut high_acc = [0_i32; Q4_K_X8_ROWS];
    let mut bias = [0_i32; Q4_K_X8_ROWS];
    let mut sb = 0;
    while sb < QK_K / 64 {
        let scale_base = 32 + sb * 24;
        let (mins_lo, scales_lo) = decode_q4_k_x8_6bit_scales(&lhs[scale_base..scale_base + 12]);
        let (mins_hi, scales_hi) =
            decode_q4_k_x8_6bit_scales(&lhs[scale_base + 12..scale_base + 24]);
        let rows = deinterleave_superblock_bl8(&lhs[128..], sb);
        let q8_base = sb * 64;
        row = 0;
        while row < Q4_K_X8_ROWS {
            let (acc_lo, acc_hi) = vdot_bl8_superblock_dots(backend, rows[row], &q8_qs, q8_base);
            low_acc[row] =
                low_acc[row].wrapping_add(i32::from(scales_lo[row]).wrapping_mul(acc_lo));
            high_acc[row] =
                high_acc[row].wrapping_add(i32::from(scales_hi[row]).wrapping_mul(acc_hi));
            bias[row] = bias[row]
                .wrapping_add(i32::from(bsums_pair[2 * sb]).wrapping_mul(i32::from(mins_lo[row])))
                .wrapping_add(
                    i32::from(bsums_pair[2 * sb + 1]).wrapping_mul(i32::from(mins_hi[row])),
                );
            row += 1;
        }
        sb += 1;
    }
    row = 0;
    while row < Q4_K_X8_ROWS {
        accumulate_q4_k_scaled_sums(
            sb_d[row],
            sb_dmin[row],
            q8_d,
            bias[row],
            low_acc[row],
            high_acc[row],
            &mut acc[row],
        );
        row += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_group_block_bl8_matrix<const RHS_ROWS: usize>(
    backend: Neon,
    lhs: &[u8],
    rhs: &[u8],
    rhs_row_bytes: usize,
    block: usize,
    acc: &mut [[f32; Q4_K_X8_ROWS]; RHS_ROWS],
) {
    let mut lhs_d = [0.0_f32; Q4_K_X8_ROWS];
    let mut lhs_dmin = [0.0_f32; Q4_K_X8_ROWS];
    let mut row = 0;
    while row < Q4_K_X8_ROWS {
        lhs_d[row] = fp16_to_f32(u16::from_le_bytes([lhs[row * 2], lhs[row * 2 + 1]]));
        lhs_dmin[row] = fp16_to_f32(u16::from_le_bytes([
            lhs[16 + row * 2],
            lhs[16 + row * 2 + 1],
        ]));
        row += 1;
    }

    let mut q8_d = [0.0_f32; RHS_ROWS];
    let mut q8_qs = [[0_i8; QK_K]; RHS_ROWS];
    let mut bsums_pair = [[0_i16; Q4_K_X8_ROWS]; RHS_ROWS];
    let mut scale_prod = [[0.0_f32; Q4_K_X8_ROWS]; RHS_ROWS];
    let mut min_prod = [[0.0_f32; Q4_K_X8_ROWS]; RHS_ROWS];
    let mut rhs_row = 0;
    while rhs_row < RHS_ROWS {
        let rhs_offset = rhs_row * rhs_row_bytes + block * Q8_K_BLOCK_BYTES;
        let rhs_block = &rhs[rhs_offset..rhs_offset + Q8_K_BLOCK_BYTES];
        let mut q8_d_bytes = [0_u8; 4];
        q8_d_bytes.copy_from_slice(&rhs_block[..4]);
        q8_d[rhs_row] = f32::from_le_bytes(q8_d_bytes);
        let mut index = 0;
        while index < QK_K {
            q8_qs[rhs_row][index] = i8::from_ne_bytes([rhs_block[4 + index]]);
            index += 1;
        }
        let mut bsums = [0_i16; QK_K / 16];
        index = 0;
        while index < QK_K / 16 {
            let offset = 260 + index * 2;
            bsums[index] = i16::from_le_bytes([rhs_block[offset], rhs_block[offset + 1]]);
            index += 1;
        }
        index = 0;
        while index < Q4_K_X8_ROWS {
            bsums_pair[rhs_row][index] = bsums[2 * index].wrapping_add(bsums[2 * index + 1]);
            index += 1;
        }
        row = 0;
        while row < Q4_K_X8_ROWS {
            scale_prod[rhs_row][row] = lhs_d[row] * q8_d[rhs_row];
            min_prod[rhs_row][row] = lhs_dmin[row] * q8_d[rhs_row];
            row += 1;
        }
        rhs_row += 1;
    }

    let mut bias = [[0_i32; Q4_K_X8_ROWS]; RHS_ROWS];
    let mut sb = 0;
    while sb < QK_K / 64 {
        let scale_base = 32 + sb * 24;
        let (mins_lo, scales_lo) = decode_q4_k_x8_6bit_scales(&lhs[scale_base..scale_base + 12]);
        let (mins_hi, scales_hi) =
            decode_q4_k_x8_6bit_scales(&lhs[scale_base + 12..scale_base + 24]);
        let rows = deinterleave_superblock_bl8(&lhs[128..], sb);
        let q8_base = sb * 64;
        rhs_row = 0;
        while rhs_row < RHS_ROWS {
            row = 0;
            while row < Q4_K_X8_ROWS {
                let (acc_lo, acc_hi) =
                    vdot_bl8_superblock_dots(backend, rows[row], &q8_qs[rhs_row], q8_base);
                let sumf_0 = i32::from(scales_lo[row]).wrapping_mul(acc_lo) as f32;
                let sumf_1 = i32::from(scales_hi[row]).wrapping_mul(acc_hi) as f32;
                acc[rhs_row][row] = scale_prod[rhs_row][row].mul_add(sumf_0, acc[rhs_row][row]);
                acc[rhs_row][row] = scale_prod[rhs_row][row].mul_add(sumf_1, acc[rhs_row][row]);
                bias[rhs_row][row] = bias[rhs_row][row]
                    .wrapping_add(
                        i32::from(bsums_pair[rhs_row][2 * sb])
                            .wrapping_mul(i32::from(mins_lo[row])),
                    )
                    .wrapping_add(
                        i32::from(bsums_pair[rhs_row][2 * sb + 1])
                            .wrapping_mul(i32::from(mins_hi[row])),
                    );
                row += 1;
            }
            rhs_row += 1;
        }
        sb += 1;
    }
    rhs_row = 0;
    while rhs_row < RHS_ROWS {
        row = 0;
        while row < Q4_K_X8_ROWS {
            acc[rhs_row][row] -= min_prod[rhs_row][row] * (bias[rhs_row][row] as f32);
            row += 1;
        }
        rhs_row += 1;
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn accumulate_q4_k_scaled_sums(
    lhs_d: f32,
    lhs_dmin: f32,
    rhs_d: f32,
    min_sum: i32,
    low_sum: i32,
    high_sum: i32,
    sum: &mut f32,
) {
    let d = lhs_d * rhs_d;
    let dmin = lhs_dmin * rhs_d;
    let dot_sum = low_sum + high_sum;
    *sum -= dmin * (min_sum as f32);
    *sum = d.mul_add(dot_sum as f32, *sum);
}

fn vdot_bl8_superblock_dots(
    backend: Neon,
    row_qs: [u8; 32],
    q8_qs: &[i8; QK_K],
    q8_base: usize,
) -> (i32, i32) {
    let mut qs_lo = [0_u8; 16];
    let mut qs_hi = [0_u8; 16];
    qs_lo.copy_from_slice(&row_qs[..16]);
    qs_hi.copy_from_slice(&row_qs[16..]);
    let acc_lo = widening_i8_dot(
        backend,
        low_nibbles(qs_lo),
        load_i8x16(&q8_qs[q8_base..q8_base + 16]),
    )
    .wrapping_add(widening_i8_dot(
        backend,
        low_nibbles(qs_hi),
        load_i8x16(&q8_qs[q8_base + 16..q8_base + 32]),
    ));
    let acc_hi = widening_i8_dot(
        backend,
        high_nibbles(qs_lo),
        load_i8x16(&q8_qs[q8_base + 32..q8_base + 48]),
    )
    .wrapping_add(widening_i8_dot(
        backend,
        high_nibbles(qs_hi),
        load_i8x16(&q8_qs[q8_base + 48..q8_base + 64]),
    ));
    (acc_lo, acc_hi)
}

const fn deinterleave_superblock_bl8(qs: &[u8], sb: usize) -> [[u8; 32]; Q4_K_X8_ROWS] {
    let mut rows = [[0_u8; 32]; Q4_K_X8_ROWS];
    let mut local = 0;
    while local < QK_K / INTERLEAVE_BL8_BYTES {
        let src_row = local % Q4_K_X8_ROWS;
        let src_offset = (local / Q4_K_X8_ROWS) * INTERLEAVE_BL8_BYTES;
        let dst_offset = sb * QK_K + local * INTERLEAVE_BL8_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BL8_BYTES {
            rows[src_row][src_offset + lane] = qs[dst_offset + lane];
            lane += 1;
        }
        local += 1;
    }
    rows
}

const fn low_nibbles(qs: [u8; 16]) -> i8x16 {
    i8x16(
        i8::from_ne_bytes([qs[0] & 0x0f]),
        i8::from_ne_bytes([qs[1] & 0x0f]),
        i8::from_ne_bytes([qs[2] & 0x0f]),
        i8::from_ne_bytes([qs[3] & 0x0f]),
        i8::from_ne_bytes([qs[4] & 0x0f]),
        i8::from_ne_bytes([qs[5] & 0x0f]),
        i8::from_ne_bytes([qs[6] & 0x0f]),
        i8::from_ne_bytes([qs[7] & 0x0f]),
        i8::from_ne_bytes([qs[8] & 0x0f]),
        i8::from_ne_bytes([qs[9] & 0x0f]),
        i8::from_ne_bytes([qs[10] & 0x0f]),
        i8::from_ne_bytes([qs[11] & 0x0f]),
        i8::from_ne_bytes([qs[12] & 0x0f]),
        i8::from_ne_bytes([qs[13] & 0x0f]),
        i8::from_ne_bytes([qs[14] & 0x0f]),
        i8::from_ne_bytes([qs[15] & 0x0f]),
    )
}

const fn high_nibbles(qs: [u8; 16]) -> i8x16 {
    i8x16(
        i8::from_ne_bytes([qs[0] >> 4]),
        i8::from_ne_bytes([qs[1] >> 4]),
        i8::from_ne_bytes([qs[2] >> 4]),
        i8::from_ne_bytes([qs[3] >> 4]),
        i8::from_ne_bytes([qs[4] >> 4]),
        i8::from_ne_bytes([qs[5] >> 4]),
        i8::from_ne_bytes([qs[6] >> 4]),
        i8::from_ne_bytes([qs[7] >> 4]),
        i8::from_ne_bytes([qs[8] >> 4]),
        i8::from_ne_bytes([qs[9] >> 4]),
        i8::from_ne_bytes([qs[10] >> 4]),
        i8::from_ne_bytes([qs[11] >> 4]),
        i8::from_ne_bytes([qs[12] >> 4]),
        i8::from_ne_bytes([qs[13] >> 4]),
        i8::from_ne_bytes([qs[14] >> 4]),
        i8::from_ne_bytes([qs[15] >> 4]),
    )
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
const fn make_block_q4_k_x8_bl4(
    rows: &[[u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS],
) -> [u8; Q4_K_X8_BLOCK_BYTES] {
    let mut out = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let mut row = 0;
    while row < Q4_K_X8_ROWS {
        out[row * 2] = rows[row][0];
        out[row * 2 + 1] = rows[row][1];
        out[16 + row * 2] = rows[row][2];
        out[16 + row * 2 + 1] = rows[row][3];
        row += 1;
    }
    let end = (QK_K * 4) / INTERLEAVE_BLOCK_BYTES;
    let mut index = 0;
    while index < end {
        let src_row = index % Q4_K_X8_ROWS;
        let src_offset = (index / Q4_K_X8_ROWS) * INTERLEAVE_BLOCK_BYTES;
        let dst_offset = index * INTERLEAVE_BLOCK_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BLOCK_BYTES {
            out[128 + dst_offset + lane] = rows[src_row][16 + src_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    pack_q4_k_x8_scales(&mut out, rows);
    out
}

#[cfg(test)]
const fn pack_q4_k_x8_scales(
    out: &mut [u8; Q4_K_X8_BLOCK_BYTES],
    rows: &[[u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS],
) {
    let mut scales = [0_u8; Q4_K_X8_ROWS];
    let mut mins = [0_u8; Q4_K_X8_ROWS];
    let mut group = 0;
    while group < 4 {
        let mut row = 0;
        while row < Q4_K_X8_ROWS {
            scales[row] = rows[row][4 + group] & 63;
            mins[row] = rows[row][8 + group] & 63;
            row += 1;
        }
        write_packed_scale12(out, group * 12, scales, mins);
        group += 1;
    }
    group = 0;
    while group < 4 {
        let mut row = 0;
        while row < Q4_K_X8_ROWS {
            scales[row] = ((rows[row][4 + group] & 0xc0) >> 2) | (rows[row][12 + group] & 0x0f);
            mins[row] =
                ((rows[row][8 + group] & 0xc0) >> 2) | ((rows[row][12 + group] & 0xf0) >> 4);
            row += 1;
        }
        write_packed_scale12(out, group * 12 + 48, scales, mins);
        group += 1;
    }
}

#[cfg(test)]
const fn write_packed_scale12(
    out: &mut [u8; Q4_K_X8_BLOCK_BYTES],
    base: usize,
    scales: [u8; Q4_K_X8_ROWS],
    mins: [u8; Q4_K_X8_ROWS],
) {
    out[32 + base] = (scales[0] & 0x3f).wrapping_add((scales[4] & 0x30) << 2);
    out[32 + base + 1] = (scales[1] & 0x3f).wrapping_add((scales[5] & 0x30) << 2);
    out[32 + base + 2] = (scales[2] & 0x3f).wrapping_add((scales[6] & 0x30) << 2);
    out[32 + base + 3] = (scales[3] & 0x3f).wrapping_add((scales[7] & 0x30) << 2);
    out[32 + base + 4] = (mins[0] & 0x3f).wrapping_add((mins[4] & 0x30) << 2);
    out[32 + base + 5] = (mins[1] & 0x3f).wrapping_add((mins[5] & 0x30) << 2);
    out[32 + base + 6] = (mins[2] & 0x3f).wrapping_add((mins[6] & 0x30) << 2);
    out[32 + base + 7] = (mins[3] & 0x3f).wrapping_add((mins[7] & 0x30) << 2);
    out[32 + base + 8] = (scales[4] & 0x0f).wrapping_add((mins[4] & 0x0f) << 4);
    out[32 + base + 9] = (scales[5] & 0x0f).wrapping_add((mins[5] & 0x0f) << 4);
    out[32 + base + 10] = (scales[6] & 0x0f).wrapping_add((mins[6] & 0x0f) << 4);
    out[32 + base + 11] = (scales[7] & 0x0f).wrapping_add((mins[7] & 0x0f) << 4);
}

#[cfg(test)]
fn pack_q4_k_rows_x8_bl4(src: &[[u8; Q4_K_BLOCK_BYTES]], m: usize, blocks: usize) -> Vec<u8> {
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let mut packed = vec![0_u8; group_count * blocks * Q4_K_X8_BLOCK_BYTES];
    let mut group = 0;
    while group < group_count {
        let row_base = group * Q4_K_X8_ROWS;
        let mut block = 0;
        while block < blocks {
            let mut group_rows = [[0_u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS];
            let mut row = 0;
            while row < Q4_K_X8_ROWS {
                let logical = row_base + row;
                if logical < m {
                    group_rows[row] = src[logical * blocks + block];
                }
                row += 1;
            }
            let offset = (group * blocks + block) * Q4_K_X8_BLOCK_BYTES;
            packed[offset..offset + Q4_K_X8_BLOCK_BYTES]
                .copy_from_slice(&make_block_q4_k_x8_bl4(&group_rows));
            block += 1;
        }
        group += 1;
    }
    packed
}

#[cfg(test)]
const fn make_block_q4_k_x8_bl8(
    rows: &[[u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS],
) -> [u8; Q4_K_X8_BLOCK_BYTES] {
    let mut out = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let mut row = 0;
    while row < Q4_K_X8_ROWS {
        out[row * 2] = rows[row][0];
        out[row * 2 + 1] = rows[row][1];
        out[16 + row * 2] = rows[row][2];
        out[16 + row * 2 + 1] = rows[row][3];
        row += 1;
    }
    let end = (QK_K * 4) / INTERLEAVE_BL8_BYTES;
    let mut index = 0;
    while index < end {
        let src_row = index % Q4_K_X8_ROWS;
        let src_offset = (index / Q4_K_X8_ROWS) * INTERLEAVE_BL8_BYTES;
        let dst_offset = index * INTERLEAVE_BL8_BYTES;
        let mut lane = 0;
        while lane < INTERLEAVE_BL8_BYTES {
            out[128 + dst_offset + lane] = rows[src_row][16 + src_offset + lane];
            lane += 1;
        }
        index += 1;
    }
    pack_q4_k_x8_scales(&mut out, rows);
    out
}

#[cfg(test)]
fn pack_q4_k_rows_x8_bl8(src: &[[u8; Q4_K_BLOCK_BYTES]], m: usize, blocks: usize) -> Vec<u8> {
    let group_count = m.div_ceil(Q4_K_X8_ROWS);
    let mut packed = vec![0_u8; group_count * blocks * Q4_K_X8_BLOCK_BYTES];
    let mut group = 0;
    while group < group_count {
        let row_base = group * Q4_K_X8_ROWS;
        let mut block = 0;
        while block < blocks {
            let mut group_rows = [[0_u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS];
            let mut row = 0;
            while row < Q4_K_X8_ROWS {
                let logical = row_base + row;
                if logical < m {
                    group_rows[row] = src[logical * blocks + block];
                }
                row += 1;
            }
            let offset = (group * blocks + block) * Q4_K_X8_BLOCK_BYTES;
            packed[offset..offset + Q4_K_X8_BLOCK_BYTES]
                .copy_from_slice(&make_block_q4_k_x8_bl8(&group_rows));
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
        clippy::unreadable_literal
    )]

    use super::{
        INTERLEAVE_BL8_BYTES, INTERLEAVE_BLOCK_BYTES, MATRIX_X4_ROWS, MATRIX_X8_ROWS,
        MAX_Q8_K_BLOCKS, OpMulMatArgmaxQ4PackedF32Bl4, OpMulMatArgmaxQ4PackedF32Bl8,
        OpMulMatQ4PackedBl4, OpMulMatQ4PackedBl4MatrixX4, OpMulMatQ4PackedBl8,
        OpMulMatQ4PackedBl8MatrixX4, OpMulMatQ4PackedBl8MatrixX8, OpMulMatQ4PackedF32Bl4,
        OpMulMatQ4PackedF32Bl8, PINNED_ACTION_BLOB, PINNED_ACTION_SPAN,
        PINNED_ARGMAX_F32_BL4_EXECUTION_SPAN, PINNED_ARGMAX_F32_BL4_GUARD_SPAN,
        PINNED_ARGMAX_F32_BL4_TRANSITION_SPAN, PINNED_ARGMAX_F32_BL8_EXECUTION_SPAN,
        PINNED_ARGMAX_F32_BL8_GUARD_SPAN, PINNED_ARGMAX_F32_BL8_TRANSITION_SPAN,
        PINNED_BL4_MATRIX_ACTION_SPAN, PINNED_BL4_MATRIX_EXECUTION_SPAN,
        PINNED_BL4_MATRIX_GUARD_SPAN, PINNED_BL4_MATRIX_KERNEL_SPAN,
        PINNED_BL4_MATRIX_TRANSITION_SPAN, PINNED_BL8_ACTION_SPAN, PINNED_BL8_EXECUTION_SPAN,
        PINNED_BL8_GUARD_SPAN, PINNED_BL8_KERNEL_SPAN, PINNED_BL8_MATRIX_ACTION_SPAN,
        PINNED_BL8_MATRIX_EXECUTION_SPAN, PINNED_BL8_MATRIX_GUARD_SPAN,
        PINNED_BL8_MATRIX_KERNEL_SPAN, PINNED_BL8_MATRIX_TRANSITION_SPAN,
        PINNED_BL8_MATRIX_X8_ACTION_SPAN, PINNED_BL8_MATRIX_X8_EXECUTION_SPAN,
        PINNED_BL8_MATRIX_X8_GUARD_SPAN, PINNED_BL8_MATRIX_X8_KERNEL_SPAN,
        PINNED_BL8_MATRIX_X8_TRANSITION_SPAN, PINNED_BL8_TRANSITION_SPAN, PINNED_EMEL_CPP_COMMIT,
        PINNED_EXECUTION_SPAN, PINNED_F32_BL4_ACTION_SPAN, PINNED_F32_BL4_EXECUTION_SPAN,
        PINNED_F32_BL4_GUARD_SPAN, PINNED_F32_BL4_TRANSITION_SPAN, PINNED_F32_BL8_ACTION_SPAN,
        PINNED_F32_BL8_EXECUTION_SPAN, PINNED_F32_BL8_GUARD_SPAN, PINNED_F32_BL8_TRANSITION_SPAN,
        PINNED_GUARD_BLOB, PINNED_GUARD_SPAN, PINNED_KERNEL_SPAN, PINNED_SM_BLOB,
        PINNED_TRANSITION_SPAN, Q4_K_BLOCK_BYTES, Q4_K_X8_BLOCK_BYTES, Q4_K_X8_ROWS,
        Q4PackedBl4Error, Q4PackedBl4Event, Q4PackedBl4Kernel, Q8_K_BLOCK_BYTES, QK_K,
        SCOPE_RESIDUAL, UnexpectedQ4PackedBl4, make_block_q4_k_x8_bl4, make_block_q4_k_x8_bl8,
        pack_q4_k_rows_x8_bl4, pack_q4_k_rows_x8_bl8,
    };
    use allocation_counter::measure;

    fn native_q4_k_row(row: usize, block: usize) -> [u8; Q4_K_BLOCK_BYTES] {
        let mut out = [0_u8; Q4_K_BLOCK_BYTES];
        let d = (0x3c00 + row * 17 + block * 7) as u16;
        let dmin = (0x3400 + row * 11 + block * 5) as u16;
        out[0..2].copy_from_slice(&d.to_le_bytes());
        out[2..4].copy_from_slice(&dmin.to_le_bytes());
        let mut index = 0;
        while index < 12 {
            out[4 + index] = ((row * 37 + block * 53 + index * 13 + 9) & 0xff) as u8;
            index += 1;
        }
        index = 0;
        while index < 128 {
            out[16 + index] = ((row * 29 + block * 43 + index * 5 + 3) & 0xff) as u8;
            index += 1;
        }
        out
    }

    fn dense_f32_rhs() -> [f32; QK_K] {
        let mut out = [0.0_f32; QK_K];
        let mut index = 0;
        while index < QK_K {
            out[index] = ((index as i32 * 7) % 31 - 15) as f32 * 0.0625;
            index += 1;
        }
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
            let bsum = sum as i16;
            let offset = 260 + group * 2;
            out[offset..offset + 2].copy_from_slice(&bsum.to_le_bytes());
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
    fn packed_bl4_kernel_constructs_on_neon() {
        assert!(Q4PackedBl4Kernel::try_new().is_some());
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
            "src/emel/kernel/aarch64/guards.hpp:477-487"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1409-1441"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5695-5734"
        );
        assert_eq!(
            PINNED_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:2937-3100"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:427-428"
        );
        assert!(SCOPE_RESIDUAL.contains("dense F32 rhs n=1"));
        assert_eq!(
            PINNED_F32_BL4_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:500-511"
        );
        assert_eq!(
            PINNED_F32_BL4_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1442-1469"
        );
        assert_eq!(
            PINNED_F32_BL4_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5732-5765"
        );
        assert_eq!(
            PINNED_F32_BL4_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:430-433"
        );
        assert_eq!(
            PINNED_F32_BL8_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:524-534"
        );
        assert_eq!(
            PINNED_F32_BL8_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1625-1630"
        );
        assert_eq!(
            PINNED_F32_BL8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5864-5903"
        );
        assert_eq!(
            PINNED_F32_BL8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:415-418"
        );
        assert_eq!(
            PINNED_BL8_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:512-522"
        );
        assert_eq!(
            PINNED_BL8_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1611-1616"
        );
        assert_eq!(
            PINNED_BL8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5827-5862"
        );
        assert_eq!(
            PINNED_BL8_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:3102-3249"
        );
        assert_eq!(
            PINNED_BL8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:412-413"
        );
        assert_eq!(MAX_Q8_K_BLOCKS, 128);
        assert_eq!(Q4_K_X8_ROWS, 8);
        assert_eq!(Q4_K_X8_BLOCK_BYTES, 1152);
        assert_eq!(Q8_K_BLOCK_BYTES, 292);
        assert_eq!(INTERLEAVE_BLOCK_BYTES, 4);
        assert_eq!(INTERLEAVE_BL8_BYTES, 8);
        assert_eq!(QK_K, 256);
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
    }

    #[test]
    fn packed_bl4_matrix_x4_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_BL4_MATRIX_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:143-151"
        );
        assert_eq!(
            PINNED_BL4_MATRIX_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:489-497"
        );
        assert_eq!(
            PINNED_BL4_MATRIX_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5784-5825"
        );
        assert_eq!(
            PINNED_BL4_MATRIX_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:2937-3100"
        );
        assert_eq!(
            PINNED_BL4_MATRIX_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:420-423"
        );
        assert!(SCOPE_RESIDUAL.contains("bl4 packed q8_k matrix_x4"));
        assert_eq!(MATRIX_X4_ROWS, 4);
    }

    #[test]
    fn packed_bl8_matrix_x4_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_BL8_MATRIX_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:73-108"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:536-545"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5906-5966"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:3251-3435"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:406-408"
        );
        assert!(SCOPE_RESIDUAL.contains("bl8 packed q8_k matrix_x4"));
        assert_eq!(MATRIX_X4_ROWS, 4);
    }

    #[test]
    fn packed_bl8_matrix_x8_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_BL8_MATRIX_X8_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:184-218"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_X8_ACTION_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:547-556"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_X8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5969-6030"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_X8_KERNEL_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:3436-3622"
        );
        assert_eq!(
            PINNED_BL8_MATRIX_X8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:400-403"
        );
        assert!(SCOPE_RESIDUAL.contains("matrix_x8"));
        assert_eq!(MATRIX_X8_ROWS, 8);
    }

    #[test]
    fn packed_bl4_block_size_matches_native_layout() {
        let rows = [
            native_q4_k_row(0, 0),
            native_q4_k_row(1, 0),
            native_q4_k_row(2, 0),
            native_q4_k_row(3, 0),
            native_q4_k_row(4, 0),
            native_q4_k_row(5, 0),
            native_q4_k_row(6, 0),
            native_q4_k_row(7, 0),
        ];
        let packed = make_block_q4_k_x8_bl4(&rows);
        assert_eq!(packed.len(), 1152);
        assert_eq!(&packed[0..2], &rows[0][0..2]);
        assert_eq!(&packed[14..16], &rows[7][0..2]);
        assert_eq!(&packed[16..18], &rows[0][2..4]);
    }

    #[test]
    fn neon_packed_bl4_m8_k256_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 8, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&lhs, &rhs, 8, QK_K), &mut output),
            Ok(())
        );
        row = 0;
        while row < 8 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn neon_packed_bl4_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 5, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&lhs, &rhs, 5, QK_K), &mut output),
            Ok(())
        );
        assert_eq!(lhs.len(), Q4_K_X8_BLOCK_BYTES);
        row = 0;
        while row < 5 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl4_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&[], &[], 0, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_bl4_rejects_non_block_k() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&[1, 2], &[3], 5, 255), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn packed_bl4_rejects_unexpected_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedQ4PackedBl4, &mut []),
            Err(Q4PackedBl4Error::UnexpectedEvent)
        );
    }

    #[test]
    fn packed_bl4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 1];
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedBl4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut output,
            ),
            Ok(())
        );
        assert!(output[0].is_finite());
    }

    #[test]
    fn packed_bl8_block_size_matches_native_layout() {
        let rows = [
            native_q4_k_row(0, 0),
            native_q4_k_row(1, 0),
            native_q4_k_row(2, 0),
            native_q4_k_row(3, 0),
            native_q4_k_row(4, 0),
            native_q4_k_row(5, 0),
            native_q4_k_row(6, 0),
            native_q4_k_row(7, 0),
        ];
        let packed = make_block_q4_k_x8_bl8(&rows);
        assert_eq!(packed.len(), 1152);
        assert_eq!(&packed[0..2], &rows[0][0..2]);
        assert_eq!(&packed[14..16], &rows[7][0..2]);
        assert_eq!(&packed[16..18], &rows[0][2..4]);
        assert_eq!(&packed[128..136], &rows[0][16..24]);
        assert_eq!(&packed[136..144], &rows[1][16..24]);
    }

    #[test]
    fn neon_packed_bl8_m8_k256_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 8, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 8];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&lhs, &rhs, 8, QK_K), &mut output),
            Ok(())
        );
        row = 0;
        while row < 8 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn neon_packed_bl8_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 5, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&lhs, &rhs, 5, QK_K), &mut output),
            Ok(())
        );
        assert_eq!(lhs.len(), Q4_K_X8_BLOCK_BYTES);
        row = 0;
        while row < 5 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl8_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&[], &[], 0, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_bl8_rejects_non_block_k() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&[1, 2], &[3], 5, 255), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn packed_bl8_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut output = [f32::NAN; 1];
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedBl8::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut output,
            ),
            Ok(())
        );
        assert!(output[0].is_finite());
    }

    #[test]
    fn packed_bl8_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl8::new(&lhs, &rhs, 1, QK_K), &mut second),
            Ok(())
        );
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn packed_bl4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = pack_q8_k_row();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedBl4::new(&lhs, &rhs, 1, QK_K), &mut second),
            Ok(())
        );
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn neon_packed_f32_bl4_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 5, 1);
        let rhs = dense_f32_rhs();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 5, QK_K),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < 5 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_f32_bl4_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl4::new(&[], &[], 0, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_f32_bl4_rejects_k_zero() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl4::new(&[], &[], 5, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn packed_f32_bl4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut via_process = [f32::NAN; 1];
        let mut via_trait = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process[0].to_bits(), via_trait[0].to_bits());
    }

    #[test]
    fn packed_f32_bl4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn packed_f32_bl4_argmax_matches_f32_dots_m5() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 5, 1);
        let rhs = dense_f32_rhs();
        let mut dots = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 5, QK_K), &mut dots,),
            Ok(())
        );
        let mut expected_index = 0_i32;
        let mut expected_value = f32::NEG_INFINITY;
        let mut index = 0;
        while index < dots.len() {
            if index == 0 || dots[index] > expected_value {
                expected_value = dots[index];
                expected_index = index as i32;
            }
            index += 1;
        }

        let mut output = [f32::NAN];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ4PackedF32Bl4::new(&lhs, &rhs, 5, QK_K),
                &mut output,
            ),
            Ok(expected_index)
        );
        assert_eq!(output[0].to_bits(), expected_value.to_bits());
    }

    #[test]
    fn packed_f32_bl4_argmax_first_tie_keeps_earlier_index() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let row = native_q4_k_row(0, 0);
        let rows = [row, row];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 2, 1);
        let rhs = dense_f32_rhs();
        let mut output = [f32::NAN];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ4PackedF32Bl4::new(&lhs, &rhs, 2, QK_K),
                &mut output,
            ),
            Ok(0)
        );
        let mut dots = [f32::NAN; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl4::new(&lhs, &rhs, 2, QK_K), &mut dots,),
            Ok(())
        );
        assert_eq!(dots[0].to_bits(), dots[1].to_bits());
        assert_eq!(output[0].to_bits(), dots[0].to_bits());
    }

    #[test]
    fn packed_f32_bl4_argmax_rejects_invalid_shapes_without_mutation() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let sentinel = f32::from_bits(0x7fc0_0001);
        for (m, k) in [(0, 0), (5, 0), (5, 255)] {
            let mut output = [sentinel];
            assert_eq!(
                kernel.process_event(
                    OpMulMatArgmaxQ4PackedF32Bl4::new(&[], &[], m, k),
                    &mut output,
                ),
                Err(Q4PackedBl4Error::InvalidShape)
            );
            assert_eq!(output[0].to_bits(), sentinel.to_bits());
        }
    }

    #[test]
    fn packed_f32_bl4_argmax_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut first = [f32::NAN];
        let mut second = [f32::NAN];
        let first_index = kernel
            .process_event(
                OpMulMatArgmaxQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K),
                &mut first,
            )
            .expect("first argmax");
        let mut second_index = -1;
        let info = measure(|| {
            second_index = kernel
                .process_event(
                    OpMulMatArgmaxQ4PackedF32Bl4::new(&lhs, &rhs, 1, QK_K),
                    &mut second,
                )
                .expect("second argmax");
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first_index, second_index);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn packed_f32_bl8_argmax_matches_f32_dots_m5() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 5, 1);
        let rhs = dense_f32_rhs();
        let mut dots = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 5, QK_K), &mut dots,),
            Ok(())
        );
        let mut expected_index = 0_i32;
        let mut expected_value = f32::NEG_INFINITY;
        let mut index = 0;
        while index < dots.len() {
            if index == 0 || dots[index] > expected_value {
                expected_value = dots[index];
                expected_index = index as i32;
            }
            index += 1;
        }

        let mut output = [f32::NAN];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ4PackedF32Bl8::new(&lhs, &rhs, 5, QK_K),
                &mut output,
            ),
            Ok(expected_index)
        );
        assert_eq!(output[0].to_bits(), expected_value.to_bits());
    }

    #[test]
    fn packed_f32_bl8_argmax_first_tie_keeps_earlier_index() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let row = native_q4_k_row(0, 0);
        let rows = [row, row];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 2, 1);
        let rhs = dense_f32_rhs();
        let mut output = [f32::NAN];
        assert_eq!(
            kernel.process_event(
                OpMulMatArgmaxQ4PackedF32Bl8::new(&lhs, &rhs, 2, QK_K),
                &mut output,
            ),
            Ok(0)
        );
        let mut dots = [f32::NAN; 2];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 2, QK_K), &mut dots,),
            Ok(())
        );
        assert_eq!(dots[0].to_bits(), dots[1].to_bits());
        assert_eq!(output[0].to_bits(), dots[0].to_bits());
    }

    #[test]
    fn packed_f32_bl8_argmax_rejects_invalid_shapes_without_mutation() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let sentinel = f32::from_bits(0x7fc0_0001);
        for (m, k) in [(0, 0), (5, 0), (5, 255)] {
            let mut output = [sentinel];
            assert_eq!(
                kernel.process_event(
                    OpMulMatArgmaxQ4PackedF32Bl8::new(&[], &[], m, k),
                    &mut output,
                ),
                Err(Q4PackedBl4Error::InvalidShape)
            );
            assert_eq!(output[0].to_bits(), sentinel.to_bits());
        }
    }

    #[test]
    fn packed_f32_bl8_argmax_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut first = [f32::NAN];
        let mut second = [f32::NAN];
        let first_index = kernel
            .process_event(
                OpMulMatArgmaxQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K),
                &mut first,
            )
            .expect("first argmax");
        let mut second_index = -1;
        let info = measure(|| {
            second_index = kernel
                .process_event(
                    OpMulMatArgmaxQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K),
                    &mut second,
                )
                .expect("second argmax");
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(info.count_total, 0);
        assert_eq!(first_index, second_index);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn packed_f32_bl8_argmax_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_ARGMAX_F32_BL8_GUARD_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1769-1798"
        );
        assert_eq!(
            PINNED_ARGMAX_F32_BL8_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5644-5693"
        );
        assert_eq!(
            PINNED_ARGMAX_F32_BL8_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:467"
        );
    }

    #[test]
    fn packed_f32_bl4_argmax_pin_constants_match_pinned_source() {
        assert_eq!(
            PINNED_ARGMAX_F32_BL4_GUARD_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1769-1788"
        );
        assert_eq!(
            PINNED_ARGMAX_F32_BL4_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:5593-5640"
        );
        assert_eq!(
            PINNED_ARGMAX_F32_BL4_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:471-474"
        );
    }

    #[test]
    fn neon_packed_f32_bl8_partial_group_m5_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 5];
        let mut row = 0;
        while row < 5 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 5, 1);
        let rhs = dense_f32_rhs();
        let mut output = [f32::NAN; 5];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 5, QK_K),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < 5 {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_f32_bl8_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001)];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl8::new(&[], &[], 0, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 0x7fc0_0001);
    }

    #[test]
    fn packed_f32_bl8_rejects_k_zero() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl8::new(&[], &[], 5, 0), &mut output),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn packed_f32_bl8_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut via_process = [f32::NAN; 1];
        let mut via_trait = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process[0].to_bits(), via_trait[0].to_bits());
    }

    #[test]
    fn packed_f32_bl8_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = dense_f32_rhs();
        let mut first = [f32::NAN; 1];
        let mut second = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K), &mut first),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedF32Bl8::new(&lhs, &rhs, 1, QK_K),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first[0].to_bits(), second[0].to_bits());
    }

    #[test]
    fn neon_packed_bl8_matrix_x4_full_group_m8_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut output = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < output.len() {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl8_matrix_x4_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&[], &[], 0, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 4]);
    }

    #[test]
    fn packed_bl8_matrix_x4_rejects_k_zero() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&[], &[], 8, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_bl8_matrix_x4_rejects_rhs_rows_not_four() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 8, 1);
        let rhs = pack_q8_k_row_at(0);
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_bl8_matrix_x4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut via_process = [f32::NAN; MATRIX_X4_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn packed_bl8_matrix_x4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut first = [f32::NAN; MATRIX_X4_ROWS];
        let mut second = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }

    #[test]
    fn neon_packed_bl4_matrix_x4_full_group_m8_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut output = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < output.len() {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn neon_packed_bl4_matrix_x4_matches_independent_bl4_dots() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut matrix = [f32::NAN; MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut matrix
            ),
            Ok(())
        );
        let mut rhs_row = 0;
        while rhs_row < MATRIX_X4_ROWS {
            let packed = pack_q8_k_row_at(rhs_row);
            let mut vector = [f32::NAN; 8];
            assert_eq!(
                kernel.process_event(
                    OpMulMatQ4PackedBl4::new(&lhs, &packed, 8, QK_K),
                    &mut vector
                ),
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
    fn packed_bl4_matrix_x4_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&[], &[], 0, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 4]);
    }

    #[test]
    fn packed_bl4_matrix_x4_rejects_k_zero() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&[], &[], 8, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_bl4_matrix_x4_rejects_rhs_rows_not_four() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 8, 1);
        let rhs = pack_q8_k_row_at(0);
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X4_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X4_ROWS * 8]);
    }

    #[test]
    fn packed_bl4_matrix_x4_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut via_process = [f32::NAN; MATRIX_X4_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn packed_bl4_matrix_x4_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl4(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut first = [f32::NAN; MATRIX_X4_ROWS];
        let mut second = [f32::NAN; MATRIX_X4_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 1, QK_K),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }

    #[test]
    fn neon_packed_bl8_matrix_x8_full_group_m8_dispatches() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut output = [f32::NAN; MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Ok(())
        );
        row = 0;
        while row < output.len() {
            assert!(output[row].is_finite());
            row += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn packed_bl8_matrix_x8_rejects_zero_dims() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&[], &[], 0, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 4]);
    }

    #[test]
    fn packed_bl8_matrix_x8_rejects_k_zero() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&[], &[], 8, 0),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X8_ROWS * 8]);
    }

    #[test]
    fn packed_bl8_matrix_x8_rejects_rhs_rows_not_eight() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; 8];
        let mut row = 0;
        while row < 8 {
            rows[row] = native_q4_k_row(row, 0);
            row += 1;
        }
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 8, 1);
        let rhs = pack_q8_k_rows_x4();
        let mut output = [f32::from_bits(0x7fc0_0001); MATRIX_X8_ROWS * 8];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 8, QK_K),
                &mut output
            ),
            Err(Q4PackedBl4Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; MATRIX_X8_ROWS * 8]);
    }

    #[test]
    fn packed_bl8_matrix_x8_trait_dispatch_matches_process_event() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut via_process = [f32::NAN; MATRIX_X8_ROWS];
        let mut via_trait = [f32::NAN; MATRIX_X8_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut via_process
            ),
            Ok(())
        );
        assert_eq!(
            Q4PackedBl4Event::dispatch(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut kernel,
                &mut via_trait
            ),
            Ok(())
        );
        assert_eq!(via_process.map(f32::to_bits), via_trait.map(f32::to_bits));
    }

    #[test]
    fn packed_bl8_matrix_x8_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = Q4PackedBl4Kernel::try_new() else {
            return;
        };
        let rows = [native_q4_k_row(0, 0)];
        let lhs = pack_q4_k_rows_x8_bl8(&rows, 1, 1);
        let rhs = pack_q8_k_rows_x8();
        let mut first = [f32::NAN; MATRIX_X8_ROWS];
        let mut second = [f32::NAN; MATRIX_X8_ROWS];
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut first
            ),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(
                OpMulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 1, QK_K),
                &mut second
            ),
            Ok(())
        );
        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }
}
