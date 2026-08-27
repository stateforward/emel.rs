//! Target-owned `AArch64` actor composition for maintained NEON operation slices.
//!
//! This router composes only child actors whose safe NEON contracts are already
//! maintained in this crate. It never substitutes the portable actor and never
//! reaches into child machine internals.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use crate::any::event::GenericEvent;
use crate::any::f16_matmul::{F16MatmulError, F16MatmulKernel};
use crate::any::generic::KernelOperation;
use crate::any::get_rows::GetRowsKernel;
pub use crate::any::get_rows::{
    Bf16View, GetRowsError, GetRowsOutcome, IndexView, OpGetRows, OpGetRowsBf16, OpGetRowsF16,
    OpGetRowsQ4_0, OpGetRowsQ4K, OpGetRowsQ8_0, UnexpectedGetRows,
};
pub use crate::any::im2col::{
    F16OutputViewMut, Im2ColError, Im2ColKernel, Im2ColParams, Im2ColResult, OpIm2Col, OpIm2ColF16,
    UnexpectedIm2Col,
};
pub use crate::any::normalization::{
    NormalizationError, NormalizationKernel, NormalizationResult, OpNorm, OpRmsNorm,
};
pub use crate::any::reductions::{OpSoftMax, ReductionError, ReductionKernel, ReductionResult};
pub use crate::any::rope::{I32View, OpRope, RopeError, RopeKernel, RopeParams, UnexpectedRope};
pub use crate::detail::f16_matmul::{
    F16MatmulResult, OpMulMatF16Scalar, OpScalarMulMatF16, UnexpectedF16Matmul,
};
pub use crate::detail::scalar_trig::{
    OpScalarCos, OpScalarLog, OpScalarSin, ScalarTrigResult, UnexpectedScalarTrig,
};

use super::f16_matmul::{Aarch64F16MatmulKernel, f16_vector_supported};

/// Pinned source identity for the `AArch64` target `RoPE` transition slice.
pub const TARGET_ROPE_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for the `RoPE` slice.
pub const TARGET_ROPE_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Pinned source identity for the `AArch64` target softmax transition slice.
pub const TARGET_SOFTMAX_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for the softmax slice.
pub const TARGET_SOFTMAX_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Pinned source identity for the `AArch64` target scalar-trigonometry slice.
pub const TARGET_SCALAR_TRIG_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for scalar trigonometry.
pub const TARGET_SCALAR_TRIG_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Pinned source identity for the `AArch64` target scalar F16 matrix route.
pub const TARGET_F16_MATMUL_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for scalar F16 matrix multiplication.
pub const TARGET_F16_MATMUL_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Pinned source identity for the target vector-aligned NEON `SiLU` route.
pub const TARGET_AARCH64_UNARY_SILU_SOURCE_COMMIT: &str =
    "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned guard span for the target vector-aligned NEON `SiLU` route.
pub const TARGET_AARCH64_UNARY_SILU_SOURCE_GUARD_SPAN: &str =
    "src/emel/kernel/aarch64/guards.hpp:826-886";
/// Pinned action span for the target vector-aligned NEON `SiLU` route.
pub const TARGET_AARCH64_UNARY_SILU_SOURCE_ACTION_SPAN: &str =
    "src/emel/kernel/aarch64/actions.hpp:89-123,180-196";
/// Pinned target state-machine blob for the vector-aligned NEON `SiLU` route.
pub const TARGET_AARCH64_UNARY_SILU_SOURCE_SM_BLOB: &str =
    "865a9cc6ba6115382ed043c464f3d62bcd851357";
/// Residual intentionally excluded by the vector-aligned target guard.
pub const TARGET_AARCH64_UNARY_SILU_RESIDUAL: &str = "none for the pinned dense F32 SiLU route";

/// Pinned source identity for the `AArch64` target normalization transition slice.
pub const TARGET_NORMALIZATION_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for the normalization slice.
pub const TARGET_NORMALIZATION_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Explicit target normalization unexpected-event request.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedNormalization;

/// Explicit target softmax unexpected-event request.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedSoftMax;

/// Pinned source identity for the `AArch64` target `im2col` transition slice.
pub const TARGET_IM2COL_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for the `im2col` slice.
pub const TARGET_IM2COL_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

/// Pinned source identity for the `AArch64` target row-gather transition slice.
pub const TARGET_GET_ROWS_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned `AArch64` target state-machine blob for the row-gather slice.
pub const TARGET_GET_ROWS_SOURCE_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
pub use super::binary::{
    BinaryF32Error, BinaryF32Kernel, BinaryF32Result, OpAarch64BinaryAdd, OpAarch64BinaryDiv,
    OpAarch64BinaryMul, OpAarch64BinarySub,
};
pub use super::broadcast::{
    BroadcastF32Error, BroadcastF32Kernel, BroadcastF32Result, OpAarch64BroadcastAdd,
    OpAarch64BroadcastMul,
};
use super::conv_transpose_1d::{
    ConvTranspose1dF32Error, ConvTranspose1dF32Kernel, ConvTranspose1dF32Result,
    ConvTranspose1dF32Shape, OpAarch64ConvTranspose1dF16, OpAarch64ConvTranspose1dF32,
};
pub use super::dup::{DupF32Error, DupF32Kernel, DupF32Result, OpAarch64DupF32};
use super::gemv::{F32GemvError, F32GemvKernel, F32GemvResult, OpF32Gemv};
use super::mul_mat_f32::{F32MulMatError, F32MulMatKernel, F32MulMatResult, OpMulMatF32};
pub use super::power::{
    OpAarch64PowerSqr, OpAarch64PowerSqrt, PowerF32Error, PowerF32Kernel, PowerF32Result,
};
use super::q2_k::{OpMulMatQ2K, OpMulMatQ2KVector, Q2KError, Q2KKernel, Q2KResult};
use super::q3_k::{OpMulMatQ3K, OpMulMatQ3KVector, Q3KError, Q3KKernel, Q3KResult};
use super::q4_0::{OpMulMatQ4_0Vector, Q4_0VectorError, Q4_0VectorKernel, Q4_0VectorResult};
use super::q4_1::{OpMulMatQ4_1Vector, Q4_1VectorError, Q4_1VectorKernel, Q4_1VectorResult};
use super::q4_k::{OpMulMatQ4K, OpMulMatQ4KVector, Q4KError, Q4KKernel, Q4KResult};
use super::q4_packed::{
    OpMulMatArgmaxQ4PackedF32Bl4, OpMulMatArgmaxQ4PackedF32Bl8, OpMulMatQ4PackedBl4,
    OpMulMatQ4PackedBl4MatrixX4, OpMulMatQ4PackedBl8, OpMulMatQ4PackedBl8MatrixX4,
    OpMulMatQ4PackedBl8MatrixX8, OpMulMatQ4PackedF32Bl4, OpMulMatQ4PackedF32Bl8, Q4PackedBl4Error,
    Q4PackedBl4Kernel, Q4PackedBl4Result, Q4PackedF32Bl4ArgmaxResult, Q4PackedF32Bl8ArgmaxResult,
};
use super::q5_0::{OpMulMatQ5_0Vector, Q5_0VectorError, Q5_0VectorKernel, Q5_0VectorResult};
use super::q6::{OpMulMatQ6, OpMulMatQ6Vector, Q6VectorError, Q6VectorKernel, Q6VectorResult};
use super::q6_packed::{
    OpMulMatArgmaxQ6Packed, OpMulMatQ6Packed, OpMulMatQ6PackedMatrixX4, Q6PackedArgmaxResult,
    Q6PackedError, Q6PackedKernel, Q6PackedResult,
};
use super::q6_prepared::{
    OpMulMatQ6Prepared, OpMulMatQ6PreparedMatrixX4, OpMulMatQ6PreparedMatrixX8, Q6PreparedError,
    Q6PreparedKernel, Q6PreparedResult,
};
use super::q8_0::{
    OpMulMatQ8_0Vector, OpMulMatQ8_0VectorQ8Rhs, Q8_0VectorError, Q8_0VectorKernel,
    Q8_0VectorResult,
};
use super::q8_0_packed::{
    OpMulMatQ8_0PackedBl4, OpMulMatQ8_0PackedBl8, OpMulMatQ8_0PackedBl8MatrixX4,
    Q8_0PackedBl4Error, Q8_0PackedBl4Kernel, Q8_0PackedBl4Result,
};
use super::unary::{
    OpAarch64UnaryAbs, OpAarch64UnaryNeg, OpAarch64UnaryRelu, OpAarch64UnarySilu, UnaryF32Error,
    UnaryF32Kernel, UnaryF32Result,
};
use crate::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use crate::any::unary::{
    OpElu, OpExp, OpGelu, OpSilu, OpTanh, UnaryError, UnaryKernel, UnaryResult,
};
pub use crate::detail::scalar_unary::{
    OpScalarUnaryElu, OpScalarUnaryExp, OpScalarUnaryGelu, OpScalarUnarySilu, OpScalarUnaryTanh,
    PINNED_AARCH64_ACTION_BLOB, PINNED_AARCH64_ACTION_SPAN, PINNED_AARCH64_GUARD_BLOB,
    PINNED_AARCH64_GUARD_SPAN, PINNED_AARCH64_TRANSITION_BLOB, PINNED_AARCH64_TRANSITION_SPAN,
    PINNED_DETAIL_SPAN, PINNED_EMEL_CPP_COMMIT,
};

/// Re-export of the maintained child shape contract for router requests.
pub use super::conv_transpose_1d::ConvTranspose1dF32Shape as ConvTransposeShape;

/// A failure in the target router itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    /// The router received its explicit unexpected-event request.
    UnexpectedEvent,
    /// The generated router failed to dispatch an event.
    Internal,
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 target event"),
            Self::Internal => formatter.write_str("internal AArch64 target dispatch error"),
        }
    }
}

impl std::error::Error for KernelError {}

/// A typed unexpected-event request for the target router.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64Kernel;

macro_rules! define_unary_request {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name<'a> {
            input: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; validation remains in the child actor.
            #[must_use]
            pub const fn new(input: &'a [f32]) -> Self {
                Self { input }
            }
        }
    };
}

macro_rules! define_binary_request {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name<'a> {
            lhs: &'a [f32],
            rhs: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; validation remains in the child actor.
            #[must_use]
            pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
                Self { lhs, rhs }
            }
        }
    };
}

define_unary_request!(Dup, "A target `AArch64` duplication request.");
define_binary_request!(BinaryAdd, "A target `AArch64` binary-add request.");
define_binary_request!(BinarySub, "A target `AArch64` binary-subtract request.");
define_binary_request!(BinaryMul, "A target `AArch64` binary-multiply request.");
define_binary_request!(BinaryDiv, "A target `AArch64` binary-divide request.");
define_unary_request!(PowerSqr, "A target `AArch64` square request.");
define_unary_request!(PowerSqrt, "A target `AArch64` square-root request.");
define_unary_request!(UnaryAbs, "A target `AArch64` absolute-value request.");
define_unary_request!(UnaryNeg, "A target `AArch64` negation request.");
define_unary_request!(UnaryRelu, "A target `AArch64` `ReLU` request.");
define_unary_request!(UnarySilu, "A target `AArch64` `SiLU` request.");

macro_rules! define_broadcast_request {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name<'a> {
            input: &'a [f32],
            row: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; validation remains in the child actor.
            #[must_use]
            pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
                Self { input, row }
            }
        }
    };
}

define_broadcast_request!(
    BroadcastAdd,
    "A target `AArch64` row-broadcast addition request."
);
define_broadcast_request!(
    BroadcastMul,
    "A target `AArch64` row-broadcast multiplication request."
);

/// A target `AArch64` F32 GEMV request.
#[derive(Debug)]
pub struct Gemv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> Gemv<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` dense F32 GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct MulMatF32<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> MulMatF32<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// A target `AArch64` packed `q4_0` vector matmul request.
#[derive(Debug)]
pub struct MulMatQ4_0Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4_0Vector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_1` vector matmul request.
#[derive(Debug)]
pub struct MulMatQ4_1Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4_1Vector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q5_0` vector matmul request.
#[derive(Debug)]
pub struct MulMatQ5_0Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ5_0Vector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q8_0` vector matmul request.
#[derive(Debug)]
pub struct MulMatQ8_0Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ8_0Vector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q8_0` by packed `q8_0` vector request.
#[derive(Debug)]
pub struct MulMatQ8_0VectorQ8Rhs<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ8_0VectorQ8Rhs<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q8_0_x4_bl4` by packed `q8_0` vector request.
#[derive(Debug)]
pub struct MulMatQ8_0PackedBl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ8_0PackedBl4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl4` by packed `q8_k` vector request.
#[derive(Debug)]
pub struct MulMatQ4PackedBl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedBl4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl4` by 4 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ4PackedBl4MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedBl4MatrixX4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl8` by packed `q8_k` vector request.
#[derive(Debug)]
pub struct MulMatQ4PackedBl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedBl8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl4` by dense F32 vector request.
#[derive(Debug)]
pub struct MulMatQ4PackedF32Bl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedF32Bl4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl4` by dense F32 vector argmax request.
#[derive(Debug)]
pub struct MulMatArgmaxQ4PackedF32Bl4<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatArgmaxQ4PackedF32Bl4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl8` by dense F32 vector argmax request.
#[derive(Debug)]
pub struct MulMatArgmaxQ4PackedF32Bl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatArgmaxQ4PackedF32Bl8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl8` by dense F32 vector request.
#[derive(Debug)]
pub struct MulMatQ4PackedF32Bl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedF32Bl8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl8` by 4 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ4PackedBl8MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedBl8MatrixX4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k_x8_bl8` by 8 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ4PackedBl8MatrixX8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4PackedBl8MatrixX8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q8_0_x4_bl8` by packed `q8_0` vector request.
#[derive(Debug)]
pub struct MulMatQ8_0PackedBl8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ8_0PackedBl8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q8_0_x4_bl8` matrix-by-4 request.
#[derive(Debug)]
pub struct MulMatQ8_0PackedBl8MatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ8_0PackedBl8MatrixX4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q6_k_x8` by packed `q8_k` vector request.
#[derive(Debug)]
pub struct MulMatQ6Packed<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6Packed<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q6_k_x8` by 4 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ6PackedMatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6PackedMatrixX4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q6_k_x8` by packed `q8_k` vector argmax request.
#[derive(Debug)]
pub struct MulMatArgmaxQ6Packed<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatArgmaxQ6Packed<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` prepared `q6_k_x8_q8` by packed `q8_k` vector request.
#[derive(Debug)]
pub struct MulMatQ6Prepared<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6Prepared<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` prepared `q6_k_x8_q8` by 4 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ6PreparedMatrixX4<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6PreparedMatrixX4<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` prepared `q6_k_x8_q8` by 8 packed `q8_k` rows.
#[derive(Debug)]
pub struct MulMatQ6PreparedMatrixX8<'a> {
    lhs: &'a [u8],
    rhs: &'a [u8],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6PreparedMatrixX8<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [u8], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q6_k` vector matmul request.
#[derive(Debug)]
pub struct MulMatQ6Vector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ6Vector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q6_k` GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct MulMatQ6<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> MulMatQ6<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// A target `AArch64` packed `q4_k` vector request.
#[derive(Debug)]
pub struct MulMatQ4KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ4KVector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q4_k` GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct MulMatQ4K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> MulMatQ4K<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// A target `AArch64` packed `q2_k` vector request.
#[derive(Debug)]
pub struct MulMatQ2KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ2KVector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q2_k` GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct MulMatQ2K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> MulMatQ2K<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// A target `AArch64` packed `q3_k` vector request.
#[derive(Debug)]
pub struct MulMatQ3KVector<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> MulMatQ3KVector<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target `AArch64` packed `q3_k` GEMM request with `n >= 2`.
#[derive(Debug)]
pub struct MulMatQ3K<'a> {
    lhs: &'a [u8],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> MulMatQ3K<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [u8], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// A target `AArch64` one-dimensional transposed-convolution request.
#[derive(Debug)]
pub struct ConvTranspose1d<'a> {
    weights: &'a [f32],
    input: &'a [f32],
    shape: ConvTranspose1dF32Shape,
}

impl<'a> ConvTranspose1d<'a> {
    /// Creates a request; validation remains in the child actor.
    #[must_use]
    pub const fn new(weights: &'a [f32], input: &'a [f32], shape: ConvTranspose1dF32Shape) -> Self {
        Self {
            weights,
            input,
            shape,
        }
    }
}

struct UnaryAbsRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryF32Result>,
}

struct UnaryNegRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryF32Result>,
}

struct UnaryReluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryF32Result>,
}

struct UnarySiluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryF32Result>,
}

struct ScalarExpRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryResult>,
}

struct ScalarTanhRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryResult>,
}

struct ScalarEluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryResult>,
}

struct ScalarGeluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryResult>,
}

struct ScalarSiluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<UnaryResult>,
}

struct BinaryAddRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BinaryF32Result>,
}

struct BinarySubRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BinaryF32Result>,
}

struct BinaryMulRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BinaryF32Result>,
}

struct BinaryDivRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BinaryF32Result>,
}

struct PowerSqrRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<PowerF32Result>,
}

struct PowerSqrtRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<PowerF32Result>,
}

struct BroadcastAddRuntime<'dispatch> {
    input: &'dispatch [f32],
    row: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BroadcastF32Result>,
}

struct BroadcastMulRuntime<'dispatch> {
    input: &'dispatch [f32],
    row: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<BroadcastF32Result>,
}

struct DupRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<DupF32Result>,
}

struct MulMatF32Runtime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    n: usize,
    result: &'dispatch Cell<F32MulMatResult>,
}

struct GemvRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<F32GemvResult>,
}

struct MulMatQ4_0VectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4_0VectorResult>,
}

struct MulMatQ4_1VectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4_1VectorResult>,
}

struct MulMatQ5_0VectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q5_0VectorResult>,
}

struct MulMatQ8_0VectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q8_0VectorResult>,
}

struct MulMatQ8_0VectorQ8RhsRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q8_0VectorResult>,
}

struct MulMatQ8_0PackedBl4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q8_0PackedBl4Result>,
}

struct MulMatQ4PackedBl4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ4PackedBl4MatrixX4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ4PackedBl8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ4PackedF32Bl4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatArgmaxQ4PackedF32Bl4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedF32Bl4ArgmaxResult>,
}

struct MulMatArgmaxQ4PackedF32Bl8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedF32Bl8ArgmaxResult>,
}

struct MulMatQ4PackedF32Bl8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ4PackedBl8MatrixX4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ4PackedBl8MatrixX8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4PackedBl4Result>,
}

struct MulMatQ8_0PackedBl8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q8_0PackedBl4Result>,
}

struct MulMatQ8_0PackedBl8MatrixX4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q8_0PackedBl4Result>,
}

struct MulMatQ6PackedRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PackedResult>,
}

struct MulMatQ6PackedMatrixX4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PackedResult>,
}

struct MulMatArgmaxQ6PackedRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PackedArgmaxResult>,
}

struct MulMatQ6PreparedRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PreparedResult>,
}

struct MulMatQ6PreparedMatrixX4Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PreparedResult>,
}

struct MulMatQ6PreparedMatrixX8Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [u8],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6PreparedResult>,
}

struct MulMatQ6VectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q6VectorResult>,
}

struct MulMatQ6Runtime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    n: usize,
    result: &'dispatch Cell<Q6VectorResult>,
}

struct MulMatQ4KVectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q4KResult>,
}

struct MulMatQ4KRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    n: usize,
    result: &'dispatch Cell<Q4KResult>,
}

struct MulMatQ2KVectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q2KResult>,
}

struct MulMatQ2KRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    n: usize,
    result: &'dispatch Cell<Q2KResult>,
}

struct MulMatQ3KVectorRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<Q3KResult>,
}

struct MulMatQ3KRuntime<'dispatch> {
    lhs: &'dispatch [u8],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    n: usize,
    result: &'dispatch Cell<Q3KResult>,
}

struct ConvTransposeRuntime<'dispatch> {
    weights: &'dispatch [f32],
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    shape: ConvTranspose1dF32Shape,
    result: &'dispatch Cell<ConvTranspose1dF32Result>,
}

struct ConvTransposeF16Runtime<'dispatch> {
    event: OpAarch64ConvTranspose1dF16<'dispatch>,
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<ConvTranspose1dF32Result>,
}

struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<Result<(), KernelError>>,
}

struct GenericRuntime<'dispatch> {
    event: GenericEvent<'dispatch>,
    result: &'dispatch Cell<Result<(), crate::any::Error>>,
}

struct GetRowsRuntime<'dispatch> {
    event: OpGetRows<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsF16Runtime<'dispatch> {
    event: OpGetRowsF16<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsBf16Runtime<'dispatch> {
    event: OpGetRowsBf16<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsQ4_0Runtime<'dispatch> {
    event: OpGetRowsQ4_0<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsQ8_0Runtime<'dispatch> {
    event: OpGetRowsQ8_0<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsQ4KRuntime<'dispatch> {
    event: OpGetRowsQ4K<'dispatch>,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct GetRowsUnexpectedRuntime<'dispatch> {
    event: UnexpectedGetRows,
    result: &'dispatch Cell<GetRowsOutcome>,
}

struct Im2ColRuntime<'dispatch> {
    event: OpIm2Col<'dispatch>,
    result: &'dispatch Cell<Im2ColResult>,
}

struct Im2ColF16Runtime<'dispatch> {
    event: OpIm2ColF16<'dispatch>,
    result: &'dispatch Cell<Im2ColResult>,
}

struct Im2ColUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<Im2ColResult>,
}

struct RopeRuntime<'dispatch> {
    event: OpRope<'dispatch>,
    result: &'dispatch Cell<Result<(), RopeError>>,
}

struct RopeUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<Result<(), RopeError>>,
}

struct SoftMaxRuntime<'dispatch> {
    event: OpSoftMax<'dispatch>,
    result: &'dispatch Cell<ReductionResult>,
}

struct SoftMaxUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<ReductionResult>,
}

struct ScalarLogRuntime<'dispatch> {
    event: OpScalarLog<'dispatch>,
    result: &'dispatch Cell<ScalarTrigResult>,
}

struct ScalarSinRuntime<'dispatch> {
    event: OpScalarSin<'dispatch>,
    result: &'dispatch Cell<ScalarTrigResult>,
}

struct ScalarCosRuntime<'dispatch> {
    event: OpScalarCos<'dispatch>,
    result: &'dispatch Cell<ScalarTrigResult>,
}

struct ScalarTrigUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<ScalarTrigResult>,
}

struct F16MatmulRuntime<'dispatch> {
    event: OpScalarMulMatF16<'dispatch>,
    result: &'dispatch Cell<F16MatmulResult>,
}

struct F16MatmulUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<F16MatmulResult>,
}

struct NormRuntime<'dispatch> {
    event: OpNorm<'dispatch>,
    result: &'dispatch Cell<NormalizationResult>,
}

struct RmsNormRuntime<'dispatch> {
    event: OpRmsNorm<'dispatch>,
    result: &'dispatch Cell<NormalizationResult>,
}

struct NormalizationUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<NormalizationResult>,
}

struct Context {
    dup: DupF32Kernel,
    binary: BinaryF32Kernel,
    power: PowerF32Kernel,
    unary: UnaryF32Kernel,
    scalar_unary: UnaryKernel,
    broadcast: BroadcastF32Kernel,
    gemv: F32GemvKernel,
    mul_mat_f32: F32MulMatKernel,
    q4_0: Q4_0VectorKernel,
    q4_1: Q4_1VectorKernel,
    q5_0: Q5_0VectorKernel,
    q8_0: Q8_0VectorKernel,
    q8_0_packed: Q8_0PackedBl4Kernel,
    q4_packed: Q4PackedBl4Kernel,
    q6: Q6VectorKernel,
    q6_packed: Q6PackedKernel,
    q6_prepared: Q6PreparedKernel,
    q4_k: Q4KKernel,
    q2_k: Q2KKernel,
    q3_k: Q3KKernel,
    conv_transpose: ConvTranspose1dF32Kernel,
    get_rows: GetRowsKernel,
    im2col: Im2ColKernel,
    rope: RopeKernel,
    reductions: ReductionKernel,
    f16_matmul: F16MatmulKernel,
    f16_matmul_vector: Aarch64F16MatmulKernel,
    f16_vector_available: bool,
    normalization: NormalizationKernel,
    generic: crate::any::Kernel,
}

sml! {
    KernelMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Dup(DupRuntime<'dispatch>) / effect_dup,
        "ready"_s <= "ready"_s + BinaryAdd(BinaryAddRuntime<'dispatch>) / effect_binary_add,
        "ready"_s <= "ready"_s + BinarySub(BinarySubRuntime<'dispatch>) / effect_binary_sub,
        "ready"_s <= "ready"_s + BinaryMul(BinaryMulRuntime<'dispatch>) / effect_binary_mul,
        "ready"_s <= "ready"_s + BinaryDiv(BinaryDivRuntime<'dispatch>) / effect_binary_div,
        "ready"_s <= "ready"_s + PowerSqr(PowerSqrRuntime<'dispatch>) / effect_power_sqr,
        "ready"_s <= "ready"_s + PowerSqrt(PowerSqrtRuntime<'dispatch>) / effect_power_sqrt,
        "ready"_s <= "ready"_s + UnaryAbs(UnaryAbsRuntime<'dispatch>) / effect_unary_abs,
        "ready"_s <= "ready"_s + UnaryNeg(UnaryNegRuntime<'dispatch>) / effect_unary_neg,
        "ready"_s <= "ready"_s + UnaryRelu(UnaryReluRuntime<'dispatch>) / effect_unary_relu,
        "ready"_s <= "ready"_s + UnarySilu(UnarySiluRuntime<'dispatch>) / effect_unary_silu,

        "ready"_s <= "ready"_s + ScalarExp(ScalarExpRuntime<'dispatch>) [guard_scalar_exp_valid] / effect_scalar_exp,
        "ready"_s <= "ready"_s + ScalarExp(ScalarExpRuntime<'dispatch>) [guard_scalar_exp_shape] / effect_scalar_exp_shape,
        "ready"_s <= "ready"_s + ScalarExp(ScalarExpRuntime<'dispatch>) [guard_scalar_exp_view] / effect_scalar_exp_view,

        "ready"_s <= "ready"_s + ScalarTanh(ScalarTanhRuntime<'dispatch>) [guard_scalar_tanh_valid] / effect_scalar_tanh,
        "ready"_s <= "ready"_s + ScalarTanh(ScalarTanhRuntime<'dispatch>) [guard_scalar_tanh_shape] / effect_scalar_tanh_shape,
        "ready"_s <= "ready"_s + ScalarTanh(ScalarTanhRuntime<'dispatch>) [guard_scalar_tanh_view] / effect_scalar_tanh_view,

        "ready"_s <= "ready"_s + ScalarElu(ScalarEluRuntime<'dispatch>) [guard_scalar_elu_valid] / effect_scalar_elu,
        "ready"_s <= "ready"_s + ScalarElu(ScalarEluRuntime<'dispatch>) [guard_scalar_elu_shape] / effect_scalar_elu_shape,
        "ready"_s <= "ready"_s + ScalarElu(ScalarEluRuntime<'dispatch>) [guard_scalar_elu_view] / effect_scalar_elu_view,

        "ready"_s <= "ready"_s + ScalarGelu(ScalarGeluRuntime<'dispatch>) [guard_scalar_gelu_valid] / effect_scalar_gelu,
        "ready"_s <= "ready"_s + ScalarGelu(ScalarGeluRuntime<'dispatch>) [guard_scalar_gelu_shape] / effect_scalar_gelu_shape,
        "ready"_s <= "ready"_s + ScalarGelu(ScalarGeluRuntime<'dispatch>) [guard_scalar_gelu_view] / effect_scalar_gelu_view,

        "ready"_s <= "ready"_s + ScalarSilu(ScalarSiluRuntime<'dispatch>) [guard_scalar_silu_valid] / effect_scalar_silu,
        "ready"_s <= "ready"_s + ScalarSilu(ScalarSiluRuntime<'dispatch>) [guard_scalar_silu_shape] / effect_scalar_silu_shape,
        "ready"_s <= "ready"_s + ScalarSilu(ScalarSiluRuntime<'dispatch>) [guard_scalar_silu_view] / effect_scalar_silu_view,

        "ready"_s <= "ready"_s + BroadcastAdd(BroadcastAddRuntime<'dispatch>) / effect_broadcast_add,
        "ready"_s <= "ready"_s + BroadcastMul(BroadcastMulRuntime<'dispatch>) / effect_broadcast_mul,
        "ready"_s <= "ready"_s + Gemv(GemvRuntime<'dispatch>) / effect_gemv,
        "ready"_s <= "ready"_s + MulMatF32(MulMatF32Runtime<'dispatch>) / effect_mul_mat_f32,
        "ready"_s <= "ready"_s + MulMatQ4_0Vector(MulMatQ4_0VectorRuntime<'dispatch>) / effect_q4_0_vector,
        "ready"_s <= "ready"_s + MulMatQ4_1Vector(MulMatQ4_1VectorRuntime<'dispatch>) / effect_q4_1_vector,
        "ready"_s <= "ready"_s + MulMatQ5_0Vector(MulMatQ5_0VectorRuntime<'dispatch>) / effect_q5_0_vector,
        "ready"_s <= "ready"_s + MulMatQ8_0Vector(MulMatQ8_0VectorRuntime<'dispatch>) / effect_q8_0_vector,
        "ready"_s <= "ready"_s + MulMatQ8_0VectorQ8Rhs(MulMatQ8_0VectorQ8RhsRuntime<'dispatch>) / effect_q8_0_vector_q8_rhs,
        "ready"_s <= "ready"_s + MulMatQ8_0PackedBl4(MulMatQ8_0PackedBl4Runtime<'dispatch>) / effect_q8_0_packed_bl4,
        "ready"_s <= "ready"_s + MulMatQ4PackedBl4(MulMatQ4PackedBl4Runtime<'dispatch>) / effect_q4_packed_bl4,
        "ready"_s <= "ready"_s + MulMatQ4PackedBl4MatrixX4(MulMatQ4PackedBl4MatrixX4Runtime<'dispatch>) / effect_q4_packed_bl4_matrix_x4,
        "ready"_s <= "ready"_s + MulMatQ4PackedBl8(MulMatQ4PackedBl8Runtime<'dispatch>) / effect_q4_packed_bl8,
        "ready"_s <= "ready"_s + MulMatQ4PackedF32Bl4(MulMatQ4PackedF32Bl4Runtime<'dispatch>) / effect_q4_packed_f32_bl4,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4PackedF32Bl4(MulMatArgmaxQ4PackedF32Bl4Runtime<'dispatch>) / effect_q4_packed_f32_bl4_argmax,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4PackedF32Bl8(MulMatArgmaxQ4PackedF32Bl8Runtime<'dispatch>) / effect_q4_packed_f32_bl8_argmax,
        "ready"_s <= "ready"_s + MulMatQ4PackedF32Bl8(MulMatQ4PackedF32Bl8Runtime<'dispatch>) / effect_q4_packed_f32_bl8,
        "ready"_s <= "ready"_s + MulMatQ4PackedBl8MatrixX4(MulMatQ4PackedBl8MatrixX4Runtime<'dispatch>) / effect_q4_packed_bl8_matrix_x4,
        "ready"_s <= "ready"_s + MulMatQ4PackedBl8MatrixX8(MulMatQ4PackedBl8MatrixX8Runtime<'dispatch>) / effect_q4_packed_bl8_matrix_x8,
        "ready"_s <= "ready"_s + MulMatQ8_0PackedBl8(MulMatQ8_0PackedBl8Runtime<'dispatch>) / effect_q8_0_packed_bl8,
        "ready"_s <= "ready"_s + MulMatQ8_0PackedBl8MatrixX4(MulMatQ8_0PackedBl8MatrixX4Runtime<'dispatch>) / effect_q8_0_packed_bl8_matrix_x4,
        "ready"_s <= "ready"_s + MulMatQ6Packed(MulMatQ6PackedRuntime<'dispatch>) / effect_q6_packed,
        "ready"_s <= "ready"_s + MulMatQ6PackedMatrixX4(MulMatQ6PackedMatrixX4Runtime<'dispatch>) / effect_q6_packed_matrix_x4,
        "ready"_s <= "ready"_s + MulMatArgmaxQ6Packed(MulMatArgmaxQ6PackedRuntime<'dispatch>) / effect_q6_packed_argmax,
        "ready"_s <= "ready"_s + MulMatQ6Prepared(MulMatQ6PreparedRuntime<'dispatch>) / effect_q6_prepared,
        "ready"_s <= "ready"_s + MulMatQ6PreparedMatrixX4(MulMatQ6PreparedMatrixX4Runtime<'dispatch>) / effect_q6_prepared_matrix_x4,
        "ready"_s <= "ready"_s + MulMatQ6PreparedMatrixX8(MulMatQ6PreparedMatrixX8Runtime<'dispatch>) / effect_q6_prepared_matrix_x8,
        "ready"_s <= "ready"_s + MulMatQ6Vector(MulMatQ6VectorRuntime<'dispatch>) / effect_q6_vector,
        "ready"_s <= "ready"_s + MulMatQ6(MulMatQ6Runtime<'dispatch>) / effect_q6_gemm,
        "ready"_s <= "ready"_s + MulMatQ4KVector(MulMatQ4KVectorRuntime<'dispatch>) / effect_q4_k_vector,
        "ready"_s <= "ready"_s + MulMatQ4K(MulMatQ4KRuntime<'dispatch>) / effect_q4_k_gemm,
        "ready"_s <= "ready"_s + MulMatQ2KVector(MulMatQ2KVectorRuntime<'dispatch>) / effect_q2_k_vector,
        "ready"_s <= "ready"_s + MulMatQ2K(MulMatQ2KRuntime<'dispatch>) / effect_q2_k_gemm,
        "ready"_s <= "ready"_s + MulMatQ3KVector(MulMatQ3KVectorRuntime<'dispatch>) / effect_q3_k_vector,
        "ready"_s <= "ready"_s + MulMatQ3K(MulMatQ3KRuntime<'dispatch>) / effect_q3_k_gemm,
        "ready"_s <= "ready"_s + ConvTranspose(ConvTransposeRuntime<'dispatch>) / effect_conv_transpose,
        "ready"_s <= "ready"_s + ConvTransposeF16(ConvTransposeF16Runtime<'dispatch>) / effect_conv_transpose_f16,
        "ready"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) / effect_get_rows,
        "ready"_s <= "ready"_s + GetRowsF16(GetRowsF16Runtime<'dispatch>) / effect_get_rows_f16,
        "ready"_s <= "ready"_s + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) / effect_get_rows_bf16,
        "ready"_s <= "ready"_s + GetRowsQ4_0(GetRowsQ4_0Runtime<'dispatch>) / effect_get_rows_q4_0,
        "ready"_s <= "ready"_s + GetRowsQ8_0(GetRowsQ8_0Runtime<'dispatch>) / effect_get_rows_q8_0,
        "ready"_s <= "ready"_s + GetRowsQ4K(GetRowsQ4KRuntime<'dispatch>) / effect_get_rows_q4k,
        "ready"_s <= "ready"_s + GetRowsUnexpected(GetRowsUnexpectedRuntime<'dispatch>) / effect_get_rows_unexpected,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>) / effect_im2col,
        "ready"_s <= "ready"_s + Im2ColF16(Im2ColF16Runtime<'dispatch>) / effect_im2col_f16,
        "ready"_s <= "ready"_s + Im2ColUnexpected(Im2ColUnexpectedRuntime<'dispatch>) / effect_im2col_unexpected,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>) [guard_rope_norm] / effect_rope_norm,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>) [guard_rope_neox] / effect_rope_neox,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>) [guard_rope_timestep] / effect_rope_timestep,
        "ready"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>) [guard_rope_invalid] / effect_rope_invalid,
        "ready"_s <= "ready"_s + RopeUnexpected(RopeUnexpectedRuntime<'dispatch>) / effect_rope_unexpected,
        "ready"_s <= "ready"_s + SoftMax(SoftMaxRuntime<'dispatch>) / effect_soft_max,
        "ready"_s <= "ready"_s + SoftMaxUnexpected(SoftMaxUnexpectedRuntime<'dispatch>) / effect_soft_max_unexpected,

        "ready"_s <= "ready"_s + ScalarLog(ScalarLogRuntime<'dispatch>) [guard_scalar_log_valid] / effect_scalar_log,
        "ready"_s <= "ready"_s + ScalarLog(ScalarLogRuntime<'dispatch>) [guard_scalar_log_invalid] / effect_scalar_log,
        "ready"_s <= "ready"_s + ScalarSin(ScalarSinRuntime<'dispatch>) [guard_scalar_sin_valid] / effect_scalar_sin,
        "ready"_s <= "ready"_s + ScalarSin(ScalarSinRuntime<'dispatch>) [guard_scalar_sin_invalid] / effect_scalar_sin,
        "ready"_s <= "ready"_s + ScalarCos(ScalarCosRuntime<'dispatch>) [guard_scalar_cos_valid] / effect_scalar_cos,
        "ready"_s <= "ready"_s + ScalarCos(ScalarCosRuntime<'dispatch>) [guard_scalar_cos_invalid] / effect_scalar_cos,
        "ready"_s <= "ready"_s + ScalarTrigUnexpected(ScalarTrigUnexpectedRuntime<'dispatch>) / effect_scalar_trig_unexpected,

        "ready"_s <= "ready"_s + F16Matmul(F16MatmulRuntime<'dispatch>) [guard_f16_matmul_vector] / effect_f16_matmul_vector,
        "ready"_s <= "ready"_s + F16Matmul(F16MatmulRuntime<'dispatch>) [guard_f16_matmul_scalar] / effect_f16_matmul,
        "ready"_s <= "ready"_s + F16Matmul(F16MatmulRuntime<'dispatch>) [guard_f16_matmul_invalid] / effect_f16_matmul,
        "ready"_s <= "ready"_s + F16MatmulUnexpected(F16MatmulUnexpectedRuntime<'dispatch>) / effect_f16_matmul_unexpected,

        "ready"_s <= "ready"_s + Norm(NormRuntime<'dispatch>) / effect_norm,
        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>) / effect_rms_norm,
        "ready"_s <= "ready"_s + NormalizationUnexpected(NormalizationUnexpectedRuntime<'dispatch>) / effect_normalization_unexpected,
        // The target envelope carries the same stable operation ordinal as the
        // pinned event list. Keep every operation's valid and invalid outcome
        // explicit in this target machine; the action only executes the path
        // selected by these guards through the owned generic child actor.
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_dup_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_dup_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_id_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_id_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add1_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add1_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_acc_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_acc_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sub_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sub_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_div_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_div_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sqr_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sqr_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sqrt_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sqrt_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_log_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_log_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sin_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sin_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cos_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cos_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sum_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sum_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sum_rows_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_sum_rows_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cumsum_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cumsum_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mean_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mean_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_argmax_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_argmax_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_count_equal_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_count_equal_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_repeat_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_repeat_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_repeat_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_repeat_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_concat_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_concat_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_silu_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_silu_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_norm_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_norm_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rms_norm_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rms_norm_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rms_norm_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rms_norm_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_group_norm_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_group_norm_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_l2_norm_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_l2_norm_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_argmax_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_argmax_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_id_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_mul_mat_id_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_out_prod_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_out_prod_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_scale_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_scale_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_set_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_set_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cpy_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cpy_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cont_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cont_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_reshape_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_reshape_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_view_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_view_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_permute_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_permute_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_transpose_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_transpose_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rows_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rows_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rows_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rows_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_set_rows_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_set_rows_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_mask_inf_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_mask_inf_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_mask_zero_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_diag_mask_zero_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_soft_max_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_soft_max_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_soft_max_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_soft_max_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rope_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rope_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rope_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rope_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_clamp_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_clamp_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_transpose_1d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_transpose_1d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_3d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_im2col_3d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_2d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_2d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_3d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_3d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_2d_dw_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_2d_dw_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_transpose_2d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_conv_transpose_2d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_1d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_1d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_2d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_2d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_2d_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pool_2d_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_upscale_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_upscale_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pad_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pad_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pad_reflect_1d_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_pad_reflect_1d_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_roll_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_roll_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_arange_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_arange_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_timestep_embedding_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_timestep_embedding_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_argsort_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_argsort_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_top_k_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_top_k_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_leaky_relu_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_leaky_relu_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_tri_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_tri_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_fill_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_fill_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_flash_attn_ext_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_flash_attn_ext_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_flash_attn_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_flash_attn_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_ssm_conv_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_ssm_conv_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_ssm_scan_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_ssm_scan_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_win_part_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_win_part_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_win_unpart_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_win_unpart_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rel_pos_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_get_rel_pos_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_rel_pos_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_add_rel_pos_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rwkv_wkv6_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rwkv_wkv6_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_gated_linear_attn_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_gated_linear_attn_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rwkv_wkv7_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_rwkv_wkv7_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_solve_tri_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_solve_tri_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_unary_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_unary_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom1_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom1_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom2_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom2_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom3_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_map_custom3_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_custom_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_custom_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cross_entropy_loss_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cross_entropy_loss_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cross_entropy_loss_back_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_cross_entropy_loss_back_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_opt_step_adamw_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_opt_step_adamw_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_opt_step_sgd_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_opt_step_sgd_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_glu_valid] / effect_generic,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>) [guard_generic_glu_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Target-owned `AArch64` actor composing maintained NEON operation families.
pub struct Kernel {
    machine: KernelMachineStateMachine<Context>,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Kernel").finish_non_exhaustive()
    }
}

impl Kernel {
    /// Constructs the router after resolving every maintained child capability.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let f16_vector_available = f16_vector_supported();
        Some(Self {
            machine: KernelMachineStateMachine::new(Context {
                dup: DupF32Kernel::try_new()?,
                binary: BinaryF32Kernel::try_new()?,
                power: PowerF32Kernel::try_new()?,
                unary: UnaryF32Kernel::try_new()?,
                scalar_unary: UnaryKernel::new(),
                broadcast: BroadcastF32Kernel::try_new()?,
                gemv: F32GemvKernel::try_new()?,
                mul_mat_f32: F32MulMatKernel::try_new()?,
                q4_0: Q4_0VectorKernel::try_new()?,
                q4_1: Q4_1VectorKernel::try_new()?,
                q5_0: Q5_0VectorKernel::try_new()?,
                q8_0: Q8_0VectorKernel::try_new()?,
                q8_0_packed: Q8_0PackedBl4Kernel::try_new()?,
                q4_packed: Q4PackedBl4Kernel::try_new()?,
                q6: Q6VectorKernel::try_new()?,
                q6_packed: Q6PackedKernel::try_new()?,
                q6_prepared: Q6PreparedKernel::try_new()?,
                q4_k: Q4KKernel::try_new()?,
                q2_k: Q2KKernel::try_new()?,
                q3_k: Q3KKernel::try_new()?,
                conv_transpose: ConvTranspose1dF32Kernel::try_new()?,
                get_rows: GetRowsKernel::new(),
                im2col: Im2ColKernel::new(),
                rope: RopeKernel::new(),
                reductions: ReductionKernel::new(),
                f16_matmul: F16MatmulKernel::new(),
                f16_matmul_vector: Aarch64F16MatmulKernel::new(f16_vector_available),
                f16_vector_available,
                normalization: NormalizationKernel::new(),
                generic: crate::any::Kernel::new(),
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: KernelEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Dispatches one source-backed row-gather event while preserving its
    /// caller-owned tensor views through the target router.
    pub fn process_get_rows<E: GetRowsEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed F32 or F16-destination `im2col` event.
    pub fn process_im2col<E: Im2ColEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one explicitly guarded `RoPE` mode event.
    pub fn process_rope<E: RopeEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed softmax event through the reduction child.
    pub fn process_soft_max<E: SoftMaxEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed scalar-trigonometry event through the reduction child.
    pub fn process_scalar_trig<E: ScalarTrigEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed scalar F16 matrix event through the child actor.
    pub fn process_f16_matmul<E: F16MatmulEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed normalization event through the child.
    pub fn process_normalization<E: NormalizationEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one safe generic envelope through the target-owned actor.
    ///
    /// # Errors
    ///
    /// Returns the portable child's typed error when its guards reject the
    /// generic envelope.
    pub fn process_generic(&mut self, event: Aarch64Generic<'_>) -> Result<(), crate::any::Error> {
        event.dispatch(self, &mut [])
    }

    /// Reports whether the composed actor is ready for another event.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&KernelMachineStates::Ready)
    }
}

/// Event interface for the composed `AArch64` actor.
pub trait KernelEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output;
}

/// Event interface for target-owned row-gather routing.
pub trait GetRowsEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned `im2col` routing.
pub trait Im2ColEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned `RoPE` mode routing.
pub trait RopeEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned softmax routing.
pub trait SoftMaxEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned scalar `log`, `sin`, and `cos` routing.
pub trait ScalarTrigEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned scalar F16 matrix routing.
pub trait F16MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Event interface for target-owned normalization routing.
pub trait NormalizationEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// A target-owned wrapper for one safe generic kernel operation envelope.
#[derive(Debug)]
pub struct Aarch64Generic<'a> {
    event: GenericEvent<'a>,
}

impl<'a> Aarch64Generic<'a> {
    /// Creates a target generic event. Validation remains an SML guard.
    #[must_use]
    pub const fn new(event: GenericEvent<'a>) -> Self {
        Self { event }
    }
}

impl KernelEvent for Aarch64Generic<'_> {
    type Output = Result<(), crate::any::Error>;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(crate::any::Error::UnexpectedEvent));
        actor
            .machine
            .process_event(KernelMachineEvents::Generic(GenericRuntime {
                event: self.event,
                result: &result,
            }))
            .map_err(|_| crate::any::Error::Internal)?;
        result.get()
    }
}

macro_rules! impl_normalization_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl NormalizationEvent for $event {
            type Output = NormalizationResult;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                let result = Cell::new(Err(NormalizationError::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| NormalizationError::Internal)?;
                result.get()
            }
        }
    };
}

impl_normalization_event!(OpNorm<'_>, NormRuntime, Norm);
impl_normalization_event!(OpRmsNorm<'_>, RmsNormRuntime, RmsNorm);

impl NormalizationEvent for UnexpectedNormalization {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(NormalizationError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::NormalizationUnexpected(
                NormalizationUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| NormalizationError::Internal)?;
        result.get()
    }
}

impl SoftMaxEvent for OpSoftMax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::SoftMax(SoftMaxRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

impl SoftMaxEvent for UnexpectedSoftMax {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::SoftMaxUnexpected(
                SoftMaxUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

macro_rules! impl_scalar_trig_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl ScalarTrigEvent for $event {
            type Output = ScalarTrigResult;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                let result = Cell::new(Err(ReductionError::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| ReductionError::Internal)?;
                result.get()
            }
        }
    };
}

impl_scalar_trig_event!(OpScalarLog<'_>, ScalarLogRuntime, ScalarLog);
impl_scalar_trig_event!(OpScalarSin<'_>, ScalarSinRuntime, ScalarSin);
impl_scalar_trig_event!(OpScalarCos<'_>, ScalarCosRuntime, ScalarCos);

impl ScalarTrigEvent for UnexpectedScalarTrig {
    type Output = ScalarTrigResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::ScalarTrigUnexpected(
                ScalarTrigUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

impl F16MatmulEvent for OpScalarMulMatF16<'_> {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(F16MatmulError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::F16Matmul(F16MatmulRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

impl F16MatmulEvent for UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(F16MatmulError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::F16MatmulUnexpected(
                F16MatmulUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

impl RopeEvent for OpRope<'_> {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(RopeError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::Rope(RopeRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }
}

impl RopeEvent for UnexpectedRope {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(RopeError::Internal));
        let _ = self;
        actor
            .machine
            .process_event(KernelMachineEvents::RopeUnexpected(RopeUnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }
}

macro_rules! impl_im2col_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl Im2ColEvent for $event {
            type Output = Im2ColResult;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                let result = Cell::new(Err(Im2ColError::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| Im2ColError::Internal)?;
                result.get()
            }
        }
    };
}

impl_im2col_event!(OpIm2Col<'_>, Im2ColRuntime, Im2Col);
impl_im2col_event!(OpIm2ColF16<'_>, Im2ColF16Runtime, Im2ColF16);

impl Im2ColEvent for UnexpectedIm2Col {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(Im2ColError::Internal));
        let _ = self;
        actor
            .machine
            .process_event(KernelMachineEvents::Im2ColUnexpected(
                Im2ColUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }
}

macro_rules! impl_get_rows_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl GetRowsEvent for $event {
            type Output = GetRowsOutcome;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                let result = Cell::new(Err(GetRowsError::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| GetRowsError::Internal)?;
                result.get()
            }
        }
    };
}

impl_get_rows_event!(OpGetRows<'_>, GetRowsRuntime, GetRows);
impl_get_rows_event!(OpGetRowsF16<'_>, GetRowsF16Runtime, GetRowsF16);
impl_get_rows_event!(OpGetRowsBf16<'_>, GetRowsBf16Runtime, GetRowsBf16);
impl_get_rows_event!(OpGetRowsQ4_0<'_>, GetRowsQ4_0Runtime, GetRowsQ4_0);
impl_get_rows_event!(OpGetRowsQ8_0<'_>, GetRowsQ8_0Runtime, GetRowsQ8_0);
impl_get_rows_event!(OpGetRowsQ4K<'_>, GetRowsQ4KRuntime, GetRowsQ4K);

impl GetRowsEvent for UnexpectedGetRows {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        let result = Cell::new(Err(GetRowsError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::GetRowsUnexpected(
                GetRowsUnexpectedRuntime {
                    event: self,
                    result: &result,
                },
            ))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }
}

impl KernelEvent for Dup<'_> {
    type Output = DupF32Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(DupF32Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::Dup(DupRuntime {
                input: self.input,
                output,
                result: &result,
            }))
            .map_err(|_| DupF32Error::Internal)?;
        result.get()
    }
}

macro_rules! impl_binary_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl KernelEvent for $event {
            type Output = BinaryF32Result;

            fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(BinaryF32Error::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        lhs: self.lhs,
                        rhs: self.rhs,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| BinaryF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_binary_event!(BinaryAdd<'_>, BinaryAddRuntime, BinaryAdd);
impl_binary_event!(BinarySub<'_>, BinarySubRuntime, BinarySub);
impl_binary_event!(BinaryMul<'_>, BinaryMulRuntime, BinaryMul);
impl_binary_event!(BinaryDiv<'_>, BinaryDivRuntime, BinaryDiv);

macro_rules! impl_power_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl KernelEvent for $event {
            type Output = PowerF32Result;

            fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(PowerF32Error::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| PowerF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_power_event!(PowerSqr<'_>, PowerSqrRuntime, PowerSqr);
impl_power_event!(PowerSqrt<'_>, PowerSqrtRuntime, PowerSqrt);

macro_rules! impl_unary_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl KernelEvent for $event {
            type Output = UnaryF32Result;

            fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(UnaryF32Error::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| UnaryF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_unary_event!(UnaryAbs<'_>, UnaryAbsRuntime, UnaryAbs);
impl_unary_event!(UnaryNeg<'_>, UnaryNegRuntime, UnaryNeg);
impl_unary_event!(UnaryRelu<'_>, UnaryReluRuntime, UnaryRelu);
impl_unary_event!(UnarySilu<'_>, UnarySiluRuntime, UnarySilu);

macro_rules! impl_scalar_unary_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl KernelEvent for $event {
            type Output = UnaryResult;

            fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(UnaryError::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        input: self.input(),
                        output,
                        result: &result,
                    }))
                    .map_err(|_| UnaryError::Internal)?;
                result.get()
            }
        }
    };
}

impl_scalar_unary_event!(OpScalarUnaryExp<'_>, ScalarExpRuntime, ScalarExp);
impl_scalar_unary_event!(OpScalarUnaryTanh<'_>, ScalarTanhRuntime, ScalarTanh);
impl_scalar_unary_event!(OpScalarUnaryElu<'_>, ScalarEluRuntime, ScalarElu);
impl_scalar_unary_event!(OpScalarUnaryGelu<'_>, ScalarGeluRuntime, ScalarGelu);
impl_scalar_unary_event!(OpScalarUnarySilu<'_>, ScalarSiluRuntime, ScalarSilu);

macro_rules! impl_broadcast_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl KernelEvent for $event {
            type Output = BroadcastF32Result;

            fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(BroadcastF32Error::Internal));
                actor
                    .machine
                    .process_event(KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        row: self.row,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| BroadcastF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_broadcast_event!(BroadcastAdd<'_>, BroadcastAddRuntime, BroadcastAdd);
impl_broadcast_event!(BroadcastMul<'_>, BroadcastMulRuntime, BroadcastMul);

impl KernelEvent for MulMatF32<'_> {
    type Output = F32MulMatResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(F32MulMatError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatF32(MulMatF32Runtime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                n: self.n,
                result: &result,
            }))
            .map_err(|_| F32MulMatError::Internal)?;
        result.get()
    }
}

impl KernelEvent for Gemv<'_> {
    type Output = F32GemvResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(F32GemvError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::Gemv(GemvRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                result: &result,
            }))
            .map_err(|_| F32GemvError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4_0Vector<'_> {
    type Output = Q4_0VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4_0VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4_0Vector(
                MulMatQ4_0VectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4_0VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4_1Vector<'_> {
    type Output = Q4_1VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4_1VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4_1Vector(
                MulMatQ4_1VectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4_1VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ5_0Vector<'_> {
    type Output = Q5_0VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q5_0VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ5_0Vector(
                MulMatQ5_0VectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q5_0VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ8_0Vector<'_> {
    type Output = Q8_0VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ8_0Vector(
                MulMatQ8_0VectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ8_0VectorQ8Rhs<'_> {
    type Output = Q8_0VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ8_0VectorQ8Rhs(
                MulMatQ8_0VectorQ8RhsRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ8_0PackedBl4<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ8_0PackedBl4(
                MulMatQ8_0PackedBl4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedBl4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedBl4(
                MulMatQ4PackedBl4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedBl4MatrixX4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedBl4MatrixX4(
                MulMatQ4PackedBl4MatrixX4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedBl8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedBl8(
                MulMatQ4PackedBl8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedF32Bl4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedF32Bl4(
                MulMatQ4PackedF32Bl4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatArgmaxQ4PackedF32Bl4<'_> {
    type Output = Q4PackedF32Bl4ArgmaxResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatArgmaxQ4PackedF32Bl4(
                MulMatArgmaxQ4PackedF32Bl4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatArgmaxQ4PackedF32Bl8<'_> {
    type Output = Q4PackedF32Bl8ArgmaxResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatArgmaxQ4PackedF32Bl8(
                MulMatArgmaxQ4PackedF32Bl8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedF32Bl8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedF32Bl8(
                MulMatQ4PackedF32Bl8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedBl8MatrixX4<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedBl8MatrixX4(
                MulMatQ4PackedBl8MatrixX4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4PackedBl8MatrixX8<'_> {
    type Output = Q4PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4PackedBl8MatrixX8(
                MulMatQ4PackedBl8MatrixX8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ8_0PackedBl8<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ8_0PackedBl8(
                MulMatQ8_0PackedBl8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ8_0PackedBl8MatrixX4<'_> {
    type Output = Q8_0PackedBl4Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q8_0PackedBl4Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ8_0PackedBl8MatrixX4(
                MulMatQ8_0PackedBl8MatrixX4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q8_0PackedBl4Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4KVector<'_> {
    type Output = Q4KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4KVector(
                MulMatQ4KVectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q4KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ4K<'_> {
    type Output = Q4KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q4KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ4K(MulMatQ4KRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                n: self.n,
                result: &result,
            }))
            .map_err(|_| Q4KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ2KVector<'_> {
    type Output = Q2KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q2KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ2KVector(
                MulMatQ2KVectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q2KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ2K<'_> {
    type Output = Q2KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q2KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ2K(MulMatQ2KRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                n: self.n,
                result: &result,
            }))
            .map_err(|_| Q2KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ3KVector<'_> {
    type Output = Q3KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q3KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ3KVector(
                MulMatQ3KVectorRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q3KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ3K<'_> {
    type Output = Q3KResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q3KError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ3K(MulMatQ3KRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                n: self.n,
                result: &result,
            }))
            .map_err(|_| Q3KError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6<'_> {
    type Output = Q6VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6(MulMatQ6Runtime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                n: self.n,
                result: &result,
            }))
            .map_err(|_| Q6VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6Packed<'_> {
    type Output = Q6PackedResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PackedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6Packed(MulMatQ6PackedRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                result: &result,
            }))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6PackedMatrixX4<'_> {
    type Output = Q6PackedResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PackedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6PackedMatrixX4(
                MulMatQ6PackedMatrixX4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatArgmaxQ6Packed<'_> {
    type Output = Q6PackedArgmaxResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PackedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatArgmaxQ6Packed(
                MulMatArgmaxQ6PackedRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PackedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6Prepared<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PreparedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6Prepared(
                MulMatQ6PreparedRuntime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6PreparedMatrixX4<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PreparedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6PreparedMatrixX4(
                MulMatQ6PreparedMatrixX4Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6PreparedMatrixX8<'_> {
    type Output = Q6PreparedResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6PreparedError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6PreparedMatrixX8(
                MulMatQ6PreparedMatrixX8Runtime {
                    lhs: self.lhs,
                    rhs: self.rhs,
                    output,
                    m: self.m,
                    k: self.k,
                    result: &result,
                },
            ))
            .map_err(|_| Q6PreparedError::Internal)?;
        result.get()
    }
}

impl KernelEvent for MulMatQ6Vector<'_> {
    type Output = Q6VectorResult;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(Q6VectorError::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::MulMatQ6Vector(MulMatQ6VectorRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                result: &result,
            }))
            .map_err(|_| Q6VectorError::Internal)?;
        result.get()
    }
}

impl KernelEvent for ConvTranspose1d<'_> {
    type Output = ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(ConvTranspose1dF32Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::ConvTranspose(ConvTransposeRuntime {
                weights: self.weights,
                input: self.input,
                output,
                shape: self.shape,
                result: &result,
            }))
            .map_err(|_| ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for OpAarch64ConvTranspose1dF16<'_> {
    type Output = ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(ConvTranspose1dF32Error::Internal));
        actor
            .machine
            .process_event(KernelMachineEvents::ConvTransposeF16(
                ConvTransposeF16Runtime {
                    event: self,
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

impl KernelEvent for UnexpectedAarch64Kernel {
    type Output = Result<(), KernelError>;

    fn dispatch(self, actor: &mut Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(KernelError::UnexpectedEvent));
        actor
            .machine
            .process_event(KernelMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| KernelError::Internal)?;
        result.get()
    }
}

macro_rules! impl_scalar_context {
    (
        $valid:ident,
        $shape:ident,
        $view:ident,
        $effect:ident,
        $effect_shape:ident,
        $effect_view:ident,
        $runtime:ident,
        $event:ident
    ) => {
        fn $valid(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(dense_request_valid(event.input, event.output))
        }

        fn $shape(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(dense_request_shape(event.input, event.output))
        }

        fn $view(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(dense_request_view(event.input, event.output))
        }

        fn $effect(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            let layout = dense_layout(event.input.len());
            let input = TensorView::new(event.input, layout);
            let output = TensorViewMut::new(event.output, layout);
            event
                .result
                .set(self.scalar_unary.process_event($event::new(input, output)));
            Ok(())
        }

        fn $effect_shape(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(UnaryError::ShapeMismatch));
            Ok(())
        }

        fn $effect_view(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(UnaryError::InvalidView));
            Ok(())
        }
    };
}

impl KernelMachineStateMachineContext for Context {
    fn effect_dup(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.dup
                .process_event(OpAarch64DupF32::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_binary_add(&mut self, event: BinaryAddRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpAarch64BinaryAdd::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_sub(&mut self, event: BinarySubRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpAarch64BinarySub::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_mul(&mut self, event: BinaryMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpAarch64BinaryMul::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_div(&mut self, event: BinaryDivRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpAarch64BinaryDiv::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_power_sqr(&mut self, event: PowerSqrRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.power
                .process_event(OpAarch64PowerSqr::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_power_sqrt(&mut self, event: PowerSqrtRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.power
                .process_event(OpAarch64PowerSqrt::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_abs(&mut self, event: UnaryAbsRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpAarch64UnaryAbs::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_neg(&mut self, event: UnaryNegRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpAarch64UnaryNeg::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_relu(&mut self, event: UnaryReluRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpAarch64UnaryRelu::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_silu(&mut self, event: UnarySiluRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpAarch64UnarySilu::new(event.input), event.output),
        );
        Ok(())
    }

    impl_scalar_context!(
        guard_scalar_exp_valid,
        guard_scalar_exp_shape,
        guard_scalar_exp_view,
        effect_scalar_exp,
        effect_scalar_exp_shape,
        effect_scalar_exp_view,
        ScalarExpRuntime,
        OpExp
    );
    impl_scalar_context!(
        guard_scalar_tanh_valid,
        guard_scalar_tanh_shape,
        guard_scalar_tanh_view,
        effect_scalar_tanh,
        effect_scalar_tanh_shape,
        effect_scalar_tanh_view,
        ScalarTanhRuntime,
        OpTanh
    );
    impl_scalar_context!(
        guard_scalar_elu_valid,
        guard_scalar_elu_shape,
        guard_scalar_elu_view,
        effect_scalar_elu,
        effect_scalar_elu_shape,
        effect_scalar_elu_view,
        ScalarEluRuntime,
        OpElu
    );
    impl_scalar_context!(
        guard_scalar_gelu_valid,
        guard_scalar_gelu_shape,
        guard_scalar_gelu_view,
        effect_scalar_gelu,
        effect_scalar_gelu_shape,
        effect_scalar_gelu_view,
        ScalarGeluRuntime,
        OpGelu
    );
    impl_scalar_context!(
        guard_scalar_silu_valid,
        guard_scalar_silu_shape,
        guard_scalar_silu_view,
        effect_scalar_silu,
        effect_scalar_silu_shape,
        effect_scalar_silu_view,
        ScalarSiluRuntime,
        OpSilu
    );

    fn effect_broadcast_add(&mut self, event: BroadcastAddRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.broadcast.process_event(
            OpAarch64BroadcastAdd::new(event.input, event.row),
            event.output,
        ));
        Ok(())
    }

    fn effect_broadcast_mul(&mut self, event: BroadcastMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.broadcast.process_event(
            OpAarch64BroadcastMul::new(event.input, event.row),
            event.output,
        ));
        Ok(())
    }

    fn effect_mul_mat_f32(&mut self, event: MulMatF32Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.mul_mat_f32.process_event(
            OpMulMatF32::new(event.lhs, event.rhs, event.m, event.k, event.n),
            event.output,
        ));
        Ok(())
    }

    fn effect_gemv(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.gemv.process_event(
            OpF32Gemv::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_0_vector(&mut self, event: MulMatQ4_0VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_0.process_event(
            OpMulMatQ4_0Vector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_1_vector(&mut self, event: MulMatQ4_1VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_1.process_event(
            OpMulMatQ4_1Vector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q5_0_vector(&mut self, event: MulMatQ5_0VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q5_0.process_event(
            OpMulMatQ5_0Vector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q8_0_vector(&mut self, event: MulMatQ8_0VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q8_0.process_event(
            OpMulMatQ8_0Vector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q8_0_vector_q8_rhs(
        &mut self,
        event: MulMatQ8_0VectorQ8RhsRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q8_0.process_event(
            OpMulMatQ8_0VectorQ8Rhs::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q8_0_packed_bl4(&mut self, event: MulMatQ8_0PackedBl4Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.q8_0_packed.process_event(
            OpMulMatQ8_0PackedBl4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_bl4(&mut self, event: MulMatQ4PackedBl4Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedBl4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_bl4_matrix_x4(
        &mut self,
        event: MulMatQ4PackedBl4MatrixX4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedBl4MatrixX4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_bl8(&mut self, event: MulMatQ4PackedBl8Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedBl8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_f32_bl4(
        &mut self,
        event: MulMatQ4PackedF32Bl4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedF32Bl4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_f32_bl4_argmax(
        &mut self,
        event: MulMatArgmaxQ4PackedF32Bl4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatArgmaxQ4PackedF32Bl4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_f32_bl8_argmax(
        &mut self,
        event: MulMatArgmaxQ4PackedF32Bl8Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatArgmaxQ4PackedF32Bl8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_f32_bl8(
        &mut self,
        event: MulMatQ4PackedF32Bl8Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedF32Bl8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_bl8_matrix_x4(
        &mut self,
        event: MulMatQ4PackedBl8MatrixX4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedBl8MatrixX4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_packed_bl8_matrix_x8(
        &mut self,
        event: MulMatQ4PackedBl8MatrixX8Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q4_packed.process_event(
            OpMulMatQ4PackedBl8MatrixX8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q8_0_packed_bl8(&mut self, event: MulMatQ8_0PackedBl8Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.q8_0_packed.process_event(
            OpMulMatQ8_0PackedBl8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q8_0_packed_bl8_matrix_x4(
        &mut self,
        event: MulMatQ8_0PackedBl8MatrixX4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q8_0_packed.process_event(
            OpMulMatQ8_0PackedBl8MatrixX4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_packed(&mut self, event: MulMatQ6PackedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q6_packed.process_event(
            OpMulMatQ6Packed::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_packed_matrix_x4(
        &mut self,
        event: MulMatQ6PackedMatrixX4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q6_packed.process_event(
            OpMulMatQ6PackedMatrixX4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_packed_argmax(
        &mut self,
        event: MulMatArgmaxQ6PackedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q6_packed.process_event(
            OpMulMatArgmaxQ6Packed::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_prepared(&mut self, event: MulMatQ6PreparedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q6_prepared.process_event(
            OpMulMatQ6Prepared::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_prepared_matrix_x4(
        &mut self,
        event: MulMatQ6PreparedMatrixX4Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q6_prepared.process_event(
            OpMulMatQ6PreparedMatrixX4::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_prepared_matrix_x8(
        &mut self,
        event: MulMatQ6PreparedMatrixX8Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.q6_prepared.process_event(
            OpMulMatQ6PreparedMatrixX8::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_vector(&mut self, event: MulMatQ6VectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q6.process_event(
            OpMulMatQ6Vector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q6_gemm(&mut self, event: MulMatQ6Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.q6.process_event(
            OpMulMatQ6::new(event.lhs, event.rhs, event.m, event.k, event.n),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_k_vector(&mut self, event: MulMatQ4KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_k.process_event(
            OpMulMatQ4KVector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q4_k_gemm(&mut self, event: MulMatQ4KRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q4_k.process_event(
            OpMulMatQ4K::new(event.lhs, event.rhs, event.m, event.k, event.n),
            event.output,
        ));
        Ok(())
    }

    fn effect_q2_k_vector(&mut self, event: MulMatQ2KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q2_k.process_event(
            OpMulMatQ2KVector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q2_k_gemm(&mut self, event: MulMatQ2KRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q2_k.process_event(
            OpMulMatQ2K::new(event.lhs, event.rhs, event.m, event.k, event.n),
            event.output,
        ));
        Ok(())
    }

    fn effect_q3_k_vector(&mut self, event: MulMatQ3KVectorRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q3_k.process_event(
            OpMulMatQ3KVector::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_q3_k_gemm(&mut self, event: MulMatQ3KRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.q3_k.process_event(
            OpMulMatQ3K::new(event.lhs, event.rhs, event.m, event.k, event.n),
            event.output,
        ));
        Ok(())
    }

    fn effect_conv_transpose(&mut self, event: ConvTransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.conv_transpose.process_event(
            OpAarch64ConvTranspose1dF32::new(event.weights, event.input, event.shape),
            event.output,
        ));
        Ok(())
    }

    fn effect_conv_transpose_f16(&mut self, event: ConvTransposeF16Runtime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.conv_transpose.process_event(event.event, event.output));
        Ok(())
    }

    fn effect_get_rows(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_f16(&mut self, event: GetRowsF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_bf16(&mut self, event: GetRowsBf16Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_q4_0(&mut self, event: GetRowsQ4_0Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_q8_0(&mut self, event: GetRowsQ8_0Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_q4k(&mut self, event: GetRowsQ4KRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_unexpected(
        &mut self,
        event: GetRowsUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_im2col(&mut self, event: Im2ColRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.im2col.process_event(event.event));
        Ok(())
    }

    fn effect_im2col_f16(&mut self, event: Im2ColF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.im2col.process_event(event.event));
        Ok(())
    }

    fn effect_im2col_unexpected(&mut self, event: Im2ColUnexpectedRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.im2col.process_event(UnexpectedIm2Col));
        Ok(())
    }

    fn guard_rope_norm(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.mode() == crate::any::rope::ROPE_MODE_NORM)
    }

    fn guard_rope_neox(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.mode() == crate::any::rope::ROPE_MODE_NEOX)
    }

    fn guard_rope_timestep(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.mode() == crate::any::rope::ROPE_MODE_TIMESTEP)
    }

    fn guard_rope_invalid(&self, event: &RopeRuntime<'_>) -> Result<bool, ()> {
        let mode = event.event.mode();
        Ok(mode != crate::any::rope::ROPE_MODE_NORM
            && mode != crate::any::rope::ROPE_MODE_NEOX
            && mode != crate::any::rope::ROPE_MODE_TIMESTEP)
    }

    fn effect_rope_norm(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
        Ok(())
    }

    fn effect_rope_neox(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
        Ok(())
    }

    fn effect_rope_timestep(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
        Ok(())
    }

    fn effect_rope_invalid(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
        Ok(())
    }

    fn effect_rope_unexpected(&mut self, event: RopeUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(UnexpectedRope));
        Ok(())
    }

    fn effect_soft_max(&mut self, event: SoftMaxRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_soft_max_unexpected(
        &mut self,
        event: SoftMaxUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(ReductionError::UnexpectedEvent));
        Ok(())
    }

    fn guard_scalar_log_valid(&self, event: &ScalarLogRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.is_valid())
    }

    fn guard_scalar_log_invalid(&self, event: &ScalarLogRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.is_valid())
    }

    fn effect_scalar_log(&mut self, event: ScalarLogRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.reductions.process_event(event.event.into_reduction()));
        Ok(())
    }

    fn guard_scalar_sin_valid(&self, event: &ScalarSinRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.is_valid())
    }

    fn guard_scalar_sin_invalid(&self, event: &ScalarSinRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.is_valid())
    }

    fn effect_scalar_sin(&mut self, event: ScalarSinRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.reductions.process_event(event.event.into_reduction()));
        Ok(())
    }

    fn guard_scalar_cos_valid(&self, event: &ScalarCosRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.is_valid())
    }

    fn guard_scalar_cos_invalid(&self, event: &ScalarCosRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.is_valid())
    }

    fn effect_scalar_cos(&mut self, event: ScalarCosRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.reductions.process_event(event.event.into_reduction()));
        Ok(())
    }

    fn effect_scalar_trig_unexpected(
        &mut self,
        event: ScalarTrigUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(ReductionError::UnexpectedEvent));
        Ok(())
    }

    fn guard_f16_matmul_vector(&self, event: &F16MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(self.f16_vector_available && event.event.is_valid())
    }

    fn guard_f16_matmul_scalar(&self, event: &F16MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.f16_vector_available && event.event.is_valid())
    }

    fn guard_f16_matmul_invalid(&self, event: &F16MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.is_valid())
    }

    fn effect_f16_matmul(&mut self, event: F16MatmulRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.f16_matmul.process_event(event.event.into_reduction()));
        Ok(())
    }

    fn effect_f16_matmul_vector(&mut self, event: F16MatmulRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.f16_matmul_vector.process_event(event.event));
        Ok(())
    }

    fn effect_f16_matmul_unexpected(
        &mut self,
        event: F16MatmulUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(F16MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_norm(&mut self, event: NormRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.normalization.process_event(event.event));
        Ok(())
    }

    fn effect_rms_norm(&mut self, event: RmsNormRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.normalization.process_event(event.event));
        Ok(())
    }

    fn effect_normalization_unexpected(
        &mut self,
        event: NormalizationUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(NormalizationError::UnexpectedEvent));
        Ok(())
    }

    fn guard_generic_dup_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Dup
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_dup_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Dup
            && event.event.request().validate().is_err())
    }

    fn guard_generic_add_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Add
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_add_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Add
            && event.event.request().validate().is_err())
    }

    fn guard_generic_add_id_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::AddId
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_add_id_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::AddId
            && event.event.request().validate().is_err())
    }

    fn guard_generic_add1_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Add1
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_add1_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Add1
            && event.event.request().validate().is_err())
    }

    fn guard_generic_acc_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Acc
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_acc_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Acc
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sub_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sub
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sub_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sub
            && event.event.request().validate().is_err())
    }

    fn guard_generic_mul_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Mul
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_mul_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Mul
            && event.event.request().validate().is_err())
    }

    fn guard_generic_div_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Div
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_div_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Div
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sqr_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sqr
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sqr_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sqr
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sqrt_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sqrt
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sqrt_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sqrt
            && event.event.request().validate().is_err())
    }

    fn guard_generic_log_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Log
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_log_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Log
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sin_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sin
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sin_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sin
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cos_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cos
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_cos_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cos
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sum_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sum
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sum_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Sum
            && event.event.request().validate().is_err())
    }

    fn guard_generic_sum_rows_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SumRows
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_sum_rows_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SumRows
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cumsum_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cumsum
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_cumsum_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cumsum
            && event.event.request().validate().is_err())
    }

    fn guard_generic_mean_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Mean
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_mean_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Mean
            && event.event.request().validate().is_err())
    }

    fn guard_generic_argmax_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Argmax
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_argmax_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Argmax
            && event.event.request().validate().is_err())
    }

    fn guard_generic_count_equal_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::CountEqual
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_count_equal_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::CountEqual
            && event.event.request().validate().is_err())
    }

    fn guard_generic_repeat_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Repeat
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_repeat_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Repeat
            && event.event.request().validate().is_err())
    }

    fn guard_generic_repeat_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RepeatBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_repeat_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RepeatBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_concat_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Concat
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_concat_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Concat
            && event.event.request().validate().is_err())
    }

    fn guard_generic_silu_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SiluBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_silu_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SiluBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Norm
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_norm_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Norm
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rms_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RmsNorm
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rms_norm_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RmsNorm
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rms_norm_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RmsNormBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rms_norm_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RmsNormBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_group_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GroupNorm
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_group_norm_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GroupNorm
            && event.event.request().validate().is_err())
    }

    fn guard_generic_l2_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::L2Norm
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_l2_norm_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::L2Norm
            && event.event.request().validate().is_err())
    }

    fn guard_generic_mul_mat_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMat
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_mul_mat_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMat
            && event.event.request().validate().is_err())
    }

    fn guard_generic_mul_mat_argmax_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMatArgmax
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_mul_mat_argmax_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMatArgmax
            && event.event.request().validate().is_err())
    }

    fn guard_generic_mul_mat_id_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMatId
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_mul_mat_id_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MulMatId
            && event.event.request().validate().is_err())
    }

    fn guard_generic_out_prod_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OutProd
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_out_prod_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OutProd
            && event.event.request().validate().is_err())
    }

    fn guard_generic_scale_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Scale
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_scale_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Scale
            && event.event.request().validate().is_err())
    }

    fn guard_generic_set_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Set
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_set_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Set
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cpy_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cpy
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_cpy_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cpy
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cont_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cont
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_cont_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Cont
            && event.event.request().validate().is_err())
    }

    fn guard_generic_reshape_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Reshape
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_reshape_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Reshape
            && event.event.request().validate().is_err())
    }

    fn guard_generic_view_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::View
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_view_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::View
            && event.event.request().validate().is_err())
    }

    fn guard_generic_permute_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Permute
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_permute_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Permute
            && event.event.request().validate().is_err())
    }

    fn guard_generic_transpose_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Transpose
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_transpose_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Transpose
            && event.event.request().validate().is_err())
    }

    fn guard_generic_get_rows_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRows
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_get_rows_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRows
            && event.event.request().validate().is_err())
    }

    fn guard_generic_get_rows_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRowsBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_get_rows_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRowsBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_set_rows_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SetRows
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_set_rows_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SetRows
            && event.event.request().validate().is_err())
    }

    fn guard_generic_diag_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Diag
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_diag_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Diag
            && event.event.request().validate().is_err())
    }

    fn guard_generic_diag_mask_inf_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::DiagMaskInf
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_diag_mask_inf_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::DiagMaskInf
            && event.event.request().validate().is_err())
    }

    fn guard_generic_diag_mask_zero_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::DiagMaskZero
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_diag_mask_zero_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::DiagMaskZero
            && event.event.request().validate().is_err())
    }

    fn guard_generic_soft_max_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SoftMax
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_soft_max_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SoftMax
            && event.event.request().validate().is_err())
    }

    fn guard_generic_soft_max_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SoftMaxBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_soft_max_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SoftMaxBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rope_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Rope
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rope_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Rope
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rope_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RopeBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rope_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RopeBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_clamp_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Clamp
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_clamp_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Clamp
            && event.event.request().validate().is_err())
    }

    fn guard_generic_conv_transpose_1d_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::ConvTranspose1d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_conv_transpose_1d_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::ConvTranspose1d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_im2col_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2Col
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_im2col_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2Col
            && event.event.request().validate().is_err())
    }

    fn guard_generic_im2col_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2ColBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_im2col_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2ColBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_im2col_3d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2Col3d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_im2col_3d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Im2Col3d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_conv_2d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv2d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_conv_2d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv2d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_conv_3d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv3d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_conv_3d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv3d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_conv_2d_dw_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv2dDw
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_conv_2d_dw_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Conv2dDw
            && event.event.request().validate().is_err())
    }

    fn guard_generic_conv_transpose_2d_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::ConvTranspose2d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_conv_transpose_2d_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::ConvTranspose2d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_pool_1d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool1d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_pool_1d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool1d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_pool_2d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool2d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_pool_2d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool2d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_pool_2d_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool2dBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_pool_2d_back_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pool2dBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_upscale_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Upscale
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_upscale_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Upscale
            && event.event.request().validate().is_err())
    }

    fn guard_generic_pad_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pad
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_pad_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Pad
            && event.event.request().validate().is_err())
    }

    fn guard_generic_pad_reflect_1d_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::PadReflect1d
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_pad_reflect_1d_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::PadReflect1d
            && event.event.request().validate().is_err())
    }

    fn guard_generic_roll_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Roll
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_roll_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Roll
            && event.event.request().validate().is_err())
    }

    fn guard_generic_arange_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Arange
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_arange_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Arange
            && event.event.request().validate().is_err())
    }

    fn guard_generic_timestep_embedding_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(
            event.event.operation() == KernelOperation::TimestepEmbedding
                && event.event.request().validate().is_ok(),
        )
    }

    fn guard_generic_timestep_embedding_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(
            event.event.operation() == KernelOperation::TimestepEmbedding
                && event.event.request().validate().is_err(),
        )
    }

    fn guard_generic_argsort_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Argsort
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_argsort_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Argsort
            && event.event.request().validate().is_err())
    }

    fn guard_generic_top_k_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::TopK
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_top_k_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::TopK
            && event.event.request().validate().is_err())
    }

    fn guard_generic_leaky_relu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::LeakyRelu
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_leaky_relu_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::LeakyRelu
            && event.event.request().validate().is_err())
    }

    fn guard_generic_tri_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Tri
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_tri_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Tri
            && event.event.request().validate().is_err())
    }

    fn guard_generic_fill_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Fill
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_fill_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Fill
            && event.event.request().validate().is_err())
    }

    fn guard_generic_flash_attn_ext_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::FlashAttnExt
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_flash_attn_ext_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::FlashAttnExt
            && event.event.request().validate().is_err())
    }

    fn guard_generic_flash_attn_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::FlashAttnBack
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_flash_attn_back_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::FlashAttnBack
            && event.event.request().validate().is_err())
    }

    fn guard_generic_ssm_conv_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SsmConv
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_ssm_conv_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SsmConv
            && event.event.request().validate().is_err())
    }

    fn guard_generic_ssm_scan_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SsmScan
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_ssm_scan_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SsmScan
            && event.event.request().validate().is_err())
    }

    fn guard_generic_win_part_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::WinPart
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_win_part_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::WinPart
            && event.event.request().validate().is_err())
    }

    fn guard_generic_win_unpart_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::WinUnpart
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_win_unpart_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::WinUnpart
            && event.event.request().validate().is_err())
    }

    fn guard_generic_get_rel_pos_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRelPos
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_get_rel_pos_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GetRelPos
            && event.event.request().validate().is_err())
    }

    fn guard_generic_add_rel_pos_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::AddRelPos
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_add_rel_pos_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::AddRelPos
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rwkv_wkv6_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RwkvWkv6
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rwkv_wkv6_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RwkvWkv6
            && event.event.request().validate().is_err())
    }

    fn guard_generic_gated_linear_attn_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GatedLinearAttn
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_gated_linear_attn_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::GatedLinearAttn
            && event.event.request().validate().is_err())
    }

    fn guard_generic_rwkv_wkv7_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RwkvWkv7
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_rwkv_wkv7_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::RwkvWkv7
            && event.event.request().validate().is_err())
    }

    fn guard_generic_solve_tri_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SolveTri
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_solve_tri_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::SolveTri
            && event.event.request().validate().is_err())
    }

    fn guard_generic_unary_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Unary
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_unary_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Unary
            && event.event.request().validate().is_err())
    }

    fn guard_generic_map_custom1_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom1
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_map_custom1_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom1
            && event.event.request().validate().is_err())
    }

    fn guard_generic_map_custom2_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom2
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_map_custom2_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom2
            && event.event.request().validate().is_err())
    }

    fn guard_generic_map_custom3_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom3
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_map_custom3_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::MapCustom3
            && event.event.request().validate().is_err())
    }

    fn guard_generic_custom_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Custom
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_custom_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Custom
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cross_entropy_loss_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::CrossEntropyLoss
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_cross_entropy_loss_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::CrossEntropyLoss
            && event.event.request().validate().is_err())
    }

    fn guard_generic_cross_entropy_loss_back_valid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(
            event.event.operation() == KernelOperation::CrossEntropyLossBack
                && event.event.request().validate().is_ok(),
        )
    }

    fn guard_generic_cross_entropy_loss_back_invalid(
        &self,
        event: &GenericRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(
            event.event.operation() == KernelOperation::CrossEntropyLossBack
                && event.event.request().validate().is_err(),
        )
    }

    fn guard_generic_opt_step_adamw_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OptStepAdamw
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_opt_step_adamw_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OptStepAdamw
            && event.event.request().validate().is_err())
    }

    fn guard_generic_opt_step_sgd_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OptStepSgd
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_opt_step_sgd_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::OptStepSgd
            && event.event.request().validate().is_err())
    }

    fn guard_generic_glu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Glu
            && event.event.request().validate().is_ok())
    }

    fn guard_generic_glu_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == KernelOperation::Glu
            && event.event.request().validate().is_err())
    }

    fn effect_generic(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.generic.process_event(event.event));
        Ok(())
    }

    fn effect_generic_invalid(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(crate::any::Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(KernelError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn dense_request_valid(input: &[f32], output: &[f32]) -> bool {
    !input.is_empty() && input.len() == output.len()
}

const fn dense_request_shape(input: &[f32], output: &[f32]) -> bool {
    !input.is_empty() && input.len() != output.len()
}

const fn dense_request_view(input: &[f32], _output: &[f32]) -> bool {
    input.is_empty()
}

const fn dense_layout(len: usize) -> Layout {
    Layout::new(DType::F32, [len as u64, 1, 1, 1], [4, 4, 4, 4])
}
