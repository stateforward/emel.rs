//! Target-owned x86 actor composition for the maintained kernel operation set.
//!
//! Target-specific actors own the optimized paths, while the safe generic
//! envelope preserves all 95 pinned operation ordinals as explicit SML
//! valid/invalid routes. Each operation remains a distinct typed event and
//! delegates through a public child actor process_event wrapper.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::large_stack_frames)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::f16_matmul::{X86F16MatmulKernel, f16_vector_supported};

use crate::any::event::GenericEvent;
use crate::any::f16_matmul::{F16MatmulError, F16MatmulKernel};
use crate::any::flash_attn::{FlashAttnError, FlashAttnKernel, FlashAttnResult, OpFlashAttnExt};
pub use crate::any::flash_attn::{FlashAttnOptions, OpFlashAttnExt as X86OpFlashAttnExt};
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
pub use crate::any::matmul::{
    MatmulArgmaxKernel, MatmulArgmaxResult, MatmulError, MatmulKernel, MatmulResult, OpMulMat,
    OpMulMatArgmax, OpMulMatArgmaxQ2K, OpMulMatArgmaxQ3K, OpMulMatArgmaxQ4_0, OpMulMatArgmaxQ4_1,
    OpMulMatArgmaxQ4K, OpMulMatArgmaxQ5_0, OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, OpMulMatQ2K,
    OpMulMatQ3K, OpMulMatQ4_0, OpMulMatQ4_1, OpMulMatQ4K, OpMulMatQ5_0, OpMulMatQ6K, OpMulMatQ8_0,
    QuantizedView,
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

/// Pinned source identity for the x86-64 target row-gather transition slice.
pub const TARGET_GET_ROWS_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for the row-gather slice.
pub const TARGET_GET_ROWS_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target `im2col` transition slice.
pub const TARGET_IM2COL_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for the `im2col` slice.
pub const TARGET_IM2COL_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target `RoPE` transition slice.
pub const TARGET_ROPE_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for the `RoPE` slice.
pub const TARGET_ROPE_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target softmax transition slice.
pub const TARGET_SOFTMAX_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for the softmax slice.
pub const TARGET_SOFTMAX_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target scalar-trigonometry slice.
pub const TARGET_SCALAR_TRIG_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for scalar trigonometry.
pub const TARGET_SCALAR_TRIG_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target scalar F16 matrix route.
pub const TARGET_F16_MATMUL_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for scalar F16 matrix multiplication.
pub const TARGET_F16_MATMUL_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Pinned source identity for the x86-64 target normalization transition slice.
pub const TARGET_NORMALIZATION_SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
/// Pinned x86-64 target state-machine blob for the normalization slice.
pub const TARGET_NORMALIZATION_SOURCE_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";

/// Explicit target normalization unexpected-event request.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedNormalization;

/// Explicit target softmax unexpected-event request.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedSoftMax;
use super::binary::{
    OpX86BinaryAdd, OpX86BinaryDiv, OpX86BinaryMul, OpX86BinarySub, X86BinaryF32Error,
    X86BinaryF32Kernel, X86BinaryF32Result,
};
use super::broadcast::{
    OpX86BroadcastAdd, OpX86BroadcastMul, X86BroadcastF32Error, X86BroadcastF32Kernel,
    X86BroadcastF32Result,
};
use super::conv_transpose_1d::{
    OpX86ConvTranspose1dF32, UnexpectedX86ConvTranspose1dF32, X86ConvTranspose1dF32Error,
    X86ConvTranspose1dF32Kernel, X86ConvTranspose1dF32Result, X86ConvTranspose1dF32Shape,
};
use super::dup::{OpX86DupF32, X86DupF32Error, X86DupF32Kernel, X86DupF32Result};
use super::fma::{F32FmaKernel, F32FmaResult, OpF32Fma};
use super::gemv::{OpX86F32Gemv, X86F32GemvError, X86F32GemvKernel, X86F32GemvResult};
use super::matmul::X86MatmulKernel;
use super::power::{
    OpX86PowerSqr, OpX86PowerSqrt, X86PowerF32Error, X86PowerF32Kernel, X86PowerF32Result,
};
use super::unary::{
    OpX86UnaryAbs, OpX86UnaryNeg, OpX86UnaryRelu, X86UnaryF32Error, X86UnaryF32Kernel,
    X86UnaryF32Result,
};
use crate::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use crate::any::unary::{
    OpElu, OpExp, OpGelu, OpSilu, OpTanh, UnaryError, UnaryKernel, UnaryResult,
};
pub use crate::detail::scalar_unary::{
    OpScalarUnaryElu, OpScalarUnaryExp, OpScalarUnaryGelu, OpScalarUnarySilu, OpScalarUnaryTanh,
    PINNED_DETAIL_SPAN, PINNED_EMEL_CPP_COMMIT, PINNED_X86_ACTION_BLOB, PINNED_X86_ACTION_SPAN,
    PINNED_X86_GUARD_BLOB, PINNED_X86_GUARD_SPAN, PINNED_X86_TRANSITION_BLOB,
    PINNED_X86_TRANSITION_SPAN,
};

/// A failure in the target router itself, distinct from child operation errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86KernelError {
    /// The generated target router rejected an unexpected event.
    UnexpectedEvent,
    /// The generated target router failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 target event"),
            Self::Internal => formatter.write_str("internal x86 target dispatch error"),
        }
    }
}

impl std::error::Error for X86KernelError {}

/// A typed unexpected-event request for the target router.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86Kernel;

/// Explicit unexpected-event request for the target F32/quantized matmul actor.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86Matmul;

/// Explicit unexpected-event request for the target matmul-argmax actor.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86MatmulArgmax;

/// Explicit unexpected-event request for the target flash-attention actor.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86FlashAttn;

macro_rules! define_unary_request {
    ($name:ident) => {
        /// A target x86 unary request over an immutable F32 input slice.
        #[derive(Debug)]
        pub struct $name<'a> {
            input: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; shape validation remains in the child actor.
            #[must_use]
            pub const fn new(input: &'a [f32]) -> Self {
                Self { input }
            }
        }
    };
}

macro_rules! define_binary_request {
    ($name:ident) => {
        /// A target x86 binary request over immutable F32 input slices.
        #[derive(Debug)]
        pub struct $name<'a> {
            lhs: &'a [f32],
            rhs: &'a [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a request; shape validation remains in the child actor.
            #[must_use]
            pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
                Self { lhs, rhs }
            }
        }
    };
}

define_unary_request!(X86Dup);
define_binary_request!(X86BinaryAdd);
define_binary_request!(X86BinarySub);
define_binary_request!(X86BinaryMul);
define_binary_request!(X86BinaryDiv);
define_unary_request!(X86PowerSqr);
define_unary_request!(X86PowerSqrt);
define_unary_request!(X86UnaryAbs);
define_unary_request!(X86UnaryNeg);
define_unary_request!(X86UnaryRelu);

/// A target x86 row-broadcast request over immutable F32 input slices.
#[derive(Debug)]
pub struct X86BroadcastAdd<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> X86BroadcastAdd<'a> {
    /// Creates a request; shape validation remains in the child actor.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// A target x86 row-broadcast request over immutable F32 input slices.
#[derive(Debug)]
pub struct X86BroadcastMul<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> X86BroadcastMul<'a> {
    /// Creates a request; shape validation remains in the child actor.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// A target x86 dense F32 matrix request with ggml's `[k,m]`/`[n,k]` layout.
#[derive(Debug)]
pub struct X86Fma<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    n: usize,
    k: usize,
}

impl<'a> X86Fma<'a> {
    /// Creates a request; shape validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, n: usize, k: usize) -> Self {
        Self { lhs, rhs, m, n, k }
    }
}

/// A target x86 F32 single-right-hand-side GEMV request.
#[derive(Debug)]
pub struct X86Gemv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> X86Gemv<'a> {
    /// Creates a request; shape validation remains in the child actor.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// A target x86 dense F32 transposed-convolution request.
#[derive(Debug)]
pub struct X86ConvTransposeF32<'a> {
    weights: &'a [f32],
    input: &'a [f32],
    shape: X86ConvTranspose1dF32Shape,
}

impl<'a> X86ConvTransposeF32<'a> {
    /// Creates a request; shape validation remains in the child actor.
    #[must_use]
    pub const fn new(
        weights: &'a [f32],
        input: &'a [f32],
        shape: X86ConvTranspose1dF32Shape,
    ) -> Self {
        Self {
            weights,
            input,
            shape,
        }
    }
}

/// Explicitly reports an unexpected target transposed-convolution event.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86ConvTranspose;

struct DupRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86DupF32Result>,
}

struct BinaryAddRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BinaryF32Result>,
}

struct BinarySubRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BinaryF32Result>,
}

struct BinaryMulRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BinaryF32Result>,
}

struct BinaryDivRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BinaryF32Result>,
}

struct PowerSqrRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86PowerF32Result>,
}

struct PowerSqrtRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86PowerF32Result>,
}

struct UnaryAbsRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86UnaryF32Result>,
}

struct UnaryNegRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86UnaryF32Result>,
}

struct UnaryReluRuntime<'dispatch> {
    input: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86UnaryF32Result>,
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

struct BroadcastAddRuntime<'dispatch> {
    input: &'dispatch [f32],
    row: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BroadcastF32Result>,
}

struct BroadcastMulRuntime<'dispatch> {
    input: &'dispatch [f32],
    row: &'dispatch [f32],
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86BroadcastF32Result>,
}

struct FmaRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    n: usize,
    k: usize,
    result: &'dispatch Cell<F32FmaResult>,
}

struct GemvRuntime<'dispatch> {
    lhs: &'dispatch [f32],
    rhs: &'dispatch [f32],
    output: &'dispatch mut [f32],
    m: usize,
    k: usize,
    result: &'dispatch Cell<X86F32GemvResult>,
}

macro_rules! define_matmul_runtime {
    ($runtime:ident, $event:ident) => {
        struct $runtime<'dispatch> {
            event: $event<'dispatch>,
            result: &'dispatch Cell<MatmulResult>,
        }
    };
}

macro_rules! define_matmul_argmax_runtime {
    ($runtime:ident, $event:ident) => {
        struct $runtime<'dispatch> {
            event: $event<'dispatch>,
            result: &'dispatch Cell<MatmulArgmaxResult>,
        }
    };
}

define_matmul_runtime!(MatmulRuntime, OpMulMat);
define_matmul_runtime!(MatmulQ4_0Runtime, OpMulMatQ4_0);
define_matmul_runtime!(MatmulQ4_1Runtime, OpMulMatQ4_1);
define_matmul_runtime!(MatmulQ5_0Runtime, OpMulMatQ5_0);
define_matmul_runtime!(MatmulQ8_0Runtime, OpMulMatQ8_0);
define_matmul_runtime!(MatmulQ2KRuntime, OpMulMatQ2K);
define_matmul_runtime!(MatmulQ3KRuntime, OpMulMatQ3K);
define_matmul_runtime!(MatmulQ4KRuntime, OpMulMatQ4K);
define_matmul_runtime!(MatmulQ6KRuntime, OpMulMatQ6K);

define_matmul_argmax_runtime!(MatmulArgmaxRuntime, OpMulMatArgmax);
define_matmul_argmax_runtime!(MatmulArgmaxQ4_0Runtime, OpMulMatArgmaxQ4_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ4_1Runtime, OpMulMatArgmaxQ4_1);
define_matmul_argmax_runtime!(MatmulArgmaxQ5_0Runtime, OpMulMatArgmaxQ5_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ8_0Runtime, OpMulMatArgmaxQ8_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ2KRuntime, OpMulMatArgmaxQ2K);
define_matmul_argmax_runtime!(MatmulArgmaxQ3KRuntime, OpMulMatArgmaxQ3K);
define_matmul_argmax_runtime!(MatmulArgmaxQ4KRuntime, OpMulMatArgmaxQ4K);
define_matmul_argmax_runtime!(MatmulArgmaxQ6KRuntime, OpMulMatArgmaxQ6K);

macro_rules! define_x86_matmul_effect {
    ($method:ident, $runtime:ident, $backend_method:ident) => {
        fn $method(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event
                .result
                .set(self.x86_matmul.$backend_method(event.event));
            Ok(())
        }
    };
}

macro_rules! define_x86_matmul_argmax_effect {
    ($method:ident, $runtime:ident, $backend_method:ident) => {
        fn $method(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event
                .result
                .set(self.x86_matmul.$backend_method(event.event));
            Ok(())
        }
    };
}

struct MatmulUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<MatmulResult>,
}

struct MatmulArgmaxUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<MatmulArgmaxResult>,
}

struct FlashAttnRuntime<'dispatch> {
    event: OpFlashAttnExt<'dispatch>,
    result: &'dispatch Cell<FlashAttnResult>,
}

struct FlashAttnUnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<FlashAttnResult>,
}

struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<Result<(), X86KernelError>>,
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

struct ConvTransposeRuntime<'dispatch> {
    event: OpX86ConvTranspose1dF32<'dispatch>,
    output: &'dispatch mut [f32],
    result: &'dispatch Cell<X86ConvTranspose1dF32Result>,
}

struct UnexpectedConvTransposeRuntime<'dispatch> {
    result: &'dispatch Cell<X86ConvTranspose1dF32Result>,
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
    dup: X86DupF32Kernel,
    binary: X86BinaryF32Kernel,
    power: X86PowerF32Kernel,
    unary: X86UnaryF32Kernel,
    scalar_unary: UnaryKernel,
    broadcast: X86BroadcastF32Kernel,
    fma: F32FmaKernel,
    gemv: X86F32GemvKernel,
    matmul: MatmulKernel,
    matmul_argmax: MatmulArgmaxKernel,
    x86_matmul: X86MatmulKernel,
    flash_attn: FlashAttnKernel,
    get_rows: GetRowsKernel,
    im2col: Im2ColKernel,
    conv_transpose: X86ConvTranspose1dF32Kernel,
    rope: RopeKernel,
    reductions: ReductionKernel,
    f16_matmul: F16MatmulKernel,
    f16_matmul_vector: X86F16MatmulKernel,
    f16_vector_available: bool,
    normalization: NormalizationKernel,
    generic: crate::any::Kernel,
}

sml! {
    X86KernelMachine<'dispatch> {
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
        "ready"_s <= "ready"_s + Fma(FmaRuntime<'dispatch>) / effect_fma,
        "ready"_s <= "ready"_s + Gemv(GemvRuntime<'dispatch>) / effect_gemv,

        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_vector] / effect_matmul_vector,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_matrix] / effect_matmul_matrix,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_portable] / effect_matmul_portable,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_shape] / effect_matmul_shape,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_view] / effect_matmul_view,
        "ready"_s <= "ready"_s + MatmulQ4_0(MatmulQ4_0Runtime<'dispatch>) / effect_matmul_q4_0,
        "ready"_s <= "ready"_s + MatmulQ4_1(MatmulQ4_1Runtime<'dispatch>) / effect_matmul_q4_1,
        "ready"_s <= "ready"_s + MatmulQ5_0(MatmulQ5_0Runtime<'dispatch>) / effect_matmul_q5_0,
        "ready"_s <= "ready"_s + MatmulQ8_0(MatmulQ8_0Runtime<'dispatch>) / effect_matmul_q8_0,
        "ready"_s <= "ready"_s + MatmulQ2K(MatmulQ2KRuntime<'dispatch>) / effect_matmul_q2_k,
        "ready"_s <= "ready"_s + MatmulQ3K(MatmulQ3KRuntime<'dispatch>) / effect_matmul_q3_k,
        "ready"_s <= "ready"_s + MatmulQ4K(MatmulQ4KRuntime<'dispatch>) / effect_matmul_q4_k,
        "ready"_s <= "ready"_s + MatmulQ6K(MatmulQ6KRuntime<'dispatch>) / effect_matmul_q6_k,
        "ready"_s <= "ready"_s + MatmulUnexpected(MatmulUnexpectedRuntime<'dispatch>)
            / effect_matmul_unexpected,

        "ready"_s <= "ready"_s + MatmulArgmax(MatmulArgmaxRuntime<'dispatch>)
            / effect_matmul_argmax,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            / effect_matmul_argmax_q4_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            / effect_matmul_argmax_q4_1,
        "ready"_s <= "ready"_s + MatmulArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            / effect_matmul_argmax_q5_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            / effect_matmul_argmax_q8_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            / effect_matmul_argmax_q2_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            / effect_matmul_argmax_q3_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            / effect_matmul_argmax_q4_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            / effect_matmul_argmax_q6_k,
        "ready"_s <= "ready"_s + MatmulArgmaxUnexpected(MatmulArgmaxUnexpectedRuntime<'dispatch>)
            / effect_matmul_argmax_unexpected,

        "ready"_s <= "ready"_s + FlashAttn(FlashAttnRuntime<'dispatch>) / effect_flash_attn,
        "ready"_s <= "ready"_s + FlashAttnUnexpected(FlashAttnUnexpectedRuntime<'dispatch>)
            / effect_flash_attn_unexpected,

        "ready"_s <= "ready"_s + ConvTranspose(ConvTransposeRuntime<'dispatch>)
            / effect_conv_transpose,
        "ready"_s <= "ready"_s + ConvTransposeUnexpected(UnexpectedConvTransposeRuntime<'dispatch>)
            / effect_conv_transpose_unexpected,
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

/// Target-owned x86 actor composing the maintained SIMD operation families.
pub struct X86Kernel {
    machine: X86KernelMachineStateMachine<Context>,
}

impl fmt::Debug for X86Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("X86Kernel").finish_non_exhaustive()
    }
}

impl X86Kernel {
    /// Constructs the router only when every maintained child resolves its
    /// required target capability before dispatch begins.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let f16_vector_available = f16_vector_supported();
        Some(Self {
            machine: X86KernelMachineStateMachine::new(Context {
                dup: X86DupF32Kernel::try_new()?,
                binary: X86BinaryF32Kernel::try_new()?,
                power: X86PowerF32Kernel::try_new()?,
                unary: X86UnaryF32Kernel::try_new()?,
                scalar_unary: UnaryKernel::new(),
                broadcast: X86BroadcastF32Kernel::try_new()?,
                fma: F32FmaKernel::try_new()?,
                gemv: X86F32GemvKernel::try_new()?,
                matmul: MatmulKernel::new(),
                matmul_argmax: MatmulArgmaxKernel::new(),
                x86_matmul: X86MatmulKernel::try_new()?,
                flash_attn: FlashAttnKernel::new(),
                conv_transpose: X86ConvTranspose1dF32Kernel::try_new()?,
                get_rows: GetRowsKernel::new(),
                im2col: Im2ColKernel::new(),
                rope: RopeKernel::new(),
                reductions: ReductionKernel::new(),
                f16_matmul: F16MatmulKernel::new(),
                f16_matmul_vector: X86F16MatmulKernel::try_new()?,
                f16_vector_available,
                normalization: NormalizationKernel::new(),
                generic: crate::any::Kernel::new(),
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86KernelEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Dispatches one source-backed row-gather event while preserving its
    /// caller-owned tensor views through the target router.
    pub fn process_get_rows<E: X86GetRowsEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed F32 or F16-destination `im2col` event.
    pub fn process_im2col<E: X86Im2ColEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one explicitly guarded `RoPE` mode event.
    pub fn process_rope<E: X86RopeEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed softmax event through the reduction child.
    pub fn process_soft_max<E: X86SoftMaxEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed scalar-trigonometry event through the reduction child.
    pub fn process_scalar_trig<E: X86ScalarTrigEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed scalar F16 matrix event through the child actor.
    pub fn process_f16_matmul<E: X86F16MatmulEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one source-backed normalization event through the child.
    pub fn process_normalization<E: X86NormalizationEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    /// Dispatches one safe generic envelope through the target-owned actor.
    ///
    /// # Errors
    ///
    /// Returns the portable child's typed error when the safe envelope is
    /// rejected.
    pub fn process_generic(&mut self, event: X86Generic<'_>) -> Result<(), crate::any::Error> {
        event.dispatch(self, &mut [])
    }

    /// Reports whether the composed target actor is ready for another event.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86KernelMachineStates::Ready)
    }
}

/// Event interface for the composed target actor.
pub trait X86KernelEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output;
}

/// Event interface for target-owned row-gather routing.
pub trait X86GetRowsEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned `im2col` routing.
pub trait X86Im2ColEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned `RoPE` mode routing.
pub trait X86RopeEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned softmax routing.
pub trait X86SoftMaxEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned scalar `log`, `sin`, and `cos` routing.
pub trait X86ScalarTrigEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned scalar F16 matrix routing.
pub trait X86F16MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// Event interface for target-owned normalization routing.
pub trait X86NormalizationEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output;
}

/// A target-owned wrapper for one safe generic kernel operation envelope.
#[derive(Debug)]
pub struct X86Generic<'a> {
    event: GenericEvent<'a>,
}

impl<'a> X86Generic<'a> {
    /// Creates a target generic event. Validation remains an SML guard.
    #[must_use]
    pub const fn new(event: GenericEvent<'a>) -> Self {
        Self { event }
    }
}

impl X86KernelEvent for X86Generic<'_> {
    type Output = Result<(), crate::any::Error>;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(crate::any::Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Generic(GenericRuntime {
                event: self.event,
                result: &result,
            }))
            .map_err(|_| crate::any::Error::Internal)?;
        result.get()
    }
}

macro_rules! impl_normalization_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86NormalizationEvent for $event {
            type Output = NormalizationResult;

            fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
                let result = Cell::new(Err(NormalizationError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
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

impl X86NormalizationEvent for UnexpectedNormalization {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(NormalizationError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::NormalizationUnexpected(
                NormalizationUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| NormalizationError::Internal)?;
        result.get()
    }
}

impl X86SoftMaxEvent for OpSoftMax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::SoftMax(SoftMaxRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

impl X86SoftMaxEvent for UnexpectedSoftMax {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::SoftMaxUnexpected(
                SoftMaxUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

macro_rules! impl_scalar_trig_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86ScalarTrigEvent for $event {
            type Output = ScalarTrigResult;

            fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
                let result = Cell::new(Err(ReductionError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
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

impl X86ScalarTrigEvent for UnexpectedScalarTrig {
    type Output = ScalarTrigResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(ReductionError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::ScalarTrigUnexpected(
                ScalarTrigUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }
}

macro_rules! impl_f16_matmul_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86F16MatmulEvent for $event {
            type Output = F16MatmulResult;

            fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
                let result = Cell::new(Err(F16MatmulError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| F16MatmulError::Internal)?;
                result.get()
            }
        }
    };
}

impl_f16_matmul_event!(OpScalarMulMatF16<'_>, F16MatmulRuntime, F16Matmul);

impl X86F16MatmulEvent for UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(F16MatmulError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::F16MatmulUnexpected(
                F16MatmulUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for X86ConvTransposeF32<'_> {
    type Output = X86ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(X86ConvTranspose1dF32Error::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::ConvTranspose(
                ConvTransposeRuntime {
                    event: OpX86ConvTranspose1dF32::new(self.weights, self.input, self.shape),
                    output,
                    result: &result,
                },
            ))
            .map_err(|_| X86ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for UnexpectedX86ConvTranspose {
    type Output = X86ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86ConvTranspose1dF32Error::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::ConvTransposeUnexpected(
                UnexpectedConvTransposeRuntime { result: &result },
            ))
            .map_err(|_| X86ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

impl X86RopeEvent for OpRope<'_> {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(RopeError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Rope(RopeRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }
}

impl X86RopeEvent for UnexpectedRope {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(RopeError::Internal));
        let _ = self;
        actor
            .machine
            .process_event(X86KernelMachineEvents::RopeUnexpected(
                RopeUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }
}

macro_rules! impl_im2col_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86Im2ColEvent for $event {
            type Output = Im2ColResult;

            fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
                let result = Cell::new(Err(Im2ColError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
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

impl X86Im2ColEvent for UnexpectedIm2Col {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(Im2ColError::Internal));
        let _ = self;
        actor
            .machine
            .process_event(X86KernelMachineEvents::Im2ColUnexpected(
                Im2ColUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }
}

macro_rules! impl_get_rows_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86GetRowsEvent for $event {
            type Output = GetRowsOutcome;

            fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
                let result = Cell::new(Err(GetRowsError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
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

impl X86GetRowsEvent for UnexpectedGetRows {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut X86Kernel) -> Self::Output {
        let result = Cell::new(Err(GetRowsError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::GetRowsUnexpected(
                GetRowsUnexpectedRuntime {
                    event: self,
                    result: &result,
                },
            ))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for X86Dup<'_> {
    type Output = X86DupF32Result;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(X86DupF32Error::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Dup(DupRuntime {
                input: self.input,
                output,
                result: &result,
            }))
            .map_err(|_| X86DupF32Error::Internal)?;
        result.get()
    }
}

macro_rules! impl_binary_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = X86BinaryF32Result;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(X86BinaryF32Error::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        lhs: self.lhs,
                        rhs: self.rhs,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| X86BinaryF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_binary_event!(X86BinaryAdd<'_>, BinaryAddRuntime, BinaryAdd);
impl_binary_event!(X86BinarySub<'_>, BinarySubRuntime, BinarySub);
impl_binary_event!(X86BinaryMul<'_>, BinaryMulRuntime, BinaryMul);
impl_binary_event!(X86BinaryDiv<'_>, BinaryDivRuntime, BinaryDiv);

macro_rules! impl_power_event {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = X86PowerF32Result;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(X86PowerF32Error::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| X86PowerF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_power_event!(X86PowerSqr<'_>, PowerSqrRuntime, PowerSqr);
impl_power_event!(X86PowerSqrt<'_>, PowerSqrtRuntime, PowerSqrt);

macro_rules! impl_unary_route {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = X86UnaryF32Result;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(X86UnaryF32Error::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| X86UnaryF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_unary_route!(X86UnaryAbs<'_>, UnaryAbsRuntime, UnaryAbs);
impl_unary_route!(X86UnaryNeg<'_>, UnaryNegRuntime, UnaryNeg);
impl_unary_route!(X86UnaryRelu<'_>, UnaryReluRuntime, UnaryRelu);

macro_rules! impl_scalar_unary_route {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = UnaryResult;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(UnaryError::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
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

impl_scalar_unary_route!(OpScalarUnaryExp<'_>, ScalarExpRuntime, ScalarExp);
impl_scalar_unary_route!(OpScalarUnaryTanh<'_>, ScalarTanhRuntime, ScalarTanh);
impl_scalar_unary_route!(OpScalarUnaryElu<'_>, ScalarEluRuntime, ScalarElu);
impl_scalar_unary_route!(OpScalarUnaryGelu<'_>, ScalarGeluRuntime, ScalarGelu);
impl_scalar_unary_route!(OpScalarUnarySilu<'_>, ScalarSiluRuntime, ScalarSilu);

macro_rules! impl_broadcast_route {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = X86BroadcastF32Result;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let result = Cell::new(Err(X86BroadcastF32Error::Internal));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        input: self.input,
                        row: self.row,
                        output,
                        result: &result,
                    }))
                    .map_err(|_| X86BroadcastF32Error::Internal)?;
                result.get()
            }
        }
    };
}

impl_broadcast_route!(X86BroadcastAdd<'_>, BroadcastAddRuntime, BroadcastAdd);
impl_broadcast_route!(X86BroadcastMul<'_>, BroadcastMulRuntime, BroadcastMul);

impl X86KernelEvent for X86Fma<'_> {
    type Output = F32FmaResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(super::fma::F32FmaError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Fma(FmaRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                n: self.n,
                k: self.k,
                result: &result,
            }))
            .map_err(|_| super::fma::F32FmaError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for X86Gemv<'_> {
    type Output = X86F32GemvResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(X86F32GemvError::Internal));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Gemv(GemvRuntime {
                lhs: self.lhs,
                rhs: self.rhs,
                output,
                m: self.m,
                k: self.k,
                result: &result,
            }))
            .map_err(|_| X86F32GemvError::Internal)?;
        result.get()
    }
}

macro_rules! impl_matmul_route {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = MatmulResult;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let _ = output;
                let result = Cell::new(Err(MatmulError::UnexpectedEvent));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| MatmulError::Internal)?;
                result.get()
            }
        }
    };
}

impl_matmul_route!(OpMulMat<'_>, MatmulRuntime, Matmul);
impl_matmul_route!(OpMulMatQ4_0<'_>, MatmulQ4_0Runtime, MatmulQ4_0);
impl_matmul_route!(OpMulMatQ4_1<'_>, MatmulQ4_1Runtime, MatmulQ4_1);
impl_matmul_route!(OpMulMatQ5_0<'_>, MatmulQ5_0Runtime, MatmulQ5_0);
impl_matmul_route!(OpMulMatQ8_0<'_>, MatmulQ8_0Runtime, MatmulQ8_0);
impl_matmul_route!(OpMulMatQ2K<'_>, MatmulQ2KRuntime, MatmulQ2K);
impl_matmul_route!(OpMulMatQ3K<'_>, MatmulQ3KRuntime, MatmulQ3K);
impl_matmul_route!(OpMulMatQ4K<'_>, MatmulQ4KRuntime, MatmulQ4K);
impl_matmul_route!(OpMulMatQ6K<'_>, MatmulQ6KRuntime, MatmulQ6K);

impl X86KernelEvent for UnexpectedX86Matmul {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = (self, output);
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::MatmulUnexpected(
                MatmulUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }
}

macro_rules! impl_matmul_argmax_route {
    ($event:ty, $runtime:ident, $variant:ident) => {
        impl X86KernelEvent for $event {
            type Output = MatmulArgmaxResult;

            fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
                let _ = output;
                let result = Cell::new(Err(MatmulError::UnexpectedEvent));
                actor
                    .machine
                    .process_event(X86KernelMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| MatmulError::Internal)?;
                result.get()
            }
        }
    };
}

impl_matmul_argmax_route!(OpMulMatArgmax<'_>, MatmulArgmaxRuntime, MatmulArgmax);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ4_0<'_>,
    MatmulArgmaxQ4_0Runtime,
    MatmulArgmaxQ4_0
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ4_1<'_>,
    MatmulArgmaxQ4_1Runtime,
    MatmulArgmaxQ4_1
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ5_0<'_>,
    MatmulArgmaxQ5_0Runtime,
    MatmulArgmaxQ5_0
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ8_0<'_>,
    MatmulArgmaxQ8_0Runtime,
    MatmulArgmaxQ8_0
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ2K<'_>,
    MatmulArgmaxQ2KRuntime,
    MatmulArgmaxQ2K
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ3K<'_>,
    MatmulArgmaxQ3KRuntime,
    MatmulArgmaxQ3K
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ4K<'_>,
    MatmulArgmaxQ4KRuntime,
    MatmulArgmaxQ4K
);
impl_matmul_argmax_route!(
    OpMulMatArgmaxQ6K<'_>,
    MatmulArgmaxQ6KRuntime,
    MatmulArgmaxQ6K
);

impl X86KernelEvent for UnexpectedX86MatmulArgmax {
    type Output = MatmulArgmaxResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = (self, output);
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::MatmulArgmaxUnexpected(
                MatmulArgmaxUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for OpFlashAttnExt<'_> {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(FlashAttnError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::FlashAttn(FlashAttnRuntime {
                event: self,
                result: &result,
            }))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for UnexpectedX86FlashAttn {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = (self, output);
        let result = Cell::new(Err(FlashAttnError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::FlashAttnUnexpected(
                FlashAttnUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }
}

impl X86KernelEvent for UnexpectedX86Kernel {
    type Output = Result<(), X86KernelError>;

    fn dispatch(self, actor: &mut X86Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86KernelError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86KernelMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86KernelError::Internal)?;
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

impl X86KernelMachineStateMachineContext for Context {
    fn effect_dup(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.dup
                .process_event(OpX86DupF32::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_binary_add(&mut self, event: BinaryAddRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpX86BinaryAdd::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_sub(&mut self, event: BinarySubRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpX86BinarySub::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_mul(&mut self, event: BinaryMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpX86BinaryMul::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_binary_div(&mut self, event: BinaryDivRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.binary
                .process_event(OpX86BinaryDiv::new(event.lhs, event.rhs), event.output),
        );
        Ok(())
    }

    fn effect_power_sqr(&mut self, event: PowerSqrRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.power
                .process_event(OpX86PowerSqr::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_power_sqrt(&mut self, event: PowerSqrtRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.power
                .process_event(OpX86PowerSqrt::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_abs(&mut self, event: UnaryAbsRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpX86UnaryAbs::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_neg(&mut self, event: UnaryNegRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpX86UnaryNeg::new(event.input), event.output),
        );
        Ok(())
    }

    fn effect_unary_relu(&mut self, event: UnaryReluRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.unary
                .process_event(OpX86UnaryRelu::new(event.input), event.output),
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
        event.result.set(
            self.broadcast
                .process_event(OpX86BroadcastAdd::new(event.input, event.row), event.output),
        );
        Ok(())
    }

    fn effect_broadcast_mul(&mut self, event: BroadcastMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(
            self.broadcast
                .process_event(OpX86BroadcastMul::new(event.input, event.row), event.output),
        );
        Ok(())
    }

    fn effect_fma(&mut self, event: FmaRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.fma.process_event(
            OpF32Fma::new(event.lhs, event.rhs, event.m, event.n, event.k),
            event.output,
        ));
        Ok(())
    }

    fn effect_gemv(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.gemv.process_event(
            OpX86F32Gemv::new(event.lhs, event.rhs, event.m, event.k),
            event.output,
        ));
        Ok(())
    }

    fn guard_matmul_vector(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_dense_views_valid()
            && event.event.target_shape_valid()
            && event.event.target_rhs_columns() == 1)
    }

    fn guard_matmul_matrix(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_dense_views_valid()
            && event.event.target_shape_valid()
            && event.event.target_rhs_columns() != 1)
    }

    fn guard_matmul_portable(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_views_valid()
            && event.event.target_shape_valid()
            && !event.event.target_dense_views_valid())
    }

    fn guard_matmul_shape(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_views_valid() && !event.event.target_shape_valid())
    }

    fn guard_matmul_view(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.target_views_valid())
    }

    fn effect_matmul_vector(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.x86_matmul.process_f32(event.event));
        Ok(())
    }

    fn effect_matmul_matrix(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.x86_matmul.process_f32(event.event));
        Ok(())
    }

    fn effect_matmul_portable(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.matmul.process_event(event.event));
        Ok(())
    }

    fn effect_matmul_shape(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_matmul_view(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    define_x86_matmul_effect!(effect_matmul_q4_0, MatmulQ4_0Runtime, process_q4_0);
    define_x86_matmul_effect!(effect_matmul_q4_1, MatmulQ4_1Runtime, process_q4_1);
    define_x86_matmul_effect!(effect_matmul_q5_0, MatmulQ5_0Runtime, process_q5_0);
    define_x86_matmul_effect!(effect_matmul_q8_0, MatmulQ8_0Runtime, process_q8_0);
    define_x86_matmul_effect!(effect_matmul_q2_k, MatmulQ2KRuntime, process_q2_k);
    define_x86_matmul_effect!(effect_matmul_q3_k, MatmulQ3KRuntime, process_q3_k);
    define_x86_matmul_effect!(effect_matmul_q4_k, MatmulQ4KRuntime, process_q4_k);
    define_x86_matmul_effect!(effect_matmul_q6_k, MatmulQ6KRuntime, process_q6_k);

    fn effect_matmul_unexpected(&mut self, event: MatmulUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_matmul_argmax(&mut self, event: MatmulArgmaxRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.matmul_argmax.process_event(event.event));
        Ok(())
    }

    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q4_0,
        MatmulArgmaxQ4_0Runtime,
        process_argmax_q4_0
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q4_1,
        MatmulArgmaxQ4_1Runtime,
        process_argmax_q4_1
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q5_0,
        MatmulArgmaxQ5_0Runtime,
        process_argmax_q5_0
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q8_0,
        MatmulArgmaxQ8_0Runtime,
        process_argmax_q8_0
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q2_k,
        MatmulArgmaxQ2KRuntime,
        process_argmax_q2_k
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q3_k,
        MatmulArgmaxQ3KRuntime,
        process_argmax_q3_k
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q4_k,
        MatmulArgmaxQ4KRuntime,
        process_argmax_q4_k
    );
    define_x86_matmul_argmax_effect!(
        effect_matmul_argmax_q6_k,
        MatmulArgmaxQ6KRuntime,
        process_argmax_q6_k
    );

    fn effect_matmul_argmax_unexpected(
        &mut self,
        event: MatmulArgmaxUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_flash_attn(&mut self, event: FlashAttnRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.flash_attn.process_event(event.event));
        Ok(())
    }

    fn effect_flash_attn_unexpected(
        &mut self,
        event: FlashAttnUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(FlashAttnError::UnexpectedEvent));
        Ok(())
    }

    fn effect_conv_transpose(&mut self, event: ConvTransposeRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.conv_transpose.process_event(event.event, event.output));
        Ok(())
    }

    fn effect_conv_transpose_unexpected(
        &mut self,
        event: UnexpectedConvTransposeRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(
            self.conv_transpose
                .process_event(UnexpectedX86ConvTranspose1dF32, &mut []),
        );
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
        event.result.set(Err(X86KernelError::UnexpectedEvent));
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
