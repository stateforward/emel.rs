//! Public, typed kernel operation events.

use super::actor::{Error, Kernel};

use super::activation::ActivationResult;
pub use super::activation::{ActivationError, OpClamp, OpLeakyRelu, OpScale, OpSiluBack};
pub use super::backward::{
    BackwardError, BackwardKernel, BackwardResult, OpGetRowsBack, OpRmsNormBack, OpSet, OpSetRows,
    OpSoftMaxBack,
};
pub use super::binary::{
    BinaryError, BinaryResult, OpAdd as OpAddTensorView, OpDiv as OpDivTensorView,
    OpMul as OpMulTensorView, OpSub as OpSubTensorView, UnexpectedBinary as UnexpectedTensorBinary,
};
pub use super::broadcast::{
    BroadcastError, BroadcastOutcome, OpAddBroadcastRow, OpMulBroadcastRow, UnexpectedBroadcast,
};
pub use super::conv::{
    Conv2dParams, Conv3dParams, ConvError, ConvKernel, ConvResult, OpConv2d, OpConv2dDw, OpConv3d,
    OpConvTranspose2d,
};
pub use super::conv_transpose_1d::{
    ConvTranspose1dError, ConvTranspose1dParams, ConvTranspose1dResult, OpConvTranspose1d,
    UnexpectedConvTranspose1d,
};
pub use super::count_equal::{CountEqualError, CountEqualKernel, CountEqualResult, OpCountEqual};
pub use super::custom::{
    CustomError, CustomKernel, CustomResult, OpCustom, OpMapCustom1, OpMapCustom2, OpMapCustom3,
};
pub use super::diag::{DiagError, DiagKernel, DiagResult, OpDiag, OpDiagMaskInf, OpDiagMaskZero};
pub use super::elementwise::{ElementwiseError, ElementwiseResult, OpAcc, OpAdd1};
pub use super::f16_matmul::{
    F16MatmulError, F16MatmulResult, F16View, OpMulMatF16, UnexpectedF16Matmul,
};
pub use super::fill_arange::{
    FillArangeError, FillArangeResult, OpArange, OpFill, UnexpectedFillArange,
};
pub use super::generic::{
    GENERIC_OPERATION_COUNT, GenericDType, GenericEvent, GenericLayout, GenericRequest,
    GenericViewError, KernelOperation, OpParams, RawTensorView, RawTensorViewMut,
};

impl Event for GenericEvent<'_> {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_generic(self)
    }
}
pub use super::flash_attn::{
    FlashAttnError, FlashAttnOptions, FlashAttnResult, OpFlashAttnExt, UnexpectedFlashAttn,
};
pub use super::get_rows::{
    Bf16View, GetRowsError, GetRowsOutcome, IndexView, OpGetRows, OpGetRowsBf16, OpGetRowsF16,
    OpGetRowsF32Bytes, OpGetRowsQ4_0, OpGetRowsQ4K, OpGetRowsQ8_0, UnexpectedGetRows,
};
pub use super::gla::{
    FlashAttnBackDst, FlashAttnBackParams, FlashAttnBackSrc, GlaError, GlaKernel, GlaParams,
    GlaResult, OpFlashAttnBack, OpGatedLinearAttn,
};
pub use super::glu::{GluError, GluKernel, GluResult, GluSubOp, OpGlu};
pub use super::group_norm::{
    GroupNormError, GroupNormKernel, GroupNormResult, OpGroupNorm, OpL2Norm,
};
pub use super::im2col::{
    F16OutputViewMut, Im2ColError, Im2ColParams, Im2ColResult, OpIm2Col, OpIm2ColF16,
    UnexpectedIm2Col,
};
pub use super::indexed::{
    IndexedError, IndexedKernel, IndexedResult, MulMatIdParams, OpAddId, OpMulMatId,
};
pub use super::linalg::{LinalgError, LinalgKernel, LinalgResult, OpOutProd, OpSolveTri};
pub use super::matmul::{
    MatmulArgmaxResult, MatmulError, MatmulResult, OpMulMat, OpMulMatArgmax, OpMulMatArgmaxQ2K,
    OpMulMatArgmaxQ3K, OpMulMatArgmaxQ4_0, OpMulMatArgmaxQ4_1, OpMulMatArgmaxQ4K,
    OpMulMatArgmaxQ5_0, OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, OpMulMatQ2K, OpMulMatQ3K,
    OpMulMatQ4_0, OpMulMatQ4_1, OpMulMatQ4K, OpMulMatQ5_0, OpMulMatQ6K, OpMulMatQ8_0,
    QuantizedView, UnexpectedMatmul,
};
pub use super::normalization::{NormalizationError, NormalizationResult, OpNorm, OpRmsNorm};
pub use super::pad::{OpPad, OpPadReflect1d, OpTri, PadError, PadKernel, PadResult};
pub use super::pool::{
    OpPool1d, OpPool2d, OpPool2dBack, Pool2dParams, PoolError, PoolKernel, PoolResult, PoolSubOp,
};
pub use super::positional::{
    OpRopeBack, OpTimestepEmbedding, PositionalError, PositionalKernel, PositionalResult,
};
pub use super::reductions::{
    OpArgmax, OpCos, OpLog, OpMean, OpSin, OpSoftMax, OpSum, OpSumRows, ReductionError,
    ReductionResult,
};
pub use super::rel::{OpAddRelPos, OpGetRelPos, RelError, RelKernel, RelResult};
pub use super::reorder::{
    ArgsortOrder, OpArgsort, OpRoll, OpTopK, OpUpscale, ReorderError, ReorderKernel, ReorderResult,
};
pub use super::rope::{
    I32View, OpRope, ROPE_MODE_NEOX, ROPE_MODE_NORM, ROPE_MODE_TIMESTEP, RopeError, RopeParams,
    UnexpectedRope,
};
pub use super::rwkv::{OpRwkvWkv6, OpRwkvWkv7, RwkvError, RwkvKernel, RwkvParams, RwkvResult};
pub use super::sequence::{
    OpConcat, OpCumsum, OpRepeat, OpRepeatBack, SequenceError, SequenceResult,
};
pub use super::shape::{
    CopyOutcome, LayoutOutcome, OpCont, OpCpy, OpPermute, OpReshape, OpTranspose, OpView,
    ShapeError,
};
pub use super::ssm::{OpSsmConv, OpSsmScan, SsmError, SsmKernel, SsmResult, SsmScanParams};
pub use super::train::{
    AdamWParams, OpCrossEntropyLoss, OpCrossEntropyLossBack, OpOptStepAdamW, OpOptStepSgd,
    TrainError, TrainKernel, TrainResult,
};
pub use super::unary::{
    OpAbs, OpCeil, OpElu, OpExp, OpExpm1, OpFloor, OpGelu, OpGeluErf, OpGeluQuick, OpHardsigmoid,
    OpHardswish, OpNeg, OpRelu, OpRound, OpSgn, OpSigmoid, OpSilu, OpSoftplus, OpStep, OpTanh,
    OpTrunc, OpUnary, UnaryError, UnaryResult, UnarySubOp, UnexpectedUnary,
};
pub use super::window::{
    Im2Col3dParams, Im2ColBackParams, OpIm2Col3d, OpIm2ColBack, OpWinPart, OpWinUnpart,
    WinPartParams, WindowError, WindowKernel, WindowResult,
};

/// An operation accepted by [`Kernel`].
pub trait Event: Sized {
    /// The result returned after the synchronous dispatch completes.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Kernel) -> Self::Output;
}

/// Copies one contiguous F32 buffer into another.
#[derive(Debug)]
pub struct OpDup<'a> {
    pub(super) input: &'a [f32],
    pub(super) output: &'a mut [f32],
}

impl<'a> OpDup<'a> {
    /// Creates a copy request. Shape validation occurs in the actor guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { input, output }
    }

    /// Returns the input element count without dispatching.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.input.len()
    }

    /// Returns the output element count without dispatching.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output.len()
    }
}

impl Event for OpDup<'_> {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_dup(self)
    }
}

/// Adds two contiguous F32 buffers element-wise into a third buffer.
#[derive(Debug)]
pub struct OpAdd<'a> {
    pub(super) lhs: &'a [f32],
    pub(super) rhs: &'a [f32],
    pub(super) output: &'a mut [f32],
}

impl<'a> OpAdd<'a> {
    /// Creates an element-wise addition request. Shape validation occurs in
    /// the actor guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { lhs, rhs, output }
    }

    /// Returns the left-hand element count without dispatching.
    #[must_use]
    pub const fn lhs_len(&self) -> usize {
        self.lhs.len()
    }

    /// Returns the right-hand element count without dispatching.
    #[must_use]
    pub const fn rhs_len(&self) -> usize {
        self.rhs.len()
    }

    /// Returns the output element count without dispatching.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output.len()
    }
}

impl Event for OpAdd<'_> {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add(self)
    }
}

macro_rules! define_binary_event {
    ($name:ident, $dispatch:ident, $lhs_len:ident, $rhs_len:ident, $output_len:ident) => {
        /// A dense contiguous F32 binary operation request.
        #[derive(Debug)]
        pub struct $name<'a> {
            pub(super) lhs: &'a [f32],
            pub(super) rhs: &'a [f32],
            pub(super) output: &'a mut [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a binary operation request. Shape validation occurs in
            /// the actor guard.
            #[must_use]
            pub const fn new(lhs: &'a [f32], rhs: &'a [f32], output: &'a mut [f32]) -> Self {
                Self { lhs, rhs, output }
            }

            /// Returns the left-hand element count without dispatching.
            #[must_use]
            pub const fn $lhs_len(&self) -> usize {
                self.lhs.len()
            }

            /// Returns the right-hand element count without dispatching.
            #[must_use]
            pub const fn $rhs_len(&self) -> usize {
                self.rhs.len()
            }

            /// Returns the output element count without dispatching.
            #[must_use]
            pub const fn $output_len(&self) -> usize {
                self.output.len()
            }
        }

        impl Event for $name<'_> {
            type Output = Result<(), Error>;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                actor.$dispatch(self)
            }
        }
    };
}

define_binary_event!(OpSub, op_sub, lhs_len, rhs_len, output_len);
define_binary_event!(OpMul, op_mul, lhs_len, rhs_len, output_len);
define_binary_event!(OpDiv, op_div, lhs_len, rhs_len, output_len);

impl Event for super::binary::OpAdd<'_> {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add_tensor_view(self)
    }
}

impl Event for super::binary::OpSub<'_> {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_sub_tensor_view(self)
    }
}

impl Event for super::binary::OpMul<'_> {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_tensor_view(self)
    }
}

impl Event for super::binary::OpDiv<'_> {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_div_tensor_view(self)
    }
}

impl Event for super::binary::UnexpectedBinary {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_tensor_binary(self)
    }
}

impl Event for super::broadcast::OpAddBroadcastRow<'_> {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add_broadcast(self)
    }
}

impl Event for super::broadcast::OpMulBroadcastRow<'_> {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_broadcast(self)
    }
}

impl Event for super::broadcast::UnexpectedBroadcast {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_broadcast(self)
    }
}

impl Event for super::conv_transpose_1d::OpConvTranspose1d<'_> {
    type Output = Result<(), ConvTranspose1dError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_conv_transpose_1d(self)
    }
}

impl Event for super::conv_transpose_1d::UnexpectedConvTranspose1d {
    type Output = Result<(), ConvTranspose1dError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_conv_transpose_1d(self)
    }
}

impl Event for super::f16_matmul::OpMulMatF16<'_> {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_mat_f16(self)
    }
}

impl Event for super::f16_matmul::UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_f16_matmul()
    }
}

impl Event for OpMulMat<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_mat(self)
    }
}

macro_rules! impl_root_matmul_event {
    ($event:ident, $dispatch:ident) => {
        impl Event for $event<'_> {
            type Output = MatmulResult;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                actor.$dispatch(self)
            }
        }
    };
}

impl_root_matmul_event!(OpMulMatQ4_0, op_mul_mat_q4_0);
impl_root_matmul_event!(OpMulMatQ4_1, op_mul_mat_q4_1);
impl_root_matmul_event!(OpMulMatQ5_0, op_mul_mat_q5_0);
impl_root_matmul_event!(OpMulMatQ8_0, op_mul_mat_q8_0);
impl_root_matmul_event!(OpMulMatQ2K, op_mul_mat_q2_k);
impl_root_matmul_event!(OpMulMatQ3K, op_mul_mat_q3_k);
impl_root_matmul_event!(OpMulMatQ4K, op_mul_mat_q4_k);
impl_root_matmul_event!(OpMulMatQ6K, op_mul_mat_q6_k);

impl Event for UnexpectedMatmul {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_matmul(self)
    }
}

/// Explicit unexpected-event route for the matrix-vector argmax child.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedMatmulArgmax;

impl Event for UnexpectedMatmulArgmax {
    type Output = MatmulArgmaxResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_matmul_argmax(UnexpectedMatmul)
    }
}

impl Event for OpMulMatArgmax<'_> {
    type Output = MatmulArgmaxResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_mat_argmax(self)
    }
}

macro_rules! impl_root_matmul_argmax_event {
    ($event:ident, $dispatch:ident) => {
        impl Event for $event<'_> {
            type Output = MatmulArgmaxResult;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                actor.$dispatch(self)
            }
        }
    };
}

impl_root_matmul_argmax_event!(OpMulMatArgmaxQ4_0, op_mul_mat_argmax_q4_0);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ4_1, op_mul_mat_argmax_q4_1);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ5_0, op_mul_mat_argmax_q5_0);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ8_0, op_mul_mat_argmax_q8_0);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ2K, op_mul_mat_argmax_q2_k);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ3K, op_mul_mat_argmax_q3_k);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ4K, op_mul_mat_argmax_q4_k);
impl_root_matmul_argmax_event!(OpMulMatArgmaxQ6K, op_mul_mat_argmax_q6_k);

impl Event for OpIm2Col<'_> {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_im2col(self)
    }
}

impl Event for OpIm2ColF16<'_> {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_im2col_f16(self)
    }
}

impl Event for UnexpectedIm2Col {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_im2col(self)
    }
}

impl Event for OpGetRows<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows(self)
    }
}

impl Event for OpGetRowsF32Bytes<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_f32_bytes(self)
    }
}

impl Event for OpGetRowsF16<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_f16(self)
    }
}

impl Event for OpGetRowsBf16<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_bf16(self)
    }
}

impl Event for OpGetRowsQ4_0<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_q4_0(self)
    }
}

impl Event for OpGetRowsQ8_0<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_q8_0(self)
    }
}

impl Event for OpGetRowsQ4K<'_> {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_q4_k(self)
    }
}

impl Event for UnexpectedGetRows {
    type Output = GetRowsOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_get_rows(self)
    }
}

macro_rules! define_unary_event {
    ($name:ident, $dispatch:ident) => {
        /// A dense contiguous F32 unary operation request.
        #[derive(Debug)]
        pub struct $name<'a> {
            pub(super) input: &'a [f32],
            pub(super) output: &'a mut [f32],
        }

        impl<'a> $name<'a> {
            /// Creates a unary operation request. Shape validation occurs in
            /// the actor guard.
            #[must_use]
            pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
                Self { input, output }
            }

            /// Returns the input element count without dispatching.
            #[must_use]
            pub const fn input_len(&self) -> usize {
                self.input.len()
            }

            /// Returns the output element count without dispatching.
            #[must_use]
            pub const fn output_len(&self) -> usize {
                self.output.len()
            }

            pub(crate) const fn into_parts(self) -> (&'a [f32], &'a mut [f32]) {
                (self.input, self.output)
            }
        }

        impl Event for $name<'_> {
            type Output = Result<(), Error>;

            fn dispatch(self, actor: &mut Kernel) -> Self::Output {
                actor.$dispatch(self)
            }
        }
    };
}

define_unary_event!(OpSqr, op_sqr);
define_unary_event!(OpSqrt, op_sqrt);

/// Operation labels reserved for an operation envelope that is not represented
/// by the typed Rust event API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedOperation {
    /// Generic matrix multiplication envelope.
    MulMat,
    /// A validated raw generic envelope without a typed safe operation.
    Generic(KernelOperation),
}

/// Explicitly rejects an operation outside the typed event API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Unsupported {
    pub(super) operation: UnsupportedOperation,
}

impl Unsupported {
    /// Creates an explicit unsupported-operation request.
    #[must_use]
    pub const fn new(operation: UnsupportedOperation) -> Self {
        Self { operation }
    }

    /// Returns the rejected operation family.
    #[must_use]
    pub const fn operation(self) -> UnsupportedOperation {
        self.operation
    }
}

impl Event for Unsupported {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unsupported(self)
    }
}

impl Event for OpLog<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_log(self)
    }
}

impl Event for OpSin<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_sin(self)
    }
}

impl Event for OpCos<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cos(self)
    }
}

impl Event for OpSum<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_sum(self)
    }
}

impl Event for OpSumRows<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_sum_rows(self)
    }
}

impl Event for OpMean<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mean(self)
    }
}

impl Event for OpArgmax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_argmax(self)
    }
}

impl Event for OpCountEqual<'_> {
    type Output = CountEqualResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_count_equal(self)
    }
}

impl Event for OpGlu<'_> {
    type Output = GluResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_glu(self)
    }
}

impl Event for OpDiag<'_> {
    type Output = DiagResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_diag(self)
    }
}

impl Event for OpDiagMaskInf<'_> {
    type Output = DiagResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_diag_mask_inf(self)
    }
}

impl Event for OpDiagMaskZero<'_> {
    type Output = DiagResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_diag_mask_zero(self)
    }
}

impl Event for OpPad<'_> {
    type Output = PadResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_pad(self)
    }
}

impl Event for OpPadReflect1d<'_> {
    type Output = PadResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_pad_reflect_1d(self)
    }
}

impl Event for OpTri<'_> {
    type Output = PadResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_tri(self)
    }
}

impl Event for OpPool1d<'_> {
    type Output = PoolResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_pool_1d(self)
    }
}

impl Event for OpPool2d<'_> {
    type Output = PoolResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_pool_2d(self)
    }
}

impl Event for OpPool2dBack<'_> {
    type Output = PoolResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_pool_2d_back(self)
    }
}

impl Event for OpRoll<'_> {
    type Output = ReorderResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_roll(self)
    }
}

impl Event for OpUpscale<'_> {
    type Output = ReorderResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_upscale(self)
    }
}

impl Event for OpArgsort<'_> {
    type Output = ReorderResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_argsort(self)
    }
}

impl Event for OpTopK<'_> {
    type Output = ReorderResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_top_k(self)
    }
}

impl Event for OpConv2d<'_> {
    type Output = ConvResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_conv_2d(self)
    }
}

impl Event for OpConv2dDw<'_> {
    type Output = ConvResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_conv_2d_dw(self)
    }
}

impl Event for OpConv3d<'_> {
    type Output = ConvResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_conv_3d(self)
    }
}

impl Event for OpConvTranspose2d<'_> {
    type Output = ConvResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_conv_transpose_2d(self)
    }
}

impl Event for OpSoftMaxBack<'_> {
    type Output = BackwardResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_soft_max_back(self)
    }
}

impl Event for OpRmsNormBack<'_> {
    type Output = BackwardResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rms_norm_back(self)
    }
}

impl Event for OpGetRowsBack<'_> {
    type Output = BackwardResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rows_back(self)
    }
}

impl Event for OpSetRows<'_> {
    type Output = BackwardResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_set_rows(self)
    }
}

impl Event for OpSet<'_> {
    type Output = BackwardResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_set(self)
    }
}

impl Event for OpWinPart<'_> {
    type Output = WindowResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_win_part(self)
    }
}

impl Event for OpWinUnpart<'_> {
    type Output = WindowResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_win_unpart(self)
    }
}

impl Event for OpIm2ColBack<'_> {
    type Output = WindowResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_im2col_back(self)
    }
}

impl Event for OpIm2Col3d<'_> {
    type Output = WindowResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_im2col_3d(self)
    }
}

impl Event for OpRopeBack<'_> {
    type Output = PositionalResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rope_back(self)
    }
}

impl Event for OpTimestepEmbedding<'_> {
    type Output = PositionalResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_timestep_embedding(self)
    }
}

impl Event for OpSsmConv<'_> {
    type Output = SsmResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_ssm_conv(self)
    }
}

impl Event for OpSsmScan<'_> {
    type Output = SsmResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_ssm_scan(self)
    }
}

impl Event for OpOutProd<'_> {
    type Output = LinalgResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_out_prod(self)
    }
}

impl Event for OpSolveTri<'_> {
    type Output = LinalgResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_solve_tri(self)
    }
}

impl Event for OpCrossEntropyLoss<'_> {
    type Output = TrainResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cross_entropy_loss(self)
    }
}
impl Event for OpCrossEntropyLossBack<'_> {
    type Output = TrainResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cross_entropy_loss_back(self)
    }
}
impl Event for OpOptStepAdamW<'_> {
    type Output = TrainResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_opt_step_adamw(self)
    }
}
impl Event for OpOptStepSgd<'_> {
    type Output = TrainResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_opt_step_sgd(self)
    }
}
impl Event for OpGetRelPos<'_> {
    type Output = RelResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_get_rel_pos(self)
    }
}
impl Event for OpAddRelPos<'_> {
    type Output = RelResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add_rel_pos(self)
    }
}

impl Event for OpAddId<'_> {
    type Output = IndexedResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add_id(self)
    }
}
impl Event for OpMulMatId<'_> {
    type Output = IndexedResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_mul_mat_id(self)
    }
}
impl Event for OpRwkvWkv6<'_> {
    type Output = RwkvResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rwkv_wkv6(self)
    }
}
impl Event for OpRwkvWkv7<'_> {
    type Output = RwkvResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rwkv_wkv7(self)
    }
}

impl Event for OpGatedLinearAttn<'_> {
    type Output = GlaResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_gated_linear_attn(self)
    }
}
impl Event for OpFlashAttnBack<'_> {
    type Output = GlaResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_flash_attn_back(self)
    }
}
impl Event for OpMapCustom1<'_> {
    type Output = CustomResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_map_custom1(self)
    }
}
impl Event for OpMapCustom2<'_> {
    type Output = CustomResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_map_custom2(self)
    }
}
impl Event for OpMapCustom3<'_> {
    type Output = CustomResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_map_custom3(self)
    }
}
impl Event for OpCustom<'_> {
    type Output = CustomResult;
    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_custom(self)
    }
}

impl Event for OpSoftMax<'_> {
    type Output = ReductionResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_soft_max(self)
    }
}

impl Event for OpScale<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_scale(self)
    }
}

impl Event for OpClamp<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_clamp(self)
    }
}

impl Event for OpSiluBack<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_silu_back(self)
    }
}

impl Event for OpLeakyRelu<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_leaky_relu(self)
    }
}

impl Event for OpAdd1<'_> {
    type Output = ElementwiseResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_add1(self)
    }
}

impl Event for OpAcc<'_> {
    type Output = ElementwiseResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_acc(self)
    }
}

impl Event for OpNorm<'_> {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_norm(self)
    }
}

impl Event for OpRmsNorm<'_> {
    type Output = NormalizationResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rms_norm(self)
    }
}

impl Event for OpGroupNorm<'_> {
    type Output = GroupNormResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_group_norm(self)
    }
}

impl Event for OpL2Norm<'_> {
    type Output = GroupNormResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_l2_norm(self)
    }
}

impl Event for OpCumsum<'_> {
    type Output = SequenceResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cumsum(self)
    }
}

impl Event for OpFill<'_> {
    type Output = FillArangeResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_fill(self)
    }
}

impl Event for OpArange<'_> {
    type Output = FillArangeResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_arange(self)
    }
}

impl Event for OpRepeat<'_> {
    type Output = SequenceResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_repeat(self)
    }
}

impl Event for OpRepeatBack<'_> {
    type Output = SequenceResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_repeat_back(self)
    }
}

impl Event for OpConcat<'_> {
    type Output = SequenceResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_concat(self)
    }
}

impl Event for OpCpy<'_> {
    type Output = CopyOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cpy(self)
    }
}

impl Event for OpCont<'_> {
    type Output = CopyOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_cont(self)
    }
}

impl Event for OpReshape<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_reshape(self)
    }
}

impl Event for OpView<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_view(self)
    }
}

impl Event for OpPermute<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_permute(self)
    }
}

impl Event for OpTranspose<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_transpose(self)
    }
}

impl Event for OpRope<'_> {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_rope(self)
    }
}

impl Event for UnexpectedRope {
    type Output = Result<(), RopeError>;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_rope(self)
    }
}

impl Event for OpFlashAttnExt<'_> {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_flash_attn(self)
    }
}

impl Event for UnexpectedFlashAttn {
    type Output = FlashAttnResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_flash_attn(self)
    }
}

impl Event for OpUnary<'_> {
    type Output = UnaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.op_unary(self)
    }
}

impl Event for UnexpectedUnary {
    type Output = UnaryResult;

    fn dispatch(self, actor: &mut Kernel) -> Self::Output {
        actor.unexpected_unary(self)
    }
}
