//! Run-to-completion kernel actor and safe portable F32 actions.

#![allow(missing_debug_implementations)]

use core::cell::Cell;
use core::fmt;

use super::activation::{
    ActivationKernel, ActivationResult, OpClamp, OpLeakyRelu, OpScale, OpSiluBack,
};
use super::backward::{
    BackwardError, BackwardKernel, BackwardResult, OpGetRowsBack, OpRmsNormBack, OpSet, OpSetRows,
    OpSoftMaxBack,
};
use super::binary::{BinaryError, BinaryKernel, BinaryResult};
use super::broadcast::{BroadcastError, BroadcastKernel, BroadcastOutcome};
use super::conv::{
    ConvError, ConvKernel, ConvResult, OpConv2d, OpConv2dDw, OpConv3d, OpConvTranspose2d,
};
use super::conv_transpose_1d::{ConvTranspose1dError, ConvTranspose1dKernel};
use super::count_equal::{CountEqualError, CountEqualKernel, CountEqualResult, OpCountEqual};
use super::custom::{
    CustomError, CustomKernel, CustomResult, OpCustom, OpMapCustom1, OpMapCustom2, OpMapCustom3,
};
use super::diag::{DiagError, DiagKernel, DiagResult, OpDiag, OpDiagMaskInf, OpDiagMaskZero};
use super::elementwise::{ElementwiseKernel, ElementwiseResult, OpAcc, OpAdd1};
use super::event;
use super::f16_matmul::{F16MatmulError, F16MatmulKernel, F16MatmulResult, OpMulMatF16};
use super::fill_arange::{FillArangeError, FillArangeKernel, FillArangeResult, OpArange, OpFill};
use super::flash_attn::{
    FlashAttnError, FlashAttnKernel, FlashAttnResult, OpFlashAttnExt, UnexpectedFlashAttn,
};
use super::generic;
use super::get_rows::{
    GetRowsError, GetRowsKernel, GetRowsOutcome, OpGetRows, OpGetRowsBf16, OpGetRowsF16,
    OpGetRowsF32Bytes, OpGetRowsQ4_0, OpGetRowsQ4K, OpGetRowsQ8_0, UnexpectedGetRows,
};
use super::gla::{GlaError, GlaKernel, GlaResult, OpFlashAttnBack, OpGatedLinearAttn};
use super::glu::{GluError, GluKernel, GluResult, OpGlu};
use super::group_norm::{GroupNormError, GroupNormKernel, GroupNormResult, OpGroupNorm, OpL2Norm};
use super::im2col::{
    Im2ColError, Im2ColKernel, Im2ColResult, OpIm2Col, OpIm2ColF16, UnexpectedIm2Col,
};
use super::indexed::{IndexedError, IndexedKernel, IndexedResult, OpAddId, OpMulMatId};
use super::linalg::{LinalgError, LinalgKernel, LinalgResult, OpOutProd, OpSolveTri};
use super::matmul::{
    MatmulArgmaxKernel, MatmulArgmaxResult, MatmulError, MatmulKernel, MatmulResult, OpMulMat,
    OpMulMatArgmax, OpMulMatArgmaxQ2K, OpMulMatArgmaxQ3K, OpMulMatArgmaxQ4_0, OpMulMatArgmaxQ4_1,
    OpMulMatArgmaxQ4K, OpMulMatArgmaxQ5_0, OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, OpMulMatQ2K,
    OpMulMatQ3K, OpMulMatQ4_0, OpMulMatQ4_1, OpMulMatQ4K, OpMulMatQ5_0, OpMulMatQ6K, OpMulMatQ8_0,
    UnexpectedMatmul,
};
use super::normalization::{NormalizationKernel, NormalizationResult, OpNorm, OpRmsNorm};
use super::pad::{OpPad, OpPadReflect1d, OpTri, PadError, PadKernel, PadResult};
use super::pool::{OpPool1d, OpPool2d, OpPool2dBack, PoolError, PoolKernel, PoolResult};
use super::positional::{
    OpRopeBack, OpTimestepEmbedding, PositionalError, PositionalKernel, PositionalResult,
};
use super::power::{PowerError, PowerKernel, PowerResult};
use super::reductions::{ReductionError, ReductionKernel, ReductionResult};
use super::rel::{OpAddRelPos, OpGetRelPos, RelError, RelKernel, RelResult};
use super::reorder::{
    OpArgsort, OpRoll, OpTopK, OpUpscale, ReorderError, ReorderKernel, ReorderResult,
};
use super::rope::{OpRope, RopeError, RopeKernel, UnexpectedRope};
use super::rwkv::{OpRwkvWkv6, OpRwkvWkv7, RwkvError, RwkvKernel, RwkvResult};
use super::sequence::{OpConcat, OpCumsum, OpRepeat, OpRepeatBack, SequenceKernel, SequenceResult};
use super::shape::{
    CopyOutcome, LayoutOutcome, OpCont, OpCpy, OpPermute, OpReshape, OpTranspose, OpView,
    ShapeKernel,
};
use super::sm::{
    KernelMachineEvents, KernelMachineStateMachine, KernelMachineStateMachineContext,
    KernelMachineStates,
};
use super::ssm::{OpSsmConv, OpSsmScan, SsmError, SsmKernel, SsmResult};
use super::train::{
    OpCrossEntropyLoss, OpCrossEntropyLossBack, OpOptStepAdamW, OpOptStepSgd, TrainError,
    TrainKernel, TrainResult,
};
use super::unary::{OpUnary, UnaryError, UnaryKernel, UnaryResult, UnexpectedUnary};
use super::window::{
    OpIm2Col3d, OpIm2ColBack, OpWinPart, OpWinUnpart, WindowError, WindowKernel, WindowResult,
};

/// Errors produced by a kernel dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// A request has an empty or mismatched shape.
    InvalidShape,
    /// The requested operation is outside the maintained implementation set.
    UnsupportedOperation(event::UnsupportedOperation),
    /// The generated machine rejected an otherwise expected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
    /// The requested architecture is not available in this build.
    UnsupportedKernelKind(crate::KernelKind),
    /// The requested architecture was compiled but could not initialize.
    KernelUnavailable(crate::KernelKind),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid kernel shape"),
            Self::UnsupportedOperation(_) => formatter.write_str("unsupported kernel operation"),
            Self::UnexpectedEvent => formatter.write_str("unexpected kernel event"),
            Self::Internal => formatter.write_str("internal kernel dispatch error"),
            Self::UnsupportedKernelKind(kind) => {
                write!(formatter, "unsupported kernel kind: {kind:?}")
            }
            Self::KernelUnavailable(kind) => {
                write!(formatter, "kernel kind unavailable: {kind:?}")
            }
        }
    }
}

impl std::error::Error for Error {}

fn dense_power_layout(length: usize) -> Option<super::tensor_view::Layout> {
    let extent = u64::try_from(length).ok()?;
    super::tensor_view::Layout::contiguous(super::tensor_view::DType::F32, [extent, 1, 1, 1])
}

const fn map_power_error(error: PowerError) -> Error {
    match error {
        PowerError::InvalidView | PowerError::ShapeMismatch => Error::InvalidShape,
        PowerError::UnexpectedEvent => Error::UnexpectedEvent,
        PowerError::Internal => Error::Internal,
    }
}

pub struct DupRuntime<'a> {
    pub(super) event: event::OpDup<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct AddRuntime<'a> {
    pub(super) event: event::OpAdd<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct SubRuntime<'a> {
    pub(super) event: event::OpSub<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct MulRuntime<'a> {
    pub(super) event: event::OpMul<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct DivRuntime<'a> {
    pub(super) event: event::OpDiv<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct GenericRuntime<'a> {
    pub(super) event: event::GenericEvent<'a>,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct BinaryAddRuntime<'a> {
    pub(super) event: event::OpAddTensorView<'a>,
    pub(super) result: &'a Cell<BinaryResult>,
}

pub struct BinarySubRuntime<'a> {
    pub(super) event: event::OpSubTensorView<'a>,
    pub(super) result: &'a Cell<BinaryResult>,
}

pub struct BinaryMulRuntime<'a> {
    pub(super) event: event::OpMulTensorView<'a>,
    pub(super) result: &'a Cell<BinaryResult>,
}

pub struct BinaryDivRuntime<'a> {
    pub(super) event: event::OpDivTensorView<'a>,
    pub(super) result: &'a Cell<BinaryResult>,
}

pub struct BinaryUnexpectedRuntime<'a> {
    pub(super) event: event::UnexpectedTensorBinary,
    pub(super) result: &'a Cell<BinaryResult>,
}

pub struct BroadcastAddRuntime<'a> {
    pub(super) event: event::OpAddBroadcastRow<'a>,
    pub(super) result: &'a Cell<BroadcastOutcome>,
}

pub struct BroadcastMulRuntime<'a> {
    pub(super) event: event::OpMulBroadcastRow<'a>,
    pub(super) result: &'a Cell<BroadcastOutcome>,
}

pub struct BroadcastUnexpectedRuntime<'a> {
    pub(super) event: event::UnexpectedBroadcast,
    pub(super) result: &'a Cell<BroadcastOutcome>,
}

pub struct ConvTranspose1dRuntime<'a> {
    pub(super) event: event::OpConvTranspose1d<'a>,
    pub(super) result: &'a Cell<Result<(), ConvTranspose1dError>>,
}

pub struct ConvTranspose1dUnexpectedRuntime<'a> {
    pub(super) event: event::UnexpectedConvTranspose1d,
    pub(super) result: &'a Cell<Result<(), ConvTranspose1dError>>,
}

pub struct F16MatmulRuntime<'a> {
    pub(super) event: OpMulMatF16<'a>,
    pub(super) result: &'a Cell<F16MatmulResult>,
}

pub struct F16MatmulUnexpectedRuntime<'a> {
    pub(super) result: &'a Cell<F16MatmulResult>,
}

pub struct Im2ColRuntime<'a> {
    pub(super) event: OpIm2Col<'a>,
    pub(super) result: &'a Cell<Im2ColResult>,
}

pub struct Im2ColF16Runtime<'a> {
    pub(super) event: OpIm2ColF16<'a>,
    pub(super) result: &'a Cell<Im2ColResult>,
}

pub struct Im2ColUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedIm2Col,
    pub(super) result: &'a Cell<Im2ColResult>,
}

pub struct GetRowsRuntime<'a> {
    pub(super) event: OpGetRows<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsF32BytesRuntime<'a> {
    pub(super) event: OpGetRowsF32Bytes<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsF16Runtime<'a> {
    pub(super) event: OpGetRowsF16<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsBf16Runtime<'a> {
    pub(super) event: OpGetRowsBf16<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsQ4_0Runtime<'a> {
    pub(super) event: OpGetRowsQ4_0<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsQ8_0Runtime<'a> {
    pub(super) event: OpGetRowsQ8_0<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsQ4KRuntime<'a> {
    pub(super) event: OpGetRowsQ4K<'a>,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct GetRowsUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedGetRows,
    pub(super) result: &'a Cell<GetRowsOutcome>,
}

pub struct RopeRuntime<'a> {
    pub(super) event: OpRope<'a>,
    pub(super) result: &'a Cell<Result<(), RopeError>>,
}

pub struct RopeUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedRope,
    pub(super) result: &'a Cell<Result<(), RopeError>>,
}

pub struct FlashAttnRuntime<'a> {
    pub(super) event: OpFlashAttnExt<'a>,
    pub(super) result: &'a Cell<FlashAttnResult>,
}

pub struct FlashAttnUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedFlashAttn,
    pub(super) result: &'a Cell<FlashAttnResult>,
}

pub struct UnaryRuntime<'a> {
    pub(super) event: OpUnary<'a>,
    pub(super) result: &'a Cell<UnaryResult>,
}

pub struct UnaryUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedUnary,
    pub(super) result: &'a Cell<UnaryResult>,
}

pub struct MatmulRuntime<'a> {
    pub(super) event: OpMulMat<'a>,
    pub(super) result: &'a Cell<MatmulResult>,
}

macro_rules! define_matmul_runtime {
    ($name:ident, $event:ident) => {
        pub struct $name<'a> {
            pub(super) event: $event<'a>,
            pub(super) result: &'a Cell<MatmulResult>,
        }
    };
}

define_matmul_runtime!(MatmulQ4_0Runtime, OpMulMatQ4_0);
define_matmul_runtime!(MatmulQ4_1Runtime, OpMulMatQ4_1);
define_matmul_runtime!(MatmulQ5_0Runtime, OpMulMatQ5_0);
define_matmul_runtime!(MatmulQ8_0Runtime, OpMulMatQ8_0);

define_matmul_runtime!(MatmulQ2KRuntime, OpMulMatQ2K);
define_matmul_runtime!(MatmulQ3KRuntime, OpMulMatQ3K);
define_matmul_runtime!(MatmulQ4KRuntime, OpMulMatQ4K);
define_matmul_runtime!(MatmulQ6KRuntime, OpMulMatQ6K);

pub struct MatmulUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedMatmul,
    pub(super) result: &'a Cell<MatmulResult>,
}

pub struct MatmulArgmaxRuntime<'a> {
    pub(super) event: OpMulMatArgmax<'a>,
    pub(super) result: &'a Cell<MatmulArgmaxResult>,
}

macro_rules! define_matmul_argmax_runtime {
    ($name:ident, $event:ident) => {
        pub struct $name<'a> {
            pub(super) event: $event<'a>,
            pub(super) result: &'a Cell<MatmulArgmaxResult>,
        }
    };
}

define_matmul_argmax_runtime!(MatmulArgmaxQ4_0Runtime, OpMulMatArgmaxQ4_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ4_1Runtime, OpMulMatArgmaxQ4_1);
define_matmul_argmax_runtime!(MatmulArgmaxQ5_0Runtime, OpMulMatArgmaxQ5_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ8_0Runtime, OpMulMatArgmaxQ8_0);
define_matmul_argmax_runtime!(MatmulArgmaxQ2KRuntime, OpMulMatArgmaxQ2K);
define_matmul_argmax_runtime!(MatmulArgmaxQ3KRuntime, OpMulMatArgmaxQ3K);
define_matmul_argmax_runtime!(MatmulArgmaxQ4KRuntime, OpMulMatArgmaxQ4K);
define_matmul_argmax_runtime!(MatmulArgmaxQ6KRuntime, OpMulMatArgmaxQ6K);

pub struct MatmulArgmaxUnexpectedRuntime<'a> {
    pub(super) event: UnexpectedMatmul,
    pub(super) result: &'a Cell<MatmulArgmaxResult>,
}

macro_rules! root_matmul_method {
    ($method:ident, $event:ty, $runtime:ident, $variant:ident) => {
        pub(super) fn $method(&mut self, event: $event) -> MatmulResult {
            let result = Cell::new(Err(MatmulError::UnexpectedEvent));
            self.machine
                .process_event(KernelMachineEvents::$variant($runtime {
                    event,
                    result: &result,
                }))
                .map_err(|_| MatmulError::Internal)?;
            result.get()
        }
    };
}

macro_rules! root_matmul_argmax_method {
    ($method:ident, $event:ty, $runtime:ident, $variant:ident) => {
        pub(super) fn $method(&mut self, event: $event) -> MatmulArgmaxResult {
            let result = Cell::new(Err(MatmulError::UnexpectedEvent));
            self.machine
                .process_event(KernelMachineEvents::$variant($runtime {
                    event,
                    result: &result,
                }))
                .map_err(|_| MatmulError::Internal)?;
            result.get()
        }
    };
}

macro_rules! root_matmul_effect {
    ($name:ident, $runtime:ident) => {
        fn $name(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(self.matmul.process_event(event.event));
            Ok(())
        }
    };
}

macro_rules! root_matmul_argmax_effect {
    ($name:ident, $runtime:ident) => {
        fn $name(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event
                .result
                .set(self.matmul_argmax.process_event(event.event));
            Ok(())
        }
    };
}

pub struct SqrRuntime<'a> {
    pub(super) event: super::power::OpSqr<'a>,
    pub(super) result: &'a Cell<PowerResult>,
}

pub struct SqrtRuntime<'a> {
    pub(super) event: super::power::OpSqrt<'a>,
    pub(super) result: &'a Cell<PowerResult>,
}

pub struct UnsupportedRuntime<'a> {
    pub(super) event: event::Unsupported,
    pub(super) result: &'a Cell<Result<(), Error>>,
}

pub struct LogRuntime<'a> {
    pub(super) event: event::OpLog<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct SinRuntime<'a> {
    pub(super) event: event::OpSin<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct CosRuntime<'a> {
    pub(super) event: event::OpCos<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct SumRuntime<'a> {
    pub(super) event: event::OpSum<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct SumRowsRuntime<'a> {
    pub(super) event: event::OpSumRows<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct MeanRuntime<'a> {
    pub(super) event: event::OpMean<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct ArgmaxRuntime<'a> {
    pub(super) event: event::OpArgmax<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct CountEqualRuntime<'a> {
    pub(super) event: OpCountEqual<'a>,
    pub(super) result: &'a Cell<CountEqualResult>,
}

pub struct GluRuntime<'a> {
    pub(super) event: OpGlu<'a>,
    pub(super) result: &'a Cell<GluResult>,
}

pub struct DiagRuntime<'a> {
    pub(super) event: OpDiag<'a>,
    pub(super) result: &'a Cell<DiagResult>,
}

pub struct DiagMaskInfRuntime<'a> {
    pub(super) event: OpDiagMaskInf<'a>,
    pub(super) result: &'a Cell<DiagResult>,
}

pub struct DiagMaskZeroRuntime<'a> {
    pub(super) event: OpDiagMaskZero<'a>,
    pub(super) result: &'a Cell<DiagResult>,
}

pub struct PadRuntime<'a> {
    pub(super) event: OpPad<'a>,
    pub(super) result: &'a Cell<PadResult>,
}

pub struct PadReflectRuntime<'a> {
    pub(super) event: OpPadReflect1d<'a>,
    pub(super) result: &'a Cell<PadResult>,
}

pub struct TriRuntime<'a> {
    pub(super) event: OpTri<'a>,
    pub(super) result: &'a Cell<PadResult>,
}

pub struct Pool1dRuntime<'a> {
    pub(super) event: OpPool1d<'a>,
    pub(super) result: &'a Cell<PoolResult>,
}

pub struct Pool2dRuntime<'a> {
    pub(super) event: OpPool2d<'a>,
    pub(super) result: &'a Cell<PoolResult>,
}

pub struct Pool2dBackRuntime<'a> {
    pub(super) event: OpPool2dBack<'a>,
    pub(super) result: &'a Cell<PoolResult>,
}

pub struct RollRuntime<'a> {
    pub(super) event: OpRoll<'a>,
    pub(super) result: &'a Cell<ReorderResult>,
}

pub struct UpscaleRuntime<'a> {
    pub(super) event: OpUpscale<'a>,
    pub(super) result: &'a Cell<ReorderResult>,
}

pub struct ArgsortRuntime<'a> {
    pub(super) event: OpArgsort<'a>,
    pub(super) result: &'a Cell<ReorderResult>,
}

pub struct TopKRuntime<'a> {
    pub(super) event: OpTopK<'a>,
    pub(super) result: &'a Cell<ReorderResult>,
}

pub struct Conv2dRuntime<'a> {
    pub(super) event: OpConv2d<'a>,
    pub(super) result: &'a Cell<ConvResult>,
}

pub struct Conv2dDwRuntime<'a> {
    pub(super) event: OpConv2dDw<'a>,
    pub(super) result: &'a Cell<ConvResult>,
}

pub struct Conv3dRuntime<'a> {
    pub(super) event: OpConv3d<'a>,
    pub(super) result: &'a Cell<ConvResult>,
}

pub struct ConvTranspose2dRuntime<'a> {
    pub(super) event: OpConvTranspose2d<'a>,
    pub(super) result: &'a Cell<ConvResult>,
}

pub struct SoftMaxBackRuntime<'a> {
    pub(super) event: OpSoftMaxBack<'a>,
    pub(super) result: &'a Cell<BackwardResult>,
}

pub struct RmsNormBackRuntime<'a> {
    pub(super) event: OpRmsNormBack<'a>,
    pub(super) result: &'a Cell<BackwardResult>,
}

pub struct GetRowsBackRuntime<'a> {
    pub(super) event: OpGetRowsBack<'a>,
    pub(super) result: &'a Cell<BackwardResult>,
}

pub struct SetRowsRuntime<'a> {
    pub(super) event: OpSetRows<'a>,
    pub(super) result: &'a Cell<BackwardResult>,
}

pub struct SetRuntime<'a> {
    pub(super) event: OpSet<'a>,
    pub(super) result: &'a Cell<BackwardResult>,
}

pub struct WinPartRuntime<'a> {
    pub(super) event: OpWinPart<'a>,
    pub(super) result: &'a Cell<WindowResult>,
}

pub struct WinUnpartRuntime<'a> {
    pub(super) event: OpWinUnpart<'a>,
    pub(super) result: &'a Cell<WindowResult>,
}

pub struct Im2ColBackRuntime<'a> {
    pub(super) event: OpIm2ColBack<'a>,
    pub(super) result: &'a Cell<WindowResult>,
}

pub struct Im2Col3dRuntime<'a> {
    pub(super) event: OpIm2Col3d<'a>,
    pub(super) result: &'a Cell<WindowResult>,
}

pub struct RopeBackRuntime<'a> {
    pub(super) event: OpRopeBack<'a>,
    pub(super) result: &'a Cell<PositionalResult>,
}

pub struct TimestepRuntime<'a> {
    pub(super) event: OpTimestepEmbedding<'a>,
    pub(super) result: &'a Cell<PositionalResult>,
}

pub struct SsmConvRuntime<'a> {
    pub(super) event: OpSsmConv<'a>,
    pub(super) result: &'a Cell<SsmResult>,
}

pub struct SsmScanRuntime<'a> {
    pub(super) event: OpSsmScan<'a>,
    pub(super) result: &'a Cell<SsmResult>,
}

pub struct OutProdRuntime<'a> {
    pub(super) event: OpOutProd<'a>,
    pub(super) result: &'a Cell<LinalgResult>,
}

pub struct SolveTriRuntime<'a> {
    pub(super) event: OpSolveTri<'a>,
    pub(super) result: &'a Cell<LinalgResult>,
}

pub struct CrossEntropyRuntime<'a> {
    pub(super) event: OpCrossEntropyLoss<'a>,
    pub(super) result: &'a Cell<TrainResult>,
}
pub struct CrossEntropyBackRuntime<'a> {
    pub(super) event: OpCrossEntropyLossBack<'a>,
    pub(super) result: &'a Cell<TrainResult>,
}
pub struct AdamWRuntime<'a> {
    pub(super) event: OpOptStepAdamW<'a>,
    pub(super) result: &'a Cell<TrainResult>,
}
pub struct SgdRuntime<'a> {
    pub(super) event: OpOptStepSgd<'a>,
    pub(super) result: &'a Cell<TrainResult>,
}
pub struct GetRelPosRuntime<'a> {
    pub(super) event: OpGetRelPos<'a>,
    pub(super) result: &'a Cell<RelResult>,
}
pub struct AddRelPosRuntime<'a> {
    pub(super) event: OpAddRelPos<'a>,
    pub(super) result: &'a Cell<RelResult>,
}

pub struct AddIdRuntime<'a> {
    pub(super) event: OpAddId<'a>,
    pub(super) result: &'a Cell<IndexedResult>,
}
pub struct MulMatIdRuntime<'a> {
    pub(super) event: OpMulMatId<'a>,
    pub(super) result: &'a Cell<IndexedResult>,
}
pub struct Wkv6Runtime<'a> {
    pub(super) event: OpRwkvWkv6<'a>,
    pub(super) result: &'a Cell<RwkvResult>,
}
pub struct Wkv7Runtime<'a> {
    pub(super) event: OpRwkvWkv7<'a>,
    pub(super) result: &'a Cell<RwkvResult>,
}

pub struct GlaRuntime<'a> {
    pub(super) event: OpGatedLinearAttn<'a>,
    pub(super) result: &'a Cell<GlaResult>,
}
pub struct FlashBackRuntime<'a> {
    pub(super) event: OpFlashAttnBack<'a>,
    pub(super) result: &'a Cell<GlaResult>,
}
pub struct Map1Runtime<'a> {
    pub(super) event: OpMapCustom1<'a>,
    pub(super) result: &'a Cell<CustomResult>,
}
pub struct Map2Runtime<'a> {
    pub(super) event: OpMapCustom2<'a>,
    pub(super) result: &'a Cell<CustomResult>,
}
pub struct Map3Runtime<'a> {
    pub(super) event: OpMapCustom3<'a>,
    pub(super) result: &'a Cell<CustomResult>,
}
pub struct CustomRuntime<'a> {
    pub(super) event: OpCustom<'a>,
    pub(super) result: &'a Cell<CustomResult>,
}

pub struct SoftMaxRuntime<'a> {
    pub(super) event: event::OpSoftMax<'a>,
    pub(super) result: &'a Cell<ReductionResult>,
}

pub struct ScaleRuntime<'a> {
    pub(super) event: OpScale<'a>,
    pub(super) result: &'a Cell<ActivationResult>,
}

pub struct ClampRuntime<'a> {
    pub(super) event: OpClamp<'a>,
    pub(super) result: &'a Cell<ActivationResult>,
}

pub struct SiluBackRuntime<'a> {
    pub(super) event: OpSiluBack<'a>,
    pub(super) result: &'a Cell<ActivationResult>,
}

pub struct LeakyReluRuntime<'a> {
    pub(super) event: OpLeakyRelu<'a>,
    pub(super) result: &'a Cell<ActivationResult>,
}

pub struct Add1Runtime<'a> {
    pub(super) event: OpAdd1<'a>,
    pub(super) result: &'a Cell<ElementwiseResult>,
}

pub struct AccRuntime<'a> {
    pub(super) event: OpAcc<'a>,
    pub(super) result: &'a Cell<ElementwiseResult>,
}

pub struct FillRuntime<'a> {
    pub(super) event: OpFill<'a>,
    pub(super) result: &'a Cell<FillArangeResult>,
}

pub struct ArangeRuntime<'a> {
    pub(super) event: OpArange<'a>,
    pub(super) result: &'a Cell<FillArangeResult>,
}

pub struct NormRuntime<'a> {
    pub(super) event: OpNorm<'a>,
    pub(super) result: &'a Cell<NormalizationResult>,
}

pub struct RmsNormRuntime<'a> {
    pub(super) event: OpRmsNorm<'a>,
    pub(super) result: &'a Cell<NormalizationResult>,
}

pub struct GroupNormRuntime<'a> {
    pub(super) event: OpGroupNorm<'a>,
    pub(super) result: &'a Cell<GroupNormResult>,
}

pub struct L2NormRuntime<'a> {
    pub(super) event: OpL2Norm<'a>,
    pub(super) result: &'a Cell<GroupNormResult>,
}

pub struct CumsumRuntime<'a> {
    pub(super) event: OpCumsum<'a>,
    pub(super) result: &'a Cell<SequenceResult>,
}

pub struct RepeatRuntime<'a> {
    pub(super) event: OpRepeat<'a>,
    pub(super) result: &'a Cell<SequenceResult>,
}

pub struct RepeatBackRuntime<'a> {
    pub(super) event: OpRepeatBack<'a>,
    pub(super) result: &'a Cell<SequenceResult>,
}

pub struct ConcatRuntime<'a> {
    pub(super) event: OpConcat<'a>,
    pub(super) result: &'a Cell<SequenceResult>,
}

pub struct CpyRuntime<'a> {
    pub(super) event: OpCpy<'a>,
    pub(super) result: &'a Cell<CopyOutcome>,
}

pub struct ContRuntime<'a> {
    pub(super) event: OpCont<'a>,
    pub(super) result: &'a Cell<CopyOutcome>,
}

pub struct ReshapeRuntime<'a> {
    pub(super) event: OpReshape<'a>,
    pub(super) result: &'a Cell<LayoutOutcome>,
}

pub struct ViewRuntime<'a> {
    pub(super) event: OpView<'a>,
    pub(super) result: &'a Cell<LayoutOutcome>,
}

pub struct PermuteRuntime<'a> {
    pub(super) event: OpPermute<'a>,
    pub(super) result: &'a Cell<LayoutOutcome>,
}

pub struct TransposeRuntime<'a> {
    pub(super) event: OpTranspose<'a>,
    pub(super) result: &'a Cell<LayoutOutcome>,
}

pub struct Context {
    pub(super) activation: ActivationKernel,
    pub(super) binary: BinaryKernel,
    pub(super) broadcast: BroadcastKernel,
    pub(super) conv_transpose_1d: ConvTranspose1dKernel,
    pub(super) f16_matmul: F16MatmulKernel,
    pub(super) flash_attn: FlashAttnKernel,
    pub(super) get_rows: GetRowsKernel,
    pub(super) matmul: MatmulKernel,
    pub(super) matmul_argmax: MatmulArgmaxKernel,
    pub(super) im2col: Im2ColKernel,
    pub(super) elementwise: ElementwiseKernel,
    pub(super) fill_arange: FillArangeKernel,
    pub(super) normalization: NormalizationKernel,
    pub(super) group_norm: GroupNormKernel,
    pub(super) reductions: ReductionKernel,
    pub(super) count_equal: CountEqualKernel,
    pub(super) glu: GluKernel,
    pub(super) diag: DiagKernel,
    pub(super) pad: PadKernel,
    pub(super) pool: PoolKernel,
    pub(super) reorder: ReorderKernel,
    pub(super) conv: ConvKernel,
    pub(super) backward: BackwardKernel,
    pub(super) window: WindowKernel,
    pub(super) positional: PositionalKernel,
    pub(super) ssm: SsmKernel,
    pub(super) linalg: LinalgKernel,
    pub(super) train: TrainKernel,
    pub(super) rel: RelKernel,
    pub(super) indexed: IndexedKernel,
    pub(super) rwkv: RwkvKernel,
    pub(super) gla: GlaKernel,
    pub(super) custom: CustomKernel,
    pub(super) rope: RopeKernel,
    pub(super) unary: UnaryKernel,
    pub(super) sequence: SequenceKernel,
    pub(super) shape: ShapeKernel,
    pub(super) power: PowerKernel,
}

impl Context {
    pub(super) fn new() -> Self {
        Self {
            activation: ActivationKernel::new(),
            binary: BinaryKernel::new(),
            broadcast: BroadcastKernel::new(),
            conv_transpose_1d: ConvTranspose1dKernel::new(),
            f16_matmul: F16MatmulKernel::new(),
            flash_attn: FlashAttnKernel::new(),
            get_rows: GetRowsKernel::new(),
            matmul: MatmulKernel::new(),
            matmul_argmax: MatmulArgmaxKernel::new(),
            im2col: Im2ColKernel::new(),
            elementwise: ElementwiseKernel::new(),
            fill_arange: FillArangeKernel::new(),
            normalization: NormalizationKernel::new(),
            group_norm: GroupNormKernel::new(),
            reductions: ReductionKernel::new(),
            count_equal: CountEqualKernel::new(),
            glu: GluKernel::new(),
            diag: DiagKernel::new(),
            pad: PadKernel::new(),
            pool: PoolKernel::new(),
            reorder: ReorderKernel::new(),
            conv: ConvKernel::new(),
            backward: BackwardKernel::new(),
            window: WindowKernel::new(),
            positional: PositionalKernel::new(),
            ssm: SsmKernel::new(),
            linalg: LinalgKernel::new(),
            train: TrainKernel::new(),
            rel: RelKernel::new(),
            indexed: IndexedKernel::new(),
            rwkv: RwkvKernel::new(),
            gla: GlaKernel::new(),
            custom: CustomKernel::new(),
            rope: RopeKernel::new(),
            unary: UnaryKernel::new(),
            sequence: SequenceKernel::new(),
            shape: ShapeKernel::new(),
            power: PowerKernel::new(),
        }
    }
}

/// Single-writer, run-to-completion actor for numerical kernels.
pub struct Kernel {
    machine: KernelMachineStateMachine<Context>,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Kernel").finish_non_exhaustive()
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel {
    /// Constructs an actor with its own local context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: KernelMachineStateMachine::new(Context::new()),
        }
    }

    /// Dispatches one typed event synchronously and to completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated root machine does not return to `Ready`.
    #[inline]
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(
            self.machine.is(&KernelMachineStates::Ready),
            "kernel machine must return to ready after dispatch"
        );
        result
    }

    /// Reports whether the generated root kernel machine is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&KernelMachineStates::Ready)
    }

    pub(super) fn op_generic(&mut self, event: event::GenericEvent<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Generic(GenericRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_dup(&mut self, event: event::OpDup<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Dup(DupRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_add(&mut self, event: event::OpAdd<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Add(AddRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_sub(&mut self, event: event::OpSub<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sub(SubRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_mul(&mut self, event: event::OpMul<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Mul(MulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_div(&mut self, event: event::OpDiv<'_>) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Div(DivRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_add_tensor_view(&mut self, event: event::OpAddTensorView<'_>) -> BinaryResult {
        let result = Cell::new(Err(BinaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BinaryAdd(BinaryAddRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BinaryError::Internal)?;
        result.get()
    }

    pub(super) fn op_sub_tensor_view(&mut self, event: event::OpSubTensorView<'_>) -> BinaryResult {
        let result = Cell::new(Err(BinaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BinarySub(BinarySubRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BinaryError::Internal)?;
        result.get()
    }

    pub(super) fn op_mul_tensor_view(&mut self, event: event::OpMulTensorView<'_>) -> BinaryResult {
        let result = Cell::new(Err(BinaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BinaryMul(BinaryMulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BinaryError::Internal)?;
        result.get()
    }

    pub(super) fn op_div_tensor_view(&mut self, event: event::OpDivTensorView<'_>) -> BinaryResult {
        let result = Cell::new(Err(BinaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BinaryDiv(BinaryDivRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BinaryError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_tensor_binary(
        &mut self,
        event: event::UnexpectedTensorBinary,
    ) -> BinaryResult {
        let result = Cell::new(Err(BinaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BinaryUnexpected(
                BinaryUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| BinaryError::Internal)?;
        result.get()
    }

    pub(super) fn op_add_broadcast(
        &mut self,
        event: event::OpAddBroadcastRow<'_>,
    ) -> BroadcastOutcome {
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BroadcastAdd(BroadcastAddRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BroadcastError::Internal)?;
        result.get()
    }

    pub(super) fn op_mul_broadcast(
        &mut self,
        event: event::OpMulBroadcastRow<'_>,
    ) -> BroadcastOutcome {
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BroadcastMul(BroadcastMulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BroadcastError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_broadcast(
        &mut self,
        event: event::UnexpectedBroadcast,
    ) -> BroadcastOutcome {
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::BroadcastUnexpected(
                BroadcastUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| BroadcastError::Internal)?;
        result.get()
    }

    pub(super) fn op_conv_transpose_1d(
        &mut self,
        event: event::OpConvTranspose1d<'_>,
    ) -> Result<(), ConvTranspose1dError> {
        let result = Cell::new(Err(ConvTranspose1dError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::ConvTranspose1d(
                ConvTranspose1dRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| ConvTranspose1dError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_conv_transpose_1d(
        &mut self,
        event: event::UnexpectedConvTranspose1d,
    ) -> Result<(), ConvTranspose1dError> {
        let result = Cell::new(Err(ConvTranspose1dError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::ConvTranspose1dUnexpected(
                ConvTranspose1dUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| ConvTranspose1dError::Internal)?;
        result.get()
    }

    pub(super) fn op_mul_mat_f16(&mut self, event: OpMulMatF16<'_>) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::F16Matmul(F16MatmulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_f16_matmul(&mut self) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::F16MatmulUnexpected(
                F16MatmulUnexpectedRuntime { result: &result },
            ))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }

    pub(super) fn op_mul_mat(&mut self, event: OpMulMat<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Matmul(MatmulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    root_matmul_method!(
        op_mul_mat_q4_0,
        OpMulMatQ4_0<'_>,
        MatmulQ4_0Runtime,
        MatmulQ4_0
    );
    root_matmul_method!(
        op_mul_mat_q4_1,
        OpMulMatQ4_1<'_>,
        MatmulQ4_1Runtime,
        MatmulQ4_1
    );
    root_matmul_method!(
        op_mul_mat_q5_0,
        OpMulMatQ5_0<'_>,
        MatmulQ5_0Runtime,
        MatmulQ5_0
    );

    root_matmul_method!(
        op_mul_mat_q8_0,
        OpMulMatQ8_0<'_>,
        MatmulQ8_0Runtime,
        MatmulQ8_0
    );
    root_matmul_method!(
        op_mul_mat_q2_k,
        OpMulMatQ2K<'_>,
        MatmulQ2KRuntime,
        MatmulQ2K
    );
    root_matmul_method!(
        op_mul_mat_q3_k,
        OpMulMatQ3K<'_>,
        MatmulQ3KRuntime,
        MatmulQ3K
    );
    root_matmul_method!(
        op_mul_mat_q4_k,
        OpMulMatQ4K<'_>,
        MatmulQ4KRuntime,
        MatmulQ4K
    );
    root_matmul_method!(
        op_mul_mat_q6_k,
        OpMulMatQ6K<'_>,
        MatmulQ6KRuntime,
        MatmulQ6K
    );

    pub(super) fn unexpected_matmul(&mut self, event: UnexpectedMatmul) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::MatmulUnexpected(
                MatmulUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(super) fn op_mul_mat_argmax(&mut self, event: OpMulMatArgmax<'_>) -> MatmulArgmaxResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::MatmulArgmax(MatmulArgmaxRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    root_matmul_argmax_method!(
        op_mul_mat_argmax_q4_0,
        OpMulMatArgmaxQ4_0<'_>,
        MatmulArgmaxQ4_0Runtime,
        MatmulArgmaxQ4_0
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q4_1,
        OpMulMatArgmaxQ4_1<'_>,
        MatmulArgmaxQ4_1Runtime,
        MatmulArgmaxQ4_1
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q5_0,
        OpMulMatArgmaxQ5_0<'_>,
        MatmulArgmaxQ5_0Runtime,
        MatmulArgmaxQ5_0
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q8_0,
        OpMulMatArgmaxQ8_0<'_>,
        MatmulArgmaxQ8_0Runtime,
        MatmulArgmaxQ8_0
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q2_k,
        OpMulMatArgmaxQ2K<'_>,
        MatmulArgmaxQ2KRuntime,
        MatmulArgmaxQ2K
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q3_k,
        OpMulMatArgmaxQ3K<'_>,
        MatmulArgmaxQ3KRuntime,
        MatmulArgmaxQ3K
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q4_k,
        OpMulMatArgmaxQ4K<'_>,
        MatmulArgmaxQ4KRuntime,
        MatmulArgmaxQ4K
    );
    root_matmul_argmax_method!(
        op_mul_mat_argmax_q6_k,
        OpMulMatArgmaxQ6K<'_>,
        MatmulArgmaxQ6KRuntime,
        MatmulArgmaxQ6K
    );

    pub(super) fn unexpected_matmul_argmax(
        &mut self,
        event: UnexpectedMatmul,
    ) -> MatmulArgmaxResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::MatmulArgmaxUnexpected(
                MatmulArgmaxUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(super) fn op_im2col(&mut self, event: OpIm2Col<'_>) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Im2Col(Im2ColRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }

    pub(super) fn op_im2col_f16(&mut self, event: OpIm2ColF16<'_>) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Im2ColF16(Im2ColF16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_im2col(&mut self, event: UnexpectedIm2Col) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Im2ColUnexpected(
                Im2ColUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows(&mut self, event: OpGetRows<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRows(GetRowsRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_f32_bytes(&mut self, event: OpGetRowsF32Bytes<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsF32Bytes(
                GetRowsF32BytesRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_f16(&mut self, event: OpGetRowsF16<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsF16(GetRowsF16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_bf16(&mut self, event: OpGetRowsBf16<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsBf16(GetRowsBf16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_q4_0(&mut self, event: OpGetRowsQ4_0<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsQ4_0(GetRowsQ4_0Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_q8_0(&mut self, event: OpGetRowsQ8_0<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsQ8_0(GetRowsQ8_0Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_q4_k(&mut self, event: OpGetRowsQ4K<'_>) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsQ4K(GetRowsQ4KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_get_rows(&mut self, event: UnexpectedGetRows) -> GetRowsOutcome {
        let result = Cell::new(Err(GetRowsError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsUnexpected(
                GetRowsUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| GetRowsError::Internal)?;
        result.get()
    }

    pub(super) fn op_rope(&mut self, event: OpRope<'_>) -> Result<(), RopeError> {
        let result = Cell::new(Err(RopeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Rope(RopeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_rope(&mut self, event: UnexpectedRope) -> Result<(), RopeError> {
        let result = Cell::new(Err(RopeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::RopeUnexpected(RopeUnexpectedRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RopeError::Internal)?;
        result.get()
    }

    pub(super) fn op_flash_attn(&mut self, event: OpFlashAttnExt<'_>) -> FlashAttnResult {
        let result = Cell::new(Err(FlashAttnError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::FlashAttn(FlashAttnRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_flash_attn(&mut self, event: UnexpectedFlashAttn) -> FlashAttnResult {
        let result = Cell::new(Err(FlashAttnError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::FlashAttnUnexpected(
                FlashAttnUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| FlashAttnError::Internal)?;
        result.get()
    }

    pub(super) fn op_unary(&mut self, event: OpUnary<'_>) -> UnaryResult {
        let result = Cell::new(Err(UnaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Unary(UnaryRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| UnaryError::Internal)?;
        result.get()
    }

    pub(super) fn unexpected_unary(&mut self, event: UnexpectedUnary) -> UnaryResult {
        let result = Cell::new(Err(UnaryError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::UnaryUnexpected(
                UnaryUnexpectedRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| UnaryError::Internal)?;
        result.get()
    }

    pub(super) fn op_sqr(&mut self, event: event::OpSqr<'_>) -> Result<(), Error> {
        let (input, output) = event.into_parts();
        let input_layout = dense_power_layout(input.len()).ok_or(Error::InvalidShape)?;
        let output_layout = dense_power_layout(output.len()).ok_or(Error::InvalidShape)?;
        let result = Cell::new(Err(PowerError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sqr(SqrRuntime {
                event: super::power::OpSqr::new(
                    super::tensor_view::TensorView::new(input, input_layout),
                    super::tensor_view::TensorViewMut::new(output, output_layout),
                ),
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get().map_err(map_power_error)
    }

    pub(super) fn op_sqrt(&mut self, event: event::OpSqrt<'_>) -> Result<(), Error> {
        let (input, output) = event.into_parts();
        let input_layout = dense_power_layout(input.len()).ok_or(Error::InvalidShape)?;
        let output_layout = dense_power_layout(output.len()).ok_or(Error::InvalidShape)?;
        let result = Cell::new(Err(PowerError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sqrt(SqrtRuntime {
                event: super::power::OpSqrt::new(
                    super::tensor_view::TensorView::new(input, input_layout),
                    super::tensor_view::TensorViewMut::new(output, output_layout),
                ),
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get().map_err(map_power_error)
    }

    pub(super) fn unsupported(&mut self, event: event::Unsupported) -> Result<(), Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Unsupported(UnsupportedRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(super) fn op_log(&mut self, event: event::OpLog<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Log(LogRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_sin(&mut self, event: event::OpSin<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sin(SinRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_cos(&mut self, event: event::OpCos<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Cos(CosRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_sum(&mut self, event: event::OpSum<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sum(SumRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_sum_rows(&mut self, event: event::OpSumRows<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SumRows(SumRowsRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_mean(&mut self, event: event::OpMean<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Mean(MeanRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_argmax(&mut self, event: event::OpArgmax<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Argmax(ArgmaxRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_count_equal(&mut self, event: OpCountEqual<'_>) -> CountEqualResult {
        let result = Cell::new(Err(CountEqualError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::CountEqual(CountEqualRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| CountEqualError::Internal)?;
        result.get()
    }

    pub(super) fn op_glu(&mut self, event: OpGlu<'_>) -> GluResult {
        let result = Cell::new(Err(GluError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Glu(GluRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GluError::Internal)?;
        result.get()
    }

    pub(super) fn op_diag(&mut self, event: OpDiag<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Diag(DiagRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        result.get()
    }

    pub(super) fn op_diag_mask_inf(&mut self, event: OpDiagMaskInf<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DiagMaskInf(DiagMaskInfRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        result.get()
    }

    pub(super) fn op_diag_mask_zero(&mut self, event: OpDiagMaskZero<'_>) -> DiagResult {
        let result = Cell::new(Err(DiagError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DiagMaskZero(DiagMaskZeroRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| DiagError::Internal)?;
        result.get()
    }

    pub(super) fn op_pad(&mut self, event: OpPad<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Pad(PadRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        result.get()
    }

    pub(super) fn op_pad_reflect_1d(&mut self, event: OpPadReflect1d<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::PadReflect(PadReflectRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        result.get()
    }

    pub(super) fn op_tri(&mut self, event: OpTri<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Tri(TriRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        result.get()
    }

    pub(super) fn op_pool_1d(&mut self, event: OpPool1d<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Pool1d(Pool1dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        result.get()
    }

    pub(super) fn op_pool_2d(&mut self, event: OpPool2d<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Pool2d(Pool2dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        result.get()
    }

    pub(super) fn op_pool_2d_back(&mut self, event: OpPool2dBack<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Pool2dBack(Pool2dBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        result.get()
    }

    pub(super) fn op_roll(&mut self, event: OpRoll<'_>) -> ReorderResult {
        let result = Cell::new(Err(ReorderError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Roll(RollRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReorderError::Internal)?;
        result.get()
    }

    pub(super) fn op_upscale(&mut self, event: OpUpscale<'_>) -> ReorderResult {
        let result = Cell::new(Err(ReorderError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Upscale(UpscaleRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReorderError::Internal)?;
        result.get()
    }

    pub(super) fn op_argsort(&mut self, event: OpArgsort<'_>) -> ReorderResult {
        let result = Cell::new(Err(ReorderError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Argsort(ArgsortRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReorderError::Internal)?;
        result.get()
    }

    pub(super) fn op_top_k(&mut self, event: OpTopK<'_>) -> ReorderResult {
        let result = Cell::new(Err(ReorderError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::TopK(TopKRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReorderError::Internal)?;
        result.get()
    }

    pub(super) fn op_conv_2d(&mut self, event: OpConv2d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseConv2d(Conv2dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        result.get()
    }

    pub(super) fn op_conv_2d_dw(&mut self, event: OpConv2dDw<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseConv2dDw(Conv2dDwRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        result.get()
    }

    pub(super) fn op_conv_3d(&mut self, event: OpConv3d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseConv3d(Conv3dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        result.get()
    }

    pub(super) fn op_conv_transpose_2d(&mut self, event: OpConvTranspose2d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseConvTranspose2d(
                ConvTranspose2dRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| ConvError::Internal)?;
        result.get()
    }

    pub(super) fn op_soft_max_back(&mut self, event: OpSoftMaxBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SoftMaxBack(SoftMaxBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        result.get()
    }

    pub(super) fn op_rms_norm_back(&mut self, event: OpRmsNormBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::RmsNormBack(RmsNormBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        result.get()
    }

    pub(super) fn op_get_rows_back(&mut self, event: OpGetRowsBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRowsBack(GetRowsBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        result.get()
    }

    pub(super) fn op_set_rows(&mut self, event: OpSetRows<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SetRows(SetRowsRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        result.get()
    }

    pub(super) fn op_set(&mut self, event: OpSet<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SetSlice(SetRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        result.get()
    }

    pub(super) fn op_win_part(&mut self, event: OpWinPart<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::WinPart(WinPartRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        result.get()
    }

    pub(super) fn op_win_unpart(&mut self, event: OpWinUnpart<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::WinUnpart(WinUnpartRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        result.get()
    }

    pub(super) fn op_im2col_back(&mut self, event: OpIm2ColBack<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseIm2ColBack(Im2ColBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        result.get()
    }

    pub(super) fn op_im2col_3d(&mut self, event: OpIm2Col3d<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::DenseIm2Col3d(Im2Col3dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        result.get()
    }

    pub(super) fn op_rope_back(&mut self, event: OpRopeBack<'_>) -> PositionalResult {
        let result = Cell::new(Err(PositionalError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::RopeBack(RopeBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PositionalError::Internal)?;
        result.get()
    }

    pub(super) fn op_timestep_embedding(
        &mut self,
        event: OpTimestepEmbedding<'_>,
    ) -> PositionalResult {
        let result = Cell::new(Err(PositionalError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Timestep(TimestepRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PositionalError::Internal)?;
        result.get()
    }

    pub(super) fn op_ssm_conv(&mut self, event: OpSsmConv<'_>) -> SsmResult {
        let result = Cell::new(Err(SsmError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SsmConv(SsmConvRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| SsmError::Internal)?;
        result.get()
    }

    pub(super) fn op_ssm_scan(&mut self, event: OpSsmScan<'_>) -> SsmResult {
        let result = Cell::new(Err(SsmError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SsmScan(SsmScanRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| SsmError::Internal)?;
        result.get()
    }

    pub(super) fn op_out_prod(&mut self, event: OpOutProd<'_>) -> LinalgResult {
        let result = Cell::new(Err(LinalgError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::OutProd(OutProdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| LinalgError::Internal)?;
        result.get()
    }

    pub(super) fn op_solve_tri(&mut self, event: OpSolveTri<'_>) -> LinalgResult {
        let result = Cell::new(Err(LinalgError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SolveTri(SolveTriRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| LinalgError::Internal)?;
        result.get()
    }

    pub(super) fn op_cross_entropy_loss(&mut self, event: OpCrossEntropyLoss<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::CrossEntropy(CrossEntropyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        result.get()
    }
    pub(super) fn op_cross_entropy_loss_back(
        &mut self,
        event: OpCrossEntropyLossBack<'_>,
    ) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::CrossEntropyBack(
                CrossEntropyBackRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| TrainError::Internal)?;
        result.get()
    }
    pub(super) fn op_opt_step_adamw(&mut self, event: OpOptStepAdamW<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::AdamW(AdamWRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        result.get()
    }
    pub(super) fn op_opt_step_sgd(&mut self, event: OpOptStepSgd<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Sgd(SgdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        result.get()
    }
    pub(super) fn op_get_rel_pos(&mut self, event: OpGetRelPos<'_>) -> RelResult {
        let result = Cell::new(Err(RelError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GetRelPos(GetRelPosRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RelError::Internal)?;
        result.get()
    }
    pub(super) fn op_add_rel_pos(&mut self, event: OpAddRelPos<'_>) -> RelResult {
        let result = Cell::new(Err(RelError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::AddRelPos(AddRelPosRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| RelError::Internal)?;
        result.get()
    }

    pub(super) fn op_add_id(&mut self, event: OpAddId<'_>) -> IndexedResult {
        let result = Cell::new(Err(IndexedError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::AddId(AddIdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| IndexedError::Internal)?;
        result.get()
    }
    pub(super) fn op_mul_mat_id(&mut self, event: OpMulMatId<'_>) -> IndexedResult {
        let result = Cell::new(Err(IndexedError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::MulMatId(MulMatIdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| IndexedError::Internal)?;
        result.get()
    }
    pub(super) fn op_rwkv_wkv6(&mut self, event: OpRwkvWkv6<'_>) -> RwkvResult {
        let result = Cell::new(Err(RwkvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Wkv6(Wkv6Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| RwkvError::Internal)?;
        result.get()
    }
    pub(super) fn op_rwkv_wkv7(&mut self, event: OpRwkvWkv7<'_>) -> RwkvResult {
        let result = Cell::new(Err(RwkvError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Wkv7(Wkv7Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| RwkvError::Internal)?;
        result.get()
    }

    pub(super) fn op_gated_linear_attn(&mut self, event: OpGatedLinearAttn<'_>) -> GlaResult {
        let result = Cell::new(Err(GlaError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Gla(GlaRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GlaError::Internal)?;
        result.get()
    }
    pub(super) fn op_flash_attn_back(&mut self, event: OpFlashAttnBack<'_>) -> GlaResult {
        let result = Cell::new(Err(GlaError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::FlashBack(FlashBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GlaError::Internal)?;
        result.get()
    }
    pub(super) fn op_map_custom1(&mut self, event: OpMapCustom1<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Map1(Map1Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        result.get()
    }
    pub(super) fn op_map_custom2(&mut self, event: OpMapCustom2<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Map2(Map2Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        result.get()
    }
    pub(super) fn op_map_custom3(&mut self, event: OpMapCustom3<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Map3(Map3Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        result.get()
    }
    pub(super) fn op_custom(&mut self, event: OpCustom<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Custom(CustomRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        result.get()
    }

    pub(super) fn op_soft_max(&mut self, event: event::OpSoftMax<'_>) -> ReductionResult {
        let result = Cell::new(Err(ReductionError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SoftMax(SoftMaxRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ReductionError::Internal)?;
        result.get()
    }

    pub(super) fn op_scale(&mut self, event: OpScale<'_>) -> ActivationResult {
        let result = Cell::new(Err(super::activation::ActivationError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Scale(ScaleRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::activation::ActivationError::Internal)?;
        result.get()
    }

    pub(super) fn op_clamp(&mut self, event: OpClamp<'_>) -> ActivationResult {
        let result = Cell::new(Err(super::activation::ActivationError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Clamp(ClampRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::activation::ActivationError::Internal)?;
        result.get()
    }

    pub(super) fn op_silu_back(&mut self, event: OpSiluBack<'_>) -> ActivationResult {
        let result = Cell::new(Err(super::activation::ActivationError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::SiluBack(SiluBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::activation::ActivationError::Internal)?;
        result.get()
    }

    pub(super) fn op_leaky_relu(&mut self, event: OpLeakyRelu<'_>) -> ActivationResult {
        let result = Cell::new(Err(super::activation::ActivationError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::LeakyRelu(LeakyReluRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::activation::ActivationError::Internal)?;
        result.get()
    }

    pub(super) fn op_add1(&mut self, event: OpAdd1<'_>) -> ElementwiseResult {
        let result = Cell::new(Err(super::elementwise::ElementwiseError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Add1(Add1Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| super::elementwise::ElementwiseError::Internal)?;
        result.get()
    }

    pub(super) fn op_acc(&mut self, event: OpAcc<'_>) -> ElementwiseResult {
        let result = Cell::new(Err(super::elementwise::ElementwiseError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Acc(AccRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::elementwise::ElementwiseError::Internal)?;
        result.get()
    }

    pub(super) fn op_norm(&mut self, event: OpNorm<'_>) -> NormalizationResult {
        let result = Cell::new(Err(
            super::normalization::NormalizationError::UnexpectedEvent,
        ));
        self.machine
            .process_event(KernelMachineEvents::Norm(NormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::normalization::NormalizationError::Internal)?;
        result.get()
    }

    pub(super) fn op_rms_norm(&mut self, event: OpRmsNorm<'_>) -> NormalizationResult {
        let result = Cell::new(Err(
            super::normalization::NormalizationError::UnexpectedEvent,
        ));
        self.machine
            .process_event(KernelMachineEvents::RmsNorm(RmsNormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::normalization::NormalizationError::Internal)?;
        result.get()
    }

    pub(super) fn op_group_norm(&mut self, event: OpGroupNorm<'_>) -> GroupNormResult {
        let result = Cell::new(Err(GroupNormError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::GroupNorm(GroupNormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GroupNormError::Internal)?;
        result.get()
    }

    pub(super) fn op_l2_norm(&mut self, event: OpL2Norm<'_>) -> GroupNormResult {
        let result = Cell::new(Err(GroupNormError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::L2Norm(L2NormRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| GroupNormError::Internal)?;
        result.get()
    }

    pub(super) fn op_cumsum(&mut self, event: OpCumsum<'_>) -> SequenceResult {
        let result = Cell::new(Err(super::sequence::SequenceError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Cumsum(CumsumRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::sequence::SequenceError::Internal)?;
        result.get()
    }

    pub(super) fn op_fill(&mut self, event: OpFill<'_>) -> FillArangeResult {
        let result = Cell::new(Err(FillArangeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Fill(FillRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| FillArangeError::Internal)?;
        result.get()
    }

    pub(super) fn op_arange(&mut self, event: OpArange<'_>) -> FillArangeResult {
        let result = Cell::new(Err(FillArangeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Arange(ArangeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| FillArangeError::Internal)?;
        result.get()
    }

    pub(super) fn op_repeat(&mut self, event: OpRepeat<'_>) -> SequenceResult {
        let result = Cell::new(Err(super::sequence::SequenceError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Repeat(RepeatRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::sequence::SequenceError::Internal)?;
        result.get()
    }

    pub(super) fn op_repeat_back(&mut self, event: OpRepeatBack<'_>) -> SequenceResult {
        let result = Cell::new(Err(super::sequence::SequenceError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::RepeatBack(RepeatBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::sequence::SequenceError::Internal)?;
        result.get()
    }

    pub(super) fn op_concat(&mut self, event: OpConcat<'_>) -> SequenceResult {
        let result = Cell::new(Err(super::sequence::SequenceError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Concat(ConcatRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::sequence::SequenceError::Internal)?;
        result.get()
    }

    pub(super) fn op_cpy(&mut self, event: OpCpy<'_>) -> CopyOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Cpy(CpyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }

    pub(super) fn op_cont(&mut self, event: OpCont<'_>) -> CopyOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Cont(ContRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }

    pub(super) fn op_reshape(&mut self, event: OpReshape<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Reshape(ReshapeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }

    pub(super) fn op_view(&mut self, event: OpView<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::View(ViewRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }

    pub(super) fn op_permute(&mut self, event: OpPermute<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Permute(PermuteRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }

    pub(super) fn op_transpose(&mut self, event: OpTranspose<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(super::shape::ShapeError::UnexpectedEvent));
        self.machine
            .process_event(KernelMachineEvents::Transpose(TransposeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| super::shape::ShapeError::Internal)?;
        result.get()
    }
}

fn guard_generic_unary_valid(event: &GenericRuntime<'_>, subop: event::UnarySubOp) -> bool {
    event.event.operation() == event::KernelOperation::Unary
        && event.event.request().unary() == Some(subop)
        && generic::generic_f32_unary_valid(event.event.request())
}

fn guard_generic_scalar_valid(
    event: &GenericRuntime<'_>,
    operation: event::KernelOperation,
) -> bool {
    event.event.operation() == operation && generic::generic_f32_scalar_valid(event.event.request())
}

fn guard_generic_concat_axis_valid(event: &GenericRuntime<'_>, axis: usize) -> bool {
    event.event.operation() == event::KernelOperation::Concat
        && match axis {
            0 => generic::generic_f32_concat_axis_valid::<0>(event.event.request()),
            1 => generic::generic_f32_concat_axis_valid::<1>(event.event.request()),
            2 => generic::generic_f32_concat_axis_valid::<2>(event.event.request()),
            3 => generic::generic_f32_concat_axis_valid::<3>(event.event.request()),
            _ => false,
        }
}

fn effect_generic_binary<const OP: u8>(event: GenericRuntime<'_>) {
    let mut request = event.event.into_request();
    generic::execute_generic_binary::<OP>(&mut request);
    event.result.set(Ok(()));
}

fn effect_generic_unary<const OP: u8>(event: GenericRuntime<'_>) {
    let mut request = event.event.into_request();
    generic::execute_generic_unary::<OP>(&mut request);
    event.result.set(Ok(()));
}

fn effect_generic_scalar<const OP: u8>(event: GenericRuntime<'_>) {
    let mut request = event.event.into_request();
    generic::execute_generic_scalar::<OP>(&mut request);
    event.result.set(Ok(()));
}

fn effect_generic_concat<const AXIS: usize>(event: GenericRuntime<'_>) {
    let mut request = event.event.into_request();
    generic::execute_generic_concat::<AXIS>(&mut request);
    event.result.set(Ok(()));
}

fn effect_generic_diag_mask<const INF: bool>(event: GenericRuntime<'_>) {
    let mut request = event.event.into_request();
    generic::execute_generic_diag_mask::<INF>(&mut request);
    event.result.set(Ok(()));
}

impl KernelMachineStateMachineContext for Context {
    fn guard_dup_valid(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.input_len() > 0 && event.event.input_len() == event.event.output_len())
    }

    fn guard_dup_invalid(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_dup_valid(event)?)
    }

    fn effect_dup_execute(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        event.event.output.copy_from_slice(event.event.input);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_dup_reject(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn guard_add_valid(&self, event: &AddRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs_len() > 0
            && event.event.lhs_len() == event.event.rhs_len()
            && event.event.lhs_len() == event.event.output_len())
    }

    fn guard_add_invalid(&self, event: &AddRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_add_valid(event)?)
    }

    fn effect_add_execute(&mut self, event: AddRuntime<'_>) -> Result<(), ()> {
        for ((output, lhs), rhs) in event
            .event
            .output
            .iter_mut()
            .zip(event.event.lhs.iter())
            .zip(event.event.rhs.iter())
        {
            *output = *lhs + *rhs;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_add_reject(&mut self, event: AddRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn guard_sub_valid(&self, event: &SubRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs_len() > 0
            && event.event.lhs_len() == event.event.rhs_len()
            && event.event.lhs_len() == event.event.output_len())
    }

    fn guard_sub_invalid(&self, event: &SubRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_sub_valid(event)?)
    }

    fn effect_sub_execute(&mut self, event: SubRuntime<'_>) -> Result<(), ()> {
        for ((output, lhs), rhs) in event
            .event
            .output
            .iter_mut()
            .zip(event.event.lhs.iter())
            .zip(event.event.rhs.iter())
        {
            *output = *lhs - *rhs;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sub_reject(&mut self, event: SubRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn guard_mul_valid(&self, event: &MulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs_len() > 0
            && event.event.lhs_len() == event.event.rhs_len()
            && event.event.lhs_len() == event.event.output_len())
    }

    fn guard_mul_invalid(&self, event: &MulRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_mul_valid(event)?)
    }

    fn effect_mul_execute(&mut self, event: MulRuntime<'_>) -> Result<(), ()> {
        for ((output, lhs), rhs) in event
            .event
            .output
            .iter_mut()
            .zip(event.event.lhs.iter())
            .zip(event.event.rhs.iter())
        {
            *output = *lhs * *rhs;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_reject(&mut self, event: MulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn guard_div_valid(&self, event: &DivRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs_len() > 0
            && event.event.lhs_len() == event.event.rhs_len()
            && event.event.lhs_len() == event.event.output_len())
    }

    fn guard_div_invalid(&self, event: &DivRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_div_valid(event)?)
    }

    fn effect_div_execute(&mut self, event: DivRuntime<'_>) -> Result<(), ()> {
        for ((output, lhs), rhs) in event
            .event
            .output
            .iter_mut()
            .zip(event.event.lhs.iter())
            .zip(event.event.rhs.iter())
        {
            *output = *lhs / *rhs;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_div_reject(&mut self, event: DivRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn effect_sqr_execute(&mut self, event: SqrRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.power.process_event(event.event));
        Ok(())
    }

    fn effect_sqrt_execute(&mut self, event: SqrtRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.power.process_event(event.event));
        Ok(())
    }

    fn effect_unsupported(&mut self, event: UnsupportedRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(Err(Error::UnsupportedOperation(event.event.operation())));
        Ok(())
    }

    fn guard_generic_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.request().validate().is_ok())
    }

    fn guard_generic_dup_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Dup
            && generic::generic_f32_copy_valid(event.event.request()))
    }

    fn guard_generic_add_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Add
            && generic::generic_f32_binary_valid(event.event.request()))
    }

    fn guard_generic_sub_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Sub
            && generic::generic_f32_binary_valid(event.event.request()))
    }

    fn guard_generic_mul_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Mul
            && generic::generic_f32_binary_valid(event.event.request()))
    }

    fn guard_generic_div_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Div
            && generic::generic_f32_binary_valid(event.event.request()))
    }

    fn guard_generic_add1_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_scalar_valid(
            event,
            event::KernelOperation::Add1,
        ))
    }

    fn guard_generic_scale_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_scalar_valid(
            event,
            event::KernelOperation::Scale,
        ))
    }

    fn guard_generic_clamp_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Clamp
            && generic::generic_f32_clamp_valid(event.event.request()))
    }

    fn guard_generic_leaky_relu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_scalar_valid(
            event,
            event::KernelOperation::LeakyRelu,
        ))
    }

    fn guard_generic_silu_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::SiluBack
            && generic::generic_f32_silu_back_valid(event.event.request()))
    }

    fn guard_generic_cumsum_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Cumsum
            && generic::generic_f32_cumsum_valid(event.event.request()))
    }

    fn guard_generic_repeat_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Repeat
            && generic::generic_f32_repeat_valid(event.event.request()))
    }

    fn guard_generic_repeat_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::RepeatBack
                && generic::generic_f32_repeat_back_valid(event.event.request()),
        )
    }

    fn guard_generic_concat_0_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_concat_axis_valid(event, 0))
    }

    fn guard_generic_concat_1_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_concat_axis_valid(event, 1))
    }

    fn guard_generic_concat_2_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_concat_axis_valid(event, 2))
    }

    fn guard_generic_concat_3_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_concat_axis_valid(event, 3))
    }

    fn guard_generic_diag_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Diag
            && generic::generic_f32_diag_valid(event.event.request()))
    }

    fn guard_generic_diag_mask_inf_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::DiagMaskInf
                && generic::generic_f32_mask_valid(event.event.request()),
        )
    }

    fn guard_generic_diag_mask_zero_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::DiagMaskZero
                && generic::generic_f32_mask_valid(event.event.request()),
        )
    }

    fn guard_generic_soft_max_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::SoftMaxBack
                && generic::generic_f32_softmax_back_valid(event.event.request()),
        )
    }

    fn guard_generic_rms_norm_back_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::RmsNormBack
                && generic::generic_f32_rms_norm_back_valid(event.event.request()),
        )
    }

    fn guard_generic_l2_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::L2Norm
            && generic::generic_f32_l2_norm_valid(event.event.request()))
    }

    fn guard_generic_count_equal_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::CountEqual
                && generic::generic_f32_count_equal_valid(event.event.request()),
        )
    }

    fn guard_generic_fill_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Fill
            && generic::generic_f32_fill_valid(event.event.request()))
    }

    fn guard_generic_arange_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Arange
            && generic::generic_f32_arange_valid(event.event.request()))
    }

    fn guard_generic_mul_mat_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::MulMat
            && generic::generic_f32_matmul_valid(event.event.request()))
    }

    fn guard_generic_mul_mat_f16_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::MulMat
            && generic::generic_f16_matmul_valid(event.event.request()))
    }

    fn guard_generic_mul_mat_argmax_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(
            event.event.operation() == event::KernelOperation::MulMatArgmax
                && generic::generic_f32_matmul_argmax_valid(event.event.request()),
        )
    }

    fn guard_generic_sqr_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Sqr
            && generic::generic_f32_unary_valid(event.event.request()))
    }

    fn guard_generic_sqrt_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Sqrt
            && generic::generic_f32_unary_valid(event.event.request()))
    }

    fn guard_generic_log_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Log
            && generic::generic_f32_unary_valid(event.event.request()))
    }

    fn guard_generic_sin_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Sin
            && generic::generic_f32_unary_valid(event.event.request()))
    }

    fn guard_generic_cos_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Cos
            && generic::generic_f32_unary_valid(event.event.request()))
    }

    fn guard_generic_sum_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Sum
            && generic::generic_f32_reduce_valid(event.event.request()))
    }

    fn guard_generic_mean_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Mean
            && generic::generic_f32_reduce_valid(event.event.request()))
    }

    fn guard_generic_argmax_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Argmax
            && generic::generic_f32_reduce_valid(event.event.request()))
    }

    fn guard_generic_soft_max_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::SoftMax
            && generic::generic_f32_rows_valid(event.event.request()))
    }

    fn guard_generic_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::Norm
            && generic::generic_f32_norm_valid(event.event.request()))
    }

    fn guard_generic_rms_norm_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.operation() == event::KernelOperation::RmsNorm
            && generic::generic_f32_norm_valid(event.event.request()))
    }

    fn guard_generic_unary_abs_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Abs))
    }

    fn guard_generic_unary_neg_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Neg))
    }

    fn guard_generic_unary_tanh_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Tanh))
    }

    fn guard_generic_unary_elu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Elu))
    }

    fn guard_generic_unary_relu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Relu))
    }

    fn guard_generic_unary_gelu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Gelu))
    }

    fn guard_generic_unary_silu_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Silu))
    }

    fn guard_generic_unary_exp_valid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(guard_generic_unary_valid(event, event::UnarySubOp::Exp))
    }

    fn guard_generic_known_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        let request = event.event.request();
        if request.validate().is_err() {
            return Ok(false);
        }
        let invalid = match event.event.operation() {
            event::KernelOperation::Dup => !generic::generic_f32_copy_valid(request),
            event::KernelOperation::Add
            | event::KernelOperation::Sub
            | event::KernelOperation::Mul
            | event::KernelOperation::Div => !generic::generic_f32_binary_valid(request),
            event::KernelOperation::Add1
            | event::KernelOperation::Scale
            | event::KernelOperation::LeakyRelu => !generic::generic_f32_scalar_valid(request),
            event::KernelOperation::Clamp => !generic::generic_f32_clamp_valid(request),
            event::KernelOperation::SiluBack => !generic::generic_f32_silu_back_valid(request),
            event::KernelOperation::Cumsum => !generic::generic_f32_cumsum_valid(request),
            event::KernelOperation::Repeat => !generic::generic_f32_repeat_valid(request),
            event::KernelOperation::RepeatBack => !generic::generic_f32_repeat_back_valid(request),
            event::KernelOperation::Concat => {
                !generic::generic_f32_concat_axis_valid::<0>(request)
                    && !generic::generic_f32_concat_axis_valid::<1>(request)
                    && !generic::generic_f32_concat_axis_valid::<2>(request)
                    && !generic::generic_f32_concat_axis_valid::<3>(request)
            }
            event::KernelOperation::Diag => !generic::generic_f32_diag_valid(request),
            event::KernelOperation::DiagMaskInf | event::KernelOperation::DiagMaskZero => {
                !generic::generic_f32_mask_valid(request)
            }
            event::KernelOperation::SoftMaxBack => {
                !generic::generic_f32_softmax_back_valid(request)
            }
            event::KernelOperation::RmsNormBack => {
                !generic::generic_f32_rms_norm_back_valid(request)
            }
            event::KernelOperation::L2Norm => !generic::generic_f32_l2_norm_valid(request),
            event::KernelOperation::CountEqual => !generic::generic_f32_count_equal_valid(request),
            event::KernelOperation::Fill => !generic::generic_f32_fill_valid(request),
            event::KernelOperation::Arange => !generic::generic_f32_arange_valid(request),
            event::KernelOperation::MulMat => {
                !generic::generic_f32_matmul_valid(request)
                    && !generic::generic_f16_matmul_valid(request)
            }
            event::KernelOperation::MulMatArgmax => {
                !generic::generic_f32_matmul_argmax_valid(request)
            }
            event::KernelOperation::Sqr
            | event::KernelOperation::Sqrt
            | event::KernelOperation::Log
            | event::KernelOperation::Sin
            | event::KernelOperation::Cos => !generic::generic_f32_unary_valid(request),
            event::KernelOperation::Sum
            | event::KernelOperation::Mean
            | event::KernelOperation::Argmax => !generic::generic_f32_reduce_valid(request),
            event::KernelOperation::SoftMax => !generic::generic_f32_rows_valid(request),
            event::KernelOperation::Norm | event::KernelOperation::RmsNorm => {
                !generic::generic_f32_norm_valid(request)
            }
            event::KernelOperation::Unary => {
                !matches!(
                    request.unary(),
                    Some(
                        event::UnarySubOp::Abs
                            | event::UnarySubOp::Neg
                            | event::UnarySubOp::Tanh
                            | event::UnarySubOp::Elu
                            | event::UnarySubOp::Relu
                            | event::UnarySubOp::Gelu
                            | event::UnarySubOp::Silu
                            | event::UnarySubOp::Exp,
                    )
                ) || !generic::generic_f32_unary_valid(request)
            }
            _ => false,
        };
        Ok(invalid)
    }

    fn effect_generic_dup(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_copy(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_add(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_binary::<0>(event);
        Ok(())
    }

    fn effect_generic_sub(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_binary::<1>(event);
        Ok(())
    }

    fn effect_generic_mul(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_binary::<2>(event);
        Ok(())
    }

    fn effect_generic_div(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_binary::<3>(event);
        Ok(())
    }

    fn effect_generic_add1(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_scalar::<0>(event);
        Ok(())
    }

    fn effect_generic_scale(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_scalar::<1>(event);
        Ok(())
    }

    fn effect_generic_clamp(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_clamp(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_leaky_relu(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_scalar::<2>(event);
        Ok(())
    }

    fn effect_generic_silu_back(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_silu_back(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_cumsum(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_cumsum(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_repeat(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_repeat(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_repeat_back(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_repeat_back(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_concat_0(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_concat::<0>(event);
        Ok(())
    }

    fn effect_generic_concat_1(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_concat::<1>(event);
        Ok(())
    }

    fn effect_generic_concat_2(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_concat::<2>(event);
        Ok(())
    }

    fn effect_generic_concat_3(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_concat::<3>(event);
        Ok(())
    }

    fn effect_generic_diag(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_diag(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_diag_mask_inf(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_diag_mask::<true>(event);
        Ok(())
    }

    fn effect_generic_diag_mask_zero(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_diag_mask::<false>(event);
        Ok(())
    }

    fn effect_generic_soft_max_back(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_softmax_back(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_rms_norm_back(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_rms_norm_back(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_l2_norm(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_l2_norm(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_count_equal(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_count_equal(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_fill(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_fill(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_arange(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_arange(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_mul_mat(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_matmul(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_mul_mat_f16(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_f16_matmul(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_mul_mat_argmax(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_matmul_argmax(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_sqr(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<8>(event);
        Ok(())
    }

    fn effect_generic_sqrt(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<9>(event);
        Ok(())
    }

    fn effect_generic_log(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<10>(event);
        Ok(())
    }

    fn effect_generic_sin(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<11>(event);
        Ok(())
    }

    fn effect_generic_cos(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<12>(event);
        Ok(())
    }

    fn effect_generic_sum(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_reduce_sum(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_mean(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_reduce_mean(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_argmax(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_argmax(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_soft_max(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_soft_max(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_norm(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_norm::<false>(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_rms_norm(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        let mut request = event.event.into_request();
        generic::execute_generic_norm::<true>(&mut request);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_generic_unary_abs(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<0>(event);
        Ok(())
    }

    fn effect_generic_unary_neg(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<1>(event);
        Ok(())
    }

    fn effect_generic_unary_tanh(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<2>(event);
        Ok(())
    }

    fn effect_generic_unary_elu(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<3>(event);
        Ok(())
    }

    fn effect_generic_unary_relu(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<4>(event);
        Ok(())
    }

    fn effect_generic_unary_gelu(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<5>(event);
        Ok(())
    }

    fn effect_generic_unary_silu(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<6>(event);
        Ok(())
    }

    fn effect_generic_unary_exp(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        effect_generic_unary::<7>(event);
        Ok(())
    }

    fn guard_generic_invalid(&self, event: &GenericRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.request().validate().is_err())
    }

    fn effect_generic_unsupported(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::UnsupportedOperation(
            event::UnsupportedOperation::Generic(event.event.operation()),
        )));
        Ok(())
    }

    fn effect_generic_invalid(&mut self, event: GenericRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidShape));
        Ok(())
    }

    fn effect_log(&mut self, event: LogRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_sin(&mut self, event: SinRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_cos(&mut self, event: CosRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_sum(&mut self, event: SumRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_sum_rows(&mut self, event: SumRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_mean(&mut self, event: MeanRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_argmax(&mut self, event: ArgmaxRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_count_equal(&mut self, event: CountEqualRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.count_equal.process_event(event.event));
        Ok(())
    }

    fn effect_glu(&mut self, event: GluRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.glu.process_event(event.event));
        Ok(())
    }

    fn effect_diag(&mut self, event: DiagRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.diag.process_event(event.event));
        Ok(())
    }

    fn effect_diag_mask_inf(&mut self, event: DiagMaskInfRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.diag.process_event(event.event));
        Ok(())
    }

    fn effect_diag_mask_zero(&mut self, event: DiagMaskZeroRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.diag.process_event(event.event));
        Ok(())
    }

    fn effect_pad(&mut self, event: PadRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pad.process_event(event.event));
        Ok(())
    }

    fn effect_pad_reflect(&mut self, event: PadReflectRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pad.process_event(event.event));
        Ok(())
    }

    fn effect_tri(&mut self, event: TriRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pad.process_event(event.event));
        Ok(())
    }

    fn effect_pool_1d(&mut self, event: Pool1dRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pool.process_event(event.event));
        Ok(())
    }

    fn effect_pool_2d(&mut self, event: Pool2dRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pool.process_event(event.event));
        Ok(())
    }

    fn effect_pool_2d_back(&mut self, event: Pool2dBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.pool.process_event(event.event));
        Ok(())
    }

    fn effect_roll(&mut self, event: RollRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reorder.process_event(event.event));
        Ok(())
    }

    fn effect_upscale(&mut self, event: UpscaleRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reorder.process_event(event.event));
        Ok(())
    }

    fn effect_argsort(&mut self, event: ArgsortRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reorder.process_event(event.event));
        Ok(())
    }

    fn effect_top_k(&mut self, event: TopKRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reorder.process_event(event.event));
        Ok(())
    }

    fn effect_dense_conv_2d(&mut self, event: Conv2dRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.conv.process_event(event.event));
        Ok(())
    }

    fn effect_dense_conv_2d_dw(&mut self, event: Conv2dDwRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.conv.process_event(event.event));
        Ok(())
    }

    fn effect_dense_conv_3d(&mut self, event: Conv3dRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.conv.process_event(event.event));
        Ok(())
    }

    fn effect_dense_conv_transpose_2d(
        &mut self,
        event: ConvTranspose2dRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.conv.process_event(event.event));
        Ok(())
    }

    fn effect_soft_max_back(&mut self, event: SoftMaxBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.backward.process_event(event.event));
        Ok(())
    }

    fn effect_rms_norm_back(&mut self, event: RmsNormBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.backward.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_back(&mut self, event: GetRowsBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.backward.process_event(event.event));
        Ok(())
    }

    fn effect_set_rows(&mut self, event: SetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.backward.process_event(event.event));
        Ok(())
    }

    fn effect_set_slice(&mut self, event: SetRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.backward.process_event(event.event));
        Ok(())
    }

    fn effect_win_part(&mut self, event: WinPartRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.window.process_event(event.event));
        Ok(())
    }

    fn effect_win_unpart(&mut self, event: WinUnpartRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.window.process_event(event.event));
        Ok(())
    }

    fn effect_im2col_back(&mut self, event: Im2ColBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.window.process_event(event.event));
        Ok(())
    }

    fn effect_im2col_3d_dense(&mut self, event: Im2Col3dRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.window.process_event(event.event));
        Ok(())
    }

    fn effect_rope_back(&mut self, event: RopeBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.positional.process_event(event.event));
        Ok(())
    }

    fn effect_timestep_embedding(&mut self, event: TimestepRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.positional.process_event(event.event));
        Ok(())
    }

    fn effect_ssm_conv(&mut self, event: SsmConvRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.ssm.process_event(event.event));
        Ok(())
    }

    fn effect_ssm_scan(&mut self, event: SsmScanRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.ssm.process_event(event.event));
        Ok(())
    }

    fn effect_out_prod(&mut self, event: OutProdRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.linalg.process_event(event.event));
        Ok(())
    }

    fn effect_solve_tri(&mut self, event: SolveTriRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.linalg.process_event(event.event));
        Ok(())
    }

    fn effect_cross_entropy(&mut self, event: CrossEntropyRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.train.process_event(event.event));
        Ok(())
    }
    fn effect_cross_entropy_back(&mut self, event: CrossEntropyBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.train.process_event(event.event));
        Ok(())
    }
    fn effect_adamw(&mut self, event: AdamWRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.train.process_event(event.event));
        Ok(())
    }
    fn effect_sgd(&mut self, event: SgdRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.train.process_event(event.event));
        Ok(())
    }
    fn effect_get_rel_pos(&mut self, event: GetRelPosRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rel.process_event(event.event));
        Ok(())
    }
    fn effect_add_rel_pos(&mut self, event: AddRelPosRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rel.process_event(event.event));
        Ok(())
    }

    fn effect_add_id(&mut self, event: AddIdRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.indexed.process_event(event.event));
        Ok(())
    }
    fn effect_mul_mat_id(&mut self, event: MulMatIdRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.indexed.process_event(event.event));
        Ok(())
    }
    fn effect_wkv6(&mut self, event: Wkv6Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.rwkv.process_event(event.event));
        Ok(())
    }
    fn effect_wkv7(&mut self, event: Wkv7Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.rwkv.process_event(event.event));
        Ok(())
    }

    fn effect_gla(&mut self, event: GlaRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.gla.process_event(event.event));
        Ok(())
    }
    fn effect_flash_back(&mut self, event: FlashBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.gla.process_event(event.event));
        Ok(())
    }
    fn effect_map1(&mut self, event: Map1Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.custom.process_event(event.event));
        Ok(())
    }
    fn effect_map2(&mut self, event: Map2Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.custom.process_event(event.event));
        Ok(())
    }
    fn effect_map3(&mut self, event: Map3Runtime<'_>) -> Result<(), ()> {
        event.result.set(self.custom.process_event(event.event));
        Ok(())
    }
    fn effect_custom(&mut self, event: CustomRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.custom.process_event(event.event));
        Ok(())
    }

    fn effect_soft_max(&mut self, event: SoftMaxRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.reductions.process_event(event.event));
        Ok(())
    }

    fn effect_binary_add(&mut self, event: BinaryAddRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.binary.process_event(event.event));
        Ok(())
    }

    fn effect_binary_sub(&mut self, event: BinarySubRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.binary.process_event(event.event));
        Ok(())
    }

    fn effect_binary_mul(&mut self, event: BinaryMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.binary.process_event(event.event));
        Ok(())
    }

    fn effect_binary_div(&mut self, event: BinaryDivRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.binary.process_event(event.event));
        Ok(())
    }

    fn effect_binary_unexpected(&mut self, event: BinaryUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.binary.process_event(event.event));
        Ok(())
    }

    fn effect_broadcast_add(&mut self, event: BroadcastAddRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.broadcast.process_event(event.event));
        Ok(())
    }

    fn effect_broadcast_mul(&mut self, event: BroadcastMulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.broadcast.process_event(event.event));
        Ok(())
    }

    fn effect_broadcast_unexpected(
        &mut self,
        event: BroadcastUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(self.broadcast.process_event(event.event));
        Ok(())
    }

    fn effect_conv_transpose_1d(&mut self, event: ConvTranspose1dRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.conv_transpose_1d.process_event(event.event));
        Ok(())
    }

    fn effect_conv_transpose_1d_unexpected(
        &mut self,
        event: ConvTranspose1dUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .set(self.conv_transpose_1d.process_event(event.event));
        Ok(())
    }

    fn effect_f16_matmul(&mut self, event: F16MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.f16_matmul.process_event(event.event));
        Ok(())
    }

    fn effect_f16_matmul_unexpected(
        &mut self,
        event: F16MatmulUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(
            self.f16_matmul
                .process_event(super::f16_matmul::UnexpectedF16Matmul),
        );
        Ok(())
    }

    fn effect_matmul(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.matmul.process_event(event.event));
        Ok(())
    }

    root_matmul_effect!(effect_matmul_q4_0, MatmulQ4_0Runtime);
    root_matmul_effect!(effect_matmul_q4_1, MatmulQ4_1Runtime);
    root_matmul_effect!(effect_matmul_q5_0, MatmulQ5_0Runtime);
    root_matmul_effect!(effect_matmul_q8_0, MatmulQ8_0Runtime);

    root_matmul_effect!(effect_matmul_q2_k, MatmulQ2KRuntime);
    root_matmul_effect!(effect_matmul_q3_k, MatmulQ3KRuntime);
    root_matmul_effect!(effect_matmul_q4_k, MatmulQ4KRuntime);
    root_matmul_effect!(effect_matmul_q6_k, MatmulQ6KRuntime);

    fn effect_matmul_unexpected(&mut self, event: MatmulUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.matmul.process_event(event.event));
        Ok(())
    }

    fn effect_matmul_argmax(&mut self, event: MatmulArgmaxRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.matmul_argmax.process_event(event.event));
        Ok(())
    }

    root_matmul_argmax_effect!(effect_matmul_argmax_q4_0, MatmulArgmaxQ4_0Runtime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q4_1, MatmulArgmaxQ4_1Runtime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q5_0, MatmulArgmaxQ5_0Runtime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q8_0, MatmulArgmaxQ8_0Runtime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q2_k, MatmulArgmaxQ2KRuntime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q3_k, MatmulArgmaxQ3KRuntime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q4_k, MatmulArgmaxQ4KRuntime);
    root_matmul_argmax_effect!(effect_matmul_argmax_q6_k, MatmulArgmaxQ6KRuntime);

    fn effect_matmul_argmax_unexpected(
        &mut self,
        event: MatmulArgmaxUnexpectedRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .set(self.matmul_argmax.process_event(event.event));
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
        event.result.set(self.im2col.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows(&mut self, event: GetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.get_rows.process_event(event.event));
        Ok(())
    }

    fn effect_get_rows_f32_bytes(&mut self, event: GetRowsF32BytesRuntime<'_>) -> Result<(), ()> {
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

    fn effect_get_rows_q4_k(&mut self, event: GetRowsQ4KRuntime<'_>) -> Result<(), ()> {
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

    fn effect_rope(&mut self, event: RopeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
        Ok(())
    }

    fn effect_rope_unexpected(&mut self, event: RopeUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.rope.process_event(event.event));
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
        event.result.set(self.flash_attn.process_event(event.event));
        Ok(())
    }

    fn effect_unary(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.unary.process_event(event.event));
        Ok(())
    }

    fn effect_unary_unexpected(&mut self, event: UnaryUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.unary.process_event(event.event));
        Ok(())
    }

    fn effect_scale(&mut self, event: ScaleRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.activation.process_event(event.event));
        Ok(())
    }

    fn effect_clamp(&mut self, event: ClampRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.activation.process_event(event.event));
        Ok(())
    }

    fn effect_silu_back(&mut self, event: SiluBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.activation.process_event(event.event));
        Ok(())
    }

    fn effect_leaky_relu(&mut self, event: LeakyReluRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.activation.process_event(event.event));
        Ok(())
    }

    fn effect_add1(&mut self, event: Add1Runtime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.elementwise.process_event(event.event));
        Ok(())
    }

    fn effect_acc(&mut self, event: AccRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.elementwise.process_event(event.event));
        Ok(())
    }

    fn effect_fill(&mut self, event: FillRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.fill_arange.process_event(event.event));
        Ok(())
    }

    fn effect_arange(&mut self, event: ArangeRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(self.fill_arange.process_event(event.event));
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

    fn effect_group_norm(&mut self, event: GroupNormRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.group_norm.process_event(event.event));
        Ok(())
    }

    fn effect_l2_norm(&mut self, event: L2NormRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.group_norm.process_event(event.event));
        Ok(())
    }

    fn effect_cumsum(&mut self, event: CumsumRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.sequence.process_event(event.event));
        Ok(())
    }

    fn effect_repeat(&mut self, event: RepeatRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.sequence.process_event(event.event));
        Ok(())
    }

    fn effect_repeat_back(&mut self, event: RepeatBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.sequence.process_event(event.event));
        Ok(())
    }

    fn effect_concat(&mut self, event: ConcatRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.sequence.process_event(event.event));
        Ok(())
    }

    fn effect_cpy(&mut self, event: CpyRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_cont(&mut self, event: ContRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_reshape(&mut self, event: ReshapeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_view(&mut self, event: ViewRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_permute(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_transpose(&mut self, event: TransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(self.shape.process_event(event.event));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
