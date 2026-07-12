//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

// --- machine KernelAarch64 from emel.cpp/src/emel/kernel/aarch64/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpAcc;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpAdd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpAdd1;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpAddId;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpAddRelPos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpArange;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpArgmax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpArgsort;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpClamp;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConcat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCont;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConv2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConv2dDw;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConv3d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConvTranspose1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpConvTranspose2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCountEqual;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCpy;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCrossEntropyLoss;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCrossEntropyLossBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCumsum;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpCustom;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpDiag;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpDiagMaskInf;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpDiagMaskZero;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpDiv;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpDup;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpFill;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpFlashAttnBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpFlashAttnExt;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGatedLinearAttn;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGetRelPos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGetRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGetRowsBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGlu;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpGroupNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpIm2col;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpIm2col3d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpIm2colBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpL2Norm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpLeakyRelu;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpLog;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMapCustom1;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMapCustom2;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMapCustom3;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMean;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMul;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMulMat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMulMatArgmax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpMulMatId;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpOptStepAdamw;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpOptStepSgd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpOutProd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPad;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPadReflect1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPermute;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPool1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPool2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpPool2dBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRepeat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRepeatBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpReshape;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRmsNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRmsNormBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRoll;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRope;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRopeBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRwkvWkv6;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpRwkvWkv7;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpScale;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSet;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSetRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSiluBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSin;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSoftMax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSoftMaxBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSolveTri;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSqr;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSqrt;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSsmConv;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSsmScan;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSub;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSum;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpSumRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpTimestepEmbedding;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpTopK;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpTranspose;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpTri;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpUnary;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpUpscale;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpView;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpWinPart;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchOpWinUnpart;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelAarch64EventDispatchRequest;

sml! {
    KernelAarch64 {
        "ready"_s <= *"ready"_s + event<EmelKernelAarch64EventDispatchRequest> / exec_dispatch,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDup> [simd_op_dup] / exec_simd_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDup> [valid_op_dup] / exec_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDup> [invalid_op_dup] / reject_invalid_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd> [simd_op_add] / exec_simd_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd> [valid_op_add_equal] / exec_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd> [valid_op_add_broadcast_row] / exec_scalar_op_add_broadcast_row,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd> [invalid_op_add] / reject_invalid_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAddId> [valid_op_add_id] / exec_op_add_id,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAddId> [invalid_op_add_id] / reject_invalid_op_add_id,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd1> [valid_op_add1] / exec_op_add1,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAdd1> [invalid_op_add1] / reject_invalid_op_add1,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAcc> [valid_op_acc] / exec_op_acc,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAcc> [invalid_op_acc] / reject_invalid_op_acc,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSub> [simd_op_sub] / exec_simd_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSub> [valid_op_sub] / exec_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSub> [invalid_op_sub] / reject_invalid_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMul> [simd_op_mul] / exec_simd_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMul> [valid_op_mul_equal] / exec_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMul> [valid_op_mul_broadcast_row] / exec_scalar_op_mul_broadcast_row,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMul> [invalid_op_mul] / reject_invalid_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiv> [simd_op_div] / exec_simd_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiv> [valid_op_div] / exec_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiv> [invalid_op_div] / reject_invalid_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqr> [simd_op_sqr] / exec_simd_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqr> [valid_op_sqr] / exec_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqr> [invalid_op_sqr] / reject_invalid_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqrt> [simd_op_sqrt] / exec_simd_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqrt> [valid_op_sqrt] / exec_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSqrt> [invalid_op_sqrt] / reject_invalid_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpLog> [valid_op_log] / exec_op_log,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpLog> [invalid_op_log] / reject_invalid_op_log,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSin> [valid_op_sin] / exec_op_sin,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSin> [invalid_op_sin] / reject_invalid_op_sin,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCos> [valid_op_cos] / exec_op_cos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCos> [invalid_op_cos] / reject_invalid_op_cos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSum> [valid_op_sum] / exec_op_sum,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSum> [invalid_op_sum] / reject_invalid_op_sum,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSumRows> [valid_op_sum_rows] / exec_op_sum_rows,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSumRows> [invalid_op_sum_rows] / reject_invalid_op_sum_rows,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCumsum> [valid_op_cumsum] / exec_op_cumsum,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCumsum> [invalid_op_cumsum] / reject_invalid_op_cumsum,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMean> [valid_op_mean] / exec_op_mean,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMean> [invalid_op_mean] / reject_invalid_op_mean,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArgmax> [valid_op_argmax] / exec_op_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArgmax> [invalid_op_argmax] / reject_invalid_op_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCountEqual> [valid_op_count_equal] / exec_op_count_equal,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCountEqual> [invalid_op_count_equal] / reject_invalid_op_count_equal,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRepeat> [valid_op_repeat] / exec_op_repeat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRepeat> [invalid_op_repeat] / reject_invalid_op_repeat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRepeatBack> [valid_op_repeat_back] / exec_op_repeat_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRepeatBack> [invalid_op_repeat_back] / reject_invalid_op_repeat_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConcat> [valid_op_concat] / exec_op_concat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConcat> [invalid_op_concat] / reject_invalid_op_concat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSiluBack> [valid_op_silu_back] / exec_op_silu_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSiluBack> [invalid_op_silu_back] / reject_invalid_op_silu_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpNorm> [valid_op_norm] / exec_op_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpNorm> [invalid_op_norm] / reject_invalid_op_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRmsNorm> [valid_op_rms_norm] / exec_op_rms_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRmsNorm> [invalid_op_rms_norm] / reject_invalid_op_rms_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRmsNormBack> [valid_op_rms_norm_back] / exec_op_rms_norm_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRmsNormBack> [invalid_op_rms_norm_back] / reject_invalid_op_rms_norm_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGroupNorm> [valid_op_group_norm] / exec_op_group_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGroupNorm> [invalid_op_group_norm] / reject_invalid_op_group_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpL2Norm> [valid_op_l2_norm] / exec_op_l2_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpL2Norm> [invalid_op_l2_norm] / reject_invalid_op_l2_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q5_0_vector] / exec_simd_op_mul_mat_q5_0_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_0_vector] / exec_simd_op_mul_mat_q4_0_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_1_vector] / exec_simd_op_mul_mat_q4_1_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_packed_bl8_matrix_x4] / exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_packed_bl8_full_groups] / exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_packed_bl8] / exec_simd_op_mul_mat_q8_0_packed_bl8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_packed_bl4] / exec_simd_op_mul_mat_q8_0_packed_bl4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_vector_q8_rhs] / exec_simd_op_mul_mat_q8_0_vector_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q8_0_vector] / exec_simd_op_mul_mat_q8_0_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8] / exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4] / exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8] / exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8] / exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4] / exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4] / exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4] / exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_q8_rhs] / exec_simd_op_mul_mat_q4_vector_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q4_vector_f32_rhs] / exec_simd_op_mul_mat_q4_vector_f32_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8] / exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4] / exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm] / exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_prepared_q8_rhs] / exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4] / exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_packed_q8_rhs] / exec_simd_op_mul_mat_q6_vector_packed_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_packed] / exec_simd_op_mul_mat_q6_vector_packed,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector_q8_rhs] / exec_simd_op_mul_mat_q6_vector_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_q6_vector] / exec_simd_op_mul_mat_q6_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_f32_vector] / exec_simd_op_mul_mat_f32_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_generic] / exec_simd_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [simd_op_mul_mat_f16_vector] / exec_simd_op_mul_mat_f16_vector,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [valid_op_mul_mat_f16] / exec_scalar_op_mul_mat_f16,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [valid_op_mul_mat] / exec_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMat> [invalid_op_mul_mat] / reject_invalid_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm] / exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8] / exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4] / exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs] / exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm] / exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [valid_op_mul_mat_argmax] / exec_op_mul_mat_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatArgmax> [invalid_op_mul_mat_argmax] / reject_invalid_op_mul_mat_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatId> [valid_op_mul_mat_id] / exec_op_mul_mat_id,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMulMatId> [invalid_op_mul_mat_id] / reject_invalid_op_mul_mat_id,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOutProd> [valid_op_out_prod] / exec_op_out_prod,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOutProd> [invalid_op_out_prod] / reject_invalid_op_out_prod,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpScale> [valid_op_scale] / exec_op_scale,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpScale> [invalid_op_scale] / reject_invalid_op_scale,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSet> [valid_op_set] / exec_op_set,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSet> [invalid_op_set] / reject_invalid_op_set,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCpy> [valid_op_cpy] / exec_op_cpy,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCpy> [invalid_op_cpy] / reject_invalid_op_cpy,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCont> [valid_op_cont] / exec_op_cont,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCont> [invalid_op_cont] / reject_invalid_op_cont,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpReshape> [valid_op_reshape] / exec_op_reshape,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpReshape> [invalid_op_reshape] / reject_invalid_op_reshape,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpView> [valid_op_view] / exec_op_view,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpView> [invalid_op_view] / reject_invalid_op_view,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPermute> [valid_op_permute] / exec_op_permute,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPermute> [invalid_op_permute] / reject_invalid_op_permute,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTranspose> [valid_op_transpose] / exec_op_transpose,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTranspose> [invalid_op_transpose] / reject_invalid_op_transpose,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_f32] / exec_scalar_op_get_rows_f32,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_f16] / exec_scalar_op_get_rows_f16,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_bf16] / exec_scalar_op_get_rows_bf16,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_q4_0] / exec_scalar_op_get_rows_q4_0,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_q8_0] / exec_scalar_op_get_rows_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [valid_op_get_rows_q4_k] / exec_scalar_op_get_rows_q4_k,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRows> [invalid_op_get_rows] / reject_invalid_op_get_rows,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRowsBack> [valid_op_get_rows_back] / exec_op_get_rows_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRowsBack> [invalid_op_get_rows_back] / reject_invalid_op_get_rows_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSetRows> [valid_op_set_rows] / exec_op_set_rows,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSetRows> [invalid_op_set_rows] / reject_invalid_op_set_rows,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiag> [valid_op_diag] / exec_op_diag,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiag> [invalid_op_diag] / reject_invalid_op_diag,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiagMaskInf> [valid_op_diag_mask_inf] / exec_op_diag_mask_inf,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiagMaskInf> [invalid_op_diag_mask_inf] / reject_invalid_op_diag_mask_inf,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiagMaskZero> [valid_op_diag_mask_zero] / exec_op_diag_mask_zero,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpDiagMaskZero> [invalid_op_diag_mask_zero] / reject_invalid_op_diag_mask_zero,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSoftMax> [valid_op_soft_max] / exec_op_soft_max,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSoftMax> [invalid_op_soft_max] / reject_invalid_op_soft_max,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSoftMaxBack> [valid_op_soft_max_back] / exec_op_soft_max_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSoftMaxBack> [invalid_op_soft_max_back] / reject_invalid_op_soft_max_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRope> [valid_op_rope_norm] / exec_scalar_op_rope_norm,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRope> [valid_op_rope_neox] / exec_scalar_op_rope_neox,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRope> [valid_op_rope_timestep] / exec_scalar_op_rope_timestep,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRope> [invalid_op_rope] / reject_invalid_op_rope,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRopeBack> [valid_op_rope_back] / exec_op_rope_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRopeBack> [invalid_op_rope_back] / reject_invalid_op_rope_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpClamp> [valid_op_clamp] / exec_op_clamp,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpClamp> [invalid_op_clamp] / reject_invalid_op_clamp,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose1d> [simd_op_conv_transpose_1d_f32] / exec_simd_op_conv_transpose_1d_f32,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose1d> [valid_op_conv_transpose_1d_f32] / exec_scalar_op_conv_transpose_1d_f32,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose1d> [valid_op_conv_transpose_1d_f16] / exec_scalar_op_conv_transpose_1d_f16,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose1d> [invalid_op_conv_transpose_1d] / reject_invalid_op_conv_transpose_1d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2col> [valid_op_im2col_f32] / exec_scalar_op_im2col_f32,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2col> [valid_op_im2col_f16] / exec_scalar_op_im2col_f16,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2col> [invalid_op_im2col] / reject_invalid_op_im2col,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2colBack> [valid_op_im2col_back] / exec_op_im2col_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2colBack> [invalid_op_im2col_back] / reject_invalid_op_im2col_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2col3d> [valid_op_im2col_3d] / exec_op_im2col_3d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpIm2col3d> [invalid_op_im2col_3d] / reject_invalid_op_im2col_3d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv2d> [valid_op_conv_2d] / exec_op_conv_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv2d> [invalid_op_conv_2d] / reject_invalid_op_conv_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv3d> [valid_op_conv_3d] / exec_op_conv_3d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv3d> [invalid_op_conv_3d] / reject_invalid_op_conv_3d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv2dDw> [valid_op_conv_2d_dw] / exec_op_conv_2d_dw,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConv2dDw> [invalid_op_conv_2d_dw] / reject_invalid_op_conv_2d_dw,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose2d> [valid_op_conv_transpose_2d] / exec_op_conv_transpose_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpConvTranspose2d> [invalid_op_conv_transpose_2d] / reject_invalid_op_conv_transpose_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool1d> [valid_op_pool_1d] / exec_op_pool_1d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool1d> [invalid_op_pool_1d] / reject_invalid_op_pool_1d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool2d> [valid_op_pool_2d] / exec_op_pool_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool2d> [invalid_op_pool_2d] / reject_invalid_op_pool_2d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool2dBack> [valid_op_pool_2d_back] / exec_op_pool_2d_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPool2dBack> [invalid_op_pool_2d_back] / reject_invalid_op_pool_2d_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUpscale> [valid_op_upscale] / exec_op_upscale,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUpscale> [invalid_op_upscale] / reject_invalid_op_upscale,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPad> [valid_op_pad] / exec_op_pad,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPad> [invalid_op_pad] / reject_invalid_op_pad,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPadReflect1d> [valid_op_pad_reflect_1d] / exec_op_pad_reflect_1d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpPadReflect1d> [invalid_op_pad_reflect_1d] / reject_invalid_op_pad_reflect_1d,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRoll> [valid_op_roll] / exec_op_roll,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRoll> [invalid_op_roll] / reject_invalid_op_roll,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArange> [valid_op_arange] / exec_op_arange,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArange> [invalid_op_arange] / reject_invalid_op_arange,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTimestepEmbedding> [valid_op_timestep_embedding] / exec_op_timestep_embedding,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTimestepEmbedding> [invalid_op_timestep_embedding] / reject_invalid_op_timestep_embedding,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArgsort> [valid_op_argsort] / exec_op_argsort,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpArgsort> [invalid_op_argsort] / reject_invalid_op_argsort,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTopK> [valid_op_top_k] / exec_op_top_k,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTopK> [invalid_op_top_k] / reject_invalid_op_top_k,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpLeakyRelu> [valid_op_leaky_relu] / exec_op_leaky_relu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpLeakyRelu> [invalid_op_leaky_relu] / reject_invalid_op_leaky_relu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTri> [valid_op_tri] / exec_op_tri,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpTri> [invalid_op_tri] / reject_invalid_op_tri,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFill> [valid_op_fill] / exec_op_fill,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFill> [invalid_op_fill] / reject_invalid_op_fill,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFlashAttnExt> [simd_op_flash_attn_ext_f16kv_one_chunk] / exec_simd_op_flash_attn_ext_f16kv_one_chunk,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFlashAttnExt> [valid_op_flash_attn_ext_shared] / exec_op_flash_attn_ext,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFlashAttnExt> [invalid_op_flash_attn_ext] / reject_invalid_op_flash_attn_ext,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFlashAttnBack> [valid_op_flash_attn_back] / exec_op_flash_attn_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpFlashAttnBack> [invalid_op_flash_attn_back] / reject_invalid_op_flash_attn_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSsmConv> [valid_op_ssm_conv] / exec_op_ssm_conv,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSsmConv> [invalid_op_ssm_conv] / reject_invalid_op_ssm_conv,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSsmScan> [valid_op_ssm_scan] / exec_op_ssm_scan,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSsmScan> [invalid_op_ssm_scan] / reject_invalid_op_ssm_scan,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpWinPart> [valid_op_win_part] / exec_op_win_part,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpWinPart> [invalid_op_win_part] / reject_invalid_op_win_part,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpWinUnpart> [valid_op_win_unpart] / exec_op_win_unpart,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpWinUnpart> [invalid_op_win_unpart] / reject_invalid_op_win_unpart,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRelPos> [valid_op_get_rel_pos] / exec_op_get_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGetRelPos> [invalid_op_get_rel_pos] / reject_invalid_op_get_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAddRelPos> [valid_op_add_rel_pos] / exec_op_add_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpAddRelPos> [invalid_op_add_rel_pos] / reject_invalid_op_add_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRwkvWkv6> [valid_op_rwkv_wkv6] / exec_op_rwkv_wkv6,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRwkvWkv6> [invalid_op_rwkv_wkv6] / reject_invalid_op_rwkv_wkv6,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGatedLinearAttn> [valid_op_gated_linear_attn] / exec_op_gated_linear_attn,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGatedLinearAttn> [invalid_op_gated_linear_attn] / reject_invalid_op_gated_linear_attn,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRwkvWkv7> [valid_op_rwkv_wkv7] / exec_op_rwkv_wkv7,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpRwkvWkv7> [invalid_op_rwkv_wkv7] / reject_invalid_op_rwkv_wkv7,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSolveTri> [valid_op_solve_tri] / exec_op_solve_tri,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpSolveTri> [invalid_op_solve_tri] / reject_invalid_op_solve_tri,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [simd_op_unary_abs] / exec_simd_op_unary_abs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [simd_op_unary_neg] / exec_simd_op_unary_neg,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [simd_op_unary_relu] / exec_simd_op_unary_relu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [simd_op_unary_silu] / exec_simd_op_unary_silu_t,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_abs] / exec_scalar_op_unary_abs,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_neg] / exec_scalar_op_unary_neg,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_relu] / exec_scalar_op_unary_relu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_exp] / exec_scalar_op_unary_exp,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_tanh] / exec_scalar_op_unary_tanh,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_elu] / exec_scalar_op_unary_elu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_gelu] / exec_scalar_op_unary_gelu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [valid_op_unary_silu] / exec_scalar_op_unary_silu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpUnary> [invalid_op_unary] / reject_invalid_op_unary,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom1> [valid_op_map_custom1] / exec_op_map_custom1,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom1> [invalid_op_map_custom1] / reject_invalid_op_map_custom1,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom2> [valid_op_map_custom2] / exec_op_map_custom2,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom2> [invalid_op_map_custom2] / reject_invalid_op_map_custom2,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom3> [valid_op_map_custom3] / exec_op_map_custom3,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpMapCustom3> [invalid_op_map_custom3] / reject_invalid_op_map_custom3,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCustom> [valid_op_custom] / exec_op_custom,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCustom> [invalid_op_custom] / reject_invalid_op_custom,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCrossEntropyLoss> [valid_op_cross_entropy_loss] / exec_op_cross_entropy_loss,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCrossEntropyLoss> [invalid_op_cross_entropy_loss] / reject_invalid_op_cross_entropy_loss,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCrossEntropyLossBack> [valid_op_cross_entropy_loss_back] / exec_op_cross_entropy_loss_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpCrossEntropyLossBack> [invalid_op_cross_entropy_loss_back] / reject_invalid_op_cross_entropy_loss_back,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOptStepAdamw> [valid_op_opt_step_adamw] / exec_op_opt_step_adamw,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOptStepAdamw> [invalid_op_opt_step_adamw] / reject_invalid_op_opt_step_adamw,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOptStepSgd> [valid_op_opt_step_sgd] / exec_op_opt_step_sgd,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpOptStepSgd> [invalid_op_opt_step_sgd] / reject_invalid_op_opt_step_sgd,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGlu> [valid_op_glu] / exec_op_glu,
        "ready"_s <= "ready"_s + event<EmelKernelAarch64EventDispatchOpGlu> [invalid_op_glu] / reject_invalid_op_glu,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected,
    }
}

/// Context for `KernelAarch64` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct KernelAarch64Context {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl KernelAarch64StateMachineContext for KernelAarch64Context {
    fn exec_dispatch(&mut self, _event: &EmelKernelAarch64EventDispatchRequest) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_dispatch
        todo!("TODO: port action `exec_dispatch` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_acc(&mut self, _event: &EmelKernelAarch64EventDispatchOpAcc) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_acc
        todo!("TODO: port action `exec_op_acc` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_add(&mut self, _event: &EmelKernelAarch64EventDispatchOpAdd) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_add
        todo!("TODO: port action `exec_op_add` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_add1(&mut self, _event: &EmelKernelAarch64EventDispatchOpAdd1) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_add1
        todo!("TODO: port action `exec_op_add1` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_add_id(&mut self, _event: &EmelKernelAarch64EventDispatchOpAddId) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_add_id
        todo!(
            "TODO: port action `exec_op_add_id` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_add_rel_pos(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAddRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_add_rel_pos
        todo!(
            "TODO: port action `exec_op_add_rel_pos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_arange(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArange,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_arange
        todo!(
            "TODO: port action `exec_op_arange` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_argmax(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_argmax
        todo!(
            "TODO: port action `exec_op_argmax` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_argsort(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArgsort,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_argsort
        todo!(
            "TODO: port action `exec_op_argsort` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_clamp(&mut self, _event: &EmelKernelAarch64EventDispatchOpClamp) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_clamp
        todo!("TODO: port action `exec_op_clamp` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_concat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConcat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_concat
        todo!(
            "TODO: port action `exec_op_concat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_cont(&mut self, _event: &EmelKernelAarch64EventDispatchOpCont) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cont
        todo!("TODO: port action `exec_op_cont` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_conv_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_conv_2d
        todo!(
            "TODO: port action `exec_op_conv_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_conv_2d_dw(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv2dDw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_conv_2d_dw
        todo!(
            "TODO: port action `exec_op_conv_2d_dw` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_conv_3d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_conv_3d
        todo!(
            "TODO: port action `exec_op_conv_3d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_conv_transpose_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_conv_transpose_2d
        todo!(
            "TODO: port action `exec_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_cos(&mut self, _event: &EmelKernelAarch64EventDispatchOpCos) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cos
        todo!("TODO: port action `exec_op_cos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_count_equal(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCountEqual,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_count_equal
        todo!(
            "TODO: port action `exec_op_count_equal` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_cpy(&mut self, _event: &EmelKernelAarch64EventDispatchOpCpy) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cpy
        todo!("TODO: port action `exec_op_cpy` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_cross_entropy_loss(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLoss,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cross_entropy_loss
        todo!(
            "TODO: port action `exec_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_cross_entropy_loss_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLossBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cross_entropy_loss_back
        todo!(
            "TODO: port action `exec_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_cumsum(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCumsum,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_cumsum
        todo!(
            "TODO: port action `exec_op_cumsum` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_custom(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCustom,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_custom
        todo!(
            "TODO: port action `exec_op_custom` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_diag(&mut self, _event: &EmelKernelAarch64EventDispatchOpDiag) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_diag
        todo!("TODO: port action `exec_op_diag` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_diag_mask_inf(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskInf,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_diag_mask_inf
        todo!(
            "TODO: port action `exec_op_diag_mask_inf` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_diag_mask_zero(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskZero,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_diag_mask_zero
        todo!(
            "TODO: port action `exec_op_diag_mask_zero` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_div(&mut self, _event: &EmelKernelAarch64EventDispatchOpDiv) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_div
        todo!("TODO: port action `exec_op_div` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_dup(&mut self, _event: &EmelKernelAarch64EventDispatchOpDup) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_dup
        todo!("TODO: port action `exec_op_dup` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_fill(&mut self, _event: &EmelKernelAarch64EventDispatchOpFill) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_fill
        todo!("TODO: port action `exec_op_fill` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_flash_attn_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_flash_attn_back
        todo!(
            "TODO: port action `exec_op_flash_attn_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_flash_attn_ext(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_flash_attn_ext
        todo!(
            "TODO: port action `exec_op_flash_attn_ext` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_gated_linear_attn(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGatedLinearAttn,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_gated_linear_attn
        todo!(
            "TODO: port action `exec_op_gated_linear_attn` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_get_rel_pos(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_get_rel_pos
        todo!(
            "TODO: port action `exec_op_get_rel_pos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_get_rows_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRowsBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_get_rows_back
        todo!(
            "TODO: port action `exec_op_get_rows_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_glu(&mut self, _event: &EmelKernelAarch64EventDispatchOpGlu) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_glu
        todo!("TODO: port action `exec_op_glu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_group_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGroupNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_group_norm
        todo!(
            "TODO: port action `exec_op_group_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_im2col_3d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_im2col_3d
        todo!(
            "TODO: port action `exec_op_im2col_3d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_im2col_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2colBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_im2col_back
        todo!(
            "TODO: port action `exec_op_im2col_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_l2_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpL2Norm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_l2_norm
        todo!(
            "TODO: port action `exec_op_l2_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_leaky_relu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpLeakyRelu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_leaky_relu
        todo!(
            "TODO: port action `exec_op_leaky_relu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_log(&mut self, _event: &EmelKernelAarch64EventDispatchOpLog) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_log
        todo!("TODO: port action `exec_op_log` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_map_custom1(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_map_custom1
        todo!(
            "TODO: port action `exec_op_map_custom1` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_map_custom2(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom2,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_map_custom2
        todo!(
            "TODO: port action `exec_op_map_custom2` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_map_custom3(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom3,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_map_custom3
        todo!(
            "TODO: port action `exec_op_map_custom3` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_mean(&mut self, _event: &EmelKernelAarch64EventDispatchOpMean) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_mean
        todo!("TODO: port action `exec_op_mean` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_mul(&mut self, _event: &EmelKernelAarch64EventDispatchOpMul) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_mul
        todo!("TODO: port action `exec_op_mul` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_mul_mat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_mul_mat
        todo!(
            "TODO: port action `exec_op_mul_mat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_mul_mat_argmax(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_mul_mat_argmax
        todo!(
            "TODO: port action `exec_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_mul_mat_id(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_mul_mat_id
        todo!(
            "TODO: port action `exec_op_mul_mat_id` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_norm(&mut self, _event: &EmelKernelAarch64EventDispatchOpNorm) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_norm
        todo!("TODO: port action `exec_op_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_opt_step_adamw(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepAdamw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_opt_step_adamw
        todo!(
            "TODO: port action `exec_op_opt_step_adamw` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_opt_step_sgd(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepSgd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_opt_step_sgd
        todo!(
            "TODO: port action `exec_op_opt_step_sgd` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_out_prod(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOutProd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_out_prod
        todo!(
            "TODO: port action `exec_op_out_prod` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_pad(&mut self, _event: &EmelKernelAarch64EventDispatchOpPad) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_pad
        todo!("TODO: port action `exec_op_pad` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_pad_reflect_1d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPadReflect1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_pad_reflect_1d
        todo!(
            "TODO: port action `exec_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_permute(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPermute,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_permute
        todo!(
            "TODO: port action `exec_op_permute` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_pool_1d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_pool_1d
        todo!(
            "TODO: port action `exec_op_pool_1d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_pool_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_pool_2d
        todo!(
            "TODO: port action `exec_op_pool_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_pool_2d_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool2dBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_pool_2d_back
        todo!(
            "TODO: port action `exec_op_pool_2d_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_repeat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRepeat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_repeat
        todo!(
            "TODO: port action `exec_op_repeat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_repeat_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRepeatBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_repeat_back
        todo!(
            "TODO: port action `exec_op_repeat_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_reshape(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpReshape,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_reshape
        todo!(
            "TODO: port action `exec_op_reshape` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_rms_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_rms_norm
        todo!(
            "TODO: port action `exec_op_rms_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_rms_norm_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNormBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_rms_norm_back
        todo!(
            "TODO: port action `exec_op_rms_norm_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_roll(&mut self, _event: &EmelKernelAarch64EventDispatchOpRoll) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_roll
        todo!("TODO: port action `exec_op_roll` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_rope_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRopeBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_rope_back
        todo!(
            "TODO: port action `exec_op_rope_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_rwkv_wkv6(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv6,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_rwkv_wkv6
        todo!(
            "TODO: port action `exec_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_rwkv_wkv7(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv7,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_rwkv_wkv7
        todo!(
            "TODO: port action `exec_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_scale(&mut self, _event: &EmelKernelAarch64EventDispatchOpScale) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_scale
        todo!("TODO: port action `exec_op_scale` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_set(&mut self, _event: &EmelKernelAarch64EventDispatchOpSet) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_set
        todo!("TODO: port action `exec_op_set` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_set_rows(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_set_rows
        todo!(
            "TODO: port action `exec_op_set_rows` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_silu_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSiluBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_silu_back
        todo!(
            "TODO: port action `exec_op_silu_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_sin(&mut self, _event: &EmelKernelAarch64EventDispatchOpSin) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sin
        todo!("TODO: port action `exec_op_sin` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_soft_max(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_soft_max
        todo!(
            "TODO: port action `exec_op_soft_max` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_soft_max_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMaxBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_soft_max_back
        todo!(
            "TODO: port action `exec_op_soft_max_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_solve_tri(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSolveTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_solve_tri
        todo!(
            "TODO: port action `exec_op_solve_tri` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_sqr(&mut self, _event: &EmelKernelAarch64EventDispatchOpSqr) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sqr
        todo!("TODO: port action `exec_op_sqr` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_sqrt(&mut self, _event: &EmelKernelAarch64EventDispatchOpSqrt) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sqrt
        todo!("TODO: port action `exec_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_ssm_conv(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSsmConv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_ssm_conv
        todo!(
            "TODO: port action `exec_op_ssm_conv` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_ssm_scan(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSsmScan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_ssm_scan
        todo!(
            "TODO: port action `exec_op_ssm_scan` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_sub(&mut self, _event: &EmelKernelAarch64EventDispatchOpSub) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sub
        todo!("TODO: port action `exec_op_sub` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_sum(&mut self, _event: &EmelKernelAarch64EventDispatchOpSum) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sum
        todo!("TODO: port action `exec_op_sum` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_sum_rows(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSumRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_sum_rows
        todo!(
            "TODO: port action `exec_op_sum_rows` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_timestep_embedding(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTimestepEmbedding,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_timestep_embedding
        todo!(
            "TODO: port action `exec_op_timestep_embedding` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_top_k(&mut self, _event: &EmelKernelAarch64EventDispatchOpTopK) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_top_k
        todo!("TODO: port action `exec_op_top_k` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_transpose(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTranspose,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_transpose
        todo!(
            "TODO: port action `exec_op_transpose` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_tri(&mut self, _event: &EmelKernelAarch64EventDispatchOpTri) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_tri
        todo!("TODO: port action `exec_op_tri` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_upscale(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUpscale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_upscale
        todo!(
            "TODO: port action `exec_op_upscale` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_view(&mut self, _event: &EmelKernelAarch64EventDispatchOpView) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_view
        todo!("TODO: port action `exec_op_view` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn exec_op_win_part(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpWinPart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_win_part
        todo!(
            "TODO: port action `exec_op_win_part` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_op_win_unpart(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpWinUnpart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_op_win_unpart
        todo!(
            "TODO: port action `exec_op_win_unpart` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_add_broadcast_row(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAdd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_add_broadcast_row
        todo!(
            "TODO: port action `exec_scalar_op_add_broadcast_row` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_conv_transpose_1d_f16(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_conv_transpose_1d_f16
        todo!(
            "TODO: port action `exec_scalar_op_conv_transpose_1d_f16` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_conv_transpose_1d_f32(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_conv_transpose_1d_f32
        todo!(
            "TODO: port action `exec_scalar_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_bf16(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_bf16
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_bf16` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_f16(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_f16
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_f16` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_f32(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_f32
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_f32` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q4_0(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_q4_0
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q4_0` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q4_k(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_q4_k
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q4_k` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q8_0(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_get_rows_q8_0
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q8_0` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_im2col_f16(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_im2col_f16
        todo!(
            "TODO: port action `exec_scalar_op_im2col_f16` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_im2col_f32(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_im2col_f32
        todo!(
            "TODO: port action `exec_scalar_op_im2col_f32` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_mul_broadcast_row(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMul,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_mul_broadcast_row
        todo!(
            "TODO: port action `exec_scalar_op_mul_broadcast_row` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_mul_mat_f16(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_mul_mat_f16
        todo!(
            "TODO: port action `exec_scalar_op_mul_mat_f16` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_neox(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_rope_neox
        todo!(
            "TODO: port action `exec_scalar_op_rope_neox` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_rope_norm
        todo!(
            "TODO: port action `exec_scalar_op_rope_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_timestep(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_rope_timestep
        todo!(
            "TODO: port action `exec_scalar_op_rope_timestep` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_abs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_abs
        todo!(
            "TODO: port action `exec_scalar_op_unary_abs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_elu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_elu
        todo!(
            "TODO: port action `exec_scalar_op_unary_elu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_exp(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_exp
        todo!(
            "TODO: port action `exec_scalar_op_unary_exp` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_gelu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_gelu
        todo!(
            "TODO: port action `exec_scalar_op_unary_gelu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_neg(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_neg
        todo!(
            "TODO: port action `exec_scalar_op_unary_neg` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_relu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_relu
        todo!(
            "TODO: port action `exec_scalar_op_unary_relu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_silu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_silu
        todo!(
            "TODO: port action `exec_scalar_op_unary_silu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_tanh(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_scalar_op_unary_tanh
        todo!(
            "TODO: port action `exec_scalar_op_unary_tanh` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_add(&mut self, _event: &EmelKernelAarch64EventDispatchOpAdd) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_add
        todo!(
            "TODO: port action `exec_simd_op_add` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_conv_transpose_1d_f32(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_conv_transpose_1d_f32
        todo!(
            "TODO: port action `exec_simd_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_div(&mut self, _event: &EmelKernelAarch64EventDispatchOpDiv) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_div
        todo!(
            "TODO: port action `exec_simd_op_div` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_dup(&mut self, _event: &EmelKernelAarch64EventDispatchOpDup) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_dup
        todo!(
            "TODO: port action `exec_simd_op_dup` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_flash_attn_ext_f16kv_one_chunk(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_flash_attn_ext_f16kv_one_chunk
        todo!(
            "TODO: port action `exec_simd_op_flash_attn_ext_f16kv_one_chunk` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul(&mut self, _event: &EmelKernelAarch64EventDispatchOpMul) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul
        todo!(
            "TODO: port action `exec_simd_op_mul` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat
        todo!(
            "TODO: port action `exec_simd_op_mul_mat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_f16_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_f16_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_f16_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_f32_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_f32_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_f32_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_0_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_0_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_0_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_1_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_1_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_1_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_f32_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_f32_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_f32_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q4_vector_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q4_vector_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q4_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q5_0_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q5_0_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q5_0_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_packed(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_packed
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_packed` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_packed_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_packed_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_packed_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q6_vector_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q6_vector_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q6_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_packed_bl4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_packed_bl4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_packed_bl4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_packed_bl8(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_packed_bl8
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_packed_bl8` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_vector(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_vector
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_vector` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat_q8_0_vector_q8_rhs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_mul_mat_q8_0_vector_q8_rhs
        todo!(
            "TODO: port action `exec_simd_op_mul_mat_q8_0_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_sqr(&mut self, _event: &EmelKernelAarch64EventDispatchOpSqr) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_sqr
        todo!(
            "TODO: port action `exec_simd_op_sqr` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_sqrt(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSqrt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_sqrt
        todo!(
            "TODO: port action `exec_simd_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_sub(&mut self, _event: &EmelKernelAarch64EventDispatchOpSub) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_sub
        todo!(
            "TODO: port action `exec_simd_op_sub` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_abs(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_unary_abs
        todo!(
            "TODO: port action `exec_simd_op_unary_abs` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_neg(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_unary_neg
        todo!(
            "TODO: port action `exec_simd_op_unary_neg` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_relu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_unary_relu
        todo!(
            "TODO: port action `exec_simd_op_unary_relu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_silu_t(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::exec_simd_op_unary_silu_t
        todo!(
            "TODO: port action `exec_simd_op_unary_silu_t` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn invalid_op_acc(&self, _event: &EmelKernelAarch64EventDispatchOpAcc) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_acc
        todo!("TODO: port guard `invalid_op_acc` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_add(&self, _event: &EmelKernelAarch64EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_add
        todo!("TODO: port guard `invalid_op_add` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_add1(&self, _event: &EmelKernelAarch64EventDispatchOpAdd1) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_add1
        todo!("TODO: port guard `invalid_op_add1` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_add_id(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpAddId,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_add_id
        todo!(
            "TODO: port guard `invalid_op_add_id` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_add_rel_pos(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpAddRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_add_rel_pos
        todo!(
            "TODO: port guard `invalid_op_add_rel_pos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_arange(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpArange,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_arange
        todo!(
            "TODO: port guard `invalid_op_arange` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_argmax(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_argmax
        todo!(
            "TODO: port guard `invalid_op_argmax` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_argsort(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpArgsort,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_argsort
        todo!(
            "TODO: port guard `invalid_op_argsort` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_clamp(&self, _event: &EmelKernelAarch64EventDispatchOpClamp) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_clamp
        todo!(
            "TODO: port guard `invalid_op_clamp` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_concat(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConcat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_concat
        todo!(
            "TODO: port guard `invalid_op_concat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_cont(&self, _event: &EmelKernelAarch64EventDispatchOpCont) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cont
        todo!("TODO: port guard `invalid_op_cont` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_conv_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_conv_2d
        todo!(
            "TODO: port guard `invalid_op_conv_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_conv_2d_dw(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv2dDw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_conv_2d_dw
        todo!(
            "TODO: port guard `invalid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_conv_3d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_conv_3d
        todo!(
            "TODO: port guard `invalid_op_conv_3d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_conv_transpose_1d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_conv_transpose_1d
        todo!(
            "TODO: port guard `invalid_op_conv_transpose_1d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_conv_transpose_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_conv_transpose_2d
        todo!(
            "TODO: port guard `invalid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_cos(&self, _event: &EmelKernelAarch64EventDispatchOpCos) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cos
        todo!("TODO: port guard `invalid_op_cos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_count_equal(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCountEqual,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_count_equal
        todo!(
            "TODO: port guard `invalid_op_count_equal` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_cpy(&self, _event: &EmelKernelAarch64EventDispatchOpCpy) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cpy
        todo!("TODO: port guard `invalid_op_cpy` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_cross_entropy_loss(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLoss,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cross_entropy_loss
        todo!(
            "TODO: port guard `invalid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_cross_entropy_loss_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLossBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cross_entropy_loss_back
        todo!(
            "TODO: port guard `invalid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_cumsum(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCumsum,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_cumsum
        todo!(
            "TODO: port guard `invalid_op_cumsum` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_custom(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCustom,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_custom
        todo!(
            "TODO: port guard `invalid_op_custom` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_diag(&self, _event: &EmelKernelAarch64EventDispatchOpDiag) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_diag
        todo!("TODO: port guard `invalid_op_diag` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_diag_mask_inf(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskInf,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_diag_mask_inf
        todo!(
            "TODO: port guard `invalid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_diag_mask_zero(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskZero,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_diag_mask_zero
        todo!(
            "TODO: port guard `invalid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_div(&self, _event: &EmelKernelAarch64EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_div
        todo!("TODO: port guard `invalid_op_div` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_dup(&self, _event: &EmelKernelAarch64EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_dup
        todo!("TODO: port guard `invalid_op_dup` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_fill(&self, _event: &EmelKernelAarch64EventDispatchOpFill) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_fill
        todo!("TODO: port guard `invalid_op_fill` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_flash_attn_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_flash_attn_back
        todo!(
            "TODO: port guard `invalid_op_flash_attn_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_flash_attn_ext(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_flash_attn_ext
        todo!(
            "TODO: port guard `invalid_op_flash_attn_ext` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_gated_linear_attn(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGatedLinearAttn,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_gated_linear_attn
        todo!(
            "TODO: port guard `invalid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_get_rel_pos(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_get_rel_pos
        todo!(
            "TODO: port guard `invalid_op_get_rel_pos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_get_rows(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_get_rows
        todo!(
            "TODO: port guard `invalid_op_get_rows` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_get_rows_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRowsBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_get_rows_back
        todo!(
            "TODO: port guard `invalid_op_get_rows_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_glu(&self, _event: &EmelKernelAarch64EventDispatchOpGlu) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_glu
        todo!("TODO: port guard `invalid_op_glu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_group_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGroupNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_group_norm
        todo!(
            "TODO: port guard `invalid_op_group_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_im2col(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_im2col
        todo!(
            "TODO: port guard `invalid_op_im2col` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_im2col_3d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_im2col_3d
        todo!(
            "TODO: port guard `invalid_op_im2col_3d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_im2col_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2colBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_im2col_back
        todo!(
            "TODO: port guard `invalid_op_im2col_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_l2_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpL2Norm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_l2_norm
        todo!(
            "TODO: port guard `invalid_op_l2_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_leaky_relu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpLeakyRelu,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_leaky_relu
        todo!(
            "TODO: port guard `invalid_op_leaky_relu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_log(&self, _event: &EmelKernelAarch64EventDispatchOpLog) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_log
        todo!("TODO: port guard `invalid_op_log` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_map_custom1(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom1,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_map_custom1
        todo!(
            "TODO: port guard `invalid_op_map_custom1` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_map_custom2(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom2,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_map_custom2
        todo!(
            "TODO: port guard `invalid_op_map_custom2` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_map_custom3(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom3,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_map_custom3
        todo!(
            "TODO: port guard `invalid_op_map_custom3` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_mean(&self, _event: &EmelKernelAarch64EventDispatchOpMean) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_mean
        todo!("TODO: port guard `invalid_op_mean` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_mul(&self, _event: &EmelKernelAarch64EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_mul
        todo!("TODO: port guard `invalid_op_mul` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_mul_mat(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_mul_mat
        todo!(
            "TODO: port guard `invalid_op_mul_mat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_mul_mat_argmax(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_mul_mat_argmax
        todo!(
            "TODO: port guard `invalid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_mul_mat_id(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatId,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_mul_mat_id
        todo!(
            "TODO: port guard `invalid_op_mul_mat_id` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_norm(&self, _event: &EmelKernelAarch64EventDispatchOpNorm) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_norm
        todo!("TODO: port guard `invalid_op_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_opt_step_adamw(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepAdamw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_opt_step_adamw
        todo!(
            "TODO: port guard `invalid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_opt_step_sgd(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepSgd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_opt_step_sgd
        todo!(
            "TODO: port guard `invalid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_out_prod(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOutProd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_out_prod
        todo!(
            "TODO: port guard `invalid_op_out_prod` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_pad(&self, _event: &EmelKernelAarch64EventDispatchOpPad) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_pad
        todo!("TODO: port guard `invalid_op_pad` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_pad_reflect_1d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPadReflect1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_pad_reflect_1d
        todo!(
            "TODO: port guard `invalid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_permute(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPermute,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_permute
        todo!(
            "TODO: port guard `invalid_op_permute` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_pool_1d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_pool_1d
        todo!(
            "TODO: port guard `invalid_op_pool_1d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_pool_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_pool_2d
        todo!(
            "TODO: port guard `invalid_op_pool_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_pool_2d_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool2dBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_pool_2d_back
        todo!(
            "TODO: port guard `invalid_op_pool_2d_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_repeat(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRepeat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_repeat
        todo!(
            "TODO: port guard `invalid_op_repeat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_repeat_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRepeatBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_repeat_back
        todo!(
            "TODO: port guard `invalid_op_repeat_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_reshape(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpReshape,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_reshape
        todo!(
            "TODO: port guard `invalid_op_reshape` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_rms_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rms_norm
        todo!(
            "TODO: port guard `invalid_op_rms_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_rms_norm_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNormBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rms_norm_back
        todo!(
            "TODO: port guard `invalid_op_rms_norm_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_roll(&self, _event: &EmelKernelAarch64EventDispatchOpRoll) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_roll
        todo!("TODO: port guard `invalid_op_roll` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_rope(&self, _event: &EmelKernelAarch64EventDispatchOpRope) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rope
        todo!("TODO: port guard `invalid_op_rope` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_rope_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRopeBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rope_back
        todo!(
            "TODO: port guard `invalid_op_rope_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_rwkv_wkv6(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv6,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rwkv_wkv6
        todo!(
            "TODO: port guard `invalid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_rwkv_wkv7(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv7,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_rwkv_wkv7
        todo!(
            "TODO: port guard `invalid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_scale(&self, _event: &EmelKernelAarch64EventDispatchOpScale) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_scale
        todo!(
            "TODO: port guard `invalid_op_scale` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_set(&self, _event: &EmelKernelAarch64EventDispatchOpSet) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_set
        todo!("TODO: port guard `invalid_op_set` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_set_rows(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_set_rows
        todo!(
            "TODO: port guard `invalid_op_set_rows` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_silu_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSiluBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_silu_back
        todo!(
            "TODO: port guard `invalid_op_silu_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_sin(&self, _event: &EmelKernelAarch64EventDispatchOpSin) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sin
        todo!("TODO: port guard `invalid_op_sin` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_soft_max(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_soft_max
        todo!(
            "TODO: port guard `invalid_op_soft_max` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_soft_max_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMaxBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_soft_max_back
        todo!(
            "TODO: port guard `invalid_op_soft_max_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_solve_tri(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSolveTri,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_solve_tri
        todo!(
            "TODO: port guard `invalid_op_solve_tri` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_sqr(&self, _event: &EmelKernelAarch64EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sqr
        todo!("TODO: port guard `invalid_op_sqr` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_sqrt(&self, _event: &EmelKernelAarch64EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sqrt
        todo!("TODO: port guard `invalid_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_ssm_conv(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSsmConv,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_ssm_conv
        todo!(
            "TODO: port guard `invalid_op_ssm_conv` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_ssm_scan(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSsmScan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_ssm_scan
        todo!(
            "TODO: port guard `invalid_op_ssm_scan` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_sub(&self, _event: &EmelKernelAarch64EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sub
        todo!("TODO: port guard `invalid_op_sub` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_sum(&self, _event: &EmelKernelAarch64EventDispatchOpSum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sum
        todo!("TODO: port guard `invalid_op_sum` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_sum_rows(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSumRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_sum_rows
        todo!(
            "TODO: port guard `invalid_op_sum_rows` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_timestep_embedding(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpTimestepEmbedding,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_timestep_embedding
        todo!(
            "TODO: port guard `invalid_op_timestep_embedding` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_top_k(&self, _event: &EmelKernelAarch64EventDispatchOpTopK) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_top_k
        todo!(
            "TODO: port guard `invalid_op_top_k` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_transpose(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpTranspose,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_transpose
        todo!(
            "TODO: port guard `invalid_op_transpose` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_tri(&self, _event: &EmelKernelAarch64EventDispatchOpTri) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_tri
        todo!("TODO: port guard `invalid_op_tri` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_unary(&self, _event: &EmelKernelAarch64EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_unary
        todo!(
            "TODO: port guard `invalid_op_unary` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_upscale(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUpscale,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_upscale
        todo!(
            "TODO: port guard `invalid_op_upscale` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_view(&self, _event: &EmelKernelAarch64EventDispatchOpView) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_view
        todo!("TODO: port guard `invalid_op_view` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn invalid_op_win_part(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpWinPart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_win_part
        todo!(
            "TODO: port guard `invalid_op_win_part` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn invalid_op_win_unpart(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpWinUnpart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::invalid_op_win_unpart
        todo!(
            "TODO: port guard `invalid_op_win_unpart` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn on_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/kernel/aarch64/actions.hpp")
    }
    fn reject_invalid_op_acc(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAcc,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_acc
        todo!(
            "TODO: port action `reject_invalid_op_acc` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_add(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAdd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_add
        todo!(
            "TODO: port action `reject_invalid_op_add` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_add1(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAdd1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_add1
        todo!(
            "TODO: port action `reject_invalid_op_add1` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_add_id(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAddId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_add_id
        todo!(
            "TODO: port action `reject_invalid_op_add_id` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_add_rel_pos(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpAddRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_add_rel_pos
        todo!(
            "TODO: port action `reject_invalid_op_add_rel_pos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_arange(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArange,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_arange
        todo!(
            "TODO: port action `reject_invalid_op_arange` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_argmax(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_argmax
        todo!(
            "TODO: port action `reject_invalid_op_argmax` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_argsort(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpArgsort,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_argsort
        todo!(
            "TODO: port action `reject_invalid_op_argsort` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_clamp(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpClamp,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_clamp
        todo!(
            "TODO: port action `reject_invalid_op_clamp` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_concat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConcat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_concat
        todo!(
            "TODO: port action `reject_invalid_op_concat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cont(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCont,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cont
        todo!(
            "TODO: port action `reject_invalid_op_cont` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_conv_2d
        todo!(
            "TODO: port action `reject_invalid_op_conv_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_2d_dw(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv2dDw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_conv_2d_dw
        todo!(
            "TODO: port action `reject_invalid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_3d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConv3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_conv_3d
        todo!(
            "TODO: port action `reject_invalid_op_conv_3d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_transpose_1d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_conv_transpose_1d
        todo!(
            "TODO: port action `reject_invalid_op_conv_transpose_1d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_transpose_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_conv_transpose_2d
        todo!(
            "TODO: port action `reject_invalid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cos(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cos
        todo!(
            "TODO: port action `reject_invalid_op_cos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_count_equal(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCountEqual,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_count_equal
        todo!(
            "TODO: port action `reject_invalid_op_count_equal` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cpy(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCpy,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cpy
        todo!(
            "TODO: port action `reject_invalid_op_cpy` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cross_entropy_loss(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLoss,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cross_entropy_loss
        todo!(
            "TODO: port action `reject_invalid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cross_entropy_loss_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLossBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cross_entropy_loss_back
        todo!(
            "TODO: port action `reject_invalid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_cumsum(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCumsum,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_cumsum
        todo!(
            "TODO: port action `reject_invalid_op_cumsum` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_custom(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpCustom,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_custom
        todo!(
            "TODO: port action `reject_invalid_op_custom` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiag,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_diag
        todo!(
            "TODO: port action `reject_invalid_op_diag` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag_mask_inf(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskInf,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_diag_mask_inf
        todo!(
            "TODO: port action `reject_invalid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag_mask_zero(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskZero,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_diag_mask_zero
        todo!(
            "TODO: port action `reject_invalid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_div(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDiv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_div
        todo!(
            "TODO: port action `reject_invalid_op_div` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_dup(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpDup,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_dup
        todo!(
            "TODO: port action `reject_invalid_op_dup` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_fill(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFill,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_fill
        todo!(
            "TODO: port action `reject_invalid_op_fill` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_flash_attn_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_flash_attn_back
        todo!(
            "TODO: port action `reject_invalid_op_flash_attn_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_flash_attn_ext(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_flash_attn_ext
        todo!(
            "TODO: port action `reject_invalid_op_flash_attn_ext` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_gated_linear_attn(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGatedLinearAttn,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_gated_linear_attn
        todo!(
            "TODO: port action `reject_invalid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rel_pos(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_get_rel_pos
        todo!(
            "TODO: port action `reject_invalid_op_get_rel_pos` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rows(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_get_rows
        todo!(
            "TODO: port action `reject_invalid_op_get_rows` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rows_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGetRowsBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_get_rows_back
        todo!(
            "TODO: port action `reject_invalid_op_get_rows_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_glu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGlu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_glu
        todo!(
            "TODO: port action `reject_invalid_op_glu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_group_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpGroupNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_group_norm
        todo!(
            "TODO: port action `reject_invalid_op_group_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_im2col
        todo!(
            "TODO: port action `reject_invalid_op_im2col` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col_3d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_im2col_3d
        todo!(
            "TODO: port action `reject_invalid_op_im2col_3d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpIm2colBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_im2col_back
        todo!(
            "TODO: port action `reject_invalid_op_im2col_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_l2_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpL2Norm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_l2_norm
        todo!(
            "TODO: port action `reject_invalid_op_l2_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_leaky_relu(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpLeakyRelu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_leaky_relu
        todo!(
            "TODO: port action `reject_invalid_op_leaky_relu` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_log(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpLog,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_log
        todo!(
            "TODO: port action `reject_invalid_op_log` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom1(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_map_custom1
        todo!(
            "TODO: port action `reject_invalid_op_map_custom1` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom2(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom2,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_map_custom2
        todo!(
            "TODO: port action `reject_invalid_op_map_custom2` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom3(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom3,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_map_custom3
        todo!(
            "TODO: port action `reject_invalid_op_map_custom3` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_mean(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMean,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_mean
        todo!(
            "TODO: port action `reject_invalid_op_mean` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMul,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_mul
        todo!(
            "TODO: port action `reject_invalid_op_mul` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_mul_mat
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat_argmax(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_mul_mat_argmax
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat_id(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_mul_mat_id
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat_id` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_norm
        todo!(
            "TODO: port action `reject_invalid_op_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_opt_step_adamw(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepAdamw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_opt_step_adamw
        todo!(
            "TODO: port action `reject_invalid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_opt_step_sgd(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepSgd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_opt_step_sgd
        todo!(
            "TODO: port action `reject_invalid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_out_prod(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpOutProd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_out_prod
        todo!(
            "TODO: port action `reject_invalid_op_out_prod` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_pad(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPad,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_pad
        todo!(
            "TODO: port action `reject_invalid_op_pad` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_pad_reflect_1d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPadReflect1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_pad_reflect_1d
        todo!(
            "TODO: port action `reject_invalid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_permute(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPermute,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_permute
        todo!(
            "TODO: port action `reject_invalid_op_permute` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_1d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_pool_1d
        todo!(
            "TODO: port action `reject_invalid_op_pool_1d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_2d(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_pool_2d
        todo!(
            "TODO: port action `reject_invalid_op_pool_2d` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_2d_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpPool2dBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_pool_2d_back
        todo!(
            "TODO: port action `reject_invalid_op_pool_2d_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_repeat(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRepeat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_repeat
        todo!(
            "TODO: port action `reject_invalid_op_repeat` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_repeat_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRepeatBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_repeat_back
        todo!(
            "TODO: port action `reject_invalid_op_repeat_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_reshape(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpReshape,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_reshape
        todo!(
            "TODO: port action `reject_invalid_op_reshape` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rms_norm(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rms_norm
        todo!(
            "TODO: port action `reject_invalid_op_rms_norm` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rms_norm_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNormBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rms_norm_back
        todo!(
            "TODO: port action `reject_invalid_op_rms_norm_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_roll(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRoll,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_roll
        todo!(
            "TODO: port action `reject_invalid_op_roll` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rope(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rope
        todo!(
            "TODO: port action `reject_invalid_op_rope` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rope_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRopeBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rope_back
        todo!(
            "TODO: port action `reject_invalid_op_rope_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rwkv_wkv6(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv6,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rwkv_wkv6
        todo!(
            "TODO: port action `reject_invalid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_rwkv_wkv7(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv7,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_rwkv_wkv7
        todo!(
            "TODO: port action `reject_invalid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_scale(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpScale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_scale
        todo!(
            "TODO: port action `reject_invalid_op_scale` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_set(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSet,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_set
        todo!(
            "TODO: port action `reject_invalid_op_set` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_set_rows(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_set_rows
        todo!(
            "TODO: port action `reject_invalid_op_set_rows` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_silu_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSiluBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_silu_back
        todo!(
            "TODO: port action `reject_invalid_op_silu_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sin(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSin,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sin
        todo!(
            "TODO: port action `reject_invalid_op_sin` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_soft_max(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_soft_max
        todo!(
            "TODO: port action `reject_invalid_op_soft_max` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_soft_max_back(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMaxBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_soft_max_back
        todo!(
            "TODO: port action `reject_invalid_op_soft_max_back` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_solve_tri(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSolveTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_solve_tri
        todo!(
            "TODO: port action `reject_invalid_op_solve_tri` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sqr(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSqr,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sqr
        todo!(
            "TODO: port action `reject_invalid_op_sqr` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sqrt(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSqrt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sqrt
        todo!(
            "TODO: port action `reject_invalid_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_ssm_conv(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSsmConv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_ssm_conv
        todo!(
            "TODO: port action `reject_invalid_op_ssm_conv` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_ssm_scan(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSsmScan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_ssm_scan
        todo!(
            "TODO: port action `reject_invalid_op_ssm_scan` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sub(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSub,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sub
        todo!(
            "TODO: port action `reject_invalid_op_sub` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sum(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSum,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sum
        todo!(
            "TODO: port action `reject_invalid_op_sum` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_sum_rows(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpSumRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_sum_rows
        todo!(
            "TODO: port action `reject_invalid_op_sum_rows` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_timestep_embedding(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTimestepEmbedding,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_timestep_embedding
        todo!(
            "TODO: port action `reject_invalid_op_timestep_embedding` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_top_k(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTopK,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_top_k
        todo!(
            "TODO: port action `reject_invalid_op_top_k` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_transpose(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTranspose,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_transpose
        todo!(
            "TODO: port action `reject_invalid_op_transpose` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_tri(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_tri
        todo!(
            "TODO: port action `reject_invalid_op_tri` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_unary(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_unary
        todo!(
            "TODO: port action `reject_invalid_op_unary` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_upscale(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpUpscale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_upscale
        todo!(
            "TODO: port action `reject_invalid_op_upscale` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_view(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_view
        todo!(
            "TODO: port action `reject_invalid_op_view` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_win_part(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpWinPart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_win_part
        todo!(
            "TODO: port action `reject_invalid_op_win_part` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn reject_invalid_op_win_unpart(
        &mut self,
        _event: &EmelKernelAarch64EventDispatchOpWinUnpart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/actions.hpp::reject_invalid_op_win_unpart
        todo!(
            "TODO: port action `reject_invalid_op_win_unpart` from emel.cpp/src/emel/kernel/aarch64/actions.hpp"
        )
    }
    fn simd_op_add(&self, _event: &EmelKernelAarch64EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_add
        todo!("TODO: port guard `simd_op_add` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_conv_transpose_1d_f32(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_conv_transpose_1d_f32
        todo!(
            "TODO: port guard `simd_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_div(&self, _event: &EmelKernelAarch64EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_div
        todo!("TODO: port guard `simd_op_div` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_dup(&self, _event: &EmelKernelAarch64EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_dup
        todo!("TODO: port guard `simd_op_dup` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_flash_attn_ext_f16kv_one_chunk(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_flash_attn_ext_f16kv_one_chunk
        todo!(
            "TODO: port guard `simd_op_flash_attn_ext_f16kv_one_chunk` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul(&self, _event: &EmelKernelAarch64EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul
        todo!("TODO: port guard `simd_op_mul` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4
        todo!(
            "TODO: port guard `simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8
        todo!(
            "TODO: port guard `simd_op_mul_mat_argmax_q4_vector_packed_f32_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm
        todo!(
            "TODO: port guard `simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm
        todo!(
            "TODO: port guard `simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_f16_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_f16_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_f16_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_f32_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_f32_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_f32_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_generic(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_generic
        todo!(
            "TODO: port guard `simd_op_mul_mat_generic` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_0_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_0_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_0_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_1_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_1_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_1_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_f32_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_f32_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_f32_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_f32_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_f32_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q4_vector_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q4_vector_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q4_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q5_0_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q5_0_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_q5_0_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_packed(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_packed
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_packed` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_packed_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_packed_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_packed_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_prepared_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_prepared_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_prepared_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q6_vector_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q6_vector_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q6_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_packed_bl4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_packed_bl4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_packed_bl4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_packed_bl8(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_packed_bl8
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_packed_bl8` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_packed_bl8_full_groups(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_packed_bl8_full_groups
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_packed_bl8_full_groups` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_packed_bl8_matrix_x4(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_packed_bl8_matrix_x4
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_packed_bl8_matrix_x4` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_vector(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_vector
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_vector` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_mul_mat_q8_0_vector_q8_rhs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_mul_mat_q8_0_vector_q8_rhs
        todo!(
            "TODO: port guard `simd_op_mul_mat_q8_0_vector_q8_rhs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_sqr(&self, _event: &EmelKernelAarch64EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_sqr
        todo!("TODO: port guard `simd_op_sqr` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_sqrt(&self, _event: &EmelKernelAarch64EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_sqrt
        todo!("TODO: port guard `simd_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_sub(&self, _event: &EmelKernelAarch64EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_sub
        todo!("TODO: port guard `simd_op_sub` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn simd_op_unary_abs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_unary_abs
        todo!(
            "TODO: port guard `simd_op_unary_abs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_unary_neg(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_unary_neg
        todo!(
            "TODO: port guard `simd_op_unary_neg` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_unary_relu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_unary_relu
        todo!(
            "TODO: port guard `simd_op_unary_relu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn simd_op_unary_silu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::simd_op_unary_silu
        todo!(
            "TODO: port guard `simd_op_unary_silu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_acc(&self, _event: &EmelKernelAarch64EventDispatchOpAcc) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_acc
        todo!("TODO: port guard `valid_op_acc` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_add1(&self, _event: &EmelKernelAarch64EventDispatchOpAdd1) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_add1
        todo!("TODO: port guard `valid_op_add1` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_add_broadcast_row(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpAdd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_add_broadcast_row
        todo!(
            "TODO: port guard `valid_op_add_broadcast_row` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_add_equal(&self, _event: &EmelKernelAarch64EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_add_equal
        todo!(
            "TODO: port guard `valid_op_add_equal` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_add_id(&self, _event: &EmelKernelAarch64EventDispatchOpAddId) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_add_id
        todo!("TODO: port guard `valid_op_add_id` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_add_rel_pos(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpAddRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_add_rel_pos
        todo!(
            "TODO: port guard `valid_op_add_rel_pos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_arange(&self, _event: &EmelKernelAarch64EventDispatchOpArange) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_arange
        todo!("TODO: port guard `valid_op_arange` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_argmax(&self, _event: &EmelKernelAarch64EventDispatchOpArgmax) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_argmax
        todo!("TODO: port guard `valid_op_argmax` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_argsort(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpArgsort,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_argsort
        todo!(
            "TODO: port guard `valid_op_argsort` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_clamp(&self, _event: &EmelKernelAarch64EventDispatchOpClamp) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_clamp
        todo!("TODO: port guard `valid_op_clamp` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_concat(&self, _event: &EmelKernelAarch64EventDispatchOpConcat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_concat
        todo!("TODO: port guard `valid_op_concat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_cont(&self, _event: &EmelKernelAarch64EventDispatchOpCont) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cont
        todo!("TODO: port guard `valid_op_cont` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_conv_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_2d
        todo!(
            "TODO: port guard `valid_op_conv_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_conv_2d_dw(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv2dDw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_2d_dw
        todo!(
            "TODO: port guard `valid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_conv_3d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConv3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_3d
        todo!(
            "TODO: port guard `valid_op_conv_3d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_conv_transpose_1d_f16(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_transpose_1d_f16
        todo!(
            "TODO: port guard `valid_op_conv_transpose_1d_f16` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_conv_transpose_1d_f32(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_transpose_1d_f32
        todo!(
            "TODO: port guard `valid_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_conv_transpose_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpConvTranspose2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_conv_transpose_2d
        todo!(
            "TODO: port guard `valid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_cos(&self, _event: &EmelKernelAarch64EventDispatchOpCos) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cos
        todo!("TODO: port guard `valid_op_cos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_count_equal(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCountEqual,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_count_equal
        todo!(
            "TODO: port guard `valid_op_count_equal` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_cpy(&self, _event: &EmelKernelAarch64EventDispatchOpCpy) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cpy
        todo!("TODO: port guard `valid_op_cpy` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_cross_entropy_loss(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLoss,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cross_entropy_loss
        todo!(
            "TODO: port guard `valid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_cross_entropy_loss_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpCrossEntropyLossBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cross_entropy_loss_back
        todo!(
            "TODO: port guard `valid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_cumsum(&self, _event: &EmelKernelAarch64EventDispatchOpCumsum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_cumsum
        todo!("TODO: port guard `valid_op_cumsum` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_custom(&self, _event: &EmelKernelAarch64EventDispatchOpCustom) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_custom
        todo!("TODO: port guard `valid_op_custom` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_diag(&self, _event: &EmelKernelAarch64EventDispatchOpDiag) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_diag
        todo!("TODO: port guard `valid_op_diag` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_diag_mask_inf(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskInf,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_diag_mask_inf
        todo!(
            "TODO: port guard `valid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_diag_mask_zero(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpDiagMaskZero,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_diag_mask_zero
        todo!(
            "TODO: port guard `valid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_div(&self, _event: &EmelKernelAarch64EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_div
        todo!("TODO: port guard `valid_op_div` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_dup(&self, _event: &EmelKernelAarch64EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_dup
        todo!("TODO: port guard `valid_op_dup` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_fill(&self, _event: &EmelKernelAarch64EventDispatchOpFill) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_fill
        todo!("TODO: port guard `valid_op_fill` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_flash_attn_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_flash_attn_back
        todo!(
            "TODO: port guard `valid_op_flash_attn_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_flash_attn_ext_shared(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_flash_attn_ext_shared
        todo!(
            "TODO: port guard `valid_op_flash_attn_ext_shared` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_gated_linear_attn(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGatedLinearAttn,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_gated_linear_attn
        todo!(
            "TODO: port guard `valid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rel_pos(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rel_pos
        todo!(
            "TODO: port guard `valid_op_get_rel_pos` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRowsBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_back
        todo!(
            "TODO: port guard `valid_op_get_rows_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_bf16(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_bf16
        todo!(
            "TODO: port guard `valid_op_get_rows_bf16` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_f16(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_f16
        todo!(
            "TODO: port guard `valid_op_get_rows_f16` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_f32(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_f32
        todo!(
            "TODO: port guard `valid_op_get_rows_f32` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q4_0(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_q4_0
        todo!(
            "TODO: port guard `valid_op_get_rows_q4_0` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q4_k(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_q4_k
        todo!(
            "TODO: port guard `valid_op_get_rows_q4_k` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q8_0(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_get_rows_q8_0
        todo!(
            "TODO: port guard `valid_op_get_rows_q8_0` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_glu(&self, _event: &EmelKernelAarch64EventDispatchOpGlu) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_glu
        todo!("TODO: port guard `valid_op_glu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_group_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpGroupNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_group_norm
        todo!(
            "TODO: port guard `valid_op_group_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_im2col_3d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_im2col_3d
        todo!(
            "TODO: port guard `valid_op_im2col_3d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_im2col_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2colBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_im2col_back
        todo!(
            "TODO: port guard `valid_op_im2col_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_im2col_f16(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_im2col_f16
        todo!(
            "TODO: port guard `valid_op_im2col_f16` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_im2col_f32(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpIm2col,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_im2col_f32
        todo!(
            "TODO: port guard `valid_op_im2col_f32` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_l2_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpL2Norm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_l2_norm
        todo!(
            "TODO: port guard `valid_op_l2_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_leaky_relu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpLeakyRelu,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_leaky_relu
        todo!(
            "TODO: port guard `valid_op_leaky_relu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_log(&self, _event: &EmelKernelAarch64EventDispatchOpLog) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_log
        todo!("TODO: port guard `valid_op_log` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_map_custom1(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom1,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_map_custom1
        todo!(
            "TODO: port guard `valid_op_map_custom1` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_map_custom2(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom2,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_map_custom2
        todo!(
            "TODO: port guard `valid_op_map_custom2` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_map_custom3(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMapCustom3,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_map_custom3
        todo!(
            "TODO: port guard `valid_op_map_custom3` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mean(&self, _event: &EmelKernelAarch64EventDispatchOpMean) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mean
        todo!("TODO: port guard `valid_op_mean` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_mul_broadcast_row(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMul,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_broadcast_row
        todo!(
            "TODO: port guard `valid_op_mul_broadcast_row` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mul_equal(&self, _event: &EmelKernelAarch64EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_equal
        todo!(
            "TODO: port guard `valid_op_mul_equal` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mul_mat(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_mat
        todo!(
            "TODO: port guard `valid_op_mul_mat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mul_mat_argmax(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_mat_argmax
        todo!(
            "TODO: port guard `valid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mul_mat_f16(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_mat_f16
        todo!(
            "TODO: port guard `valid_op_mul_mat_f16` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_mul_mat_id(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpMulMatId,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_mul_mat_id
        todo!(
            "TODO: port guard `valid_op_mul_mat_id` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_norm(&self, _event: &EmelKernelAarch64EventDispatchOpNorm) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_norm
        todo!("TODO: port guard `valid_op_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_opt_step_adamw(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepAdamw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_opt_step_adamw
        todo!(
            "TODO: port guard `valid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_opt_step_sgd(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOptStepSgd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_opt_step_sgd
        todo!(
            "TODO: port guard `valid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_out_prod(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpOutProd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_out_prod
        todo!(
            "TODO: port guard `valid_op_out_prod` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_pad(&self, _event: &EmelKernelAarch64EventDispatchOpPad) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_pad
        todo!("TODO: port guard `valid_op_pad` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_pad_reflect_1d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPadReflect1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_pad_reflect_1d
        todo!(
            "TODO: port guard `valid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_permute(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPermute,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_permute
        todo!(
            "TODO: port guard `valid_op_permute` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_pool_1d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_pool_1d
        todo!(
            "TODO: port guard `valid_op_pool_1d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_pool_2d(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_pool_2d
        todo!(
            "TODO: port guard `valid_op_pool_2d` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_pool_2d_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpPool2dBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_pool_2d_back
        todo!(
            "TODO: port guard `valid_op_pool_2d_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_repeat(&self, _event: &EmelKernelAarch64EventDispatchOpRepeat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_repeat
        todo!("TODO: port guard `valid_op_repeat` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_repeat_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRepeatBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_repeat_back
        todo!(
            "TODO: port guard `valid_op_repeat_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_reshape(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpReshape,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_reshape
        todo!(
            "TODO: port guard `valid_op_reshape` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rms_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rms_norm
        todo!(
            "TODO: port guard `valid_op_rms_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rms_norm_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRmsNormBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rms_norm_back
        todo!(
            "TODO: port guard `valid_op_rms_norm_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_roll(&self, _event: &EmelKernelAarch64EventDispatchOpRoll) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_roll
        todo!("TODO: port guard `valid_op_roll` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_rope_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRopeBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rope_back
        todo!(
            "TODO: port guard `valid_op_rope_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rope_neox(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rope_neox
        todo!(
            "TODO: port guard `valid_op_rope_neox` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rope_norm(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rope_norm
        todo!(
            "TODO: port guard `valid_op_rope_norm` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rope_timestep(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRope,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rope_timestep
        todo!(
            "TODO: port guard `valid_op_rope_timestep` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rwkv_wkv6(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv6,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rwkv_wkv6
        todo!(
            "TODO: port guard `valid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_rwkv_wkv7(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpRwkvWkv7,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_rwkv_wkv7
        todo!(
            "TODO: port guard `valid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_scale(&self, _event: &EmelKernelAarch64EventDispatchOpScale) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_scale
        todo!("TODO: port guard `valid_op_scale` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_set(&self, _event: &EmelKernelAarch64EventDispatchOpSet) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_set
        todo!("TODO: port guard `valid_op_set` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_set_rows(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_set_rows
        todo!(
            "TODO: port guard `valid_op_set_rows` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_silu_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSiluBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_silu_back
        todo!(
            "TODO: port guard `valid_op_silu_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_sin(&self, _event: &EmelKernelAarch64EventDispatchOpSin) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sin
        todo!("TODO: port guard `valid_op_sin` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_soft_max(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_soft_max
        todo!(
            "TODO: port guard `valid_op_soft_max` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_soft_max_back(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSoftMaxBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_soft_max_back
        todo!(
            "TODO: port guard `valid_op_soft_max_back` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_solve_tri(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSolveTri,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_solve_tri
        todo!(
            "TODO: port guard `valid_op_solve_tri` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_sqr(&self, _event: &EmelKernelAarch64EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sqr
        todo!("TODO: port guard `valid_op_sqr` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_sqrt(&self, _event: &EmelKernelAarch64EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sqrt
        todo!("TODO: port guard `valid_op_sqrt` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_ssm_conv(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSsmConv,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_ssm_conv
        todo!(
            "TODO: port guard `valid_op_ssm_conv` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_ssm_scan(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSsmScan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_ssm_scan
        todo!(
            "TODO: port guard `valid_op_ssm_scan` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_sub(&self, _event: &EmelKernelAarch64EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sub
        todo!("TODO: port guard `valid_op_sub` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_sum(&self, _event: &EmelKernelAarch64EventDispatchOpSum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sum
        todo!("TODO: port guard `valid_op_sum` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_sum_rows(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpSumRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_sum_rows
        todo!(
            "TODO: port guard `valid_op_sum_rows` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_timestep_embedding(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpTimestepEmbedding,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_timestep_embedding
        todo!(
            "TODO: port guard `valid_op_timestep_embedding` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_top_k(&self, _event: &EmelKernelAarch64EventDispatchOpTopK) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_top_k
        todo!("TODO: port guard `valid_op_top_k` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_transpose(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpTranspose,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_transpose
        todo!(
            "TODO: port guard `valid_op_transpose` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_tri(&self, _event: &EmelKernelAarch64EventDispatchOpTri) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_tri
        todo!("TODO: port guard `valid_op_tri` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_unary_abs(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_abs
        todo!(
            "TODO: port guard `valid_op_unary_abs` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_elu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_elu
        todo!(
            "TODO: port guard `valid_op_unary_elu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_exp(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_exp
        todo!(
            "TODO: port guard `valid_op_unary_exp` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_gelu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_gelu
        todo!(
            "TODO: port guard `valid_op_unary_gelu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_neg(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_neg
        todo!(
            "TODO: port guard `valid_op_unary_neg` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_relu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_relu
        todo!(
            "TODO: port guard `valid_op_unary_relu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_silu(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_silu
        todo!(
            "TODO: port guard `valid_op_unary_silu` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_unary_tanh(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_unary_tanh
        todo!(
            "TODO: port guard `valid_op_unary_tanh` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_upscale(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpUpscale,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_upscale
        todo!(
            "TODO: port guard `valid_op_upscale` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_view(&self, _event: &EmelKernelAarch64EventDispatchOpView) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_view
        todo!("TODO: port guard `valid_op_view` from emel.cpp/src/emel/kernel/aarch64/guards.hpp")
    }
    fn valid_op_win_part(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpWinPart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_win_part
        todo!(
            "TODO: port guard `valid_op_win_part` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
    fn valid_op_win_unpart(
        &self,
        _event: &EmelKernelAarch64EventDispatchOpWinUnpart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/aarch64/guards.hpp::valid_op_win_unpart
        todo!(
            "TODO: port guard `valid_op_win_unpart` from emel.cpp/src/emel/kernel/aarch64/guards.hpp"
        )
    }
}
