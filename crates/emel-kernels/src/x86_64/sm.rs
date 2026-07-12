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

// --- machine KernelX8664 from emel.cpp/src/emel/kernel/x86_64/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpAcc;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpAdd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpAdd1;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpAddId;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpAddRelPos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpArange;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpArgmax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpArgsort;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpClamp;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConcat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCont;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConv2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConv2dDw;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConv3d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConvTranspose1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpConvTranspose2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCountEqual;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCpy;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCrossEntropyLoss;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCrossEntropyLossBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCumsum;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpCustom;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpDiag;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpDiagMaskInf;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpDiagMaskZero;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpDiv;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpDup;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpFill;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpFlashAttnBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpFlashAttnExt;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGatedLinearAttn;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGetRelPos;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGetRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGetRowsBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGlu;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpGroupNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpIm2col;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpIm2col3d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpIm2colBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpL2Norm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpLeakyRelu;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpLog;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMapCustom1;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMapCustom2;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMapCustom3;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMean;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMul;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMulMat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMulMatArgmax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpMulMatId;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpOptStepAdamw;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpOptStepSgd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpOutProd;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPad;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPadReflect1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPermute;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPool1d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPool2d;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpPool2dBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRepeat;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRepeatBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpReshape;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRmsNorm;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRmsNormBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRoll;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRope;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRopeBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRwkvWkv6;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpRwkvWkv7;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpScale;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSet;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSetRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSiluBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSin;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSoftMax;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSoftMaxBack;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSolveTri;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSqr;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSqrt;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSsmConv;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSsmScan;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSub;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSum;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpSumRows;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpTimestepEmbedding;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpTopK;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpTranspose;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpTri;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpUnary;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpUpscale;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpView;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpWinPart;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchOpWinUnpart;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelKernelX8664EventDispatchRequest;

sml! {
    KernelX8664 {
        "ready"_s <= *"ready"_s + event<EmelKernelX8664EventDispatchRequest> / exec_dispatch,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDup> [simd_op_dup] / exec_simd_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDup> [valid_op_dup] / exec_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDup> [invalid_op_dup] / reject_invalid_op_dup,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd> [simd_op_add] / exec_simd_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd> [valid_op_add_equal] / exec_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd> [valid_op_add_broadcast_row] / exec_scalar_op_add_broadcast_row,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd> [invalid_op_add] / reject_invalid_op_add,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAddId> [valid_op_add_id] / exec_op_add_id,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAddId> [invalid_op_add_id] / reject_invalid_op_add_id,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd1> [valid_op_add1] / exec_op_add1,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAdd1> [invalid_op_add1] / reject_invalid_op_add1,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAcc> [valid_op_acc] / exec_op_acc,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAcc> [invalid_op_acc] / reject_invalid_op_acc,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSub> [simd_op_sub] / exec_simd_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSub> [valid_op_sub] / exec_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSub> [invalid_op_sub] / reject_invalid_op_sub,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMul> [simd_op_mul] / exec_simd_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMul> [valid_op_mul_equal] / exec_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMul> [valid_op_mul_broadcast_row] / exec_scalar_op_mul_broadcast_row,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMul> [invalid_op_mul] / reject_invalid_op_mul,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiv> [simd_op_div] / exec_simd_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiv> [valid_op_div] / exec_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiv> [invalid_op_div] / reject_invalid_op_div,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqr> [simd_op_sqr] / exec_simd_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqr> [valid_op_sqr] / exec_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqr> [invalid_op_sqr] / reject_invalid_op_sqr,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqrt> [simd_op_sqrt] / exec_simd_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqrt> [valid_op_sqrt] / exec_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSqrt> [invalid_op_sqrt] / reject_invalid_op_sqrt,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpLog> [valid_op_log] / exec_op_log,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpLog> [invalid_op_log] / reject_invalid_op_log,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSin> [valid_op_sin] / exec_op_sin,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSin> [invalid_op_sin] / reject_invalid_op_sin,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCos> [valid_op_cos] / exec_op_cos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCos> [invalid_op_cos] / reject_invalid_op_cos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSum> [valid_op_sum] / exec_op_sum,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSum> [invalid_op_sum] / reject_invalid_op_sum,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSumRows> [valid_op_sum_rows] / exec_op_sum_rows,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSumRows> [invalid_op_sum_rows] / reject_invalid_op_sum_rows,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCumsum> [valid_op_cumsum] / exec_op_cumsum,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCumsum> [invalid_op_cumsum] / reject_invalid_op_cumsum,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMean> [valid_op_mean] / exec_op_mean,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMean> [invalid_op_mean] / reject_invalid_op_mean,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArgmax> [valid_op_argmax] / exec_op_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArgmax> [invalid_op_argmax] / reject_invalid_op_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCountEqual> [valid_op_count_equal] / exec_op_count_equal,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCountEqual> [invalid_op_count_equal] / reject_invalid_op_count_equal,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRepeat> [valid_op_repeat] / exec_op_repeat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRepeat> [invalid_op_repeat] / reject_invalid_op_repeat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRepeatBack> [valid_op_repeat_back] / exec_op_repeat_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRepeatBack> [invalid_op_repeat_back] / reject_invalid_op_repeat_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConcat> [valid_op_concat] / exec_op_concat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConcat> [invalid_op_concat] / reject_invalid_op_concat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSiluBack> [valid_op_silu_back] / exec_op_silu_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSiluBack> [invalid_op_silu_back] / reject_invalid_op_silu_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpNorm> [valid_op_norm] / exec_op_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpNorm> [invalid_op_norm] / reject_invalid_op_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRmsNorm> [valid_op_rms_norm] / exec_op_rms_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRmsNorm> [invalid_op_rms_norm] / reject_invalid_op_rms_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRmsNormBack> [valid_op_rms_norm_back] / exec_op_rms_norm_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRmsNormBack> [invalid_op_rms_norm_back] / reject_invalid_op_rms_norm_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGroupNorm> [valid_op_group_norm] / exec_op_group_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGroupNorm> [invalid_op_group_norm] / reject_invalid_op_group_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpL2Norm> [valid_op_l2_norm] / exec_op_l2_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpL2Norm> [invalid_op_l2_norm] / reject_invalid_op_l2_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q2_k_q8_k] / effect_exec_simd_op_mul_mat_q2_k_q8_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q3_k_q8_k] / effect_exec_simd_op_mul_mat_q3_k_q8_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q4_k_q8_k] / effect_exec_simd_op_mul_mat_q4_k_q8_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q6_k_q8_k] / effect_exec_simd_op_mul_mat_q6_k_q8_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q4_0_q8_0] / effect_exec_simd_op_mul_mat_q4_0_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q4_1_q8_0] / effect_exec_simd_op_mul_mat_q4_1_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q5_0_q8_0] / effect_exec_simd_op_mul_mat_q5_0_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_q8_0_q8_0] / effect_exec_simd_op_mul_mat_q8_0_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_f32_fma_vector] / effect_exec_simd_op_mul_mat_f32_fma_vector,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_f32_fma] / effect_exec_simd_op_mul_mat_f32_fma,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [guard_simd_op_mul_mat_f32_avx2_only] / exec_simd_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [valid_op_mul_mat_f16] / exec_scalar_op_mul_mat_f16,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [valid_op_mul_mat] / exec_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMat> [invalid_op_mul_mat] / reject_invalid_op_mul_mat,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMatArgmax> [valid_op_mul_mat_argmax] / exec_op_mul_mat_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMatArgmax> [invalid_op_mul_mat_argmax] / reject_invalid_op_mul_mat_argmax,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMatId> [valid_op_mul_mat_id] / exec_op_mul_mat_id,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMulMatId> [invalid_op_mul_mat_id] / reject_invalid_op_mul_mat_id,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOutProd> [valid_op_out_prod] / exec_op_out_prod,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOutProd> [invalid_op_out_prod] / reject_invalid_op_out_prod,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpScale> [valid_op_scale] / exec_op_scale,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpScale> [invalid_op_scale] / reject_invalid_op_scale,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSet> [valid_op_set] / exec_op_set,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSet> [invalid_op_set] / reject_invalid_op_set,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCpy> [valid_op_cpy] / exec_op_cpy,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCpy> [invalid_op_cpy] / reject_invalid_op_cpy,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCont> [valid_op_cont] / exec_op_cont,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCont> [invalid_op_cont] / reject_invalid_op_cont,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpReshape> [valid_op_reshape] / exec_op_reshape,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpReshape> [invalid_op_reshape] / reject_invalid_op_reshape,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpView> [valid_op_view] / exec_op_view,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpView> [invalid_op_view] / reject_invalid_op_view,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPermute> [valid_op_permute] / exec_op_permute,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPermute> [invalid_op_permute] / reject_invalid_op_permute,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTranspose> [valid_op_transpose] / exec_op_transpose,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTranspose> [invalid_op_transpose] / reject_invalid_op_transpose,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_f32] / exec_scalar_op_get_rows_f32,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_f16] / exec_scalar_op_get_rows_f16,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_bf16] / exec_scalar_op_get_rows_bf16,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_q4_0] / exec_scalar_op_get_rows_q4_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_q8_0] / exec_scalar_op_get_rows_q8_0,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [valid_op_get_rows_q4_k] / exec_scalar_op_get_rows_q4_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRows> [invalid_op_get_rows] / reject_invalid_op_get_rows,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRowsBack> [valid_op_get_rows_back] / exec_op_get_rows_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRowsBack> [invalid_op_get_rows_back] / reject_invalid_op_get_rows_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSetRows> [valid_op_set_rows] / exec_op_set_rows,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSetRows> [invalid_op_set_rows] / reject_invalid_op_set_rows,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiag> [valid_op_diag] / exec_op_diag,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiag> [invalid_op_diag] / reject_invalid_op_diag,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiagMaskInf> [valid_op_diag_mask_inf] / exec_op_diag_mask_inf,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiagMaskInf> [invalid_op_diag_mask_inf] / reject_invalid_op_diag_mask_inf,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiagMaskZero> [valid_op_diag_mask_zero] / exec_op_diag_mask_zero,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpDiagMaskZero> [invalid_op_diag_mask_zero] / reject_invalid_op_diag_mask_zero,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSoftMax> [valid_op_soft_max] / exec_op_soft_max,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSoftMax> [invalid_op_soft_max] / reject_invalid_op_soft_max,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSoftMaxBack> [valid_op_soft_max_back] / exec_op_soft_max_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSoftMaxBack> [invalid_op_soft_max_back] / reject_invalid_op_soft_max_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRope> [valid_op_rope_norm] / exec_scalar_op_rope_norm,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRope> [valid_op_rope_neox] / exec_scalar_op_rope_neox,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRope> [valid_op_rope_timestep] / exec_scalar_op_rope_timestep,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRope> [invalid_op_rope] / reject_invalid_op_rope,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRopeBack> [valid_op_rope_back] / exec_op_rope_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRopeBack> [invalid_op_rope_back] / reject_invalid_op_rope_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpClamp> [valid_op_clamp] / exec_op_clamp,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpClamp> [invalid_op_clamp] / reject_invalid_op_clamp,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConvTranspose1d> [valid_op_conv_transpose_1d_f32] / exec_scalar_op_conv_transpose_1d_f32,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConvTranspose1d> [valid_op_conv_transpose_1d_f16] / exec_scalar_op_conv_transpose_1d_f16,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConvTranspose1d> [invalid_op_conv_transpose_1d] / reject_invalid_op_conv_transpose_1d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2col> [valid_op_im2col_f32] / exec_scalar_op_im2col_f32,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2col> [valid_op_im2col_f16] / exec_scalar_op_im2col_f16,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2col> [invalid_op_im2col] / reject_invalid_op_im2col,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2colBack> [valid_op_im2col_back] / exec_op_im2col_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2colBack> [invalid_op_im2col_back] / reject_invalid_op_im2col_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2col3d> [valid_op_im2col_3d] / exec_op_im2col_3d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpIm2col3d> [invalid_op_im2col_3d] / reject_invalid_op_im2col_3d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv2d> [valid_op_conv_2d] / exec_op_conv_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv2d> [invalid_op_conv_2d] / reject_invalid_op_conv_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv3d> [valid_op_conv_3d] / exec_op_conv_3d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv3d> [invalid_op_conv_3d] / reject_invalid_op_conv_3d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv2dDw> [valid_op_conv_2d_dw] / exec_op_conv_2d_dw,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConv2dDw> [invalid_op_conv_2d_dw] / reject_invalid_op_conv_2d_dw,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConvTranspose2d> [valid_op_conv_transpose_2d] / exec_op_conv_transpose_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpConvTranspose2d> [invalid_op_conv_transpose_2d] / reject_invalid_op_conv_transpose_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool1d> [valid_op_pool_1d] / exec_op_pool_1d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool1d> [invalid_op_pool_1d] / reject_invalid_op_pool_1d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool2d> [valid_op_pool_2d] / exec_op_pool_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool2d> [invalid_op_pool_2d] / reject_invalid_op_pool_2d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool2dBack> [valid_op_pool_2d_back] / exec_op_pool_2d_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPool2dBack> [invalid_op_pool_2d_back] / reject_invalid_op_pool_2d_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUpscale> [valid_op_upscale] / exec_op_upscale,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUpscale> [invalid_op_upscale] / reject_invalid_op_upscale,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPad> [valid_op_pad] / exec_op_pad,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPad> [invalid_op_pad] / reject_invalid_op_pad,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPadReflect1d> [valid_op_pad_reflect_1d] / exec_op_pad_reflect_1d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpPadReflect1d> [invalid_op_pad_reflect_1d] / reject_invalid_op_pad_reflect_1d,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRoll> [valid_op_roll] / exec_op_roll,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRoll> [invalid_op_roll] / reject_invalid_op_roll,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArange> [valid_op_arange] / exec_op_arange,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArange> [invalid_op_arange] / reject_invalid_op_arange,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTimestepEmbedding> [valid_op_timestep_embedding] / exec_op_timestep_embedding,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTimestepEmbedding> [invalid_op_timestep_embedding] / reject_invalid_op_timestep_embedding,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArgsort> [valid_op_argsort] / exec_op_argsort,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpArgsort> [invalid_op_argsort] / reject_invalid_op_argsort,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTopK> [valid_op_top_k] / exec_op_top_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTopK> [invalid_op_top_k] / reject_invalid_op_top_k,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpLeakyRelu> [valid_op_leaky_relu] / exec_op_leaky_relu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpLeakyRelu> [invalid_op_leaky_relu] / reject_invalid_op_leaky_relu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTri> [valid_op_tri] / exec_op_tri,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpTri> [invalid_op_tri] / reject_invalid_op_tri,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFill> [valid_op_fill] / exec_op_fill,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFill> [invalid_op_fill] / reject_invalid_op_fill,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFlashAttnExt> [simd_op_flash_attn_ext_f16kv_one_chunk] / exec_simd_op_flash_attn_ext_f16kv_one_chunk,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFlashAttnExt> [valid_op_flash_attn_ext_shared] / exec_op_flash_attn_ext,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFlashAttnExt> [invalid_op_flash_attn_ext] / reject_invalid_op_flash_attn_ext,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFlashAttnBack> [valid_op_flash_attn_back] / exec_op_flash_attn_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpFlashAttnBack> [invalid_op_flash_attn_back] / reject_invalid_op_flash_attn_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSsmConv> [valid_op_ssm_conv] / exec_op_ssm_conv,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSsmConv> [invalid_op_ssm_conv] / reject_invalid_op_ssm_conv,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSsmScan> [valid_op_ssm_scan] / exec_op_ssm_scan,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSsmScan> [invalid_op_ssm_scan] / reject_invalid_op_ssm_scan,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpWinPart> [valid_op_win_part] / exec_op_win_part,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpWinPart> [invalid_op_win_part] / reject_invalid_op_win_part,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpWinUnpart> [valid_op_win_unpart] / exec_op_win_unpart,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpWinUnpart> [invalid_op_win_unpart] / reject_invalid_op_win_unpart,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRelPos> [valid_op_get_rel_pos] / exec_op_get_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGetRelPos> [invalid_op_get_rel_pos] / reject_invalid_op_get_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAddRelPos> [valid_op_add_rel_pos] / exec_op_add_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpAddRelPos> [invalid_op_add_rel_pos] / reject_invalid_op_add_rel_pos,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRwkvWkv6> [valid_op_rwkv_wkv6] / exec_op_rwkv_wkv6,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRwkvWkv6> [invalid_op_rwkv_wkv6] / reject_invalid_op_rwkv_wkv6,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGatedLinearAttn> [valid_op_gated_linear_attn] / exec_op_gated_linear_attn,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGatedLinearAttn> [invalid_op_gated_linear_attn] / reject_invalid_op_gated_linear_attn,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRwkvWkv7> [valid_op_rwkv_wkv7] / exec_op_rwkv_wkv7,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpRwkvWkv7> [invalid_op_rwkv_wkv7] / reject_invalid_op_rwkv_wkv7,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSolveTri> [valid_op_solve_tri] / exec_op_solve_tri,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpSolveTri> [invalid_op_solve_tri] / reject_invalid_op_solve_tri,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [simd_op_unary_abs] / exec_simd_op_unary_abs,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [simd_op_unary_neg] / exec_simd_op_unary_neg,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [simd_op_unary_relu] / exec_simd_op_unary_relu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_abs] / exec_scalar_op_unary_abs,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_neg] / exec_scalar_op_unary_neg,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_relu] / exec_scalar_op_unary_relu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_exp] / exec_scalar_op_unary_exp,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_tanh] / exec_scalar_op_unary_tanh,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_elu] / exec_scalar_op_unary_elu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_gelu] / exec_scalar_op_unary_gelu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [valid_op_unary_silu] / exec_scalar_op_unary_silu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpUnary> [invalid_op_unary] / reject_invalid_op_unary,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom1> [valid_op_map_custom1] / exec_op_map_custom1,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom1> [invalid_op_map_custom1] / reject_invalid_op_map_custom1,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom2> [valid_op_map_custom2] / exec_op_map_custom2,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom2> [invalid_op_map_custom2] / reject_invalid_op_map_custom2,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom3> [valid_op_map_custom3] / exec_op_map_custom3,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpMapCustom3> [invalid_op_map_custom3] / reject_invalid_op_map_custom3,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCustom> [valid_op_custom] / exec_op_custom,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCustom> [invalid_op_custom] / reject_invalid_op_custom,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCrossEntropyLoss> [valid_op_cross_entropy_loss] / exec_op_cross_entropy_loss,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCrossEntropyLoss> [invalid_op_cross_entropy_loss] / reject_invalid_op_cross_entropy_loss,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCrossEntropyLossBack> [valid_op_cross_entropy_loss_back] / exec_op_cross_entropy_loss_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpCrossEntropyLossBack> [invalid_op_cross_entropy_loss_back] / reject_invalid_op_cross_entropy_loss_back,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOptStepAdamw> [valid_op_opt_step_adamw] / exec_op_opt_step_adamw,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOptStepAdamw> [invalid_op_opt_step_adamw] / reject_invalid_op_opt_step_adamw,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOptStepSgd> [valid_op_opt_step_sgd] / exec_op_opt_step_sgd,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpOptStepSgd> [invalid_op_opt_step_sgd] / reject_invalid_op_opt_step_sgd,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGlu> [valid_op_glu] / exec_op_glu,
        "ready"_s <= "ready"_s + event<EmelKernelX8664EventDispatchOpGlu> [invalid_op_glu] / reject_invalid_op_glu,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected,
    }
}

/// Context for `KernelX8664` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct KernelX8664Context {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl KernelX8664StateMachineContext for KernelX8664Context {
    fn effect_exec_simd_op_mul_mat_f32_fma(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_f32_fma
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_f32_fma` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_f32_fma_vector(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_f32_fma_vector
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_f32_fma_vector` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q2_k_q8_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q2_k_q8_k
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q2_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q3_k_q8_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q3_k_q8_k
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q3_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q4_0_q8_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q4_0_q8_0
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q4_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q4_1_q8_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q4_1_q8_0
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q4_1_q8_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q4_k_q8_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q4_k_q8_k
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q4_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q5_0_q8_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q5_0_q8_0
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q5_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q6_k_q8_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q6_k_q8_k
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q6_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn effect_exec_simd_op_mul_mat_q8_0_q8_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::effect_exec_simd_op_mul_mat_q8_0_q8_0
        todo!(
            "TODO: port action `effect_exec_simd_op_mul_mat_q8_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_dispatch(&mut self, _event: &EmelKernelX8664EventDispatchRequest) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_dispatch
        todo!("TODO: port action `exec_dispatch` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_acc(&mut self, _event: &EmelKernelX8664EventDispatchOpAcc) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_acc
        todo!("TODO: port action `exec_op_acc` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_add(&mut self, _event: &EmelKernelX8664EventDispatchOpAdd) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_add
        todo!("TODO: port action `exec_op_add` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_add1(&mut self, _event: &EmelKernelX8664EventDispatchOpAdd1) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_add1
        todo!("TODO: port action `exec_op_add1` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_add_id(&mut self, _event: &EmelKernelX8664EventDispatchOpAddId) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_add_id
        todo!("TODO: port action `exec_op_add_id` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_add_rel_pos(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAddRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_add_rel_pos
        todo!(
            "TODO: port action `exec_op_add_rel_pos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_arange(&mut self, _event: &EmelKernelX8664EventDispatchOpArange) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_arange
        todo!("TODO: port action `exec_op_arange` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_argmax(&mut self, _event: &EmelKernelX8664EventDispatchOpArgmax) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_argmax
        todo!("TODO: port action `exec_op_argmax` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_argsort(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpArgsort,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_argsort
        todo!(
            "TODO: port action `exec_op_argsort` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_clamp(&mut self, _event: &EmelKernelX8664EventDispatchOpClamp) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_clamp
        todo!("TODO: port action `exec_op_clamp` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_concat(&mut self, _event: &EmelKernelX8664EventDispatchOpConcat) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_concat
        todo!("TODO: port action `exec_op_concat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_cont(&mut self, _event: &EmelKernelX8664EventDispatchOpCont) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cont
        todo!("TODO: port action `exec_op_cont` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_conv_2d(&mut self, _event: &EmelKernelX8664EventDispatchOpConv2d) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_conv_2d
        todo!(
            "TODO: port action `exec_op_conv_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_conv_2d_dw(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConv2dDw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_conv_2d_dw
        todo!(
            "TODO: port action `exec_op_conv_2d_dw` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_conv_3d(&mut self, _event: &EmelKernelX8664EventDispatchOpConv3d) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_conv_3d
        todo!(
            "TODO: port action `exec_op_conv_3d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_conv_transpose_2d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_conv_transpose_2d
        todo!(
            "TODO: port action `exec_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_cos(&mut self, _event: &EmelKernelX8664EventDispatchOpCos) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cos
        todo!("TODO: port action `exec_op_cos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_count_equal(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCountEqual,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_count_equal
        todo!(
            "TODO: port action `exec_op_count_equal` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_cpy(&mut self, _event: &EmelKernelX8664EventDispatchOpCpy) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cpy
        todo!("TODO: port action `exec_op_cpy` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_cross_entropy_loss(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLoss,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cross_entropy_loss
        todo!(
            "TODO: port action `exec_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_cross_entropy_loss_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLossBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cross_entropy_loss_back
        todo!(
            "TODO: port action `exec_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_cumsum(&mut self, _event: &EmelKernelX8664EventDispatchOpCumsum) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_cumsum
        todo!("TODO: port action `exec_op_cumsum` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_custom(&mut self, _event: &EmelKernelX8664EventDispatchOpCustom) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_custom
        todo!("TODO: port action `exec_op_custom` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_diag(&mut self, _event: &EmelKernelX8664EventDispatchOpDiag) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_diag
        todo!("TODO: port action `exec_op_diag` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_diag_mask_inf(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskInf,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_diag_mask_inf
        todo!(
            "TODO: port action `exec_op_diag_mask_inf` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_diag_mask_zero(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskZero,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_diag_mask_zero
        todo!(
            "TODO: port action `exec_op_diag_mask_zero` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_div(&mut self, _event: &EmelKernelX8664EventDispatchOpDiv) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_div
        todo!("TODO: port action `exec_op_div` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_dup(&mut self, _event: &EmelKernelX8664EventDispatchOpDup) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_dup
        todo!("TODO: port action `exec_op_dup` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_fill(&mut self, _event: &EmelKernelX8664EventDispatchOpFill) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_fill
        todo!("TODO: port action `exec_op_fill` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_flash_attn_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_flash_attn_back
        todo!(
            "TODO: port action `exec_op_flash_attn_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_flash_attn_ext(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_flash_attn_ext
        todo!(
            "TODO: port action `exec_op_flash_attn_ext` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_gated_linear_attn(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGatedLinearAttn,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_gated_linear_attn
        todo!(
            "TODO: port action `exec_op_gated_linear_attn` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_get_rel_pos(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_get_rel_pos
        todo!(
            "TODO: port action `exec_op_get_rel_pos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_get_rows_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRowsBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_get_rows_back
        todo!(
            "TODO: port action `exec_op_get_rows_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_glu(&mut self, _event: &EmelKernelX8664EventDispatchOpGlu) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_glu
        todo!("TODO: port action `exec_op_glu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_group_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGroupNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_group_norm
        todo!(
            "TODO: port action `exec_op_group_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_im2col_3d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2col3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_im2col_3d
        todo!(
            "TODO: port action `exec_op_im2col_3d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_im2col_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2colBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_im2col_back
        todo!(
            "TODO: port action `exec_op_im2col_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_l2_norm(&mut self, _event: &EmelKernelX8664EventDispatchOpL2Norm) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_l2_norm
        todo!(
            "TODO: port action `exec_op_l2_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_leaky_relu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpLeakyRelu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_leaky_relu
        todo!(
            "TODO: port action `exec_op_leaky_relu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_log(&mut self, _event: &EmelKernelX8664EventDispatchOpLog) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_log
        todo!("TODO: port action `exec_op_log` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_map_custom1(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_map_custom1
        todo!(
            "TODO: port action `exec_op_map_custom1` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_map_custom2(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom2,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_map_custom2
        todo!(
            "TODO: port action `exec_op_map_custom2` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_map_custom3(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom3,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_map_custom3
        todo!(
            "TODO: port action `exec_op_map_custom3` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_mean(&mut self, _event: &EmelKernelX8664EventDispatchOpMean) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_mean
        todo!("TODO: port action `exec_op_mean` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_mul(&mut self, _event: &EmelKernelX8664EventDispatchOpMul) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_mul
        todo!("TODO: port action `exec_op_mul` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_mul_mat(&mut self, _event: &EmelKernelX8664EventDispatchOpMulMat) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_mul_mat
        todo!(
            "TODO: port action `exec_op_mul_mat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_mul_mat_argmax(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_mul_mat_argmax
        todo!(
            "TODO: port action `exec_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_mul_mat_id(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMatId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_mul_mat_id
        todo!(
            "TODO: port action `exec_op_mul_mat_id` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_norm(&mut self, _event: &EmelKernelX8664EventDispatchOpNorm) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_norm
        todo!("TODO: port action `exec_op_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_opt_step_adamw(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOptStepAdamw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_opt_step_adamw
        todo!(
            "TODO: port action `exec_op_opt_step_adamw` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_opt_step_sgd(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOptStepSgd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_opt_step_sgd
        todo!(
            "TODO: port action `exec_op_opt_step_sgd` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_out_prod(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOutProd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_out_prod
        todo!(
            "TODO: port action `exec_op_out_prod` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_pad(&mut self, _event: &EmelKernelX8664EventDispatchOpPad) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_pad
        todo!("TODO: port action `exec_op_pad` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_pad_reflect_1d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPadReflect1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_pad_reflect_1d
        todo!(
            "TODO: port action `exec_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_permute(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPermute,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_permute
        todo!(
            "TODO: port action `exec_op_permute` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_pool_1d(&mut self, _event: &EmelKernelX8664EventDispatchOpPool1d) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_pool_1d
        todo!(
            "TODO: port action `exec_op_pool_1d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_pool_2d(&mut self, _event: &EmelKernelX8664EventDispatchOpPool2d) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_pool_2d
        todo!(
            "TODO: port action `exec_op_pool_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_pool_2d_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPool2dBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_pool_2d_back
        todo!(
            "TODO: port action `exec_op_pool_2d_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_repeat(&mut self, _event: &EmelKernelX8664EventDispatchOpRepeat) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_repeat
        todo!("TODO: port action `exec_op_repeat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_repeat_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRepeatBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_repeat_back
        todo!(
            "TODO: port action `exec_op_repeat_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_reshape(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpReshape,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_reshape
        todo!(
            "TODO: port action `exec_op_reshape` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_rms_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRmsNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_rms_norm
        todo!(
            "TODO: port action `exec_op_rms_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_rms_norm_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRmsNormBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_rms_norm_back
        todo!(
            "TODO: port action `exec_op_rms_norm_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_roll(&mut self, _event: &EmelKernelX8664EventDispatchOpRoll) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_roll
        todo!("TODO: port action `exec_op_roll` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_rope_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRopeBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_rope_back
        todo!(
            "TODO: port action `exec_op_rope_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_rwkv_wkv6(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv6,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_rwkv_wkv6
        todo!(
            "TODO: port action `exec_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_rwkv_wkv7(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv7,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_rwkv_wkv7
        todo!(
            "TODO: port action `exec_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_scale(&mut self, _event: &EmelKernelX8664EventDispatchOpScale) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_scale
        todo!("TODO: port action `exec_op_scale` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_set(&mut self, _event: &EmelKernelX8664EventDispatchOpSet) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_set
        todo!("TODO: port action `exec_op_set` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_set_rows(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_set_rows
        todo!(
            "TODO: port action `exec_op_set_rows` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_silu_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSiluBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_silu_back
        todo!(
            "TODO: port action `exec_op_silu_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_sin(&mut self, _event: &EmelKernelX8664EventDispatchOpSin) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sin
        todo!("TODO: port action `exec_op_sin` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_soft_max(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSoftMax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_soft_max
        todo!(
            "TODO: port action `exec_op_soft_max` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_soft_max_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSoftMaxBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_soft_max_back
        todo!(
            "TODO: port action `exec_op_soft_max_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_solve_tri(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSolveTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_solve_tri
        todo!(
            "TODO: port action `exec_op_solve_tri` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_sqr(&mut self, _event: &EmelKernelX8664EventDispatchOpSqr) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sqr
        todo!("TODO: port action `exec_op_sqr` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_sqrt(&mut self, _event: &EmelKernelX8664EventDispatchOpSqrt) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sqrt
        todo!("TODO: port action `exec_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_ssm_conv(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSsmConv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_ssm_conv
        todo!(
            "TODO: port action `exec_op_ssm_conv` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_ssm_scan(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSsmScan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_ssm_scan
        todo!(
            "TODO: port action `exec_op_ssm_scan` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_sub(&mut self, _event: &EmelKernelX8664EventDispatchOpSub) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sub
        todo!("TODO: port action `exec_op_sub` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_sum(&mut self, _event: &EmelKernelX8664EventDispatchOpSum) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sum
        todo!("TODO: port action `exec_op_sum` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_sum_rows(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSumRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_sum_rows
        todo!(
            "TODO: port action `exec_op_sum_rows` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_timestep_embedding(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTimestepEmbedding,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_timestep_embedding
        todo!(
            "TODO: port action `exec_op_timestep_embedding` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_top_k(&mut self, _event: &EmelKernelX8664EventDispatchOpTopK) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_top_k
        todo!("TODO: port action `exec_op_top_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_transpose(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTranspose,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_transpose
        todo!(
            "TODO: port action `exec_op_transpose` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_tri(&mut self, _event: &EmelKernelX8664EventDispatchOpTri) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_tri
        todo!("TODO: port action `exec_op_tri` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_upscale(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUpscale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_upscale
        todo!(
            "TODO: port action `exec_op_upscale` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_view(&mut self, _event: &EmelKernelX8664EventDispatchOpView) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_view
        todo!("TODO: port action `exec_op_view` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn exec_op_win_part(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpWinPart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_win_part
        todo!(
            "TODO: port action `exec_op_win_part` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_op_win_unpart(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpWinUnpart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_op_win_unpart
        todo!(
            "TODO: port action `exec_op_win_unpart` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_add_broadcast_row(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAdd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_add_broadcast_row
        todo!(
            "TODO: port action `exec_scalar_op_add_broadcast_row` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_conv_transpose_1d_f16(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_conv_transpose_1d_f16
        todo!(
            "TODO: port action `exec_scalar_op_conv_transpose_1d_f16` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_conv_transpose_1d_f32(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_conv_transpose_1d_f32
        todo!(
            "TODO: port action `exec_scalar_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_bf16(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_bf16
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_bf16` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_f16(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_f16
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_f16` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_f32(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_f32
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_f32` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q4_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_q4_0
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q4_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q4_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_q4_k
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q4_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_get_rows_q8_0(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_get_rows_q8_0
        todo!(
            "TODO: port action `exec_scalar_op_get_rows_q8_0` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_im2col_f16(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_im2col_f16
        todo!(
            "TODO: port action `exec_scalar_op_im2col_f16` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_im2col_f32(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_im2col_f32
        todo!(
            "TODO: port action `exec_scalar_op_im2col_f32` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_mul_broadcast_row(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMul,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_mul_broadcast_row
        todo!(
            "TODO: port action `exec_scalar_op_mul_broadcast_row` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_mul_mat_f16(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_mul_mat_f16
        todo!(
            "TODO: port action `exec_scalar_op_mul_mat_f16` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_neox(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_rope_neox
        todo!(
            "TODO: port action `exec_scalar_op_rope_neox` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_rope_norm
        todo!(
            "TODO: port action `exec_scalar_op_rope_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_rope_timestep(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_rope_timestep
        todo!(
            "TODO: port action `exec_scalar_op_rope_timestep` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_abs(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_abs
        todo!(
            "TODO: port action `exec_scalar_op_unary_abs` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_elu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_elu
        todo!(
            "TODO: port action `exec_scalar_op_unary_elu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_exp(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_exp
        todo!(
            "TODO: port action `exec_scalar_op_unary_exp` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_gelu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_gelu
        todo!(
            "TODO: port action `exec_scalar_op_unary_gelu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_neg(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_neg
        todo!(
            "TODO: port action `exec_scalar_op_unary_neg` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_relu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_relu
        todo!(
            "TODO: port action `exec_scalar_op_unary_relu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_silu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_silu
        todo!(
            "TODO: port action `exec_scalar_op_unary_silu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_scalar_op_unary_tanh(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_scalar_op_unary_tanh
        todo!(
            "TODO: port action `exec_scalar_op_unary_tanh` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_add(&mut self, _event: &EmelKernelX8664EventDispatchOpAdd) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_add
        todo!(
            "TODO: port action `exec_simd_op_add` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_div(&mut self, _event: &EmelKernelX8664EventDispatchOpDiv) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_div
        todo!(
            "TODO: port action `exec_simd_op_div` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_dup(&mut self, _event: &EmelKernelX8664EventDispatchOpDup) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_dup
        todo!(
            "TODO: port action `exec_simd_op_dup` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_flash_attn_ext_f16kv_one_chunk(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_flash_attn_ext_f16kv_one_chunk
        todo!(
            "TODO: port action `exec_simd_op_flash_attn_ext_f16kv_one_chunk` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_mul(&mut self, _event: &EmelKernelX8664EventDispatchOpMul) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_mul
        todo!(
            "TODO: port action `exec_simd_op_mul` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_mul_mat(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_mul_mat
        todo!(
            "TODO: port action `exec_simd_op_mul_mat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_sqr(&mut self, _event: &EmelKernelX8664EventDispatchOpSqr) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_sqr
        todo!(
            "TODO: port action `exec_simd_op_sqr` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_sqrt(&mut self, _event: &EmelKernelX8664EventDispatchOpSqrt) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_sqrt
        todo!(
            "TODO: port action `exec_simd_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_sub(&mut self, _event: &EmelKernelX8664EventDispatchOpSub) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_sub
        todo!(
            "TODO: port action `exec_simd_op_sub` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_abs(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_unary_abs
        todo!(
            "TODO: port action `exec_simd_op_unary_abs` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_neg(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_unary_neg
        todo!(
            "TODO: port action `exec_simd_op_unary_neg` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn exec_simd_op_unary_relu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::exec_simd_op_unary_relu
        todo!(
            "TODO: port action `exec_simd_op_unary_relu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn guard_simd_op_mul_mat_f32_avx2_only(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_f32_avx2_only
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_f32_avx2_only` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_f32_fma(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_f32_fma
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_f32_fma` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_f32_fma_vector(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_f32_fma_vector
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_f32_fma_vector` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q2_k_q8_k(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q2_k_q8_k
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q2_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q3_k_q8_k(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q3_k_q8_k
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q3_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q4_0_q8_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q4_0_q8_0
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q4_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q4_1_q8_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q4_1_q8_0
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q4_1_q8_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q4_k_q8_k(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q4_k_q8_k
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q4_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q5_0_q8_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q5_0_q8_0
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q5_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q6_k_q8_k(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q6_k_q8_k
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q6_k_q8_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn guard_simd_op_mul_mat_q8_0_q8_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::guard_simd_op_mul_mat_q8_0_q8_0
        todo!(
            "TODO: port guard `guard_simd_op_mul_mat_q8_0_q8_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_acc(&self, _event: &EmelKernelX8664EventDispatchOpAcc) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_acc
        todo!("TODO: port guard `invalid_op_acc` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_add(&self, _event: &EmelKernelX8664EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_add
        todo!("TODO: port guard `invalid_op_add` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_add1(&self, _event: &EmelKernelX8664EventDispatchOpAdd1) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_add1
        todo!("TODO: port guard `invalid_op_add1` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_add_id(&self, _event: &EmelKernelX8664EventDispatchOpAddId) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_add_id
        todo!(
            "TODO: port guard `invalid_op_add_id` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_add_rel_pos(
        &self,
        _event: &EmelKernelX8664EventDispatchOpAddRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_add_rel_pos
        todo!(
            "TODO: port guard `invalid_op_add_rel_pos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_arange(&self, _event: &EmelKernelX8664EventDispatchOpArange) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_arange
        todo!(
            "TODO: port guard `invalid_op_arange` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_argmax(&self, _event: &EmelKernelX8664EventDispatchOpArgmax) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_argmax
        todo!(
            "TODO: port guard `invalid_op_argmax` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_argsort(
        &self,
        _event: &EmelKernelX8664EventDispatchOpArgsort,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_argsort
        todo!(
            "TODO: port guard `invalid_op_argsort` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_clamp(&self, _event: &EmelKernelX8664EventDispatchOpClamp) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_clamp
        todo!("TODO: port guard `invalid_op_clamp` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_concat(&self, _event: &EmelKernelX8664EventDispatchOpConcat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_concat
        todo!(
            "TODO: port guard `invalid_op_concat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_cont(&self, _event: &EmelKernelX8664EventDispatchOpCont) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cont
        todo!("TODO: port guard `invalid_op_cont` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_conv_2d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConv2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_conv_2d
        todo!(
            "TODO: port guard `invalid_op_conv_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_conv_2d_dw(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConv2dDw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_conv_2d_dw
        todo!(
            "TODO: port guard `invalid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_conv_3d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConv3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_conv_3d
        todo!(
            "TODO: port guard `invalid_op_conv_3d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_conv_transpose_1d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_conv_transpose_1d
        todo!(
            "TODO: port guard `invalid_op_conv_transpose_1d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_conv_transpose_2d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_conv_transpose_2d
        todo!(
            "TODO: port guard `invalid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_cos(&self, _event: &EmelKernelX8664EventDispatchOpCos) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cos
        todo!("TODO: port guard `invalid_op_cos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_count_equal(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCountEqual,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_count_equal
        todo!(
            "TODO: port guard `invalid_op_count_equal` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_cpy(&self, _event: &EmelKernelX8664EventDispatchOpCpy) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cpy
        todo!("TODO: port guard `invalid_op_cpy` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_cross_entropy_loss(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLoss,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cross_entropy_loss
        todo!(
            "TODO: port guard `invalid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_cross_entropy_loss_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLossBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cross_entropy_loss_back
        todo!(
            "TODO: port guard `invalid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_cumsum(&self, _event: &EmelKernelX8664EventDispatchOpCumsum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_cumsum
        todo!(
            "TODO: port guard `invalid_op_cumsum` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_custom(&self, _event: &EmelKernelX8664EventDispatchOpCustom) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_custom
        todo!(
            "TODO: port guard `invalid_op_custom` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_diag(&self, _event: &EmelKernelX8664EventDispatchOpDiag) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_diag
        todo!("TODO: port guard `invalid_op_diag` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_diag_mask_inf(
        &self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskInf,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_diag_mask_inf
        todo!(
            "TODO: port guard `invalid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_diag_mask_zero(
        &self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskZero,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_diag_mask_zero
        todo!(
            "TODO: port guard `invalid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_div(&self, _event: &EmelKernelX8664EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_div
        todo!("TODO: port guard `invalid_op_div` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_dup(&self, _event: &EmelKernelX8664EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_dup
        todo!("TODO: port guard `invalid_op_dup` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_fill(&self, _event: &EmelKernelX8664EventDispatchOpFill) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_fill
        todo!("TODO: port guard `invalid_op_fill` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_flash_attn_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_flash_attn_back
        todo!(
            "TODO: port guard `invalid_op_flash_attn_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_flash_attn_ext(
        &self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_flash_attn_ext
        todo!(
            "TODO: port guard `invalid_op_flash_attn_ext` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_gated_linear_attn(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGatedLinearAttn,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_gated_linear_attn
        todo!(
            "TODO: port guard `invalid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_get_rel_pos(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_get_rel_pos
        todo!(
            "TODO: port guard `invalid_op_get_rel_pos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_get_rows(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_get_rows
        todo!(
            "TODO: port guard `invalid_op_get_rows` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_get_rows_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRowsBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_get_rows_back
        todo!(
            "TODO: port guard `invalid_op_get_rows_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_glu(&self, _event: &EmelKernelX8664EventDispatchOpGlu) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_glu
        todo!("TODO: port guard `invalid_op_glu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_group_norm(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGroupNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_group_norm
        todo!(
            "TODO: port guard `invalid_op_group_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_im2col(&self, _event: &EmelKernelX8664EventDispatchOpIm2col) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_im2col
        todo!(
            "TODO: port guard `invalid_op_im2col` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_im2col_3d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2col3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_im2col_3d
        todo!(
            "TODO: port guard `invalid_op_im2col_3d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_im2col_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2colBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_im2col_back
        todo!(
            "TODO: port guard `invalid_op_im2col_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_l2_norm(
        &self,
        _event: &EmelKernelX8664EventDispatchOpL2Norm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_l2_norm
        todo!(
            "TODO: port guard `invalid_op_l2_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_leaky_relu(
        &self,
        _event: &EmelKernelX8664EventDispatchOpLeakyRelu,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_leaky_relu
        todo!(
            "TODO: port guard `invalid_op_leaky_relu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_log(&self, _event: &EmelKernelX8664EventDispatchOpLog) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_log
        todo!("TODO: port guard `invalid_op_log` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_map_custom1(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom1,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_map_custom1
        todo!(
            "TODO: port guard `invalid_op_map_custom1` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_map_custom2(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom2,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_map_custom2
        todo!(
            "TODO: port guard `invalid_op_map_custom2` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_map_custom3(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom3,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_map_custom3
        todo!(
            "TODO: port guard `invalid_op_map_custom3` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_mean(&self, _event: &EmelKernelX8664EventDispatchOpMean) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_mean
        todo!("TODO: port guard `invalid_op_mean` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_mul(&self, _event: &EmelKernelX8664EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_mul
        todo!("TODO: port guard `invalid_op_mul` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_mul_mat(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_mul_mat
        todo!(
            "TODO: port guard `invalid_op_mul_mat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_mul_mat_argmax(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_mul_mat_argmax
        todo!(
            "TODO: port guard `invalid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_mul_mat_id(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMatId,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_mul_mat_id
        todo!(
            "TODO: port guard `invalid_op_mul_mat_id` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_norm(&self, _event: &EmelKernelX8664EventDispatchOpNorm) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_norm
        todo!("TODO: port guard `invalid_op_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_opt_step_adamw(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOptStepAdamw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_opt_step_adamw
        todo!(
            "TODO: port guard `invalid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_opt_step_sgd(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOptStepSgd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_opt_step_sgd
        todo!(
            "TODO: port guard `invalid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_out_prod(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOutProd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_out_prod
        todo!(
            "TODO: port guard `invalid_op_out_prod` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_pad(&self, _event: &EmelKernelX8664EventDispatchOpPad) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_pad
        todo!("TODO: port guard `invalid_op_pad` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_pad_reflect_1d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPadReflect1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_pad_reflect_1d
        todo!(
            "TODO: port guard `invalid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_permute(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPermute,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_permute
        todo!(
            "TODO: port guard `invalid_op_permute` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_pool_1d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPool1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_pool_1d
        todo!(
            "TODO: port guard `invalid_op_pool_1d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_pool_2d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPool2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_pool_2d
        todo!(
            "TODO: port guard `invalid_op_pool_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_pool_2d_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPool2dBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_pool_2d_back
        todo!(
            "TODO: port guard `invalid_op_pool_2d_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_repeat(&self, _event: &EmelKernelX8664EventDispatchOpRepeat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_repeat
        todo!(
            "TODO: port guard `invalid_op_repeat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_repeat_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRepeatBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_repeat_back
        todo!(
            "TODO: port guard `invalid_op_repeat_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_reshape(
        &self,
        _event: &EmelKernelX8664EventDispatchOpReshape,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_reshape
        todo!(
            "TODO: port guard `invalid_op_reshape` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_rms_norm(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRmsNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rms_norm
        todo!(
            "TODO: port guard `invalid_op_rms_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_rms_norm_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRmsNormBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rms_norm_back
        todo!(
            "TODO: port guard `invalid_op_rms_norm_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_roll(&self, _event: &EmelKernelX8664EventDispatchOpRoll) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_roll
        todo!("TODO: port guard `invalid_op_roll` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_rope(&self, _event: &EmelKernelX8664EventDispatchOpRope) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rope
        todo!("TODO: port guard `invalid_op_rope` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_rope_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRopeBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rope_back
        todo!(
            "TODO: port guard `invalid_op_rope_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_rwkv_wkv6(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv6,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rwkv_wkv6
        todo!(
            "TODO: port guard `invalid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_rwkv_wkv7(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv7,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_rwkv_wkv7
        todo!(
            "TODO: port guard `invalid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_scale(&self, _event: &EmelKernelX8664EventDispatchOpScale) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_scale
        todo!("TODO: port guard `invalid_op_scale` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_set(&self, _event: &EmelKernelX8664EventDispatchOpSet) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_set
        todo!("TODO: port guard `invalid_op_set` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_set_rows(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_set_rows
        todo!(
            "TODO: port guard `invalid_op_set_rows` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_silu_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSiluBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_silu_back
        todo!(
            "TODO: port guard `invalid_op_silu_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_sin(&self, _event: &EmelKernelX8664EventDispatchOpSin) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sin
        todo!("TODO: port guard `invalid_op_sin` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_soft_max(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSoftMax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_soft_max
        todo!(
            "TODO: port guard `invalid_op_soft_max` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_soft_max_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSoftMaxBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_soft_max_back
        todo!(
            "TODO: port guard `invalid_op_soft_max_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_solve_tri(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSolveTri,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_solve_tri
        todo!(
            "TODO: port guard `invalid_op_solve_tri` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_sqr(&self, _event: &EmelKernelX8664EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sqr
        todo!("TODO: port guard `invalid_op_sqr` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_sqrt(&self, _event: &EmelKernelX8664EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sqrt
        todo!("TODO: port guard `invalid_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_ssm_conv(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSsmConv,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_ssm_conv
        todo!(
            "TODO: port guard `invalid_op_ssm_conv` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_ssm_scan(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSsmScan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_ssm_scan
        todo!(
            "TODO: port guard `invalid_op_ssm_scan` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_sub(&self, _event: &EmelKernelX8664EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sub
        todo!("TODO: port guard `invalid_op_sub` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_sum(&self, _event: &EmelKernelX8664EventDispatchOpSum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sum
        todo!("TODO: port guard `invalid_op_sum` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_sum_rows(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSumRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_sum_rows
        todo!(
            "TODO: port guard `invalid_op_sum_rows` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_timestep_embedding(
        &self,
        _event: &EmelKernelX8664EventDispatchOpTimestepEmbedding,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_timestep_embedding
        todo!(
            "TODO: port guard `invalid_op_timestep_embedding` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_top_k(&self, _event: &EmelKernelX8664EventDispatchOpTopK) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_top_k
        todo!("TODO: port guard `invalid_op_top_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_transpose(
        &self,
        _event: &EmelKernelX8664EventDispatchOpTranspose,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_transpose
        todo!(
            "TODO: port guard `invalid_op_transpose` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_tri(&self, _event: &EmelKernelX8664EventDispatchOpTri) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_tri
        todo!("TODO: port guard `invalid_op_tri` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_unary(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_unary
        todo!("TODO: port guard `invalid_op_unary` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_upscale(
        &self,
        _event: &EmelKernelX8664EventDispatchOpUpscale,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_upscale
        todo!(
            "TODO: port guard `invalid_op_upscale` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_view(&self, _event: &EmelKernelX8664EventDispatchOpView) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_view
        todo!("TODO: port guard `invalid_op_view` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn invalid_op_win_part(
        &self,
        _event: &EmelKernelX8664EventDispatchOpWinPart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_win_part
        todo!(
            "TODO: port guard `invalid_op_win_part` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn invalid_op_win_unpart(
        &self,
        _event: &EmelKernelX8664EventDispatchOpWinUnpart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::invalid_op_win_unpart
        todo!(
            "TODO: port guard `invalid_op_win_unpart` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn on_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/kernel/x86_64/actions.hpp")
    }
    fn reject_invalid_op_acc(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAcc,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_acc
        todo!(
            "TODO: port action `reject_invalid_op_acc` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_add(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAdd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_add
        todo!(
            "TODO: port action `reject_invalid_op_add` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_add1(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAdd1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_add1
        todo!(
            "TODO: port action `reject_invalid_op_add1` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_add_id(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAddId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_add_id
        todo!(
            "TODO: port action `reject_invalid_op_add_id` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_add_rel_pos(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpAddRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_add_rel_pos
        todo!(
            "TODO: port action `reject_invalid_op_add_rel_pos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_arange(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpArange,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_arange
        todo!(
            "TODO: port action `reject_invalid_op_arange` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_argmax(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_argmax
        todo!(
            "TODO: port action `reject_invalid_op_argmax` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_argsort(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpArgsort,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_argsort
        todo!(
            "TODO: port action `reject_invalid_op_argsort` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_clamp(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpClamp,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_clamp
        todo!(
            "TODO: port action `reject_invalid_op_clamp` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_concat(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConcat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_concat
        todo!(
            "TODO: port action `reject_invalid_op_concat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cont(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCont,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cont
        todo!(
            "TODO: port action `reject_invalid_op_cont` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_2d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConv2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_conv_2d
        todo!(
            "TODO: port action `reject_invalid_op_conv_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_2d_dw(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConv2dDw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_conv_2d_dw
        todo!(
            "TODO: port action `reject_invalid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_3d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConv3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_conv_3d
        todo!(
            "TODO: port action `reject_invalid_op_conv_3d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_transpose_1d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_conv_transpose_1d
        todo!(
            "TODO: port action `reject_invalid_op_conv_transpose_1d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_conv_transpose_2d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_conv_transpose_2d
        todo!(
            "TODO: port action `reject_invalid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cos(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cos
        todo!(
            "TODO: port action `reject_invalid_op_cos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_count_equal(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCountEqual,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_count_equal
        todo!(
            "TODO: port action `reject_invalid_op_count_equal` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cpy(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCpy,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cpy
        todo!(
            "TODO: port action `reject_invalid_op_cpy` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cross_entropy_loss(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLoss,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cross_entropy_loss
        todo!(
            "TODO: port action `reject_invalid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cross_entropy_loss_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLossBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cross_entropy_loss_back
        todo!(
            "TODO: port action `reject_invalid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_cumsum(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCumsum,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_cumsum
        todo!(
            "TODO: port action `reject_invalid_op_cumsum` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_custom(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpCustom,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_custom
        todo!(
            "TODO: port action `reject_invalid_op_custom` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiag,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_diag
        todo!(
            "TODO: port action `reject_invalid_op_diag` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag_mask_inf(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskInf,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_diag_mask_inf
        todo!(
            "TODO: port action `reject_invalid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_diag_mask_zero(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskZero,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_diag_mask_zero
        todo!(
            "TODO: port action `reject_invalid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_div(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDiv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_div
        todo!(
            "TODO: port action `reject_invalid_op_div` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_dup(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpDup,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_dup
        todo!(
            "TODO: port action `reject_invalid_op_dup` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_fill(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFill,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_fill
        todo!(
            "TODO: port action `reject_invalid_op_fill` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_flash_attn_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_flash_attn_back
        todo!(
            "TODO: port action `reject_invalid_op_flash_attn_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_flash_attn_ext(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_flash_attn_ext
        todo!(
            "TODO: port action `reject_invalid_op_flash_attn_ext` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_gated_linear_attn(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGatedLinearAttn,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_gated_linear_attn
        todo!(
            "TODO: port action `reject_invalid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rel_pos(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRelPos,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_get_rel_pos
        todo!(
            "TODO: port action `reject_invalid_op_get_rel_pos` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rows(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_get_rows
        todo!(
            "TODO: port action `reject_invalid_op_get_rows` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_get_rows_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGetRowsBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_get_rows_back
        todo!(
            "TODO: port action `reject_invalid_op_get_rows_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_glu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGlu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_glu
        todo!(
            "TODO: port action `reject_invalid_op_glu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_group_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpGroupNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_group_norm
        todo!(
            "TODO: port action `reject_invalid_op_group_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2col,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_im2col
        todo!(
            "TODO: port action `reject_invalid_op_im2col` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col_3d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2col3d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_im2col_3d
        todo!(
            "TODO: port action `reject_invalid_op_im2col_3d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_im2col_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpIm2colBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_im2col_back
        todo!(
            "TODO: port action `reject_invalid_op_im2col_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_l2_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpL2Norm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_l2_norm
        todo!(
            "TODO: port action `reject_invalid_op_l2_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_leaky_relu(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpLeakyRelu,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_leaky_relu
        todo!(
            "TODO: port action `reject_invalid_op_leaky_relu` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_log(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpLog,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_log
        todo!(
            "TODO: port action `reject_invalid_op_log` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom1(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom1,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_map_custom1
        todo!(
            "TODO: port action `reject_invalid_op_map_custom1` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom2(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom2,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_map_custom2
        todo!(
            "TODO: port action `reject_invalid_op_map_custom2` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_map_custom3(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom3,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_map_custom3
        todo!(
            "TODO: port action `reject_invalid_op_map_custom3` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_mean(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMean,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_mean
        todo!(
            "TODO: port action `reject_invalid_op_mean` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMul,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_mul
        todo!(
            "TODO: port action `reject_invalid_op_mul` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_mul_mat
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat_argmax(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMatArgmax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_mul_mat_argmax
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_mul_mat_id(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpMulMatId,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_mul_mat_id
        todo!(
            "TODO: port action `reject_invalid_op_mul_mat_id` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_norm
        todo!(
            "TODO: port action `reject_invalid_op_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_opt_step_adamw(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOptStepAdamw,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_opt_step_adamw
        todo!(
            "TODO: port action `reject_invalid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_opt_step_sgd(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOptStepSgd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_opt_step_sgd
        todo!(
            "TODO: port action `reject_invalid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_out_prod(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpOutProd,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_out_prod
        todo!(
            "TODO: port action `reject_invalid_op_out_prod` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_pad(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPad,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_pad
        todo!(
            "TODO: port action `reject_invalid_op_pad` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_pad_reflect_1d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPadReflect1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_pad_reflect_1d
        todo!(
            "TODO: port action `reject_invalid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_permute(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPermute,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_permute
        todo!(
            "TODO: port action `reject_invalid_op_permute` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_1d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPool1d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_pool_1d
        todo!(
            "TODO: port action `reject_invalid_op_pool_1d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_2d(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPool2d,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_pool_2d
        todo!(
            "TODO: port action `reject_invalid_op_pool_2d` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_pool_2d_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpPool2dBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_pool_2d_back
        todo!(
            "TODO: port action `reject_invalid_op_pool_2d_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_repeat(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRepeat,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_repeat
        todo!(
            "TODO: port action `reject_invalid_op_repeat` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_repeat_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRepeatBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_repeat_back
        todo!(
            "TODO: port action `reject_invalid_op_repeat_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_reshape(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpReshape,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_reshape
        todo!(
            "TODO: port action `reject_invalid_op_reshape` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rms_norm(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRmsNorm,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rms_norm
        todo!(
            "TODO: port action `reject_invalid_op_rms_norm` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rms_norm_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRmsNormBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rms_norm_back
        todo!(
            "TODO: port action `reject_invalid_op_rms_norm_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_roll(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRoll,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_roll
        todo!(
            "TODO: port action `reject_invalid_op_roll` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rope(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRope,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rope
        todo!(
            "TODO: port action `reject_invalid_op_rope` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rope_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRopeBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rope_back
        todo!(
            "TODO: port action `reject_invalid_op_rope_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rwkv_wkv6(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv6,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rwkv_wkv6
        todo!(
            "TODO: port action `reject_invalid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_rwkv_wkv7(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv7,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_rwkv_wkv7
        todo!(
            "TODO: port action `reject_invalid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_scale(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpScale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_scale
        todo!(
            "TODO: port action `reject_invalid_op_scale` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_set(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSet,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_set
        todo!(
            "TODO: port action `reject_invalid_op_set` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_set_rows(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSetRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_set_rows
        todo!(
            "TODO: port action `reject_invalid_op_set_rows` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_silu_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSiluBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_silu_back
        todo!(
            "TODO: port action `reject_invalid_op_silu_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sin(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSin,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sin
        todo!(
            "TODO: port action `reject_invalid_op_sin` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_soft_max(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSoftMax,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_soft_max
        todo!(
            "TODO: port action `reject_invalid_op_soft_max` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_soft_max_back(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSoftMaxBack,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_soft_max_back
        todo!(
            "TODO: port action `reject_invalid_op_soft_max_back` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_solve_tri(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSolveTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_solve_tri
        todo!(
            "TODO: port action `reject_invalid_op_solve_tri` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sqr(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSqr,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sqr
        todo!(
            "TODO: port action `reject_invalid_op_sqr` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sqrt(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSqrt,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sqrt
        todo!(
            "TODO: port action `reject_invalid_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_ssm_conv(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSsmConv,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_ssm_conv
        todo!(
            "TODO: port action `reject_invalid_op_ssm_conv` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_ssm_scan(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSsmScan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_ssm_scan
        todo!(
            "TODO: port action `reject_invalid_op_ssm_scan` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sub(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSub,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sub
        todo!(
            "TODO: port action `reject_invalid_op_sub` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sum(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSum,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sum
        todo!(
            "TODO: port action `reject_invalid_op_sum` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_sum_rows(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpSumRows,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_sum_rows
        todo!(
            "TODO: port action `reject_invalid_op_sum_rows` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_timestep_embedding(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTimestepEmbedding,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_timestep_embedding
        todo!(
            "TODO: port action `reject_invalid_op_timestep_embedding` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_top_k(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTopK,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_top_k
        todo!(
            "TODO: port action `reject_invalid_op_top_k` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_transpose(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTranspose,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_transpose
        todo!(
            "TODO: port action `reject_invalid_op_transpose` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_tri(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpTri,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_tri
        todo!(
            "TODO: port action `reject_invalid_op_tri` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_unary(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_unary
        todo!(
            "TODO: port action `reject_invalid_op_unary` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_upscale(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpUpscale,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_upscale
        todo!(
            "TODO: port action `reject_invalid_op_upscale` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_view(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_view
        todo!(
            "TODO: port action `reject_invalid_op_view` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_win_part(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpWinPart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_win_part
        todo!(
            "TODO: port action `reject_invalid_op_win_part` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn reject_invalid_op_win_unpart(
        &mut self,
        _event: &EmelKernelX8664EventDispatchOpWinUnpart,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/actions.hpp::reject_invalid_op_win_unpart
        todo!(
            "TODO: port action `reject_invalid_op_win_unpart` from emel.cpp/src/emel/kernel/x86_64/actions.hpp"
        )
    }
    fn simd_op_add(&self, _event: &EmelKernelX8664EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_add
        todo!("TODO: port guard `simd_op_add` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_div(&self, _event: &EmelKernelX8664EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_div
        todo!("TODO: port guard `simd_op_div` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_dup(&self, _event: &EmelKernelX8664EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_dup
        todo!("TODO: port guard `simd_op_dup` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_flash_attn_ext_f16kv_one_chunk(
        &self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_flash_attn_ext_f16kv_one_chunk
        todo!(
            "TODO: port guard `simd_op_flash_attn_ext_f16kv_one_chunk` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn simd_op_mul(&self, _event: &EmelKernelX8664EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_mul
        todo!("TODO: port guard `simd_op_mul` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_sqr(&self, _event: &EmelKernelX8664EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_sqr
        todo!("TODO: port guard `simd_op_sqr` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_sqrt(&self, _event: &EmelKernelX8664EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_sqrt
        todo!("TODO: port guard `simd_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_sub(&self, _event: &EmelKernelX8664EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_sub
        todo!("TODO: port guard `simd_op_sub` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn simd_op_unary_abs(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_unary_abs
        todo!(
            "TODO: port guard `simd_op_unary_abs` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn simd_op_unary_neg(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_unary_neg
        todo!(
            "TODO: port guard `simd_op_unary_neg` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn simd_op_unary_relu(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::simd_op_unary_relu
        todo!(
            "TODO: port guard `simd_op_unary_relu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_acc(&self, _event: &EmelKernelX8664EventDispatchOpAcc) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_acc
        todo!("TODO: port guard `valid_op_acc` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_add1(&self, _event: &EmelKernelX8664EventDispatchOpAdd1) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_add1
        todo!("TODO: port guard `valid_op_add1` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_add_broadcast_row(
        &self,
        _event: &EmelKernelX8664EventDispatchOpAdd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_add_broadcast_row
        todo!(
            "TODO: port guard `valid_op_add_broadcast_row` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_add_equal(&self, _event: &EmelKernelX8664EventDispatchOpAdd) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_add_equal
        todo!(
            "TODO: port guard `valid_op_add_equal` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_add_id(&self, _event: &EmelKernelX8664EventDispatchOpAddId) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_add_id
        todo!("TODO: port guard `valid_op_add_id` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_add_rel_pos(
        &self,
        _event: &EmelKernelX8664EventDispatchOpAddRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_add_rel_pos
        todo!(
            "TODO: port guard `valid_op_add_rel_pos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_arange(&self, _event: &EmelKernelX8664EventDispatchOpArange) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_arange
        todo!("TODO: port guard `valid_op_arange` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_argmax(&self, _event: &EmelKernelX8664EventDispatchOpArgmax) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_argmax
        todo!("TODO: port guard `valid_op_argmax` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_argsort(&self, _event: &EmelKernelX8664EventDispatchOpArgsort) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_argsort
        todo!("TODO: port guard `valid_op_argsort` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_clamp(&self, _event: &EmelKernelX8664EventDispatchOpClamp) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_clamp
        todo!("TODO: port guard `valid_op_clamp` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_concat(&self, _event: &EmelKernelX8664EventDispatchOpConcat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_concat
        todo!("TODO: port guard `valid_op_concat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_cont(&self, _event: &EmelKernelX8664EventDispatchOpCont) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cont
        todo!("TODO: port guard `valid_op_cont` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_conv_2d(&self, _event: &EmelKernelX8664EventDispatchOpConv2d) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_2d
        todo!("TODO: port guard `valid_op_conv_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_conv_2d_dw(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConv2dDw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_2d_dw
        todo!(
            "TODO: port guard `valid_op_conv_2d_dw` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_conv_3d(&self, _event: &EmelKernelX8664EventDispatchOpConv3d) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_3d
        todo!("TODO: port guard `valid_op_conv_3d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_conv_transpose_1d_f16(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_transpose_1d_f16
        todo!(
            "TODO: port guard `valid_op_conv_transpose_1d_f16` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_conv_transpose_1d_f32(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_transpose_1d_f32
        todo!(
            "TODO: port guard `valid_op_conv_transpose_1d_f32` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_conv_transpose_2d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpConvTranspose2d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_conv_transpose_2d
        todo!(
            "TODO: port guard `valid_op_conv_transpose_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_cos(&self, _event: &EmelKernelX8664EventDispatchOpCos) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cos
        todo!("TODO: port guard `valid_op_cos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_count_equal(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCountEqual,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_count_equal
        todo!(
            "TODO: port guard `valid_op_count_equal` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_cpy(&self, _event: &EmelKernelX8664EventDispatchOpCpy) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cpy
        todo!("TODO: port guard `valid_op_cpy` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_cross_entropy_loss(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLoss,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cross_entropy_loss
        todo!(
            "TODO: port guard `valid_op_cross_entropy_loss` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_cross_entropy_loss_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpCrossEntropyLossBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cross_entropy_loss_back
        todo!(
            "TODO: port guard `valid_op_cross_entropy_loss_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_cumsum(&self, _event: &EmelKernelX8664EventDispatchOpCumsum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_cumsum
        todo!("TODO: port guard `valid_op_cumsum` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_custom(&self, _event: &EmelKernelX8664EventDispatchOpCustom) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_custom
        todo!("TODO: port guard `valid_op_custom` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_diag(&self, _event: &EmelKernelX8664EventDispatchOpDiag) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_diag
        todo!("TODO: port guard `valid_op_diag` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_diag_mask_inf(
        &self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskInf,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_diag_mask_inf
        todo!(
            "TODO: port guard `valid_op_diag_mask_inf` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_diag_mask_zero(
        &self,
        _event: &EmelKernelX8664EventDispatchOpDiagMaskZero,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_diag_mask_zero
        todo!(
            "TODO: port guard `valid_op_diag_mask_zero` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_div(&self, _event: &EmelKernelX8664EventDispatchOpDiv) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_div
        todo!("TODO: port guard `valid_op_div` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_dup(&self, _event: &EmelKernelX8664EventDispatchOpDup) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_dup
        todo!("TODO: port guard `valid_op_dup` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_fill(&self, _event: &EmelKernelX8664EventDispatchOpFill) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_fill
        todo!("TODO: port guard `valid_op_fill` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_flash_attn_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_flash_attn_back
        todo!(
            "TODO: port guard `valid_op_flash_attn_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_flash_attn_ext_shared(
        &self,
        _event: &EmelKernelX8664EventDispatchOpFlashAttnExt,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_flash_attn_ext_shared
        todo!(
            "TODO: port guard `valid_op_flash_attn_ext_shared` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_gated_linear_attn(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGatedLinearAttn,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_gated_linear_attn
        todo!(
            "TODO: port guard `valid_op_gated_linear_attn` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rel_pos(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRelPos,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rel_pos
        todo!(
            "TODO: port guard `valid_op_get_rel_pos` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRowsBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_back
        todo!(
            "TODO: port guard `valid_op_get_rows_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_bf16(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_bf16
        todo!(
            "TODO: port guard `valid_op_get_rows_bf16` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_f16(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_f16
        todo!(
            "TODO: port guard `valid_op_get_rows_f16` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_f32(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_f32
        todo!(
            "TODO: port guard `valid_op_get_rows_f32` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q4_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_q4_0
        todo!(
            "TODO: port guard `valid_op_get_rows_q4_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q4_k(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_q4_k
        todo!(
            "TODO: port guard `valid_op_get_rows_q4_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_get_rows_q8_0(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_get_rows_q8_0
        todo!(
            "TODO: port guard `valid_op_get_rows_q8_0` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_glu(&self, _event: &EmelKernelX8664EventDispatchOpGlu) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_glu
        todo!("TODO: port guard `valid_op_glu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_group_norm(
        &self,
        _event: &EmelKernelX8664EventDispatchOpGroupNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_group_norm
        todo!(
            "TODO: port guard `valid_op_group_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_im2col_3d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2col3d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_im2col_3d
        todo!(
            "TODO: port guard `valid_op_im2col_3d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_im2col_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2colBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_im2col_back
        todo!(
            "TODO: port guard `valid_op_im2col_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_im2col_f16(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2col,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_im2col_f16
        todo!(
            "TODO: port guard `valid_op_im2col_f16` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_im2col_f32(
        &self,
        _event: &EmelKernelX8664EventDispatchOpIm2col,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_im2col_f32
        todo!(
            "TODO: port guard `valid_op_im2col_f32` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_l2_norm(&self, _event: &EmelKernelX8664EventDispatchOpL2Norm) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_l2_norm
        todo!("TODO: port guard `valid_op_l2_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_leaky_relu(
        &self,
        _event: &EmelKernelX8664EventDispatchOpLeakyRelu,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_leaky_relu
        todo!(
            "TODO: port guard `valid_op_leaky_relu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_log(&self, _event: &EmelKernelX8664EventDispatchOpLog) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_log
        todo!("TODO: port guard `valid_op_log` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_map_custom1(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom1,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_map_custom1
        todo!(
            "TODO: port guard `valid_op_map_custom1` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_map_custom2(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom2,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_map_custom2
        todo!(
            "TODO: port guard `valid_op_map_custom2` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_map_custom3(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMapCustom3,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_map_custom3
        todo!(
            "TODO: port guard `valid_op_map_custom3` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_mean(&self, _event: &EmelKernelX8664EventDispatchOpMean) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mean
        todo!("TODO: port guard `valid_op_mean` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_mul_broadcast_row(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMul,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_broadcast_row
        todo!(
            "TODO: port guard `valid_op_mul_broadcast_row` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_mul_equal(&self, _event: &EmelKernelX8664EventDispatchOpMul) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_equal
        todo!(
            "TODO: port guard `valid_op_mul_equal` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_mul_mat(&self, _event: &EmelKernelX8664EventDispatchOpMulMat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_mat
        todo!("TODO: port guard `valid_op_mul_mat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_mul_mat_argmax(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMatArgmax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_mat_argmax
        todo!(
            "TODO: port guard `valid_op_mul_mat_argmax` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_mul_mat_f16(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMat,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_mat_f16
        todo!(
            "TODO: port guard `valid_op_mul_mat_f16` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_mul_mat_id(
        &self,
        _event: &EmelKernelX8664EventDispatchOpMulMatId,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_mul_mat_id
        todo!(
            "TODO: port guard `valid_op_mul_mat_id` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_norm(&self, _event: &EmelKernelX8664EventDispatchOpNorm) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_norm
        todo!("TODO: port guard `valid_op_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_opt_step_adamw(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOptStepAdamw,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_opt_step_adamw
        todo!(
            "TODO: port guard `valid_op_opt_step_adamw` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_opt_step_sgd(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOptStepSgd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_opt_step_sgd
        todo!(
            "TODO: port guard `valid_op_opt_step_sgd` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_out_prod(
        &self,
        _event: &EmelKernelX8664EventDispatchOpOutProd,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_out_prod
        todo!(
            "TODO: port guard `valid_op_out_prod` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_pad(&self, _event: &EmelKernelX8664EventDispatchOpPad) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_pad
        todo!("TODO: port guard `valid_op_pad` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_pad_reflect_1d(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPadReflect1d,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_pad_reflect_1d
        todo!(
            "TODO: port guard `valid_op_pad_reflect_1d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_permute(&self, _event: &EmelKernelX8664EventDispatchOpPermute) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_permute
        todo!("TODO: port guard `valid_op_permute` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_pool_1d(&self, _event: &EmelKernelX8664EventDispatchOpPool1d) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_pool_1d
        todo!("TODO: port guard `valid_op_pool_1d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_pool_2d(&self, _event: &EmelKernelX8664EventDispatchOpPool2d) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_pool_2d
        todo!("TODO: port guard `valid_op_pool_2d` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_pool_2d_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpPool2dBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_pool_2d_back
        todo!(
            "TODO: port guard `valid_op_pool_2d_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_repeat(&self, _event: &EmelKernelX8664EventDispatchOpRepeat) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_repeat
        todo!("TODO: port guard `valid_op_repeat` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_repeat_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRepeatBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_repeat_back
        todo!(
            "TODO: port guard `valid_op_repeat_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_reshape(&self, _event: &EmelKernelX8664EventDispatchOpReshape) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_reshape
        todo!("TODO: port guard `valid_op_reshape` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_rms_norm(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRmsNorm,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rms_norm
        todo!(
            "TODO: port guard `valid_op_rms_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rms_norm_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRmsNormBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rms_norm_back
        todo!(
            "TODO: port guard `valid_op_rms_norm_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_roll(&self, _event: &EmelKernelX8664EventDispatchOpRoll) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_roll
        todo!("TODO: port guard `valid_op_roll` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_rope_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRopeBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rope_back
        todo!(
            "TODO: port guard `valid_op_rope_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rope_neox(&self, _event: &EmelKernelX8664EventDispatchOpRope) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rope_neox
        todo!(
            "TODO: port guard `valid_op_rope_neox` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rope_norm(&self, _event: &EmelKernelX8664EventDispatchOpRope) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rope_norm
        todo!(
            "TODO: port guard `valid_op_rope_norm` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rope_timestep(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRope,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rope_timestep
        todo!(
            "TODO: port guard `valid_op_rope_timestep` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rwkv_wkv6(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv6,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rwkv_wkv6
        todo!(
            "TODO: port guard `valid_op_rwkv_wkv6` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_rwkv_wkv7(
        &self,
        _event: &EmelKernelX8664EventDispatchOpRwkvWkv7,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_rwkv_wkv7
        todo!(
            "TODO: port guard `valid_op_rwkv_wkv7` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_scale(&self, _event: &EmelKernelX8664EventDispatchOpScale) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_scale
        todo!("TODO: port guard `valid_op_scale` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_set(&self, _event: &EmelKernelX8664EventDispatchOpSet) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_set
        todo!("TODO: port guard `valid_op_set` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_set_rows(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSetRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_set_rows
        todo!(
            "TODO: port guard `valid_op_set_rows` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_silu_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSiluBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_silu_back
        todo!(
            "TODO: port guard `valid_op_silu_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_sin(&self, _event: &EmelKernelX8664EventDispatchOpSin) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sin
        todo!("TODO: port guard `valid_op_sin` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_soft_max(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSoftMax,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_soft_max
        todo!(
            "TODO: port guard `valid_op_soft_max` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_soft_max_back(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSoftMaxBack,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_soft_max_back
        todo!(
            "TODO: port guard `valid_op_soft_max_back` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_solve_tri(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSolveTri,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_solve_tri
        todo!(
            "TODO: port guard `valid_op_solve_tri` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_sqr(&self, _event: &EmelKernelX8664EventDispatchOpSqr) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sqr
        todo!("TODO: port guard `valid_op_sqr` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_sqrt(&self, _event: &EmelKernelX8664EventDispatchOpSqrt) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sqrt
        todo!("TODO: port guard `valid_op_sqrt` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_ssm_conv(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSsmConv,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_ssm_conv
        todo!(
            "TODO: port guard `valid_op_ssm_conv` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_ssm_scan(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSsmScan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_ssm_scan
        todo!(
            "TODO: port guard `valid_op_ssm_scan` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_sub(&self, _event: &EmelKernelX8664EventDispatchOpSub) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sub
        todo!("TODO: port guard `valid_op_sub` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_sum(&self, _event: &EmelKernelX8664EventDispatchOpSum) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sum
        todo!("TODO: port guard `valid_op_sum` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_sum_rows(
        &self,
        _event: &EmelKernelX8664EventDispatchOpSumRows,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_sum_rows
        todo!(
            "TODO: port guard `valid_op_sum_rows` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_timestep_embedding(
        &self,
        _event: &EmelKernelX8664EventDispatchOpTimestepEmbedding,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_timestep_embedding
        todo!(
            "TODO: port guard `valid_op_timestep_embedding` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_top_k(&self, _event: &EmelKernelX8664EventDispatchOpTopK) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_top_k
        todo!("TODO: port guard `valid_op_top_k` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_transpose(
        &self,
        _event: &EmelKernelX8664EventDispatchOpTranspose,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_transpose
        todo!(
            "TODO: port guard `valid_op_transpose` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_tri(&self, _event: &EmelKernelX8664EventDispatchOpTri) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_tri
        todo!("TODO: port guard `valid_op_tri` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_unary_abs(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_abs
        todo!(
            "TODO: port guard `valid_op_unary_abs` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_elu(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_elu
        todo!(
            "TODO: port guard `valid_op_unary_elu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_exp(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_exp
        todo!(
            "TODO: port guard `valid_op_unary_exp` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_gelu(
        &self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_gelu
        todo!(
            "TODO: port guard `valid_op_unary_gelu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_neg(&self, _event: &EmelKernelX8664EventDispatchOpUnary) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_neg
        todo!(
            "TODO: port guard `valid_op_unary_neg` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_relu(
        &self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_relu
        todo!(
            "TODO: port guard `valid_op_unary_relu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_silu(
        &self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_silu
        todo!(
            "TODO: port guard `valid_op_unary_silu` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_unary_tanh(
        &self,
        _event: &EmelKernelX8664EventDispatchOpUnary,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_unary_tanh
        todo!(
            "TODO: port guard `valid_op_unary_tanh` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_upscale(&self, _event: &EmelKernelX8664EventDispatchOpUpscale) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_upscale
        todo!("TODO: port guard `valid_op_upscale` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_view(&self, _event: &EmelKernelX8664EventDispatchOpView) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_view
        todo!("TODO: port guard `valid_op_view` from emel.cpp/src/emel/kernel/x86_64/guards.hpp")
    }
    fn valid_op_win_part(
        &self,
        _event: &EmelKernelX8664EventDispatchOpWinPart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_win_part
        todo!(
            "TODO: port guard `valid_op_win_part` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
    fn valid_op_win_unpart(
        &self,
        _event: &EmelKernelX8664EventDispatchOpWinUnpart,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/kernel/x86_64/guards.hpp::valid_op_win_unpart
        todo!(
            "TODO: port guard `valid_op_win_unpart` from emel.cpp/src/emel/kernel/x86_64/guards.hpp"
        )
    }
}
