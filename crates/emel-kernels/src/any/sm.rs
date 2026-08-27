//! Explicit kernel operation routing.

#![allow(clippy::derive_partial_eq_without_eq)]

use sml::sml;

use super::actor::{
    AccRuntime, AdamWRuntime, Add1Runtime, AddIdRuntime, AddRelPosRuntime, AddRuntime,
    ArangeRuntime, ArgmaxRuntime, ArgsortRuntime, BinaryAddRuntime, BinaryDivRuntime,
    BinaryMulRuntime, BinarySubRuntime, BinaryUnexpectedRuntime, BroadcastAddRuntime,
    BroadcastMulRuntime, BroadcastUnexpectedRuntime, ClampRuntime, ConcatRuntime, ContRuntime,
    Conv2dDwRuntime, Conv2dRuntime, Conv3dRuntime, ConvTranspose1dRuntime,
    ConvTranspose1dUnexpectedRuntime, ConvTranspose2dRuntime, CosRuntime, CountEqualRuntime,
    CpyRuntime, CrossEntropyBackRuntime, CrossEntropyRuntime, CumsumRuntime, CustomRuntime,
    DiagMaskInfRuntime, DiagMaskZeroRuntime, DiagRuntime, DivRuntime, DupRuntime, F16MatmulRuntime,
    F16MatmulUnexpectedRuntime, FillRuntime, FlashAttnRuntime, FlashAttnUnexpectedRuntime,
    FlashBackRuntime, GenericRuntime, GetRelPosRuntime, GetRowsBackRuntime, GetRowsBf16Runtime,
    GetRowsF16Runtime, GetRowsF32BytesRuntime, GetRowsQ4_0Runtime, GetRowsQ4KRuntime,
    GetRowsQ8_0Runtime, GetRowsRuntime, GetRowsUnexpectedRuntime, GlaRuntime, GluRuntime,
    GroupNormRuntime, Im2Col3dRuntime, Im2ColBackRuntime, Im2ColF16Runtime, Im2ColRuntime,
    Im2ColUnexpectedRuntime, L2NormRuntime, LeakyReluRuntime, LogRuntime, Map1Runtime, Map2Runtime,
    Map3Runtime, MatmulArgmaxQ2KRuntime, MatmulArgmaxQ3KRuntime, MatmulArgmaxQ4_0Runtime,
    MatmulArgmaxQ4_1Runtime, MatmulArgmaxQ4KRuntime, MatmulArgmaxQ5_0Runtime,
    MatmulArgmaxQ6KRuntime, MatmulArgmaxQ8_0Runtime, MatmulArgmaxRuntime,
    MatmulArgmaxUnexpectedRuntime, MatmulQ2KRuntime, MatmulQ3KRuntime, MatmulQ4_0Runtime,
    MatmulQ4_1Runtime, MatmulQ4KRuntime, MatmulQ5_0Runtime, MatmulQ6KRuntime, MatmulQ8_0Runtime,
    MatmulRuntime, MatmulUnexpectedRuntime, MeanRuntime, MulMatIdRuntime, MulRuntime, NormRuntime,
    OutProdRuntime, PadReflectRuntime, PadRuntime, PermuteRuntime, Pool1dRuntime,
    Pool2dBackRuntime, Pool2dRuntime, RepeatBackRuntime, RepeatRuntime, ReshapeRuntime,
    RmsNormBackRuntime, RmsNormRuntime, RollRuntime, RopeBackRuntime, RopeRuntime,
    RopeUnexpectedRuntime, ScaleRuntime, SetRowsRuntime, SetRuntime, SgdRuntime, SiluBackRuntime,
    SinRuntime, SoftMaxBackRuntime, SoftMaxRuntime, SolveTriRuntime, SqrRuntime, SqrtRuntime,
    SsmConvRuntime, SsmScanRuntime, SubRuntime, SumRowsRuntime, SumRuntime, TimestepRuntime,
    TopKRuntime, TransposeRuntime, TriRuntime, UnaryRuntime, UnaryUnexpectedRuntime,
    UnsupportedRuntime, UpscaleRuntime, ViewRuntime, WinPartRuntime, WinUnpartRuntime, Wkv6Runtime,
    Wkv7Runtime,
};

sml! {
    KernelMachine<'dispatch> {
        // Every operation has an explicit validation guard before its action.
        "ready"_s <= *"ready"_s + Dup(DupRuntime<'dispatch>) [guard_dup_valid] / effect_dup_execute,
        "ready"_s <= "ready"_s + Dup(DupRuntime<'dispatch>) [guard_dup_invalid] / effect_dup_reject,

        "ready"_s <= "ready"_s + Add(AddRuntime<'dispatch>) [guard_add_valid] / effect_add_execute,
        "ready"_s <= "ready"_s + Add(AddRuntime<'dispatch>) [guard_add_invalid] / effect_add_reject,

        "ready"_s <= "ready"_s + Sub(SubRuntime<'dispatch>) [guard_sub_valid] / effect_sub_execute,
        "ready"_s <= "ready"_s + Sub(SubRuntime<'dispatch>) [guard_sub_invalid] / effect_sub_reject,

        "ready"_s <= "ready"_s + Mul(MulRuntime<'dispatch>) [guard_mul_valid] / effect_mul_execute,
        "ready"_s <= "ready"_s + Mul(MulRuntime<'dispatch>) [guard_mul_invalid] / effect_mul_reject,

        "ready"_s <= "ready"_s + Div(DivRuntime<'dispatch>) [guard_div_valid] / effect_div_execute,
        "ready"_s <= "ready"_s + Div(DivRuntime<'dispatch>) [guard_div_invalid] / effect_div_reject,

        // Tensor-view binary operations are composed through the owned
        // BinaryKernel child. These five forwarding rows intentionally enter
        // a transient handoff state; the completion rows below join the child
        // synchronously before returning to `ready`. The sole `*ready` marker
        // above establishes the machine's initial lifecycle state: in
        // stateforward-sml, `*` is an initial-state marker (not a per-row
        // self-transition), so duplicating it on these five routes would be
        // invalid. Keeping the forwarding states explicit preserves the
        // single-writer RTC lifecycle without an orthogonal root region.
        "binary_add_dispatch"_s <= "ready"_s + BinaryAdd(BinaryAddRuntime<'dispatch>)
            / effect_binary_add,
        "ready"_s <= "binary_add_dispatch"_s + completion<_>,
        "binary_sub_dispatch"_s <= "ready"_s + BinarySub(BinarySubRuntime<'dispatch>)
            / effect_binary_sub,
        "ready"_s <= "binary_sub_dispatch"_s + completion<_>,
        "binary_mul_dispatch"_s <= "ready"_s + BinaryMul(BinaryMulRuntime<'dispatch>)
            / effect_binary_mul,
        "ready"_s <= "binary_mul_dispatch"_s + completion<_>,
        "binary_div_dispatch"_s <= "ready"_s + BinaryDiv(BinaryDivRuntime<'dispatch>)
            / effect_binary_div,
        "ready"_s <= "binary_div_dispatch"_s + completion<_>,
        "binary_unexpected_dispatch"_s <= "ready"_s
            + BinaryUnexpected(BinaryUnexpectedRuntime<'dispatch>) / effect_binary_unexpected,
        "ready"_s <= "binary_unexpected_dispatch"_s + completion<_>,

        "ready"_s <= "binary_add_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "binary_sub_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "binary_mul_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "binary_div_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "binary_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Transposed convolution is composed through the owned F32 child.
        "conv_transpose_1d_dispatch"_s <= "ready"_s
            + ConvTranspose1d(ConvTranspose1dRuntime<'dispatch>)
            / effect_conv_transpose_1d,
        "ready"_s <= "conv_transpose_1d_dispatch"_s + completion<_>,
        "conv_transpose_1d_unexpected_dispatch"_s <= "ready"_s
            + ConvTranspose1dUnexpected(ConvTranspose1dUnexpectedRuntime<'dispatch>)
            / effect_conv_transpose_1d_unexpected,
        "ready"_s <= "conv_transpose_1d_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "conv_transpose_1d_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "conv_transpose_1d_unexpected_dispatch"_s
            + unexpected_event<_> / effect_unexpected,

        // F16 matrix multiplication is synchronously handed to the owned
        // F16MatmulKernel child through an explicit transient state.
        "f16_matmul_dispatch"_s <= "ready"_s
            + F16Matmul(F16MatmulRuntime<'dispatch>) / effect_f16_matmul,
        "ready"_s <= "f16_matmul_dispatch"_s + completion<_>,
        "f16_matmul_unexpected_dispatch"_s <= "ready"_s
            + F16MatmulUnexpected(F16MatmulUnexpectedRuntime<'dispatch>)
            / effect_f16_matmul_unexpected,
        "ready"_s <= "f16_matmul_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "f16_matmul_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "f16_matmul_unexpected_dispatch"_s
            + unexpected_event<_> / effect_unexpected,

        // Regular F32 and packed matrix multiplication are handed to the
        // maintained MatmulKernel child through explicit transient states.
        "matmul_dispatch"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>) / effect_matmul,
        "ready"_s <= "matmul_dispatch"_s + completion<_>,
        "matmul_q4_0_dispatch"_s <= "ready"_s
            + MatmulQ4_0(MatmulQ4_0Runtime<'dispatch>) / effect_matmul_q4_0,
        "ready"_s <= "matmul_q4_0_dispatch"_s + completion<_>,
        "matmul_q4_1_dispatch"_s <= "ready"_s
            + MatmulQ4_1(MatmulQ4_1Runtime<'dispatch>) / effect_matmul_q4_1,
        "ready"_s <= "matmul_q4_1_dispatch"_s + completion<_>,
        "matmul_q5_0_dispatch"_s <= "ready"_s
            + MatmulQ5_0(MatmulQ5_0Runtime<'dispatch>) / effect_matmul_q5_0,
        "ready"_s <= "matmul_q5_0_dispatch"_s + completion<_>,
        "matmul_q8_0_dispatch"_s <= "ready"_s
            + MatmulQ8_0(MatmulQ8_0Runtime<'dispatch>) / effect_matmul_q8_0,
        "ready"_s <= "matmul_q8_0_dispatch"_s + completion<_>,
        "matmul_q2_k_dispatch"_s <= "ready"_s
            + MatmulQ2K(MatmulQ2KRuntime<'dispatch>) / effect_matmul_q2_k,
        "ready"_s <= "matmul_q2_k_dispatch"_s + completion<_>,
        "matmul_q3_k_dispatch"_s <= "ready"_s
            + MatmulQ3K(MatmulQ3KRuntime<'dispatch>) / effect_matmul_q3_k,
        "ready"_s <= "matmul_q3_k_dispatch"_s + completion<_>,
        "matmul_q4_k_dispatch"_s <= "ready"_s
            + MatmulQ4K(MatmulQ4KRuntime<'dispatch>) / effect_matmul_q4_k,
        "ready"_s <= "matmul_q4_k_dispatch"_s + completion<_>,
        "matmul_q6_k_dispatch"_s <= "ready"_s
            + MatmulQ6K(MatmulQ6KRuntime<'dispatch>) / effect_matmul_q6_k,
        "ready"_s <= "matmul_q6_k_dispatch"_s + completion<_>,
        "matmul_unexpected_dispatch"_s <= "ready"_s
            + MatmulUnexpected(MatmulUnexpectedRuntime<'dispatch>) / effect_matmul_unexpected,
        "ready"_s <= "matmul_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "matmul_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q4_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q4_1_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q5_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q8_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q2_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q3_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q4_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_q6_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Matrix-vector argmax remains a separate actor because its result is
        // an index and score rather than a unit regular-matmul result.
        "matmul_argmax_dispatch"_s <= "ready"_s
            + MatmulArgmax(MatmulArgmaxRuntime<'dispatch>) / effect_matmul_argmax,
        "ready"_s <= "matmul_argmax_dispatch"_s + completion<_>,
        "matmul_argmax_q4_0_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            / effect_matmul_argmax_q4_0,
        "ready"_s <= "matmul_argmax_q4_0_dispatch"_s + completion<_>,
        "matmul_argmax_q4_1_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            / effect_matmul_argmax_q4_1,
        "ready"_s <= "matmul_argmax_q4_1_dispatch"_s + completion<_>,
        "matmul_argmax_q5_0_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            / effect_matmul_argmax_q5_0,
        "ready"_s <= "matmul_argmax_q5_0_dispatch"_s + completion<_>,
        "matmul_argmax_q8_0_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            / effect_matmul_argmax_q8_0,
        "ready"_s <= "matmul_argmax_q8_0_dispatch"_s + completion<_>,
        "matmul_argmax_q2_k_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            / effect_matmul_argmax_q2_k,
        "ready"_s <= "matmul_argmax_q2_k_dispatch"_s + completion<_>,
        "matmul_argmax_q3_k_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            / effect_matmul_argmax_q3_k,
        "ready"_s <= "matmul_argmax_q3_k_dispatch"_s + completion<_>,
        "matmul_argmax_q4_k_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            / effect_matmul_argmax_q4_k,
        "ready"_s <= "matmul_argmax_q4_k_dispatch"_s + completion<_>,
        "matmul_argmax_q6_k_dispatch"_s <= "ready"_s
            + MatmulArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            / effect_matmul_argmax_q6_k,
        "ready"_s <= "matmul_argmax_q6_k_dispatch"_s + completion<_>,
        "matmul_argmax_unexpected_dispatch"_s <= "ready"_s
            + MatmulArgmaxUnexpected(MatmulArgmaxUnexpectedRuntime<'dispatch>)
            / effect_matmul_argmax_unexpected,
        "ready"_s <= "matmul_argmax_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "matmul_argmax_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q4_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q4_1_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q5_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q8_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q2_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q3_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q4_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_q6_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "matmul_argmax_unexpected_dispatch"_s
            + unexpected_event<_> / effect_unexpected,

        // One-dimensional im2col operations are synchronously handed to the
        // owned child actor through explicit transient states.
        "im2col_dispatch"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>) / effect_im2col,
        "ready"_s <= "im2col_dispatch"_s + completion<_>,
        "im2col_f16_dispatch"_s <= "ready"_s
            + Im2ColF16(Im2ColF16Runtime<'dispatch>) / effect_im2col_f16,
        "ready"_s <= "im2col_f16_dispatch"_s + completion<_>,
        "im2col_unexpected_dispatch"_s <= "ready"_s
            + Im2ColUnexpected(Im2ColUnexpectedRuntime<'dispatch>) / effect_im2col_unexpected,
        "ready"_s <= "im2col_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "im2col_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "im2col_f16_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "im2col_unexpected_dispatch"_s
            + unexpected_event<_> / effect_unexpected,

        // Row-gather operations are synchronously handed to the maintained
        // GetRowsKernel child through explicit transient states.
        "get_rows_dispatch"_s <= "ready"_s + GetRows(GetRowsRuntime<'dispatch>) / effect_get_rows,
        "ready"_s <= "get_rows_dispatch"_s + completion<_>,
        "get_rows_f32_bytes_dispatch"_s <= "ready"_s
            + GetRowsF32Bytes(GetRowsF32BytesRuntime<'dispatch>) / effect_get_rows_f32_bytes,
        "ready"_s <= "get_rows_f32_bytes_dispatch"_s + completion<_>,
        "get_rows_f16_dispatch"_s <= "ready"_s
            + GetRowsF16(GetRowsF16Runtime<'dispatch>) / effect_get_rows_f16,
        "ready"_s <= "get_rows_f16_dispatch"_s + completion<_>,
        "get_rows_bf16_dispatch"_s <= "ready"_s
            + GetRowsBf16(GetRowsBf16Runtime<'dispatch>) / effect_get_rows_bf16,
        "ready"_s <= "get_rows_bf16_dispatch"_s + completion<_>,
        "get_rows_q4_0_dispatch"_s <= "ready"_s
            + GetRowsQ4_0(GetRowsQ4_0Runtime<'dispatch>) / effect_get_rows_q4_0,
        "ready"_s <= "get_rows_q4_0_dispatch"_s + completion<_>,
        "get_rows_q8_0_dispatch"_s <= "ready"_s
            + GetRowsQ8_0(GetRowsQ8_0Runtime<'dispatch>) / effect_get_rows_q8_0,
        "ready"_s <= "get_rows_q8_0_dispatch"_s + completion<_>,
        "get_rows_q4_k_dispatch"_s <= "ready"_s
            + GetRowsQ4K(GetRowsQ4KRuntime<'dispatch>) / effect_get_rows_q4_k,
        "ready"_s <= "get_rows_q4_k_dispatch"_s + completion<_>,
        "get_rows_unexpected_dispatch"_s <= "ready"_s
            + GetRowsUnexpected(GetRowsUnexpectedRuntime<'dispatch>)
            / effect_get_rows_unexpected,
        "ready"_s <= "get_rows_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "get_rows_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_f32_bytes_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_f16_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_bf16_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_q4_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_q8_0_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_q4_k_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "get_rows_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Rotary-position operations are synchronously handed to the owned
        // RopeKernel child through explicit transient states.
        "rope_dispatch"_s <= "ready"_s + Rope(RopeRuntime<'dispatch>) / effect_rope,
        "ready"_s <= "rope_dispatch"_s + completion<_>,
        "rope_unexpected_dispatch"_s <= "ready"_s
            + RopeUnexpected(RopeUnexpectedRuntime<'dispatch>) / effect_rope_unexpected,
        "ready"_s <= "rope_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "rope_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "rope_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Flash-attention is synchronously handed to the owned
        // FlashAttnKernel child through explicit transient states.
        "flash_attn_dispatch"_s <= "ready"_s
            + FlashAttn(FlashAttnRuntime<'dispatch>) / effect_flash_attn,
        "ready"_s <= "flash_attn_dispatch"_s + completion<_>,
        "flash_attn_unexpected_dispatch"_s <= "ready"_s
            + FlashAttnUnexpected(FlashAttnUnexpectedRuntime<'dispatch>)
            / effect_flash_attn_unexpected,
        "ready"_s <= "flash_attn_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "flash_attn_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "flash_attn_unexpected_dispatch"_s
            + unexpected_event<_> / effect_unexpected,

        // Generic unary operations are synchronously handed to the maintained
        // UnaryKernel child. The child owns the explicit suboperation guards.
        "unary_dispatch"_s <= "ready"_s + Unary(UnaryRuntime<'dispatch>) / effect_unary,
        "ready"_s <= "unary_dispatch"_s + completion<_>,
        "unary_unexpected_dispatch"_s <= "ready"_s
            + UnaryUnexpected(UnaryUnexpectedRuntime<'dispatch>) / effect_unary_unexpected,
        "ready"_s <= "unary_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unary_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "unary_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Row-broadcast operations are synchronously handed to the owned
        // BroadcastKernel child through explicit transient states.
        "broadcast_add_dispatch"_s <= "ready"_s
            + BroadcastAdd(BroadcastAddRuntime<'dispatch>) / effect_broadcast_add,
        "ready"_s <= "broadcast_add_dispatch"_s + completion<_>,
        "broadcast_mul_dispatch"_s <= "ready"_s
            + BroadcastMul(BroadcastMulRuntime<'dispatch>) / effect_broadcast_mul,
        "ready"_s <= "broadcast_mul_dispatch"_s + completion<_>,
        "broadcast_unexpected_dispatch"_s <= "ready"_s
            + BroadcastUnexpected(BroadcastUnexpectedRuntime<'dispatch>)
            / effect_broadcast_unexpected,
        "ready"_s <= "broadcast_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "broadcast_add_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "broadcast_mul_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "broadcast_unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Power operations are delegated to the owned PowerKernel child. The
        // transient handoff makes the synchronous child join explicit and
        // leaves all view/shape validation in the child actor.
        "power_sqr_dispatch"_s <= "ready"_s + Sqr(SqrRuntime<'dispatch>) / effect_sqr_execute,
        "ready"_s <= "power_sqr_dispatch"_s + completion<_>,
        "power_sqrt_dispatch"_s <= "ready"_s + Sqrt(SqrtRuntime<'dispatch>) / effect_sqrt_execute,
        "ready"_s <= "power_sqrt_dispatch"_s + completion<_>,
        "ready"_s <= "power_sqr_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "power_sqrt_dispatch"_s + unexpected_event<_> / effect_unexpected,

        // Unsupported operations are never silently routed to a fallback.
        "ready"_s <= "ready"_s + Unsupported(UnsupportedRuntime<'dispatch>) / effect_unsupported,

        // The fixed-size safe envelope preserves the pinned generic operation
        // boundary. Each maintained scalar operation has an explicit guard
        // and action; the const-selected actions contain only their bounded
        // data-plane loops. Unsupported operations remain an explicit result.
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_dup_valid] / effect_generic_dup,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_add_valid] / effect_generic_add,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_sub_valid] / effect_generic_sub,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_mul_valid] / effect_generic_mul,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_div_valid] / effect_generic_div,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_add1_valid] / effect_generic_add1,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_scale_valid] / effect_generic_scale,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_clamp_valid] / effect_generic_clamp,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_leaky_relu_valid] / effect_generic_leaky_relu,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_silu_back_valid] / effect_generic_silu_back,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_cumsum_valid] / effect_generic_cumsum,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_repeat_valid] / effect_generic_repeat,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_repeat_back_valid] / effect_generic_repeat_back,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_concat_0_valid] / effect_generic_concat_0,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_concat_1_valid] / effect_generic_concat_1,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_concat_2_valid] / effect_generic_concat_2,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_concat_3_valid] / effect_generic_concat_3,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_diag_valid] / effect_generic_diag,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_diag_mask_inf_valid] / effect_generic_diag_mask_inf,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_diag_mask_zero_valid] / effect_generic_diag_mask_zero,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_soft_max_back_valid] / effect_generic_soft_max_back,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_rms_norm_back_valid] / effect_generic_rms_norm_back,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_l2_norm_valid] / effect_generic_l2_norm,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_count_equal_valid] / effect_generic_count_equal,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_fill_valid] / effect_generic_fill,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_arange_valid] / effect_generic_arange,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_mul_mat_valid] / effect_generic_mul_mat,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_mul_mat_f16_valid] / effect_generic_mul_mat_f16,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_mul_mat_argmax_valid] / effect_generic_mul_mat_argmax,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_sqr_valid] / effect_generic_sqr,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_sqrt_valid] / effect_generic_sqrt,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_log_valid] / effect_generic_log,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_sin_valid] / effect_generic_sin,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_cos_valid] / effect_generic_cos,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_sum_valid] / effect_generic_sum,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_mean_valid] / effect_generic_mean,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_argmax_valid] / effect_generic_argmax,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_soft_max_valid] / effect_generic_soft_max,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_norm_valid] / effect_generic_norm,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_rms_norm_valid] / effect_generic_rms_norm,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_abs_valid] / effect_generic_unary_abs,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_neg_valid] / effect_generic_unary_neg,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_tanh_valid] / effect_generic_unary_tanh,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_elu_valid] / effect_generic_unary_elu,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_relu_valid] / effect_generic_unary_relu,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_gelu_valid] / effect_generic_unary_gelu,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_silu_valid] / effect_generic_unary_silu,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_unary_exp_valid] / effect_generic_unary_exp,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_known_invalid] / effect_generic_invalid,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_valid] / effect_generic_unsupported,
        "ready"_s <= "ready"_s + Generic(GenericRuntime<'dispatch>)
            [guard_generic_invalid] / effect_generic_invalid,

        // Maintained portable reduction actor composition.
        "ready"_s <= "ready"_s + Log(LogRuntime<'dispatch>) / effect_log,
        "ready"_s <= "ready"_s + Sin(SinRuntime<'dispatch>) / effect_sin,
        "ready"_s <= "ready"_s + Cos(CosRuntime<'dispatch>) / effect_cos,
        "ready"_s <= "ready"_s + Sum(SumRuntime<'dispatch>) / effect_sum,
        "ready"_s <= "ready"_s + SumRows(SumRowsRuntime<'dispatch>) / effect_sum_rows,
        "ready"_s <= "ready"_s + Mean(MeanRuntime<'dispatch>) / effect_mean,
        "ready"_s <= "ready"_s + Argmax(ArgmaxRuntime<'dispatch>) / effect_argmax,
        "ready"_s <= "ready"_s + CountEqual(CountEqualRuntime<'dispatch>) / effect_count_equal,
        "ready"_s <= "ready"_s + Glu(GluRuntime<'dispatch>) / effect_glu,
        "ready"_s <= "ready"_s + Diag(DiagRuntime<'dispatch>) / effect_diag,
        "ready"_s <= "ready"_s + DiagMaskInf(DiagMaskInfRuntime<'dispatch>) / effect_diag_mask_inf,
        "ready"_s <= "ready"_s + DiagMaskZero(DiagMaskZeroRuntime<'dispatch>) / effect_diag_mask_zero,
        "ready"_s <= "ready"_s + Pad(PadRuntime<'dispatch>) / effect_pad,
        "ready"_s <= "ready"_s + PadReflect(PadReflectRuntime<'dispatch>) / effect_pad_reflect,
        "ready"_s <= "ready"_s + Tri(TriRuntime<'dispatch>) / effect_tri,
        "ready"_s <= "ready"_s + Pool1d(Pool1dRuntime<'dispatch>) / effect_pool_1d,
        "ready"_s <= "ready"_s + Pool2d(Pool2dRuntime<'dispatch>) / effect_pool_2d,
        "ready"_s <= "ready"_s + Pool2dBack(Pool2dBackRuntime<'dispatch>) / effect_pool_2d_back,
        "ready"_s <= "ready"_s + Roll(RollRuntime<'dispatch>) / effect_roll,
        "ready"_s <= "ready"_s + Upscale(UpscaleRuntime<'dispatch>) / effect_upscale,
        "ready"_s <= "ready"_s + Argsort(ArgsortRuntime<'dispatch>) / effect_argsort,
        "ready"_s <= "ready"_s + TopK(TopKRuntime<'dispatch>) / effect_top_k,
        "ready"_s <= "ready"_s + DenseConv2d(Conv2dRuntime<'dispatch>) / effect_dense_conv_2d,
        "ready"_s <= "ready"_s + DenseConv2dDw(Conv2dDwRuntime<'dispatch>) / effect_dense_conv_2d_dw,
        "ready"_s <= "ready"_s + DenseConv3d(Conv3dRuntime<'dispatch>) / effect_dense_conv_3d,
        "ready"_s <= "ready"_s + DenseConvTranspose2d(ConvTranspose2dRuntime<'dispatch>) / effect_dense_conv_transpose_2d,
        "ready"_s <= "ready"_s + SoftMaxBack(SoftMaxBackRuntime<'dispatch>) / effect_soft_max_back,
        "ready"_s <= "ready"_s + RmsNormBack(RmsNormBackRuntime<'dispatch>) / effect_rms_norm_back,
        "ready"_s <= "ready"_s + GetRowsBack(GetRowsBackRuntime<'dispatch>) / effect_get_rows_back,
        "ready"_s <= "ready"_s + SetRows(SetRowsRuntime<'dispatch>) / effect_set_rows,
        "ready"_s <= "ready"_s + SetSlice(SetRuntime<'dispatch>) / effect_set_slice,
        "ready"_s <= "ready"_s + WinPart(WinPartRuntime<'dispatch>) / effect_win_part,
        "ready"_s <= "ready"_s + WinUnpart(WinUnpartRuntime<'dispatch>) / effect_win_unpart,
        "ready"_s <= "ready"_s + DenseIm2ColBack(Im2ColBackRuntime<'dispatch>) / effect_im2col_back,
        "ready"_s <= "ready"_s + DenseIm2Col3d(Im2Col3dRuntime<'dispatch>) / effect_im2col_3d_dense,
        "ready"_s <= "ready"_s + RopeBack(RopeBackRuntime<'dispatch>) / effect_rope_back,
        "ready"_s <= "ready"_s + Timestep(TimestepRuntime<'dispatch>) / effect_timestep_embedding,
        "ready"_s <= "ready"_s + SsmConv(SsmConvRuntime<'dispatch>) / effect_ssm_conv,
        "ready"_s <= "ready"_s + SsmScan(SsmScanRuntime<'dispatch>) / effect_ssm_scan,
        "ready"_s <= "ready"_s + OutProd(OutProdRuntime<'dispatch>) / effect_out_prod,
        "ready"_s <= "ready"_s + SolveTri(SolveTriRuntime<'dispatch>) / effect_solve_tri,
        "ready"_s <= "ready"_s + CrossEntropy(CrossEntropyRuntime<'dispatch>) / effect_cross_entropy,
        "ready"_s <= "ready"_s + CrossEntropyBack(CrossEntropyBackRuntime<'dispatch>) / effect_cross_entropy_back,
        "ready"_s <= "ready"_s + AdamW(AdamWRuntime<'dispatch>) / effect_adamw,
        "ready"_s <= "ready"_s + Sgd(SgdRuntime<'dispatch>) / effect_sgd,
        "ready"_s <= "ready"_s + GetRelPos(GetRelPosRuntime<'dispatch>) / effect_get_rel_pos,
        "ready"_s <= "ready"_s + AddRelPos(AddRelPosRuntime<'dispatch>) / effect_add_rel_pos,
        "ready"_s <= "ready"_s + AddId(AddIdRuntime<'dispatch>) / effect_add_id,
        "ready"_s <= "ready"_s + MulMatId(MulMatIdRuntime<'dispatch>) / effect_mul_mat_id,
        "ready"_s <= "ready"_s + Wkv6(Wkv6Runtime<'dispatch>) / effect_wkv6,
        "ready"_s <= "ready"_s + Wkv7(Wkv7Runtime<'dispatch>) / effect_wkv7,
        "ready"_s <= "ready"_s + Gla(GlaRuntime<'dispatch>) / effect_gla,
        "ready"_s <= "ready"_s + FlashBack(FlashBackRuntime<'dispatch>) / effect_flash_back,
        "ready"_s <= "ready"_s + Map1(Map1Runtime<'dispatch>) / effect_map1,
        "ready"_s <= "ready"_s + Map2(Map2Runtime<'dispatch>) / effect_map2,
        "ready"_s <= "ready"_s + Map3(Map3Runtime<'dispatch>) / effect_map3,
        "ready"_s <= "ready"_s + Custom(CustomRuntime<'dispatch>) / effect_custom,
        "ready"_s <= "ready"_s + SoftMax(SoftMaxRuntime<'dispatch>) / effect_soft_max,

        // Maintained portable activation and elementwise actors are composed
        // through explicit child-dispatch transitions.
        "ready"_s <= "ready"_s + Scale(ScaleRuntime<'dispatch>) / effect_scale,
        "ready"_s <= "ready"_s + Clamp(ClampRuntime<'dispatch>) / effect_clamp,
        "ready"_s <= "ready"_s + SiluBack(SiluBackRuntime<'dispatch>) / effect_silu_back,
        "ready"_s <= "ready"_s + LeakyRelu(LeakyReluRuntime<'dispatch>) / effect_leaky_relu,
        "ready"_s <= "ready"_s + Add1(Add1Runtime<'dispatch>) / effect_add1,
        "ready"_s <= "ready"_s + Acc(AccRuntime<'dispatch>) / effect_acc,

        "ready"_s <= "ready"_s + Norm(NormRuntime<'dispatch>) / effect_norm,
        "ready"_s <= "ready"_s + RmsNorm(RmsNormRuntime<'dispatch>) / effect_rms_norm,
        "ready"_s <= "ready"_s + GroupNorm(GroupNormRuntime<'dispatch>) / effect_group_norm,
        "ready"_s <= "ready"_s + L2Norm(L2NormRuntime<'dispatch>) / effect_l2_norm,

        "ready"_s <= "ready"_s + Cumsum(CumsumRuntime<'dispatch>) / effect_cumsum,
        "ready"_s <= "ready"_s + Repeat(RepeatRuntime<'dispatch>) / effect_repeat,
        "ready"_s <= "ready"_s + RepeatBack(RepeatBackRuntime<'dispatch>) / effect_repeat_back,
        "ready"_s <= "ready"_s + Concat(ConcatRuntime<'dispatch>) / effect_concat,

        "ready"_s <= "ready"_s + Fill(FillRuntime<'dispatch>) / effect_fill,
        "ready"_s <= "ready"_s + Arange(ArangeRuntime<'dispatch>) / effect_arange,

        "ready"_s <= "ready"_s + Cpy(CpyRuntime<'dispatch>) / effect_cpy,
        "ready"_s <= "ready"_s + Cont(ContRuntime<'dispatch>) / effect_cont,
        "ready"_s <= "ready"_s + Reshape(ReshapeRuntime<'dispatch>) / effect_reshape,
        "ready"_s <= "ready"_s + View(ViewRuntime<'dispatch>) / effect_view,
        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) / effect_permute,
        "ready"_s <= "ready"_s + Transpose(TransposeRuntime<'dispatch>) / effect_transpose,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
