//! Safe representation of the pinned generic kernel event envelope.
//!
//! The reference event contract uses one operation struct per GGML opcode,
//! with four-dimensional tensor metadata and a fixed operation-parameter
//! buffer.  Rust's maintained operators expose typed events for operations
//! that have a safe implementation.  This module preserves the complete wire
//! envelope for boundary validation without exposing raw pointers or making a
//! byte-backed request look like a typed tensor.

use core::fmt;

use super::glu::GluSubOp;
use super::pool::PoolSubOp;
use super::tensor_view::Layout;
pub use super::tensor_view::{DType as GenericDType, Layout as GenericLayout};
use super::unary::UnarySubOp;

/// Number of generic operations in the pinned `EMEL_KERNEL_OP_EVENT_LIST`.
pub const GENERIC_OPERATION_COUNT: usize = 95;

/// The operation names in the pinned generic kernel event list.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelOperation {
    /// `op_dup`.
    Dup = 0,
    /// `op_add`.
    Add,
    /// `op_add_id`.
    AddId,
    /// `op_add1`.
    Add1,
    /// `op_acc`.
    Acc,
    /// `op_sub`.
    Sub,
    /// `op_mul`.
    Mul,
    /// `op_div`.
    Div,
    /// `op_sqr`.
    Sqr,
    /// `op_sqrt`.
    Sqrt,
    /// `op_log`.
    Log,
    /// `op_sin`.
    Sin,
    /// `op_cos`.
    Cos,
    /// `op_sum`.
    Sum,
    /// `op_sum_rows`.
    SumRows,
    /// `op_cumsum`.
    Cumsum,
    /// `op_mean`.
    Mean,
    /// `op_argmax`.
    Argmax,
    /// `op_count_equal`.
    CountEqual,
    /// `op_repeat`.
    Repeat,
    /// `op_repeat_back`.
    RepeatBack,
    /// `op_concat`.
    Concat,
    /// `op_silu_back`.
    SiluBack,
    /// `op_norm`.
    Norm,
    /// `op_rms_norm`.
    RmsNorm,
    /// `op_rms_norm_back`.
    RmsNormBack,
    /// `op_group_norm`.
    GroupNorm,
    /// `op_l2_norm`.
    L2Norm,
    /// `op_mul_mat`.
    MulMat,
    /// `op_mul_mat_argmax`.
    MulMatArgmax,
    /// `op_mul_mat_id`.
    MulMatId,
    /// `op_out_prod`.
    OutProd,
    /// `op_scale`.
    Scale,
    /// `op_set`.
    Set,
    /// `op_cpy`.
    Cpy,
    /// `op_cont`.
    Cont,
    /// `op_reshape`.
    Reshape,
    /// `op_view`.
    View,
    /// `op_permute`.
    Permute,
    /// `op_transpose`.
    Transpose,
    /// `op_get_rows`.
    GetRows,
    /// `op_get_rows_back`.
    GetRowsBack,
    /// `op_set_rows`.
    SetRows,
    /// `op_diag`.
    Diag,
    /// `op_diag_mask_inf`.
    DiagMaskInf,
    /// `op_diag_mask_zero`.
    DiagMaskZero,
    /// `op_soft_max`.
    SoftMax,
    /// `op_soft_max_back`.
    SoftMaxBack,
    /// `op_rope`.
    Rope,
    /// `op_rope_back`.
    RopeBack,
    /// `op_clamp`.
    Clamp,
    /// `op_conv_transpose_1d`.
    ConvTranspose1d,
    /// `op_im2col`.
    Im2Col,
    /// `op_im2col_back`.
    Im2ColBack,
    /// `op_im2col_3d`.
    Im2Col3d,
    /// `op_conv_2d`.
    Conv2d,
    /// `op_conv_3d`.
    Conv3d,
    /// `op_conv_2d_dw`.
    Conv2dDw,
    /// `op_conv_transpose_2d`.
    ConvTranspose2d,
    /// `op_pool_1d`.
    Pool1d,
    /// `op_pool_2d`.
    Pool2d,
    /// `op_pool_2d_back`.
    Pool2dBack,
    /// `op_upscale`.
    Upscale,
    /// `op_pad`.
    Pad,
    /// `op_pad_reflect_1d`.
    PadReflect1d,
    /// `op_roll`.
    Roll,
    /// `op_arange`.
    Arange,
    /// `op_timestep_embedding`.
    TimestepEmbedding,
    /// `op_argsort`.
    Argsort,
    /// `op_top_k`.
    TopK,
    /// `op_leaky_relu`.
    LeakyRelu,
    /// `op_tri`.
    Tri,
    /// `op_fill`.
    Fill,
    /// `op_flash_attn_ext`.
    FlashAttnExt,
    /// `op_flash_attn_back`.
    FlashAttnBack,
    /// `op_ssm_conv`.
    SsmConv,
    /// `op_ssm_scan`.
    SsmScan,
    /// `op_win_part`.
    WinPart,
    /// `op_win_unpart`.
    WinUnpart,
    /// `op_get_rel_pos`.
    GetRelPos,
    /// `op_add_rel_pos`.
    AddRelPos,
    /// `op_rwkv_wkv6`.
    RwkvWkv6,
    /// `op_gated_linear_attn`.
    GatedLinearAttn,
    /// `op_rwkv_wkv7`.
    RwkvWkv7,
    /// `op_solve_tri`.
    SolveTri,
    /// `op_unary`.
    Unary,
    /// `op_map_custom1`.
    MapCustom1,
    /// `op_map_custom2`.
    MapCustom2,
    /// `op_map_custom3`.
    MapCustom3,
    /// `op_custom`.
    Custom,
    /// `op_cross_entropy_loss`.
    CrossEntropyLoss,
    /// `op_cross_entropy_loss_back`.
    CrossEntropyLossBack,
    /// `op_opt_step_adamw`.
    OptStepAdamw,
    /// `op_opt_step_sgd`.
    OptStepSgd,
    /// `op_glu`.
    Glu,
}

impl KernelOperation {
    /// Converts the stable inventory ordinal to an operation.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Dup,
            1 => Self::Add,
            2 => Self::AddId,
            3 => Self::Add1,
            4 => Self::Acc,
            5 => Self::Sub,
            6 => Self::Mul,
            7 => Self::Div,
            8 => Self::Sqr,
            9 => Self::Sqrt,
            10 => Self::Log,
            11 => Self::Sin,
            12 => Self::Cos,
            13 => Self::Sum,
            14 => Self::SumRows,
            15 => Self::Cumsum,
            16 => Self::Mean,
            17 => Self::Argmax,
            18 => Self::CountEqual,
            19 => Self::Repeat,
            20 => Self::RepeatBack,
            21 => Self::Concat,
            22 => Self::SiluBack,
            23 => Self::Norm,
            24 => Self::RmsNorm,
            25 => Self::RmsNormBack,
            26 => Self::GroupNorm,
            27 => Self::L2Norm,
            28 => Self::MulMat,
            29 => Self::MulMatArgmax,
            30 => Self::MulMatId,
            31 => Self::OutProd,
            32 => Self::Scale,
            33 => Self::Set,
            34 => Self::Cpy,
            35 => Self::Cont,
            36 => Self::Reshape,
            37 => Self::View,
            38 => Self::Permute,
            39 => Self::Transpose,
            40 => Self::GetRows,
            41 => Self::GetRowsBack,
            42 => Self::SetRows,
            43 => Self::Diag,
            44 => Self::DiagMaskInf,
            45 => Self::DiagMaskZero,
            46 => Self::SoftMax,
            47 => Self::SoftMaxBack,
            48 => Self::Rope,
            49 => Self::RopeBack,
            50 => Self::Clamp,
            51 => Self::ConvTranspose1d,
            52 => Self::Im2Col,
            53 => Self::Im2ColBack,
            54 => Self::Im2Col3d,
            55 => Self::Conv2d,
            56 => Self::Conv3d,
            57 => Self::Conv2dDw,
            58 => Self::ConvTranspose2d,
            59 => Self::Pool1d,
            60 => Self::Pool2d,
            61 => Self::Pool2dBack,
            62 => Self::Upscale,
            63 => Self::Pad,
            64 => Self::PadReflect1d,
            65 => Self::Roll,
            66 => Self::Arange,
            67 => Self::TimestepEmbedding,
            68 => Self::Argsort,
            69 => Self::TopK,
            70 => Self::LeakyRelu,
            71 => Self::Tri,
            72 => Self::Fill,
            73 => Self::FlashAttnExt,
            74 => Self::FlashAttnBack,
            75 => Self::SsmConv,
            76 => Self::SsmScan,
            77 => Self::WinPart,
            78 => Self::WinUnpart,
            79 => Self::GetRelPos,
            80 => Self::AddRelPos,
            81 => Self::RwkvWkv6,
            82 => Self::GatedLinearAttn,
            83 => Self::RwkvWkv7,
            84 => Self::SolveTri,
            85 => Self::Unary,
            86 => Self::MapCustom1,
            87 => Self::MapCustom2,
            88 => Self::MapCustom3,
            89 => Self::Custom,
            90 => Self::CrossEntropyLoss,
            91 => Self::CrossEntropyLossBack,
            92 => Self::OptStepAdamw,
            93 => Self::OptStepSgd,
            94 => Self::Glu,
            _ => return None,
        })
    }

    /// Returns the source operation name without allocating.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dup => "op_dup",
            Self::Add => "op_add",
            Self::AddId => "op_add_id",
            Self::Add1 => "op_add1",
            Self::Acc => "op_acc",
            Self::Sub => "op_sub",
            Self::Mul => "op_mul",
            Self::Div => "op_div",
            Self::Sqr => "op_sqr",
            Self::Sqrt => "op_sqrt",
            Self::Log => "op_log",
            Self::Sin => "op_sin",
            Self::Cos => "op_cos",
            Self::Sum => "op_sum",
            Self::SumRows => "op_sum_rows",
            Self::Cumsum => "op_cumsum",
            Self::Mean => "op_mean",
            Self::Argmax => "op_argmax",
            Self::CountEqual => "op_count_equal",
            Self::Repeat => "op_repeat",
            Self::RepeatBack => "op_repeat_back",
            Self::Concat => "op_concat",
            Self::SiluBack => "op_silu_back",
            Self::Norm => "op_norm",
            Self::RmsNorm => "op_rms_norm",
            Self::RmsNormBack => "op_rms_norm_back",
            Self::GroupNorm => "op_group_norm",
            Self::L2Norm => "op_l2_norm",
            Self::MulMat => "op_mul_mat",
            Self::MulMatArgmax => "op_mul_mat_argmax",
            Self::MulMatId => "op_mul_mat_id",
            Self::OutProd => "op_out_prod",
            Self::Scale => "op_scale",
            Self::Set => "op_set",
            Self::Cpy => "op_cpy",
            Self::Cont => "op_cont",
            Self::Reshape => "op_reshape",
            Self::View => "op_view",
            Self::Permute => "op_permute",
            Self::Transpose => "op_transpose",
            Self::GetRows => "op_get_rows",
            Self::GetRowsBack => "op_get_rows_back",
            Self::SetRows => "op_set_rows",
            Self::Diag => "op_diag",
            Self::DiagMaskInf => "op_diag_mask_inf",
            Self::DiagMaskZero => "op_diag_mask_zero",
            Self::SoftMax => "op_soft_max",
            Self::SoftMaxBack => "op_soft_max_back",
            Self::Rope => "op_rope",
            Self::RopeBack => "op_rope_back",
            Self::Clamp => "op_clamp",
            Self::ConvTranspose1d => "op_conv_transpose_1d",
            Self::Im2Col => "op_im2col",
            Self::Im2ColBack => "op_im2col_back",
            Self::Im2Col3d => "op_im2col_3d",
            Self::Conv2d => "op_conv_2d",
            Self::Conv3d => "op_conv_3d",
            Self::Conv2dDw => "op_conv_2d_dw",
            Self::ConvTranspose2d => "op_conv_transpose_2d",
            Self::Pool1d => "op_pool_1d",
            Self::Pool2d => "op_pool_2d",
            Self::Pool2dBack => "op_pool_2d_back",
            Self::Upscale => "op_upscale",
            Self::Pad => "op_pad",
            Self::PadReflect1d => "op_pad_reflect_1d",
            Self::Roll => "op_roll",
            Self::Arange => "op_arange",
            Self::TimestepEmbedding => "op_timestep_embedding",
            Self::Argsort => "op_argsort",
            Self::TopK => "op_top_k",
            Self::LeakyRelu => "op_leaky_relu",
            Self::Tri => "op_tri",
            Self::Fill => "op_fill",
            Self::FlashAttnExt => "op_flash_attn_ext",
            Self::FlashAttnBack => "op_flash_attn_back",
            Self::SsmConv => "op_ssm_conv",
            Self::SsmScan => "op_ssm_scan",
            Self::WinPart => "op_win_part",
            Self::WinUnpart => "op_win_unpart",
            Self::GetRelPos => "op_get_rel_pos",
            Self::AddRelPos => "op_add_rel_pos",
            Self::RwkvWkv6 => "op_rwkv_wkv6",
            Self::GatedLinearAttn => "op_gated_linear_attn",
            Self::RwkvWkv7 => "op_rwkv_wkv7",
            Self::SolveTri => "op_solve_tri",
            Self::Unary => "op_unary",
            Self::MapCustom1 => "op_map_custom1",
            Self::MapCustom2 => "op_map_custom2",
            Self::MapCustom3 => "op_map_custom3",
            Self::Custom => "op_custom",
            Self::CrossEntropyLoss => "op_cross_entropy_loss",
            Self::CrossEntropyLossBack => "op_cross_entropy_loss_back",
            Self::OptStepAdamw => "op_opt_step_adamw",
            Self::OptStepSgd => "op_opt_step_sgd",
            Self::Glu => "op_glu",
        }
    }
}

/// A borrowed immutable byte tensor matching the reference `tensor_view`.
#[derive(Clone, Copy, Debug)]
pub struct RawTensorView<'a> {
    data: &'a [u8],
    layout: Layout,
}

impl<'a> RawTensorView<'a> {
    /// Creates a byte-backed tensor view. The actor guard validates its span.
    #[must_use]
    pub const fn new(data: &'a [u8], layout: Layout) -> Self {
        Self { data, layout }
    }

    /// Returns the layout metadata.
    #[must_use]
    pub const fn layout(self) -> Layout {
        self.layout
    }

    /// Returns the borrowed storage length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.data.len()
    }

    /// Returns whether the borrowed storage is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.data.is_empty()
    }

    /// Returns the borrowed bytes for a caller-owned, already-validated view.
    #[must_use]
    pub const fn data(self) -> &'a [u8] {
        self.data
    }

    /// Validates metadata and the complete byte span without casting bytes.
    ///
    /// # Errors
    ///
    /// Returns the first invalid shape, stride, or bounds condition.
    pub fn validate(self) -> Result<usize, GenericViewError> {
        validate_span(self.data.len(), self.layout)
    }

    pub(super) fn f32_valid(self) -> bool {
        let Some(strides) = self.layout.effective_f32_strides() else {
            return false;
        };
        self.validate().is_ok() && strides.into_iter().all(|stride| stride >= 4)
    }

    pub(super) fn read_f32_index(self, index: usize) -> f32 {
        let coordinates = coordinates_for_index(index, self.layout.ne());
        let offset = byte_offset(self.layout, coordinates);
        let bytes = self
            .data
            .get(offset..offset + 4)
            .expect("generic F32 read is guard-proven");
        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    pub(super) fn f16_valid(self) -> bool {
        self.layout.dtype() == GenericDType::F16
            && self.validate().is_ok()
            && self.layout.is_dense_contiguous_f16()
    }

    pub(super) fn read_f16_index(self, index: usize) -> u16 {
        let coordinates = coordinates_for_index(index, self.layout.ne());
        let offset = byte_offset(self.layout, coordinates);
        let bytes = self
            .data
            .get(offset..offset + 2)
            .expect("generic F16 read is guard-proven");
        u16::from_ne_bytes([bytes[0], bytes[1]])
    }
}

/// A borrowed mutable byte tensor matching the reference `tensor_view_mut`.
#[derive(Debug)]
pub struct RawTensorViewMut<'a> {
    data: &'a mut [u8],
    layout: Layout,
}

impl<'a> RawTensorViewMut<'a> {
    /// Creates a byte-backed mutable tensor view. The actor guard validates it.
    #[must_use]
    pub const fn new(data: &'a mut [u8], layout: Layout) -> Self {
        Self { data, layout }
    }

    /// Returns the layout metadata.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the borrowed storage length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the borrowed storage is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Validates metadata and the complete byte span without mutating storage.
    ///
    /// # Errors
    ///
    /// Returns the first invalid shape, stride, or bounds condition.
    pub fn validate(&self) -> Result<usize, GenericViewError> {
        validate_span(self.data.len(), self.layout)
    }

    /// Returns the mutable bytes after the caller has validated the view.
    pub const fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    pub(super) fn f32_valid(&self) -> bool {
        let Some(strides) = self.layout.effective_f32_strides() else {
            return false;
        };
        self.validate().is_ok() && strides.into_iter().all(|stride| stride >= 4)
    }

    pub(super) fn write_f32_index(&mut self, index: usize, value: f32) {
        let coordinates = coordinates_for_index(index, self.layout.ne());
        let offset = byte_offset(self.layout, coordinates);
        let bytes = value.to_ne_bytes();
        let destination = self
            .data
            .get_mut(offset..offset + 4)
            .expect("generic F32 write is guard-proven");
        destination.copy_from_slice(&bytes);
    }

    pub(super) fn read_f32_index(&self, index: usize) -> f32 {
        let coordinates = coordinates_for_index(index, self.layout.ne());
        let offset = byte_offset(self.layout, coordinates);
        let bytes = self
            .data
            .get(offset..offset + 4)
            .expect("generic F32 read is guard-proven");
        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

/// Safe fixed-size operation parameters matching `op_params[64]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpParams {
    bytes: [u8; 64],
    len: usize,
}

impl OpParams {
    /// Constructs parameters and rejects lengths beyond the pinned buffer.
    #[must_use]
    pub const fn new(bytes: [u8; 64], len: usize) -> Option<Self> {
        if len > bytes.len() {
            None
        } else {
            Some(Self { bytes, len })
        }
    }

    /// Constructs an empty parameter buffer.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            bytes: [0; 64],
            len: 0,
        }
    }

    /// Returns the initialized parameter prefix.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// Returns the initialized parameter prefix borrowed from this value.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.as_slice()
    }

    /// Returns the pinned parameter length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Returns whether the initialized parameter prefix is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// The safe equivalent of one pinned generic operation struct.
#[derive(Debug)]
pub struct GenericRequest<'a> {
    operation: KernelOperation,
    pub(crate) src0: Option<RawTensorView<'a>>,
    pub(crate) src1: Option<RawTensorView<'a>>,
    pub(crate) src2: Option<RawTensorView<'a>>,
    pub(crate) dst: Option<RawTensorViewMut<'a>>,
    params: OpParams,
    pool: Option<PoolSubOp>,
    unary: Option<UnarySubOp>,
    glu: Option<GluSubOp>,
    index_out: Option<&'a mut i32>,
}

impl<'a> GenericRequest<'a> {
    /// Creates the common operation envelope.
    #[must_use]
    pub const fn new(
        operation: KernelOperation,
        src0: Option<RawTensorView<'a>>,
        src1: Option<RawTensorView<'a>>,
        src2: Option<RawTensorView<'a>>,
        dst: Option<RawTensorViewMut<'a>>,
        params: OpParams,
    ) -> Self {
        Self {
            operation,
            src0,
            src1,
            src2,
            dst,
            params,
            pool: Some(PoolSubOp::Avg),
            unary: Some(UnarySubOp::Abs),
            glu: Some(GluSubOp::ReGlu),
            index_out: None,
        }
    }

    /// Sets the pinned pooling suboperation before dispatch.
    #[must_use]
    pub const fn with_pool(mut self, pool: PoolSubOp) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Sets the pinned unary suboperation before dispatch.
    #[must_use]
    pub const fn with_unary(mut self, unary: UnarySubOp) -> Self {
        self.unary = Some(unary);
        self
    }

    /// Sets the pinned GLU suboperation before dispatch.
    #[must_use]
    pub const fn with_glu(mut self, glu: GluSubOp) -> Self {
        self.glu = Some(glu);
        self
    }

    /// Adds the safe mutable argmax result destination.
    #[must_use]
    pub const fn with_index_out(mut self, index_out: &'a mut i32) -> Self {
        self.index_out = Some(index_out);
        self
    }

    /// Returns the operation tag.
    #[must_use]
    pub const fn operation(&self) -> KernelOperation {
        self.operation
    }

    /// Returns the common source operation parameters.
    #[must_use]
    pub const fn params(&self) -> OpParams {
        self.params
    }

    /// Returns the optional pool selector.
    #[must_use]
    pub const fn pool(&self) -> Option<PoolSubOp> {
        self.pool
    }

    /// Returns the optional unary selector.
    #[must_use]
    pub const fn unary(&self) -> Option<UnarySubOp> {
        self.unary
    }

    /// Returns the optional GLU selector.
    #[must_use]
    pub const fn glu(&self) -> Option<GluSubOp> {
        self.glu
    }

    /// Returns whether a safe argmax destination was supplied.
    #[must_use]
    pub const fn has_index_out(&self) -> bool {
        self.index_out.is_some()
    }

    pub(super) fn set_index_out(&mut self, value: i32) {
        if let Some(index_out) = self.index_out.as_deref_mut() {
            *index_out = value;
        }
    }

    /// Validates all five borrowed tensors and the operation-specific fields.
    ///
    /// # Errors
    ///
    /// Returns the first invalid view or missing operation-specific field.
    pub fn validate(&self) -> Result<(), GenericViewError> {
        if let Some(src0) = self.src0 {
            src0.validate()?;
        }
        if let Some(src1) = self.src1 {
            src1.validate()?;
        }
        if let Some(src2) = self.src2 {
            src2.validate()?;
        }
        if let Some(dst) = &self.dst {
            dst.validate()?;
        }
        if self.operation == KernelOperation::MulMatArgmax && !self.has_index_out() {
            return Err(GenericViewError::MissingArgmaxDestination);
        }
        Ok(())
    }
}

/// An event boundary carrying one safe generic operation request.
#[derive(Debug)]
pub struct GenericEvent<'a> {
    request: GenericRequest<'a>,
}

impl<'a> GenericEvent<'a> {
    /// Creates an event. Validation remains an actor guard concern.
    #[must_use]
    pub const fn new(request: GenericRequest<'a>) -> Self {
        Self { request }
    }

    /// Returns the operation tag without dispatching.
    #[must_use]
    pub const fn operation(&self) -> KernelOperation {
        self.request.operation()
    }

    /// Returns the request for pre-dispatch inspection.
    #[must_use]
    pub const fn request(&self) -> &GenericRequest<'a> {
        &self.request
    }

    /// Consumes the event and returns its request.
    #[must_use]
    pub const fn into_request(self) -> GenericRequest<'a> {
        self.request
    }
}

/// Errors found while validating a safe generic view or request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenericViewError {
    /// The view has an empty or unrepresentable extent.
    InvalidShape,
    /// A stride is zero or cannot be used as a byte offset.
    InvalidStride,
    /// The computed span exceeds the borrowed byte slice.
    OutOfBounds,
    /// `op_mul_mat_argmax` did not receive a safe destination.
    MissingArgmaxDestination,
}

impl fmt::Display for GenericViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid generic tensor shape"),
            Self::InvalidStride => formatter.write_str("invalid generic tensor stride"),
            Self::OutOfBounds => formatter.write_str("generic tensor view is out of bounds"),
            Self::MissingArgmaxDestination => {
                formatter.write_str("argmax operation is missing its destination")
            }
        }
    }
}

impl std::error::Error for GenericViewError {}

fn validate_span(data_len: usize, layout: Layout) -> Result<usize, GenericViewError> {
    let ne = layout.ne();
    let nb = if layout.dtype() == GenericDType::F32 {
        layout
            .effective_f32_strides()
            .ok_or(GenericViewError::InvalidStride)?
    } else {
        layout.nb()
    };
    let mut count = 1usize;
    let mut dimension = 0;
    while dimension < 4 {
        let extent = usize::try_from(ne[dimension]).map_err(|_| GenericViewError::InvalidShape)?;
        if extent == 0 {
            return Err(GenericViewError::InvalidShape);
        }
        count = count
            .checked_mul(extent)
            .ok_or(GenericViewError::InvalidShape)?;
        if ne[dimension] > 1 && nb[dimension] == 0 {
            return Err(GenericViewError::InvalidStride);
        }
        dimension += 1;
    }

    let mut max_offset = 0u128;
    dimension = 0;
    while dimension < 4 {
        let contribution = u128::from(ne[dimension] - 1)
            .checked_mul(u128::from(nb[dimension]))
            .ok_or(GenericViewError::InvalidShape)?;
        max_offset = max_offset
            .checked_add(contribution)
            .ok_or(GenericViewError::InvalidShape)?;
        dimension += 1;
    }
    let end = max_offset
        .checked_add(1)
        .ok_or(GenericViewError::InvalidShape)?;
    if end > u128::from(data_len as u64) {
        return Err(GenericViewError::OutOfBounds);
    }
    Ok(count)
}

fn coordinates_for_index(mut index: usize, ne: [u64; 4]) -> [usize; 4] {
    let mut coordinates = [0usize; 4];
    let mut dimension = 0;
    while dimension < 4 {
        let extent = usize::try_from(ne[dimension]).expect("guard-proven extent fits usize");
        coordinates[dimension] = index % extent;
        index /= extent;
        dimension += 1;
    }
    coordinates
}

fn byte_offset(layout: Layout, coordinates: [usize; 4]) -> usize {
    let strides = if layout.dtype() == GenericDType::F32 {
        layout
            .effective_f32_strides()
            .expect("generic F32 offset has a valid layout")
    } else {
        layout.nb()
    };
    let mut offset = 0u128;
    let mut dimension = 0;
    while dimension < 4 {
        offset = offset
            .checked_add(
                u128::from(coordinates[dimension] as u64)
                    .checked_mul(u128::from(strides[dimension]))
                    .expect("guard-proven generic stride multiplication fits"),
            )
            .expect("guard-proven generic offset fits");
        dimension += 1;
    }
    usize::try_from(offset).expect("guard-proven generic offset fits usize")
}

fn parameter_f32(request: &GenericRequest<'_>, slot: usize) -> Option<f32> {
    let start = slot.checked_mul(4)?;
    let bytes = request.params.as_slice().get(start..start + 4)?;
    Some(f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn parameter_i32(request: &GenericRequest<'_>, slot: usize) -> Option<i32> {
    let start = slot.checked_mul(4)?;
    let bytes = request.params.as_slice().get(start..start + 4)?;
    Some(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn f32_pair_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && src0.layout.element_count() == dst.layout.element_count()
}

fn f32_triplet_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(src1), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    src0.f32_valid()
        && src1.f32_valid()
        && dst.f32_valid()
        && src0.layout.ne() == src1.layout.ne()
        && src0.layout.ne() == dst.layout.ne()
}

pub(super) fn generic_f32_scalar_valid(request: &GenericRequest<'_>) -> bool {
    f32_pair_valid(request) && parameter_f32(request, 0).is_some_and(f32::is_finite)
}

pub(super) fn generic_f32_clamp_valid(request: &GenericRequest<'_>) -> bool {
    if !f32_pair_valid(request) {
        return false;
    }
    let (Some(minimum), Some(maximum)) = (parameter_f32(request, 0), parameter_f32(request, 1))
    else {
        return false;
    };
    minimum.is_finite() && maximum.is_finite() && minimum <= maximum
}

pub(super) fn generic_f32_silu_back_valid(request: &GenericRequest<'_>) -> bool {
    f32_triplet_valid(request)
}

pub(super) fn generic_f32_cumsum_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && src0.layout.ne() == dst.layout.ne()
}

const fn repeat_shape_valid(source: [u64; 4], target: [u64; 4]) -> bool {
    let mut dimension = 0;
    while dimension < 4 {
        if source[dimension] == 0 || target[dimension] == 0 {
            return false;
        }
        if !target[dimension].is_multiple_of(source[dimension]) {
            return false;
        }
        dimension += 1;
    }
    true
}

pub(super) fn generic_f32_repeat_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && repeat_shape_valid(src0.layout.ne(), dst.layout.ne())
}

pub(super) fn generic_f32_repeat_back_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && repeat_shape_valid(dst.layout.ne(), src0.layout.ne())
}

fn concat_shape_valid<const AXIS: usize>(lhs: [u64; 4], rhs: [u64; 4], dst: [u64; 4]) -> bool {
    let mut dimension = 0;
    while dimension < 4 {
        if dimension == AXIS {
            let Some(sum) = lhs[dimension].checked_add(rhs[dimension]) else {
                return false;
            };
            if sum != dst[dimension] {
                return false;
            }
        } else if lhs[dimension] != rhs[dimension] || lhs[dimension] != dst[dimension] {
            return false;
        }
        dimension += 1;
    }
    true
}

pub(super) fn generic_f32_concat_axis_valid<const AXIS: usize>(
    request: &GenericRequest<'_>,
) -> bool {
    let (Some(lhs), Some(rhs), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    lhs.f32_valid()
        && rhs.f32_valid()
        && dst.f32_valid()
        && parameter_i32(request, 0) == i32::try_from(AXIS).ok()
        && concat_shape_valid::<AXIS>(lhs.layout.ne(), rhs.layout.ne(), dst.layout.ne())
}

pub(super) fn generic_f32_fill_valid(request: &GenericRequest<'_>) -> bool {
    request.src0.is_none()
        && request
            .dst
            .as_ref()
            .is_some_and(RawTensorViewMut::f32_valid)
        && parameter_f32(request, 0).is_some_and(f32::is_finite)
}

pub(super) fn generic_f32_arange_valid(request: &GenericRequest<'_>) -> bool {
    request.src0.is_none()
        && request
            .dst
            .as_ref()
            .is_some_and(RawTensorViewMut::f32_valid)
        && parameter_f32(request, 0).is_some_and(f32::is_finite)
        && parameter_f32(request, 1).is_some_and(f32::is_finite)
}

pub(super) fn generic_f32_diag_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    let Some(side) = src0.layout.ne()[0].checked_mul(src0.layout.ne()[0]) else {
        return false;
    };
    src0.f32_valid()
        && dst.f32_valid()
        && src0.layout.ne()[1..] == [1, 1, 1]
        && dst.layout.ne() == [src0.layout.ne()[0], src0.layout.ne()[0], 1, 1]
        && dst.layout.element_count() == usize::try_from(side).ok()
}

pub(super) fn generic_f32_mask_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    let Some(n_past) = parameter_i32(request, 0) else {
        return false;
    };
    let shape = src0.layout.ne();
    n_past >= 0
        && src0.f32_valid()
        && dst.f32_valid()
        && shape == dst.layout.ne()
        && shape[0] > 0
        && shape[1] == shape[0]
        && shape[2] == 1
        && shape[3] == 1
}

pub(super) fn generic_f32_l2_norm_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid()
        && dst.f32_valid()
        && src0.layout.ne() == dst.layout.ne()
        && src0.layout.element_count().is_some_and(|count| count > 0)
}

pub(super) fn generic_f32_softmax_back_valid(request: &GenericRequest<'_>) -> bool {
    if !f32_triplet_valid(request) {
        return false;
    }
    let values = request
        .src0
        .expect("softmax values are present after validation");
    let Some(row) = parameter_i32(request, 0) else {
        return false;
    };
    row > 0
        && values
            .layout
            .element_count()
            .is_some_and(|count| count.is_multiple_of(usize::try_from(row).unwrap_or(0)))
}

pub(super) fn generic_f32_rms_norm_back_valid(request: &GenericRequest<'_>) -> bool {
    if !f32_triplet_valid(request) {
        return false;
    }
    let input = request.src0.expect("RMS input is present after validation");
    let gradient = request
        .src1
        .expect("RMS gradient is present after validation");
    let dst = request
        .dst
        .as_ref()
        .expect("RMS destination is present after validation");
    let Some(epsilon) = parameter_f32(request, 0) else {
        return false;
    };
    epsilon.is_finite()
        && epsilon >= 0.0
        && input.layout.ne() == gradient.layout.ne()
        && input.layout.ne() == dst.layout.ne()
        && input.layout.ne()[0] > 0
}

pub(super) fn generic_f32_count_equal_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(lhs), Some(rhs), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    lhs.f32_valid()
        && rhs.f32_valid()
        && dst.f32_valid()
        && lhs.layout.element_count() == rhs.layout.element_count()
        && dst.layout.element_count() == Some(1)
}

pub(super) fn generic_f32_copy_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && src0.layout.element_count() == dst.layout.element_count()
}

pub(super) fn generic_f32_binary_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(src1), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    src0.f32_valid()
        && src1.f32_valid()
        && dst.f32_valid()
        && src0.layout.element_count() == src1.layout.element_count()
        && src0.layout.element_count() == dst.layout.element_count()
}

pub(super) fn generic_f32_unary_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && src0.layout.element_count() == dst.layout.element_count()
}

pub(super) fn generic_f32_reduce_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    src0.f32_valid() && dst.f32_valid() && dst.layout.element_count() == Some(1)
}

pub(super) fn generic_f32_rows_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    if !src0.f32_valid() || !dst.f32_valid() {
        return false;
    }
    src0.layout.element_count() == dst.layout.element_count()
}

pub(super) fn generic_f32_norm_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(src0), Some(dst)) = (request.src0, request.dst.as_ref()) else {
        return false;
    };
    let params = request.params();
    let epsilon = params.as_slice();
    src0.f32_valid()
        && dst.f32_valid()
        && src0.layout.ne() == dst.layout.ne()
        && epsilon.len() >= 4
        && f32::from_ne_bytes([epsilon[0], epsilon[1], epsilon[2], epsilon[3]]).is_finite()
        && f32::from_ne_bytes([epsilon[0], epsilon[1], epsilon[2], epsilon[3]]) >= 0.0
}

pub(super) fn generic_f32_matmul_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(lhs), Some(rhs), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    let lhs_shape = lhs.layout.ne();
    let rhs_shape = rhs.layout.ne();
    let dst_shape = dst.layout.ne();
    lhs.f32_valid()
        && rhs.f32_valid()
        && dst.f32_valid()
        && lhs_shape[0] > 0
        && lhs_shape[1] > 0
        && rhs_shape[0] > 0
        && rhs_shape[1] == lhs_shape[0]
        && dst_shape[0] == rhs_shape[0]
        && dst_shape[1] == lhs_shape[1]
        && lhs_shape[2] == 1
        && lhs_shape[3] == 1
        && rhs_shape[2] == 1
        && rhs_shape[3] == 1
        && dst_shape[2] == 1
        && dst_shape[3] == 1
}

pub(super) fn generic_f32_matmul_argmax_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(lhs), Some(rhs), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    let lhs_shape = lhs.layout.ne();
    let rhs_shape = rhs.layout.ne();
    let dst_shape = dst.layout.ne();
    request.has_index_out()
        && lhs.f32_valid()
        && rhs.f32_valid()
        && dst.f32_valid()
        && lhs_shape[0] > 0
        && lhs_shape[1] > 0
        && rhs_shape == [lhs_shape[0], 1, 1, 1]
        && dst_shape == [1, 1, 1, 1]
        && lhs_shape[2] == 1
        && lhs_shape[3] == 1
}

pub(super) fn generic_f16_matmul_valid(request: &GenericRequest<'_>) -> bool {
    let (Some(lhs), Some(rhs), Some(dst)) = (request.src0, request.src1, request.dst.as_ref())
    else {
        return false;
    };
    let lhs_shape = lhs.layout.ne();
    let rhs_shape = rhs.layout.ne();
    let dst_shape = dst.layout.ne();
    lhs.f16_valid()
        && rhs.f16_valid()
        && dst.f32_valid()
        && lhs_shape[0] > 0
        && lhs_shape[1] > 0
        && rhs_shape[0] == lhs_shape[0]
        && rhs_shape[1] > 0
        && dst_shape[0] == lhs_shape[1]
        && dst_shape[1] == rhs_shape[1]
        && lhs_shape[2] == 1
        && lhs_shape[3] == 1
        && rhs_shape[2] == 1
        && rhs_shape[3] == 1
        && dst_shape[2] == 1
        && dst_shape[3] == 1
}

pub(super) fn execute_generic_copy(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("copy source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("copy destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("copy count is guard-proven");
    let mut index = 0;
    while index < count {
        dst.write_f32_index(index, src0.read_f32_index(index));
        index += 1;
    }
}

pub(super) fn execute_generic_binary<const OP: u8>(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("binary lhs is guard-proven");
    let rhs = request.src1.expect("binary rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("binary destination is guard-proven");
    let count = lhs
        .layout
        .element_count()
        .expect("binary count is guard-proven");
    let mut index = 0;
    while index < count {
        let left = lhs.read_f32_index(index);
        let right = rhs.read_f32_index(index);
        let value = match OP {
            0 => left + right,
            1 => left - right,
            2 => left * right,
            3 => left / right,
            _ => unreachable!("generic binary operation is compile-time selected"),
        };
        dst.write_f32_index(index, value);
        index += 1;
    }
}

pub(super) fn execute_generic_unary<const OP: u8>(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("unary source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("unary destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("unary count is guard-proven");
    let mut index = 0;
    while index < count {
        let value = src0.read_f32_index(index);
        let output = match OP {
            0 => value.abs(),
            1 => -value,
            2 => value.tanh(),
            3 => {
                if value > 0.0 {
                    value
                } else {
                    value.exp_m1()
                }
            }
            4 => value.max(0.0),
            5 => {
                let rounded = f32::from_ne_bytes(value.to_ne_bytes());
                0.5 * rounded
                    * (1.0
                        + (0.797_884_6 * (0.044_715 * rounded * rounded).mul_add(rounded, rounded))
                            .tanh())
            }
            6 => value / (1.0 + (-value).exp()),
            7 => value.exp(),
            8 => value * value,
            9 => value.sqrt(),
            10 => value.ln(),
            11 => value.sin(),
            12 => value.cos(),
            _ => unreachable!("generic unary operation is compile-time selected"),
        };
        dst.write_f32_index(index, output);
        index += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_reduce_sum(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("reduction source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("reduction destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("reduction count is guard-proven");
    let mut sum = 0.0f64;
    let mut index = 0;
    while index < count {
        sum += f64::from(src0.read_f32_index(index));
        index += 1;
    }
    dst.write_f32_index(0, sum as f32);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn execute_generic_reduce_mean(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("reduction source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("reduction destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("reduction count is guard-proven");
    let mut sum = 0.0f64;
    let mut index = 0;
    while index < count {
        sum += f64::from(src0.read_f32_index(index));
        index += 1;
    }
    dst.write_f32_index(0, (sum / count as f64) as f32);
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn execute_generic_argmax(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("argmax source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("argmax destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("argmax count is guard-proven");
    let mut best_index = 0usize;
    let mut best_value = src0.read_f32_index(0);
    let mut index = 1;
    while index < count {
        let value = src0.read_f32_index(index);
        if value > best_value {
            best_value = value;
            best_index = index;
        }
        index += 1;
    }
    dst.write_f32_index(0, best_index as f32);
}

pub(super) fn execute_generic_matmul(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("matmul lhs is guard-proven");
    let rhs = request.src1.expect("matmul rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("matmul destination is guard-proven");
    let lhs_shape = lhs.layout.ne();
    let rhs_shape = rhs.layout.ne();
    let k = usize::try_from(lhs_shape[0]).expect("matmul k is guard-proven");
    let m = usize::try_from(lhs_shape[1]).expect("matmul m is guard-proven");
    let n = usize::try_from(rhs_shape[0]).expect("matmul n is guard-proven");
    let mut row = 0;
    while row < m {
        let mut column = 0;
        while column < n {
            let mut value = 0.0f32;
            let mut inner = 0;
            while inner < k {
                value = lhs
                    .read_f32_index(row * k + inner)
                    .mul_add(rhs.read_f32_index(inner * n + column), value);
                inner += 1;
            }
            dst.write_f32_index(row * n + column, value);
            column += 1;
        }
        row += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_matmul_argmax(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("argmax lhs is guard-proven");
    let rhs = request.src1.expect("argmax rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("argmax destination is guard-proven");
    let shape = lhs.layout.ne();
    let k = usize::try_from(shape[0]).expect("argmax k is guard-proven");
    let rows = usize::try_from(shape[1]).expect("argmax rows are guard-proven");
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    let mut row = 0;
    while row < rows {
        let mut value = 0.0f32;
        let mut inner = 0;
        while inner < k {
            value = lhs
                .read_f32_index(row * k + inner)
                .mul_add(rhs.read_f32_index(inner), value);
            inner += 1;
        }
        if row == 0 || value > best_value {
            best_value = value;
            best_index = row;
        }
        row += 1;
    }
    dst.write_f32_index(0, best_value);
    request.set_index_out(i32::try_from(best_index).expect("argmax index fits i32"));
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn execute_generic_f16_matmul(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("F16 matmul lhs is guard-proven");
    let rhs = request.src1.expect("F16 matmul rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("F16 matmul destination is guard-proven");
    let lhs_shape = lhs.layout.ne();
    let rhs_shape = rhs.layout.ne();
    let k = usize::try_from(lhs_shape[0]).expect("F16 matmul k is guard-proven");
    let m = usize::try_from(lhs_shape[1]).expect("F16 matmul m is guard-proven");
    let n = usize::try_from(rhs_shape[1]).expect("F16 matmul n is guard-proven");
    let mut column = 0;
    while column < n {
        let mut row = 0;
        while row < m {
            let mut sum = 0.0f64;
            let mut inner = 0;
            while inner < k {
                let lhs_value = super::quant::fp16_to_f32(lhs.read_f16_index(row * k + inner));
                let rhs_value = super::quant::fp16_to_f32(rhs.read_f16_index(column * k + inner));
                sum += f64::from(lhs_value * rhs_value);
                inner += 1;
            }
            dst.write_f32_index(row + column * m, sum as f32);
            row += 1;
        }
        column += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_scalar<const OP: u8>(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("scalar source is guard-proven");
    let scalar = parameter_f32(request, 0).expect("scalar parameter is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("scalar destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("scalar count is guard-proven");
    let mut index = 0;
    while index < count {
        let value = src0.read_f32_index(index);
        let output = match OP {
            0 => value + scalar,
            1 => value * scalar,
            2 => {
                if value < 0.0 {
                    value * scalar
                } else {
                    value
                }
            }
            _ => unreachable!("generic scalar operation is compile-time selected"),
        };
        dst.write_f32_index(index, output);
        index += 1;
    }
}

pub(super) fn execute_generic_clamp(request: &mut GenericRequest<'_>) {
    let minimum = parameter_f32(request, 0).expect("clamp minimum is guard-proven");
    let maximum = parameter_f32(request, 1).expect("clamp maximum is guard-proven");
    let src0 = request.src0.expect("clamp source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("clamp destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("clamp count is guard-proven");
    let mut index = 0;
    while index < count {
        dst.write_f32_index(index, src0.read_f32_index(index).clamp(minimum, maximum));
        index += 1;
    }
}

pub(super) fn execute_generic_silu_back(request: &mut GenericRequest<'_>) {
    let input = request.src0.expect("SiLU input is guard-proven");
    let gradient = request.src1.expect("SiLU gradient is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("SiLU destination is guard-proven");
    let count = input
        .layout
        .element_count()
        .expect("SiLU count is guard-proven");
    let mut index = 0;
    while index < count {
        let value = input.read_f32_index(index);
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        let derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
        dst.write_f32_index(index, gradient.read_f32_index(index) * derivative);
        index += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_cumsum(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("cumsum source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("cumsum destination is guard-proven");
    let width = usize::try_from(src0.layout.ne()[0]).expect("cumsum width fits usize");
    let count = src0
        .layout
        .element_count()
        .expect("cumsum count is guard-proven");
    let rows = count / width;
    let mut row = 0;
    while row < rows {
        let mut sum = 0.0;
        let mut column = 0;
        while column < width {
            sum += src0.read_f32_index(row * width + column);
            dst.write_f32_index(row * width + column, sum);
            column += 1;
        }
        row += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
const fn decode_ordinal(mut ordinal: usize, layout: Layout) -> [u64; 4] {
    let extents = layout.ne();
    let mut coordinates = [0_u64; 4];
    let mut dimension = 0;
    while dimension < 4 {
        let extent = extents[dimension];
        coordinates[dimension] = (ordinal as u64) % extent;
        ordinal = (ordinal as u64 / extent) as usize;
        dimension += 1;
    }
    coordinates
}

#[allow(clippy::cast_possible_truncation)]
const fn encode_ordinal(coordinates: [u64; 4], layout: Layout) -> usize {
    let extents = layout.ne();
    let mut ordinal = 0_u64;
    let mut stride = 1_u64;
    let mut dimension = 0;
    while dimension < 4 {
        ordinal += coordinates[dimension] * stride;
        stride *= extents[dimension];
        dimension += 1;
    }
    ordinal as usize
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_repeat(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("repeat source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("repeat destination is guard-proven");
    let source_shape = src0.layout.ne();
    let count = dst
        .layout
        .element_count()
        .expect("repeat count is guard-proven");
    let mut ordinal = 0;
    while ordinal < count {
        let coordinates = decode_ordinal(ordinal, dst.layout);
        let mut source_coordinates = [0_u64; 4];
        let mut dimension = 0;
        while dimension < 4 {
            source_coordinates[dimension] = coordinates[dimension] % source_shape[dimension];
            dimension += 1;
        }
        let source = encode_ordinal(source_coordinates, src0.layout);
        dst.write_f32_index(ordinal, src0.read_f32_index(source));
        ordinal += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_repeat_back(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("repeat-back source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("repeat-back destination is guard-proven");
    let target_shape = dst.layout.ne();
    let target_count = dst
        .layout
        .element_count()
        .expect("repeat-back count is guard-proven");
    let source_count = src0
        .layout
        .element_count()
        .expect("repeat-back source count is guard-proven");
    let mut ordinal = 0;
    while ordinal < target_count {
        dst.write_f32_index(ordinal, 0.0);
        ordinal += 1;
    }
    ordinal = 0;
    while ordinal < source_count {
        let mut coordinates = decode_ordinal(ordinal, src0.layout);
        let mut dimension = 0;
        while dimension < 4 {
            coordinates[dimension] %= target_shape[dimension];
            dimension += 1;
        }
        let target = encode_ordinal(coordinates, dst.layout);
        let value = dst.read_f32_index(target) + src0.read_f32_index(ordinal);
        dst.write_f32_index(target, value);
        ordinal += 1;
    }
}

pub(super) fn execute_generic_concat<const AXIS: usize>(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("concat lhs is guard-proven");
    let rhs = request.src1.expect("concat rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("concat destination is guard-proven");
    let lhs_count = lhs
        .layout
        .element_count()
        .expect("concat lhs count is guard-proven");
    let rhs_count = rhs
        .layout
        .element_count()
        .expect("concat rhs count is guard-proven");
    let lhs_shape = lhs.layout.ne();
    let mut ordinal = 0;
    while ordinal < lhs_count {
        let coordinates = decode_ordinal(ordinal, lhs.layout);
        let target = encode_ordinal(coordinates, dst.layout);
        dst.write_f32_index(target, lhs.read_f32_index(ordinal));
        ordinal += 1;
    }
    ordinal = 0;
    while ordinal < rhs_count {
        let mut coordinates = decode_ordinal(ordinal, rhs.layout);
        coordinates[AXIS] += lhs_shape[AXIS];
        let target = encode_ordinal(coordinates, dst.layout);
        dst.write_f32_index(target, rhs.read_f32_index(ordinal));
        ordinal += 1;
    }
}

pub(super) fn execute_generic_fill(request: &mut GenericRequest<'_>) {
    let value = parameter_f32(request, 0).expect("fill parameter is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("fill destination is guard-proven");
    let count = dst
        .layout
        .element_count()
        .expect("fill count is guard-proven");
    let mut ordinal = 0;
    while ordinal < count {
        dst.write_f32_index(ordinal, value);
        ordinal += 1;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn execute_generic_arange(request: &mut GenericRequest<'_>) {
    let start = parameter_f32(request, 0).expect("arange start is guard-proven");
    let step = parameter_f32(request, 1).expect("arange step is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("arange destination is guard-proven");
    let count = dst
        .layout
        .element_count()
        .expect("arange count is guard-proven");
    let mut ordinal = 0;
    while ordinal < count {
        dst.write_f32_index(ordinal, (ordinal as f32).mul_add(step, start));
        ordinal += 1;
    }
}

pub(super) fn execute_generic_diag(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("diag source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("diag destination is guard-proven");
    let side = usize::try_from(src0.layout.ne()[0]).expect("diag side fits usize");
    let mut row = 0;
    while row < side {
        let mut column = 0;
        while column < side {
            let value = if row == column {
                src0.read_f32_index(row)
            } else {
                0.0
            };
            dst.write_f32_index(row * side + column, value);
            column += 1;
        }
        row += 1;
    }
}

pub(super) fn execute_generic_diag_mask<const INF: bool>(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("diag mask source is guard-proven");
    let n_past =
        usize::try_from(parameter_i32(request, 0).expect("mask parameter is guard-proven"))
            .expect("mask parameter is non-negative");
    let dst = request
        .dst
        .as_mut()
        .expect("diag mask destination is guard-proven");
    let side = usize::try_from(src0.layout.ne()[0]).expect("diag mask side fits usize");
    let mut row = 0;
    while row < side {
        let mut column = 0;
        while column < side {
            let index = row * side + column;
            let value = if column > row + n_past {
                if INF { f32::NEG_INFINITY } else { 0.0 }
            } else {
                src0.read_f32_index(index)
            };
            dst.write_f32_index(index, value);
            column += 1;
        }
        row += 1;
    }
}

pub(super) fn execute_generic_l2_norm(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("L2 source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("L2 destination is guard-proven");
    let count = src0
        .layout
        .element_count()
        .expect("L2 count is guard-proven");
    let mut sum = 0.0f32;
    let mut index = 0;
    while index < count {
        let value = src0.read_f32_index(index);
        sum += value * value;
        index += 1;
    }
    let scale = 1.0 / sum.sqrt();
    index = 0;
    while index < count {
        dst.write_f32_index(index, src0.read_f32_index(index) * scale);
        index += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn execute_generic_softmax_back(request: &mut GenericRequest<'_>) {
    let values = request.src0.expect("softmax values are guard-proven");
    let gradient = request.src1.expect("softmax gradient is guard-proven");
    let row = usize::try_from(parameter_i32(request, 0).expect("softmax row is guard-proven"))
        .expect("softmax row is positive");
    let dst = request
        .dst
        .as_mut()
        .expect("softmax destination is guard-proven");
    let count = values
        .layout
        .element_count()
        .expect("softmax count is guard-proven");
    let rows = count / row;
    let mut current = 0;
    while current < rows {
        let start = current * row;
        let mut dot = 0.0;
        let mut column = 0;
        while column < row {
            dot = values
                .read_f32_index(start + column)
                .mul_add(gradient.read_f32_index(start + column), dot);
            column += 1;
        }
        column = 0;
        while column < row {
            let value = values.read_f32_index(start + column);
            let grad = gradient.read_f32_index(start + column);
            dst.write_f32_index(start + column, value * (grad - dot));
            column += 1;
        }
        current += 1;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn execute_generic_rms_norm_back(request: &mut GenericRequest<'_>) {
    let input = request.src0.expect("RMS input is guard-proven");
    let gradient = request.src1.expect("RMS gradient is guard-proven");
    let epsilon = parameter_f32(request, 0).expect("RMS epsilon is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("RMS destination is guard-proven");
    let row = usize::try_from(input.layout.ne()[0]).expect("RMS row fits usize");
    let count = input
        .layout
        .element_count()
        .expect("RMS count is guard-proven");
    let rows = count / row;
    let mut current = 0;
    while current < rows {
        let start = current * row;
        let mut sum = 0.0f64;
        let mut column = 0;
        while column < row {
            let value = input.read_f32_index(start + column);
            sum += f64::from(value * value);
            column += 1;
        }
        let inv = 1.0 / (sum / row as f64 + f64::from(epsilon)).sqrt();
        let mut dot = 0.0f64;
        column = 0;
        while column < row {
            dot += f64::from(
                gradient.read_f32_index(start + column) * input.read_f32_index(start + column),
            );
            column += 1;
        }
        column = 0;
        while column < row {
            let x = f64::from(input.read_f32_index(start + column));
            let dy = f64::from(gradient.read_f32_index(start + column));
            let value = dy * inv - x * inv * inv * inv * dot / row as f64;
            dst.write_f32_index(start + column, value as f32);
            column += 1;
        }
        current += 1;
    }
}

pub(super) fn execute_generic_count_equal(request: &mut GenericRequest<'_>) {
    let lhs = request.src0.expect("count-equal lhs is guard-proven");
    let rhs = request.src1.expect("count-equal rhs is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("count-equal destination is guard-proven");
    let count = lhs
        .layout
        .element_count()
        .expect("count-equal count is guard-proven");
    let mut equal = 0.0f32;
    let mut index = 0;
    while index < count {
        if lhs.read_f32_index(index).to_bits() == rhs.read_f32_index(index).to_bits() {
            equal += 1.0;
        }
        index += 1;
    }
    dst.write_f32_index(0, equal);
}

pub(super) fn execute_generic_soft_max(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("softmax source is guard-proven");
    let dst = request
        .dst
        .as_mut()
        .expect("softmax destination is guard-proven");
    let shape = src0.layout.ne();
    let width = usize::try_from(shape[0]).expect("softmax width is guard-proven");
    let rows = src0
        .layout
        .element_count()
        .expect("softmax count is guard-proven")
        / width;
    let mut row = 0;
    while row < rows {
        let mut maximum = f32::NEG_INFINITY;
        let mut column = 0;
        while column < width {
            maximum = maximum.max(src0.read_f32_index(row * width + column));
            column += 1;
        }
        let mut sum = 0.0f32;
        column = 0;
        while column < width {
            let value = (src0.read_f32_index(row * width + column) - maximum).exp();
            dst.write_f32_index(row * width + column, value);
            sum += value;
            column += 1;
        }
        column = 0;
        while column < width {
            let value = dst.read_f32_index(row * width + column);
            dst.write_f32_index(row * width + column, value / sum);
            column += 1;
        }
        row += 1;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(super) fn execute_generic_norm<const RMS: bool>(request: &mut GenericRequest<'_>) {
    let src0 = request.src0.expect("norm source is guard-proven");
    let params = request.params();
    let epsilon_bytes = params.as_slice();
    let epsilon = f32::from_ne_bytes([
        epsilon_bytes[0],
        epsilon_bytes[1],
        epsilon_bytes[2],
        epsilon_bytes[3],
    ]);
    let dst = request
        .dst
        .as_mut()
        .expect("norm destination is guard-proven");
    let shape = src0.layout.ne();
    let width = usize::try_from(shape[0]).expect("norm width is guard-proven");
    let rows = src0
        .layout
        .element_count()
        .expect("norm count is guard-proven")
        / width;
    let mut row = 0;
    while row < rows {
        let mut sum = 0.0f64;
        let mut column = 0;
        while column < width {
            let value = src0.read_f32_index(row * width + column);
            sum += if RMS {
                f64::from(value * value)
            } else {
                f64::from(value)
            };
            column += 1;
        }
        let mean = (sum / width as f64) as f32;
        let mut variance = 0.0f64;
        if !RMS {
            column = 0;
            while column < width {
                let centered = src0.read_f32_index(row * width + column) - mean;
                variance += f64::from(centered * centered);
                column += 1;
            }
            variance /= width as f64;
        }
        let scale = (1.0
            / (if RMS { f64::from(mean) } else { variance } + f64::from(epsilon)).sqrt())
            as f32;
        column = 0;
        while column < width {
            let value = src0.read_f32_index(row * width + column);
            dst.write_f32_index(
                row * width + column,
                if RMS {
                    value * scale
                } else {
                    (value - mean) * scale
                },
            );
            column += 1;
        }
        row += 1;
    }
}
