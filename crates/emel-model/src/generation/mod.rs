//! Generation graph surfaces from `emel.cpp/src/emel/model/generation/any.hpp`
//! and `any.cpp`.

use crate::data::{ModelData, TensorRecord, Data};
use crate::{
    GenerationAttentionQkNormRoute, GenerationAttentionVNormRoute,
    GenerationAttentionValueRoute, GenerationAttentionWindowRoute, GenerationResidualRoute,
};

/// Count of quantized stage families (C++ `k_quantized_stage_family_count`).
pub const K_QUANTIZED_STAGE_FAMILY_COUNT: u32 = 14;

/// Bound tensor view (C++ `generation::tensor_view`).
#[derive(Clone, Debug, Default)]
pub struct TensorView {
    /// Optional tensor record pointer shell (owned index in Rust).
    pub tensor_index: Option<usize>,
    /// Tensor name.
    pub name: String,
}

impl TensorView {
    /// Build from a model tensor record reference.
    #[must_use]
    pub fn from_record(index: usize, record: &TensorRecord, name: impl Into<String>) -> Self {
        let _ = record;
        Self {
            tensor_index: Some(index),
            name: name.into(),
        }
    }
}

/// Per-block weight view (C++ `generation::block_view`).
#[derive(Clone, Debug, Default)]
pub struct BlockView {
    /// Block index.
    pub index: i32,
    /// Whether this block uses attention.
    pub uses_attention: bool,
    /// Attention norm tensor.
    pub attention_norm: TensorView,
    /// Attention Q.
    pub attention_q: TensorView,
    /// Attention K.
    pub attention_k: TensorView,
    /// Attention V.
    pub attention_v: TensorView,
    /// Attention Q norm.
    pub attention_q_norm: TensorView,
    /// Attention K norm.
    pub attention_k_norm: TensorView,
    /// Attention output.
    pub attention_output: TensorView,
    /// Short-conv kernel.
    pub shortconv_conv: TensorView,
    /// Short-conv input projection.
    pub shortconv_in_proj: TensorView,
    /// Short-conv output projection.
    pub shortconv_out_proj: TensorView,
    /// FFN norm.
    pub feed_forward_norm: TensorView,
    /// FFN gate.
    pub feed_forward_gate: TensorView,
    /// FFN down.
    pub feed_forward_down: TensorView,
    /// FFN up.
    pub feed_forward_up: TensorView,
}

/// Full execution view over model weights (C++ `generation::execution_view`).
#[derive(Clone, Debug, Default)]
pub struct ExecutionView {
    /// Max blocks (C++ `k_max_blocks` ≈ `data::k_max_metadata_arrays`).
    pub block_count: i32,
    /// Token embedding.
    pub token_embedding: TensorView,
    /// Output norm.
    pub output_norm: TensorView,
    /// Output projection.
    pub output: TensorView,
    /// Per-block views.
    pub blocks: Vec<BlockView>,
}

impl ExecutionView {
    /// Max blocks capacity constant.
    pub const K_MAX_BLOCKS: u32 = Data::K_MAX_METADATA_ARRAYS as u32;
}

/// Per-layer generation execution routing (C++ `generation_layer_execution`).
#[derive(Clone, Debug, Default)]
pub struct GenerationLayerExecution {
    /// Residual route.
    pub residual_route: GenerationResidualRoute,
    /// QK norm route.
    pub qk_norm_route: GenerationAttentionQkNormRoute,
    /// Value route.
    pub value_route: GenerationAttentionValueRoute,
    /// V norm route.
    pub v_norm_route: GenerationAttentionVNormRoute,
    /// Window route.
    pub window_route: GenerationAttentionWindowRoute,
    /// Attention key length.
    pub attention_key_length: i32,
    /// Attention value length.
    pub attention_value_length: i32,
    /// RoPE dimension.
    pub attention_rope_dim: i32,
    /// RoPE frequency base.
    pub attention_rope_freq_base: f32,
}

/// Generation execution descriptor (C++ `generation_execution_descriptor`).
#[derive(Clone, Debug, Default)]
pub struct GenerationExecutionDescriptor {
    /// Layer count.
    pub layer_count: u32,
    /// Per-layer descriptors (capacity: `K_MAX_LAYERS`).
    pub layers: Vec<GenerationLayerExecution>,
}

impl GenerationExecutionDescriptor {
    /// Max layers (C++ `k_max_layers`).
    pub const K_MAX_LAYERS: u32 = Data::K_MAX_METADATA_ARRAYS as u32;
}

/// Topology summary (C++ `generation::topology`).
#[derive(Clone, Debug, Default)]
pub struct Topology {
    /// Node count.
    pub node_count: u32,
    /// Tensor count.
    pub tensor_count: u32,
    /// Bytes per tensor estimate.
    pub bytes_per_tensor: u64,
    /// Workspace capacity.
    pub workspace_capacity_bytes: u64,
}

/// Step kind (C++ `generation::step_kind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum StepKind {
    /// Prefill step.
    #[default]
    Prefill = 0,
    /// Decode step.
    Decode = 1,
}

/// Step plan (C++ `generation::step_plan`).
#[derive(Clone, Debug, Default)]
pub struct StepPlan {
    /// Prefill or decode.
    pub kind: StepKind,
    /// Node count.
    pub node_count: u32,
    /// Tensor count.
    pub tensor_count: u32,
    /// Expected outputs.
    pub expected_outputs: i32,
    /// Max tokens this step.
    pub max_step_tokens: i32,
}

/// Quantized stage family (C++ `quantized_stage_family`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum QuantizedStageFamily {
    /// Token embedding stage.
    #[default]
    TokenEmbedding = 0,
    /// Output norm.
    OutputNorm,
    /// Output.
    Output,
    /// Attention norm.
    AttentionNorm,
    /// Attention Q.
    AttentionQ,
    /// Attention K.
    AttentionK,
    /// Attention V.
    AttentionV,
    /// Attention Q norm.
    AttentionQNorm,
    /// Attention K norm.
    AttentionKNorm,
    /// Attention output.
    AttentionOutput,
    /// FFN norm.
    FeedForwardNorm,
    /// FFN gate.
    FeedForwardGate,
    /// FFN down.
    FeedForwardDown,
    /// FFN up.
    FeedForwardUp,
}

/// Quantized contract kind (C++ `quantized_contract_kind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum QuantizedContractKind {
    /// Native quantized path.
    NativeQuantized = 0,
    /// Approved dense f32 by contract.
    #[default]
    ApprovedDenseF32ByContract,
    /// Disallowed fallback.
    DisallowedFallback,
    /// Explicit no claim.
    ExplicitNoClaim,
    /// Not applicable.
    NotApplicable,
}

/// Per-stage quantized audit (C++ `quantized_stage_audit`).
#[derive(Clone, Debug, Default)]
pub struct QuantizedStageAudit {
    /// Stage family.
    pub family: QuantizedStageFamily,
    /// Tensor type id.
    pub tensor_type: i32,
    /// Contract classification.
    pub contract: QuantizedContractKind,
    /// Consistency across layers.
    pub consistent_across_layers: bool,
}

/// Full quantized path audit (C++ `quantized_path_audit`).
#[derive(Clone, Debug, Default)]
pub struct QuantizedPathAudit {
    /// Per-family stages.
    pub stages: Vec<QuantizedStageAudit>,
}

/// Generation contract aggregate (C++ `generation::contract`).
#[derive(Clone, Debug, Default)]
pub struct Contract {
    /// Bound execution view.
    pub execution: ExecutionView,
    /// Generation execution descriptor.
    pub generation_execution: GenerationExecutionDescriptor,
    /// Topology.
    pub topology: Topology,
    /// Prefill plan.
    pub prefill_plan: StepPlan,
    /// Decode plan.
    pub decode_plan: StepPlan,
    /// Quantized path audit.
    pub quantized_audit: QuantizedPathAudit,
}

impl Contract {
    /// Reset all fields (C++ `contract::reset`).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Build a generation contract from model data.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp
/// (`build_contract` declared in any.hpp).
pub fn build_contract(_model_data: &ModelData, _contract_out: &mut Contract) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::build_contract
    todo!(
        "TODO: port build_contract from emel.cpp/src/emel/model/generation/any.cpp \
         (any.hpp)"
    )
}

/// Whether model data contains a tensor with the given name.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`has_tensor_named`).
pub fn has_tensor_named(_model_data: &ModelData, _name: &str) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::has_tensor_named
    todo!("TODO: port has_tensor_named from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Bind a named tensor into a view.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`bind_tensor_view`).
pub fn bind_tensor_view(
    _model_data: &ModelData,
    _name: &str,
    _view_out: &mut TensorView,
) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::bind_tensor_view
    todo!("TODO: port bind_tensor_view from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Bind the output projection view (with optional tied embedding).
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`bind_output_view`).
pub fn bind_output_view(
    _model_data: &ModelData,
    _token_embedding: &TensorView,
    _allow_tied_output: bool,
    _output_out: &mut TensorView,
) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::bind_output_view
    todo!("TODO: port bind_output_view from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Bind a block-local tensor by suffix.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`bind_block_tensor_view`).
pub fn bind_block_tensor_view(
    _model_data: &ModelData,
    _block_index: i32,
    _suffix: &str,
    _view_out: &mut TensorView,
) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::bind_block_tensor_view
    todo!("TODO: port bind_block_tensor_view from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Require that a block tensor exists.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`require_block_tensor`).
pub fn require_block_tensor(_model_data: &ModelData, _block_index: i32, _suffix: &str) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::require_block_tensor
    todo!("TODO: port require_block_tensor from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Reject presence of a block tensor.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`reject_block_tensor`).
pub fn reject_block_tensor(_model_data: &ModelData, _block_index: i32, _suffix: &str) -> bool {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::reject_block_tensor
    todo!("TODO: port reject_block_tensor from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Bind an attention block view.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`bind_attention_block`).
pub fn bind_attention_block(
    _model_data: &ModelData,
    _block_index: i32,
    _require_qk_norm: bool,
    _use_shared_key_value: bool,
    _block_out: &mut BlockView,
) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::bind_attention_block
    todo!("TODO: port bind_attention_block from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Bind a short-conv block view.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`bind_shortconv_block`).
pub fn bind_shortconv_block(
    _model_data: &ModelData,
    _block_index: i32,
    _block_out: &mut BlockView,
) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::bind_shortconv_block
    todo!("TODO: port bind_shortconv_block from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Lookup a block view by index.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`lookup_block_view`).
pub fn lookup_block_view(
    _execution: &ExecutionView,
    _block_index: i32,
    _block_out: &mut BlockView,
) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::lookup_block_view
    todo!("TODO: port lookup_block_view from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Build prefill and decode step plans from topology.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`build_step_plans`).
pub fn build_step_plans(
    _topology_in: &Topology,
    _prefill_out: &mut StepPlan,
    _decode_out: &mut StepPlan,
) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::build_step_plans
    todo!("TODO: port build_step_plans from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Validate a filled contract.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`validate_contract`).
pub fn validate_contract(_contract_in: &Contract) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::validate_contract
    todo!("TODO: port validate_contract from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Complete a partially filled contract.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`complete_contract`).
pub fn complete_contract(_contract_out: &mut Contract) -> Result<(), i32> {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::complete_contract
    todo!("TODO: port complete_contract from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Build quantized path audit for an execution view.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`build_quantized_path_audit`).
#[must_use]
pub fn build_quantized_path_audit(_execution: &ExecutionView) -> QuantizedPathAudit {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::build_quantized_path_audit
    todo!("TODO: port build_quantized_path_audit from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Name for a quantized stage family.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`quantized_stage_family_name`).
#[must_use]
pub fn quantized_stage_family_name(_family: QuantizedStageFamily) -> &'static str {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::quantized_stage_family_name
    todo!("TODO: port quantized_stage_family_name from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Name for a quantized contract kind.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`quantized_contract_kind_name`).
#[must_use]
pub fn quantized_contract_kind_name(_kind: QuantizedContractKind) -> &'static str {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::quantized_contract_kind_name
    todo!("TODO: port quantized_contract_kind_name from emel.cpp/src/emel/model/generation/any.cpp")
}

/// Name for a tensor type id.
///
/// TODO: convert from emel.cpp/src/emel/model/generation/any.cpp (`tensor_type_name`).
#[must_use]
pub fn tensor_type_name(_tensor_type: i32) -> &'static str {
    // TODO: convert from emel.cpp/src/emel/model/generation/any.cpp /
    // emel.cpp/src/emel/model/generation/any.hpp::tensor_type_name
    todo!("TODO: port tensor_type_name from emel.cpp/src/emel/model/generation/any.cpp")
}
