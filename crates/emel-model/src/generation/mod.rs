//! Allocation-free common generation-contract actor.
//!
//! This module ports the architecture-neutral behavior in pinned
//! `emel.cpp/src/emel/model/generation/any.hpp` and `any.cpp`. Architecture
//! selection remains at the architecture-router boundary; callers submit the
//! already-selected layer contract one block per dispatch.
//!
//! ```compile_fail
//! use emel_model::generation::storage::BlockSlot;
//! ```
//!
//! ```compile_fail
//! let catalog = emel_model::catalog::Catalog::try_new().unwrap();
//! let capability = emel_kernels::capability::Resolver::new();
//! let builder = emel_model::generation::Builder::new(catalog, capability);
//! let _ = builder.state();
//! ```

mod actor;
pub mod event;
mod sm;
mod storage;

pub use actor::Builder;

use emel_tensor::dtype::SerializedType;

use crate::catalog::event::{ModelIdentity, TensorId};

/// Exact number of pinned quantized stage families.
pub const QUANTIZED_STAGE_FAMILY_COUNT: usize = 14;

/// Per-layer residual path selected by an owning family actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ResidualRoute {
    #[default]
    Attention = 0,
    Shortconv = 1,
}

/// Per-layer Q/K normalization contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionQkNormRoute {
    #[default]
    None = 0,
    HeadwiseRms = 1,
}

/// Per-layer value projection contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionValueRoute {
    #[default]
    DedicatedValue = 0,
    SharedKeyValue = 1,
}

/// Per-layer value normalization contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionVNormRoute {
    #[default]
    None = 0,
    Rms = 1,
}

/// Per-layer attention-window contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AttentionWindowRoute {
    #[default]
    FullContext = 0,
    SlidingWindow = 1,
}

/// Source-exact generation execution facts for one layer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerExecution {
    residual_route: ResidualRoute,
    qk_norm_route: AttentionQkNormRoute,
    value_route: AttentionValueRoute,
    v_norm_route: AttentionVNormRoute,
    window_route: AttentionWindowRoute,
    attention_key_length: i32,
    attention_value_length: i32,
    attention_rope_dim: i32,
    attention_rope_freq_base: f32,
}

impl LayerExecution {
    /// Constructs one already-selected layer contract.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        residual_route: ResidualRoute,
        qk_norm_route: AttentionQkNormRoute,
        value_route: AttentionValueRoute,
        v_norm_route: AttentionVNormRoute,
        window_route: AttentionWindowRoute,
        attention_key_length: i32,
        attention_value_length: i32,
        attention_rope_dim: i32,
        attention_rope_freq_base: f32,
    ) -> Self {
        Self {
            residual_route,
            qk_norm_route,
            value_route,
            v_norm_route,
            window_route,
            attention_key_length,
            attention_value_length,
            attention_rope_dim,
            attention_rope_freq_base,
        }
    }

    #[must_use]
    pub const fn residual_route(self) -> ResidualRoute {
        self.residual_route
    }
    #[must_use]
    pub const fn qk_norm_route(self) -> AttentionQkNormRoute {
        self.qk_norm_route
    }
    #[must_use]
    pub const fn value_route(self) -> AttentionValueRoute {
        self.value_route
    }
    #[must_use]
    pub const fn v_norm_route(self) -> AttentionVNormRoute {
        self.v_norm_route
    }
    #[must_use]
    pub const fn window_route(self) -> AttentionWindowRoute {
        self.window_route
    }
    #[must_use]
    pub const fn attention_key_length(self) -> i32 {
        self.attention_key_length
    }
    #[must_use]
    pub const fn attention_value_length(self) -> i32 {
        self.attention_value_length
    }
    #[must_use]
    pub const fn attention_rope_dim(self) -> i32 {
        self.attention_rope_dim
    }
    #[must_use]
    pub const fn attention_rope_freq_base(self) -> f32 {
        self.attention_rope_freq_base
    }
}

/// Prefill or decode plan kind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum StepKind {
    #[default]
    Prefill = 0,
    Decode = 1,
}

/// Immutable source-exact step plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StepPlan {
    kind: StepKind,
    node_count: u32,
    tensor_count: u32,
    expected_outputs: i32,
    max_step_tokens: i32,
}

impl StepPlan {
    #[must_use]
    pub const fn kind(self) -> StepKind {
        self.kind
    }
    #[must_use]
    pub const fn node_count(self) -> u32 {
        self.node_count
    }
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }
    #[must_use]
    pub const fn expected_outputs(self) -> i32 {
        self.expected_outputs
    }
    #[must_use]
    pub const fn max_step_tokens(self) -> i32 {
        self.max_step_tokens
    }
}

/// One of the fourteen source audit families.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum QuantizedStageFamily {
    #[default]
    TokenEmbedding = 0,
    OutputNorm,
    Output,
    AttentionNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionQNorm,
    AttentionKNorm,
    AttentionOutput,
    FeedForwardNorm,
    FeedForwardGate,
    FeedForwardDown,
    FeedForwardUp,
}

impl QuantizedStageFamily {
    /// All source stage families in source order.
    pub const ALL: [Self; QUANTIZED_STAGE_FAMILY_COUNT] = [
        Self::TokenEmbedding,
        Self::OutputNorm,
        Self::Output,
        Self::AttentionNorm,
        Self::AttentionQ,
        Self::AttentionK,
        Self::AttentionV,
        Self::AttentionQNorm,
        Self::AttentionKNorm,
        Self::AttentionOutput,
        Self::FeedForwardNorm,
        Self::FeedForwardGate,
        Self::FeedForwardDown,
        Self::FeedForwardUp,
    ];

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// Returns the pinned source label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::OutputNorm => "output_norm",
            Self::Output => "output",
            Self::AttentionNorm => "attention_norm",
            Self::AttentionQ => "attention_q",
            Self::AttentionK => "attention_k",
            Self::AttentionV => "attention_v",
            Self::AttentionQNorm => "attention_q_norm",
            Self::AttentionKNorm => "attention_k_norm",
            Self::AttentionOutput => "attention_output",
            Self::FeedForwardNorm => "feed_forward_norm",
            Self::FeedForwardGate => "feed_forward_gate",
            Self::FeedForwardDown => "feed_forward_down",
            Self::FeedForwardUp => "feed_forward_up",
        }
    }
}

/// Source-exact quantized contract classification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum QuantizedContractKind {
    NativeQuantized = 0,
    #[default]
    ApprovedDenseF32ByContract,
    DisallowedFallback,
    ExplicitNoClaim,
    NotApplicable,
}

impl QuantizedContractKind {
    /// Returns the pinned source label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NativeQuantized => "native_quantized",
            Self::ApprovedDenseF32ByContract => "approved_dense_f32_by_contract",
            Self::DisallowedFallback => "disallowed_fallback",
            Self::ExplicitNoClaim => "explicit_no_claim",
            Self::NotApplicable => "not_applicable",
        }
    }
}

/// Final audit for one stage family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageAudit {
    family: QuantizedStageFamily,
    tensor_type: Option<SerializedType>,
    contract: QuantizedContractKind,
    consistent_across_layers: bool,
}

impl StageAudit {
    pub(crate) const fn empty(family: QuantizedStageFamily) -> Self {
        Self {
            family,
            tensor_type: None,
            contract: QuantizedContractKind::NotApplicable,
            consistent_across_layers: true,
        }
    }
    #[must_use]
    pub const fn family(self) -> QuantizedStageFamily {
        self.family
    }
    #[must_use]
    pub const fn tensor_type(self) -> Option<SerializedType> {
        self.tensor_type
    }
    #[must_use]
    pub const fn contract(self) -> QuantizedContractKind {
        self.contract
    }
    #[must_use]
    pub const fn consistent_across_layers(self) -> bool {
        self.consistent_across_layers
    }
}

/// Immutable topology facts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TopologyDescriptor {
    node_count: u32,
    tensor_count: u32,
    bytes_per_tensor: u64,
    workspace_capacity_bytes: u64,
}

impl TopologyDescriptor {
    #[must_use]
    pub const fn node_count(self) -> u32 {
        self.node_count
    }
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }
    #[must_use]
    pub const fn bytes_per_tensor(self) -> u64 {
        self.bytes_per_tensor
    }
    #[must_use]
    pub const fn workspace_capacity_bytes(self) -> u64 {
        self.workspace_capacity_bytes
    }
}

/// Immutable completed contract descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractDescriptor {
    model: ModelIdentity,
    token_embedding: TensorId,
    output_norm: TensorId,
    output: TensorId,
    block_count: u32,
    topology: TopologyDescriptor,
    prefill: StepPlan,
    decode: StepPlan,
    audit: [StageAudit; QUANTIZED_STAGE_FAMILY_COUNT],
}

impl ContractDescriptor {
    #[must_use]
    pub const fn model(self) -> ModelIdentity {
        self.model
    }
    #[must_use]
    pub const fn token_embedding(self) -> TensorId {
        self.token_embedding
    }
    #[must_use]
    pub const fn output_norm(self) -> TensorId {
        self.output_norm
    }
    #[must_use]
    pub const fn output(self) -> TensorId {
        self.output
    }
    #[must_use]
    pub const fn block_count(self) -> u32 {
        self.block_count
    }
    #[must_use]
    pub const fn topology(self) -> TopologyDescriptor {
        self.topology
    }
    #[must_use]
    pub const fn prefill_plan(self) -> StepPlan {
        self.prefill
    }
    #[must_use]
    pub const fn decode_plan(self) -> StepPlan {
        self.decode
    }
    #[must_use]
    pub const fn audit(self) -> [StageAudit; QUANTIZED_STAGE_FAMILY_COUNT] {
        self.audit
    }
}

/// Returns the pinned source label for a serialized tensor type.
#[must_use]
pub const fn tensor_type_name(tensor_type: SerializedType) -> &'static str {
    match tensor_type {
        SerializedType::F32 => "f32",
        SerializedType::Q2K => "q2_k",
        SerializedType::Q3K => "q3_k",
        SerializedType::Q4K => "q4_k",
        SerializedType::Q6K => "q6_k",
        SerializedType::Q4_0 => "q4_0",
        _ => "unknown",
    }
}

use actor::{
    AttentionRuntime, AuditRuntime, BeginRuntime, BlockVisitRuntime, GlobalRuntime, PlanRuntime,
    RejectRuntime, ResetRuntime, ShortconvRuntime, StageRuntime, StorageBindRuntime,
    StorageReleaseRuntime, TopologyRuntime, UnexpectedRuntime, ValidateRuntime, VisitRuntime,
};

#[cfg(test)]
mod tests;
