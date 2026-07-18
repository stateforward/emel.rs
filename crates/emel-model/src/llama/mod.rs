//! Llama-family generation actor.
//!
//! This module ports the family-owned behavior in pinned
//! `emel.cpp/src/emel/model/llama/any.hpp`, `detail.hpp`, and `detail.cpp`.
//! Tensor lookup and capability resolution remain owned by the public common
//! generation builder; this actor selects only the source-fixed Llama route.

mod actor;
pub mod event;
mod hparams;
mod sm;

pub use actor::Llama;
pub use event::Parameters;
pub use hparams::load_hparams;

pub use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, QuantizedContractKind, QuantizedStageFamily, ResidualRoute,
    StageAudit, StepKind, StepPlan, TopologyDescriptor, tensor_type_name,
};

/// Pinned global tensor count used by Llama topology construction.
pub const GLOBAL_TENSOR_COUNT: u32 = 3;
/// Pinned per-block topology tensor count.
pub const BLOCK_TENSOR_COUNT: u32 = 8;
/// Source-fixed token embedding tensor name.
pub const TOKEN_EMBEDDING_NAME: &[u8] = b"token_embd.weight";
/// Source-fixed output normalization tensor name.
pub const OUTPUT_NORM_NAME: &[u8] = b"output_norm.weight";
/// Source architecture name accepted by the family actor.
pub const ARCHITECTURE_NAME: &[u8] = b"llama";

#[cfg(test)]
mod tests;
