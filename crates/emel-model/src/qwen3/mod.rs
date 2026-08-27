//! Qwen3-family generation actor.
//!
//! This module ports every family-owned behavior in pinned
//! `emel.cpp/src/emel/model/qwen3/detail.hpp` and `detail.cpp`. Tensor lookup,
//! tied-output fallback, Q/K-normalized attention binding, capability
//! resolution, planning, and audit are composed through the public common
//! generation builder.

mod actor;
pub mod event;
mod hparams;
mod sm;

pub use actor::Qwen3;
pub use event::Parameters;
pub use hparams::load_hparams;

pub use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, QuantizedContractKind, QuantizedStageFamily, ResidualRoute,
    StageAudit, StepKind, StepPlan, TopologyDescriptor, tensor_type_name,
};

/// Pinned global tensor count used by Qwen3 topology construction.
pub const GLOBAL_TENSOR_COUNT: u32 = 3;
/// Pinned per-block topology tensor count including Q/K normalization.
pub const BLOCK_TENSOR_COUNT: u32 = 10;
/// Source-fixed token embedding tensor name.
pub const TOKEN_EMBEDDING_NAME: &[u8] = b"token_embd.weight";
/// Source-fixed output normalization tensor name.
pub const OUTPUT_NORM_NAME: &[u8] = b"output_norm.weight";
/// Source architecture name accepted by the family actor.
pub const ARCHITECTURE_NAME: &[u8] = b"qwen3";

#[cfg(test)]
mod tests;
