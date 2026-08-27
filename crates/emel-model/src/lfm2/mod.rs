//! LFM2-family generation actor.
//!
//! This module ports every family-owned behavior in pinned
//! `emel.cpp/src/emel/model/lfm2/detail.hpp` and `detail.cpp`. Tensor lookup,
//! opposite-family exclusion, capability resolution, planning, and audit remain
//! owned by the public common generation builder.

mod actor;
pub mod event;
mod hparams;
mod sm;

pub use actor::Lfm2;
pub use event::{Parameters, Variant};
pub use hparams::load_hparams;

pub use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, QuantizedContractKind, QuantizedStageFamily, ResidualRoute,
    StageAudit, StepKind, StepPlan, TopologyDescriptor, tensor_type_name,
};

/// Maximum source-compatible per-layer KV-head pattern length.
pub const MAX_BLOCKS: usize = 512;
/// Pinned global tensor count used by LFM2 topology construction.
pub const GLOBAL_TENSOR_COUNT: u32 = 3;
/// Pinned attention-block topology tensor count.
pub const ATTENTION_BLOCK_TENSOR_COUNT: u32 = 11;
/// Pinned short-convolution-block topology tensor count.
pub const SHORTCONV_BLOCK_TENSOR_COUNT: u32 = 8;
/// Source-fixed token embedding tensor name.
pub const TOKEN_EMBEDDING_NAME: &[u8] = b"token_embd.weight";
/// Source-fixed output normalization tensor name.
pub const OUTPUT_NORM_NAME: &[u8] = b"token_embd_norm.weight";
/// Source architecture name accepted by the family actor.
pub const ARCHITECTURE_NAME: &[u8] = b"lfm2";

#[cfg(test)]
mod tests;
