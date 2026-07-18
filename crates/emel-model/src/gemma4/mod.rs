//! Gemma4-family generation actor.
//!
//! This module ports every family-owned behavior in pinned
//! `emel.cpp/src/emel/model/gemma4/detail.hpp` and `detail.cpp`. Tensor lookup,
//! tied-output fallback, shared-value binding, attention-window selection,
//! capability resolution, planning, and audit are composed through the public
//! common generation builder.

mod actor;
pub mod event;
mod hparams;
mod sm;

pub use actor::Gemma4;
pub use event::Parameters;
pub use hparams::load_hparams;

pub use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    ContractDescriptor, LayerExecution, QuantizedContractKind, QuantizedStageFamily, ResidualRoute,
    StageAudit, StepKind, StepPlan, TopologyDescriptor, tensor_type_name,
};

pub const MAX_BLOCKS: usize = 512;
pub const ARCHITECTURE_NAME: &[u8] = b"gemma4";
pub const TOKEN_EMBEDDING_NAME: &[u8] = b"token_embd.weight";
pub const OUTPUT_NORM_NAME: &[u8] = b"output_norm.weight";
pub const GLOBAL_TENSOR_COUNT: u32 = 3;
pub const SHARED_KV_BLOCK_TENSOR_COUNT: u32 = 9;
pub const DEDICATED_KV_BLOCK_TENSOR_COUNT: u32 = 10;

pub const BLOCK_COUNT: i32 = 35;
pub const CONTEXT_LENGTH: i32 = 131_072;
pub const EMBEDDING_LENGTH: i32 = 1_536;
pub const EMBEDDING_LENGTH_PER_LAYER_INPUT: i32 = 256;
pub const FEED_FORWARD_LENGTH: i32 = 6_144;
pub const HEAD_COUNT: i32 = 8;
pub const HEAD_COUNT_KV: i32 = 1;
pub const KEY_LENGTH: i32 = 512;
pub const KEY_LENGTH_SWA: i32 = 256;
pub const VALUE_LENGTH: i32 = 512;
pub const VALUE_LENGTH_SWA: i32 = 256;
pub const VOCAB_SIZE: i32 = 262_144;
pub const SLIDING_WINDOW: i32 = 512;
pub const SHARED_KV_LAYERS: i32 = 20;
pub const FULL_ATTENTION_INTERVAL: i32 = 5;
pub const LAYER_NORM_RMS_EPSILON: f32 = 1.0e-6;
pub const ROPE_DIMENSION_COUNT: i32 = 512;
pub const ROPE_DIMENSION_COUNT_SWA: i32 = 256;
pub const ROPE_FREQ_BASE: f32 = 1_000_000.0;
pub const ROPE_FREQ_BASE_SWA: f32 = 10_000.0;
pub const SLIDING_WINDOW_PATTERN: [u8; BLOCK_COUNT as usize] = [
    1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
    1, 1, 0,
];

#[cfg(test)]
mod tests;
