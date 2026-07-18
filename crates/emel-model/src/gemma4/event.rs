//! Public Gemma4-family events and source-exact parameter contract.

use core::fmt;

use crate::attention_family;
use crate::catalog::event::ModelIdentity;
use crate::generation::{self, ContractDescriptor};

use super::Gemma4;

pub use attention_family::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractReset, ContractVisit, PlanBuild,
    StageAudit, Storage, StorageBindError, StorageRelease, TopologyBuild,
};
pub use generation::event::Error;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Gemma4`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Gemma4) -> Self::Output;
}

/// The 21 metadata keys inspected by pinned Gemma4 `load_hparams`.
pub const HPARAM_KEYS: [&str; 21] = [
    "gemma4.context_length",
    "gemma4.embedding_length",
    "gemma4.embedding_length_per_layer_input",
    "gemma4.feed_forward_length",
    "gemma4.attention.head_count",
    "gemma4.attention.head_count_kv",
    "gemma4.attention.key_length",
    "gemma4.attention.key_length_swa",
    "gemma4.attention.value_length",
    "gemma4.attention.value_length_swa",
    "gemma4.block_count",
    "gemma4.vocab_size",
    "gemma4.attention.sliding_window",
    "gemma4.attention.shared_kv_layers",
    "gemma4.rope.dimension_count",
    "gemma4.rope.dimension_count_swa",
    "gemma4.attention.layer_norm_rms_epsilon",
    "gemma4.final_logit_softcapping",
    "gemma4.rope.freq_base",
    "gemma4.rope.freq_base_swa",
    "gemma4.attention.sliding_window_pattern",
];

/// Source-exact Gemma4 hyperparameters and bounded sliding-window pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_out: i32,
    pub embedding_length_per_layer_input: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub attention_key_length: i32,
    pub attention_key_length_swa: i32,
    pub attention_value_length: i32,
    pub attention_value_length_swa: i32,
    pub block_count: i32,
    pub vocab_size: i32,
    pub attention_sliding_window: i32,
    pub attention_shared_kv_layers: i32,
    pub rope_dimension_count: i32,
    pub rope_dimension_count_swa: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub final_logit_softcapping: f32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub full_attention_interval: i32,
    pub tie_word_embeddings: bool,
    pub rope_pair_x0_stride: i32,
    pub rope_pair_x1_stride: i32,
    pub rope_pair_x1_offset: i32,
    pub rope_pair_x1_half_rot_offset: i32,
    pub sliding_window_pattern_count: u32,
    pub sliding_window_pattern_flags: [u8; super::MAX_BLOCKS],
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            context_length: 0,
            embedding_length: 0,
            embedding_length_out: 0,
            embedding_length_per_layer_input: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            attention_head_count_kv: 0,
            attention_key_length: 0,
            attention_key_length_swa: 0,
            attention_value_length: 0,
            attention_value_length_swa: 0,
            block_count: 0,
            vocab_size: 0,
            attention_sliding_window: 0,
            attention_shared_kv_layers: 0,
            rope_dimension_count: 0,
            rope_dimension_count_swa: 0,
            attention_layer_norm_rms_epsilon: 0.0,
            final_logit_softcapping: 0.0,
            rope_freq_base: 0.0,
            rope_freq_base_swa: 0.0,
            full_attention_interval: 0,
            tie_word_embeddings: false,
            rope_pair_x0_stride: 0,
            rope_pair_x1_stride: 0,
            rope_pair_x1_offset: 0,
            rope_pair_x1_half_rot_offset: 0,
            sliding_window_pattern_count: 0,
            sliding_window_pattern_flags: [0; super::MAX_BLOCKS],
        }
    }
}

impl Parameters {
    /// Returns the exact pinned 35-layer execution metadata fixture.
    #[must_use]
    pub const fn canonical() -> Self {
        let mut flags = [0; super::MAX_BLOCKS];
        let mut index = 0;
        while index < super::SLIDING_WINDOW_PATTERN.len() {
            flags[index] = super::SLIDING_WINDOW_PATTERN[index];
            index += 1;
        }
        Self {
            context_length: super::CONTEXT_LENGTH,
            embedding_length: super::EMBEDDING_LENGTH,
            embedding_length_out: super::EMBEDDING_LENGTH,
            embedding_length_per_layer_input: super::EMBEDDING_LENGTH_PER_LAYER_INPUT,
            feed_forward_length: super::FEED_FORWARD_LENGTH,
            attention_head_count: super::HEAD_COUNT,
            attention_head_count_kv: super::HEAD_COUNT_KV,
            attention_key_length: super::KEY_LENGTH,
            attention_key_length_swa: super::KEY_LENGTH_SWA,
            attention_value_length: super::VALUE_LENGTH,
            attention_value_length_swa: super::VALUE_LENGTH_SWA,
            block_count: super::BLOCK_COUNT,
            vocab_size: super::VOCAB_SIZE,
            attention_sliding_window: super::SLIDING_WINDOW,
            attention_shared_kv_layers: super::SHARED_KV_LAYERS,
            rope_dimension_count: super::ROPE_DIMENSION_COUNT,
            rope_dimension_count_swa: super::ROPE_DIMENSION_COUNT_SWA,
            attention_layer_norm_rms_epsilon: super::LAYER_NORM_RMS_EPSILON,
            final_logit_softcapping: 30.0,
            rope_freq_base: super::ROPE_FREQ_BASE,
            rope_freq_base_swa: super::ROPE_FREQ_BASE_SWA,
            full_attention_interval: super::FULL_ATTENTION_INTERVAL,
            tie_word_embeddings: true,
            rope_pair_x0_stride: 1,
            rope_pair_x1_stride: 1,
            rope_pair_x1_offset: 0,
            rope_pair_x1_half_rot_offset: 1,
            sliding_window_pattern_count: super::BLOCK_COUNT as u32,
            sliding_window_pattern_flags: flags,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BeginContract {
    Execution,
    Validation,
}

/// Begins a Gemma4 contract while borrowing its bounded pattern payload.
#[derive(Clone, Copy, Debug)]
pub struct ContractBegin<'a> {
    pub(super) architecture: &'a [u8],
    pub(super) model: ModelIdentity,
    pub(super) parameters: &'a Parameters,
    pub(super) layer_count: i32,
    pub(super) contract: BeginContract,
}
impl<'a> ContractBegin<'a> {
    /// Begins the exact pinned 35-layer execution contract.
    #[must_use]
    pub const fn new(
        architecture: &'a [u8],
        model: ModelIdentity,
        parameters: &'a Parameters,
    ) -> Self {
        Self {
            architecture,
            model,
            parameters,
            layer_count: parameters.block_count,
            contract: BeginContract::Execution,
        }
    }

    /// Begins the non-strict source validation contract.
    #[must_use]
    pub const fn validation(
        architecture: &'a [u8],
        model: ModelIdentity,
        parameters: &'a Parameters,
        layer_count: i32,
    ) -> Self {
        Self {
            architecture,
            model,
            parameters,
            layer_count,
            contract: BeginContract::Validation,
        }
    }
}

macro_rules! impl_event {
    ($event:ty, $output:ty, $method:ident) => {
        impl sealed::Sealed for $event {}
        impl Event for $event {
            type Output = $output;
            #[inline]
            fn dispatch(self, actor: &mut Gemma4) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_event!(ContractBegin<'_>, Result<(), Error>, contract_begin);
impl_event!(BlockBuild, Result<(), Error>, block_build);
impl_event!(TopologyBuild, Result<(), Error>, topology_build);
impl_event!(PlanBuild, Result<(), Error>, plan_build);
impl_event!(BlockValidation, Result<(), Error>, block_validation);
impl_event!(BlockAudit, Result<(), Error>, block_audit);
impl_event!(StageAudit, Result<(), Error>, stage_audit);
impl_event!(ContractVisit, Result<ContractDescriptor, Error>, contract_visit);
impl_event!(BlockVisit, Result<generation::event::BlockDescriptor, Error>, block_visit);
impl_event!(ContractReset, Result<(), Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, Error>, storage_release);

impl fmt::Display for Parameters {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Gemma4 {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
