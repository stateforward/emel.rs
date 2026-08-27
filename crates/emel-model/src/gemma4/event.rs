//! Public Gemma4-family events and source-exact parameter contract.

use core::fmt;

use crate::catalog::event::ModelIdentity;
use crate::generation::{self, ContractDescriptor, QuantizedStageFamily};

use super::Gemma4;

pub use generation::event::Error;

/// Caller-preallocated common and family immutable block-view storage.
#[derive(Debug)]
pub struct Storage {
    pub(super) common: generation::event::Storage,
    pub(super) blocks: Vec<Option<generation::event::BlockDescriptor>>,
}
impl Storage {
    /// Allocates all family and common block regions before actor construction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the bound is invalid or allocation fails.
    pub fn with_block_capacity(block_capacity: usize) -> Result<Self, Error> {
        let common = generation::event::Storage::with_block_capacity(block_capacity)?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(block_capacity)
            .map_err(|_| Error::Capacity)?;
        blocks.resize(block_capacity, None);
        Ok(Self { common, blocks })
    }
    #[must_use]
    pub const fn block_capacity(&self) -> usize {
        self.blocks.len()
    }
    /// Clears family and common regions for caller-side reuse.
    pub fn clear(&mut self) {
        self.common.clear();
        self.blocks.fill(None);
    }
}

/// Rejected family construction that preserves every caller-owned region.
#[derive(Debug)]
pub struct StorageBindError {
    error: Error,
    storage: Storage,
}
impl StorageBindError {
    pub(super) const fn new(error: Error, storage: Storage) -> Self {
        Self { error, storage }
    }
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }
    #[must_use]
    pub fn into_storage(self) -> Storage {
        self.storage
    }
}
impl fmt::Display for StorageBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}
impl std::error::Error for StorageBindError {}

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Gemma4`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Gemma4) -> Self::Output;
}

/// The exact twenty-one metadata keys consumed by pinned Gemma4 `load_hparams`.
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

/// Source-exact Gemma4 hyperparameters and derived fixed-layout facts.
///
/// Allocation and metadata decoding happen before dispatch. The actor retains this
/// owned `Copy` value because its layer facts are used across later block events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_per_layer_input: i32,
    pub embedding_length_out: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub attention_key_length: i32,
    pub attention_key_length_swa: i32,
    pub attention_value_length: i32,
    pub attention_value_length_swa: i32,
    pub rope_dimension_count: i32,
    pub rope_dimension_count_swa: i32,
    pub block_count: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub final_logit_softcapping: f32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub attention_sliding_window: i32,
    pub attention_shared_kv_layers: i32,
    pub sliding_window_pattern: [u8; 35],
    pub sliding_window_pattern_count: u32,
    pub tie_word_embeddings: bool,
    pub rope_pair_x0_stride: i32,
    pub rope_pair_x1_stride: i32,
    pub rope_pair_x1_offset: i32,
    pub rope_pair_x1_half_rot_offset: i32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            context_length: 0,
            embedding_length: 0,
            embedding_length_per_layer_input: 0,
            embedding_length_out: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            attention_head_count_kv: 0,
            attention_key_length: 0,
            attention_key_length_swa: 0,
            attention_value_length: 0,
            attention_value_length_swa: 0,
            rope_dimension_count: 0,
            rope_dimension_count_swa: 0,
            block_count: 0,
            attention_layer_norm_rms_epsilon: 0.0,
            final_logit_softcapping: 0.0,
            rope_freq_base: 0.0,
            rope_freq_base_swa: 0.0,
            attention_sliding_window: 0,
            attention_shared_kv_layers: 0,
            sliding_window_pattern: [0; 35],
            sliding_window_pattern_count: 0,
            tie_word_embeddings: false,
            rope_pair_x0_stride: 0,
            rope_pair_x1_stride: 0,
            rope_pair_x1_offset: 0,
            rope_pair_x1_half_rot_offset: 0,
        }
    }
}

/// Begins the source-fixed Gemma4 contract and binds its three global tensors.
#[derive(Clone, Copy, Debug)]
pub struct ContractBegin<'a> {
    pub(super) architecture: &'a [u8],
    pub(super) model: ModelIdentity,
    pub(super) parameters: Parameters,
}
impl<'a> ContractBegin<'a> {
    #[must_use]
    pub const fn new(architecture: &'a [u8], model: ModelIdentity, parameters: Parameters) -> Self {
        Self {
            architecture,
            model,
            parameters,
        }
    }
}

/// Binds exactly one source-fixed Gemma4 block in one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct BlockBuild {
    pub(super) index: i32,
}
impl BlockBuild {
    #[must_use]
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Installs the source-fixed Gemma4 topology.
#[derive(Clone, Copy, Debug, Default)]
pub struct TopologyBuild;
impl TopologyBuild {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Builds common prefill and decode plans after topology is installed.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanBuild;
impl PlanBuild {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Validates exactly one bound block.
#[derive(Clone, Copy, Debug)]
pub struct BlockValidation {
    pub(super) index: i32,
}
impl BlockValidation {
    #[must_use]
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Audits exactly one bound block.
#[derive(Clone, Copy, Debug)]
pub struct BlockAudit {
    pub(super) index: i32,
}
impl BlockAudit {
    #[must_use]
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Finalizes exactly one common quantized stage family.
#[derive(Clone, Copy, Debug)]
pub struct StageAudit {
    pub(super) family: QuantizedStageFamily,
}
impl StageAudit {
    #[must_use]
    pub const fn new(family: QuantizedStageFamily) -> Self {
        Self { family }
    }
}

/// Visits the completed immutable Gemma4 contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Looks up one validated Gemma4 block.
#[derive(Clone, Copy, Debug)]
pub struct BlockVisit {
    pub(super) index: i32,
}
impl BlockVisit {
    #[must_use]
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Resets the active family contract while retaining preallocated storage.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractReset;
impl ContractReset {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Releases the common caller-preallocated storage after reset.
#[derive(Clone, Copy, Debug, Default)]
pub struct StorageRelease;
impl StorageRelease {
    #[must_use]
    pub const fn new() -> Self {
        Self
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

impl_event!(ContractBegin<'_>, Result<(), generation::event::Error>, contract_begin);
impl_event!(BlockBuild, Result<(), generation::event::Error>, block_build);
impl_event!(TopologyBuild, Result<(), generation::event::Error>, topology_build);
impl_event!(PlanBuild, Result<(), generation::event::Error>, plan_build);
impl_event!(BlockValidation, Result<(), generation::event::Error>, block_validation);
impl_event!(BlockAudit, Result<(), generation::event::Error>, block_audit);
impl_event!(StageAudit, Result<(), generation::event::Error>, stage_audit);
impl_event!(ContractVisit, Result<ContractDescriptor, generation::event::Error>, contract_visit);
impl_event!(BlockVisit, Result<generation::event::BlockDescriptor, generation::event::Error>, block_visit);
impl_event!(ContractReset, Result<(), generation::event::Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, generation::event::Error>, storage_release);

impl fmt::Display for Parameters {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Gemma4 {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
