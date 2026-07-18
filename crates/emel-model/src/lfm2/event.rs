//! Public Lfm2-family events and source-exact parameter contract.

use core::fmt;

use crate::catalog::event::ModelIdentity;
use crate::generation::{self, ContractDescriptor, QuantizedStageFamily};

use super::Lfm2;

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

/// Event accepted by [`Lfm2`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Lfm2) -> Self::Output;
}

/// The exact ten metadata keys consumed by pinned LFM2 `load_hparams`.
pub const HPARAM_KEYS: [&str; 10] = [
    "lfm2.context_length",
    "lfm2.embedding_length",
    "lfm2.feed_forward_length",
    "lfm2.attention.head_count",
    "lfm2.attention.head_count_kv",
    "lfm2.block_count",
    "lfm2.vocab_size",
    "lfm2.shortconv.l_cache",
    "lfm2.attention.layer_norm_rms_epsilon",
    "lfm2.rope.freq_base",
];

/// Source-maintained LFM2 geometry classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    /// LFM2.5 1.2B: 16 layers, 2048 embedding width, 32 attention heads.
    OnePointTwoB,
    /// LFM2.5 230M: 14 layers, 1024 embedding width, 16 attention heads.
    TwoHundredThirtyM,
}

/// Source-exact LFM2 hyperparameters and bounded per-layer attention pattern.
///
/// Metadata decoding allocates nothing and happens before family dispatch. Public
/// begin events borrow this value; the actor copies it into preallocated context
/// during the same RTC boundary and never retains the borrow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_out: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub block_count: i32,
    pub vocab_size: i32,
    pub shortconv_l_cache: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub attention_key_length: i32,
    pub attention_value_length: i32,
    pub rope_dimension_count: i32,
    pub attention_layer_pattern_count: u32,
    pub attention_layer_pattern_flags: [u8; super::MAX_BLOCKS],
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            context_length: 0,
            embedding_length: 0,
            embedding_length_out: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            attention_head_count_kv: 0,
            block_count: 0,
            vocab_size: 0,
            shortconv_l_cache: 0,
            attention_layer_norm_rms_epsilon: 0.0,
            rope_freq_base: 0.0,
            attention_key_length: 0,
            attention_value_length: 0,
            rope_dimension_count: 0,
            attention_layer_pattern_count: 0,
            attention_layer_pattern_flags: [0; super::MAX_BLOCKS],
        }
    }
}

impl Parameters {
    /// Classifies exactly the two maintained source geometries.
    #[must_use]
    pub const fn variant(&self) -> Option<Variant> {
        if self.block_count == 16
            && self.embedding_length == 2048
            && self.attention_head_count == 32
        {
            Some(Variant::OnePointTwoB)
        } else if self.block_count == 14
            && self.embedding_length == 1024
            && self.attention_head_count == 16
        {
            Some(Variant::TwoHundredThirtyM)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BeginContract {
    Execution,
    Validation,
}

/// Begins an Lfm2 contract and binds its three global tensors.
#[derive(Clone, Copy, Debug)]
pub struct ContractBegin<'a> {
    pub(super) architecture: &'a [u8],
    pub(super) model: ModelIdentity,
    pub(super) parameters: &'a Parameters,
    pub(super) layer_count: i32,
    pub(super) contract: BeginContract,
}
impl<'a> ContractBegin<'a> {
    /// Begins the strict maintained-geometry execution contract.
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

    /// Begins the shared non-strict `validate_builder_contract` and `validate_data` protocol.
    ///
    /// The caller must dispatch one [`BlockBuild`] for every layer after this
    /// event succeeds. Together those dispatches validate the complete source
    /// tensor contract without applying maintained execution-geometry limits.
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

/// Binds exactly one source-fixed Lfm2 block in one dispatch.
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

/// Installs the source-fixed Lfm2 topology.
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

/// Visits the completed immutable Lfm2 contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Looks up one validated Lfm2 block.
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
            fn dispatch(self, actor: &mut Lfm2) -> Self::Output {
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
            "Lfm2 {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
