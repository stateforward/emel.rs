//! Public Qwen3-family events and source-exact parameter contract.

use core::fmt;

use crate::catalog::event::ModelIdentity;
use crate::generation::{self, ContractDescriptor, QuantizedStageFamily};

use super::Qwen3;

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

/// Event accepted by [`Qwen3`].
pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Qwen3) -> Self::Output;
}

/// The exact ten metadata keys consumed by pinned Qwen3 `load_hparams`.
pub const HPARAM_KEYS: [&str; 10] = [
    "qwen3.context_length",
    "qwen3.embedding_length",
    "qwen3.feed_forward_length",
    "qwen3.attention.head_count",
    "qwen3.attention.head_count_kv",
    "qwen3.attention.key_length",
    "qwen3.attention.value_length",
    "qwen3.block_count",
    "qwen3.attention.layer_norm_rms_epsilon",
    "qwen3.rope.freq_base",
];

/// Source-exact Qwen3 hyperparameters and derived fixed-layout facts.
///
/// Allocation and metadata decoding happen before dispatch. The actor retains this
/// owned `Copy` value because its layer facts are used across later block events.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Parameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_out: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub attention_key_length: i32,
    pub attention_value_length: i32,
    pub rope_dimension_count: i32,
    pub block_count: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub tie_word_embeddings: bool,
    pub rope_pair_x0_stride: i32,
    pub rope_pair_x1_stride: i32,
    pub rope_pair_x1_offset: i32,
    pub rope_pair_x1_half_rot_offset: i32,
}

/// Begins the source-fixed Qwen3 contract and binds its three global tensors.
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

/// Binds exactly one source-fixed Qwen3 block in one dispatch.
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

/// Installs the source-fixed Qwen3 topology.
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

/// Visits the completed immutable Qwen3 contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Looks up one validated Qwen3 block.
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
            fn dispatch(self, actor: &mut Qwen3) -> Self::Output {
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
            "Qwen3 {} layers, embedding {}, context {}",
            self.block_count, self.embedding_length, self.context_length
        )
    }
}
