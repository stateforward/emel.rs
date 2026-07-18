//! Shared public attention-family event payloads and caller-owned storage.

use core::fmt;

use crate::catalog::event::ModelIdentity;
use crate::generation::{self, QuantizedStageFamily};

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
    pub(crate) const fn new(error: Error, storage: Storage) -> Self {
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

/// Begins a source-fixed attention-family contract and binds its global tensors.
#[derive(Clone, Copy, Debug)]
pub struct ContractBegin<'a, P> {
    pub(crate) architecture: &'a [u8],
    pub(crate) model: ModelIdentity,
    pub(crate) parameters: P,
}
impl<'a, P> ContractBegin<'a, P> {
    #[must_use]
    pub const fn new(architecture: &'a [u8], model: ModelIdentity, parameters: P) -> Self {
        Self {
            architecture,
            model,
            parameters,
        }
    }
}

/// Binds exactly one source-fixed attention-family block in one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct BlockBuild {
    pub(crate) index: i32,
}
impl BlockBuild {
    #[must_use]
    pub const fn new(index: i32) -> Self {
        Self { index }
    }
}

/// Installs the source-fixed attention-family topology.
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
    pub(crate) index: i32,
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
    pub(crate) index: i32,
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
    pub(crate) family: QuantizedStageFamily,
}
impl StageAudit {
    #[must_use]
    pub const fn new(family: QuantizedStageFamily) -> Self {
        Self { family }
    }
}

/// Visits the completed immutable attention-family contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Looks up one validated attention-family block.
#[derive(Clone, Copy, Debug)]
pub struct BlockVisit {
    pub(crate) index: i32,
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
