//! Public events and opaque caller-preallocated storage for the generation builder.

use core::fmt;

use crate::catalog::event::{ModelIdentity, TensorId};

use super::{
    AttentionQkNormRoute, AttentionValueRoute, Builder, ContractDescriptor, LayerExecution,
    QuantizedStageFamily, ResidualRoute,
};

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Builder`].
pub trait Event<C, K>: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Builder<C, K>) -> Self::Output;
}

/// Common generation protocol or model-validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidRequest,
    ModelInvalid,
    Capacity,
    StorageUnavailable,
    Busy,
    WrongModelIdentity,
    StaleModelIdentity,
    Dependency,
    UnexpectedEvent,
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "invalid generation request",
            Self::ModelInvalid => "generation contract is invalid",
            Self::Capacity => "generation storage capacity is unavailable",
            Self::StorageUnavailable => "generation storage is unavailable",
            Self::Busy => "generation builder is busy",
            Self::WrongModelIdentity => "model identity belongs to another catalog",
            Self::StaleModelIdentity => "model identity is stale",
            Self::Dependency => "generation dependency failed",
            Self::UnexpectedEvent => "unexpected generation event",
            Self::Internal => "internal generation error",
        })
    }
}
impl std::error::Error for Error {}

/// Opaque preallocated block, layer, and audit storage.
pub struct Storage {
    pub(super) regions: super::storage::StorageRegions,
}

impl Storage {
    /// Allocates every variable-capacity region before actor dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] for an invalid bound or allocation failure.
    pub fn with_block_capacity(block_capacity: usize) -> Result<Self, Error> {
        if block_capacity == 0 || block_capacity > 512 {
            return Err(Error::Capacity);
        }
        Ok(Self {
            regions: super::storage::StorageRegions {
                blocks: zeroed(block_capacity)?,
                views: zeroed(block_capacity)?,
                layers: zeroed(block_capacity)?,
                block_audits: zeroed(block_capacity)?,
            },
        })
    }

    #[must_use]
    pub const fn block_capacity(&self) -> usize {
        self.regions.blocks.len()
    }

    /// Clears the regions for pre-dispatch caller-side reuse.
    pub fn clear(&mut self) {
        self.regions.clear();
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storage")
            .field("block_capacity", &self.block_capacity())
            .finish_non_exhaustive()
    }
}

fn zeroed<T: Clone + Default>(length: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Capacity)?;
    values.resize(length, T::default());
    Ok(values)
}

/// Moves caller-preallocated storage into an empty builder.
#[derive(Debug)]
pub struct StorageBind {
    pub(super) storage: Storage,
}
impl StorageBind {
    #[must_use]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}

/// Rejected storage bind that preserves ownership.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for StorageBindError {}

/// Begins a common contract after an owning architecture actor selects the route.
#[derive(Clone, Copy, Debug)]
pub struct ContractBegin {
    pub(super) model: ModelIdentity,
    pub(super) block_count: i32,
    pub(super) context_length: i32,
}
impl ContractBegin {
    #[must_use]
    pub const fn new(model: ModelIdentity, block_count: i32, context_length: i32) -> Self {
        Self {
            model,
            block_count,
            context_length,
        }
    }
}

/// Binds architecture-supplied global names and the source-fixed output name.
#[derive(Clone, Copy, Debug)]
pub struct GlobalBindings<'a> {
    pub(super) token_embedding: &'a [u8],
    pub(super) output_norm: &'a [u8],
    pub(super) allow_tied_output: bool,
}
impl<'a> GlobalBindings<'a> {
    #[must_use]
    pub const fn new(
        token_embedding: &'a [u8],
        output_norm: &'a [u8],
        allow_tied_output: bool,
    ) -> Self {
        Self {
            token_embedding,
            output_norm,
            allow_tied_output,
        }
    }
}

/// Binds one attention block in one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct AttentionBlock {
    pub(super) index: i32,
    pub(super) layer: LayerExecution,
}
impl AttentionBlock {
    #[must_use]
    pub const fn new(index: i32, layer: LayerExecution) -> Self {
        Self { index, layer }
    }
    pub(super) fn requires_qk_norm(self) -> bool {
        self.layer.qk_norm_route() != AttentionQkNormRoute::None
    }
    pub(super) fn uses_shared_value(self) -> bool {
        self.layer.value_route() == AttentionValueRoute::SharedKeyValue
    }
}

/// Binds one short-convolution block in one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ShortconvBlock {
    pub(super) index: i32,
    pub(super) layer: LayerExecution,
}

/// A source-selected block tensor family that must be absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockFamily {
    Attention,
    Shortconv,
}

/// Rejects every tensor owned by one opposite block family.
///
/// The builder performs the fixed catalog queries synchronously and returns
/// [`Error::ModelInvalid`] when any rejected tensor is present.
#[derive(Clone, Copy, Debug)]
pub struct RejectBlockTensors {
    pub(super) index: i32,
    pub(super) family: BlockFamily,
}
impl RejectBlockTensors {
    #[must_use]
    pub const fn new(index: i32, family: BlockFamily) -> Self {
        Self { index, family }
    }
}
impl ShortconvBlock {
    #[must_use]
    pub const fn new(index: i32, layer: LayerExecution) -> Self {
        Self { index, layer }
    }
}

/// Installs topology facts computed by the owning family actor.
#[derive(Clone, Copy, Debug)]
pub struct Topology {
    pub(super) node_count: u32,
    pub(super) tensor_count: u32,
    pub(super) bytes_per_tensor: u64,
    pub(super) workspace_capacity_bytes: u64,
}
impl Topology {
    #[must_use]
    pub const fn new(
        node_count: u32,
        tensor_count: u32,
        bytes_per_tensor: u64,
        workspace_capacity_bytes: u64,
    ) -> Self {
        Self {
            node_count,
            tensor_count,
            bytes_per_tensor,
            workspace_capacity_bytes,
        }
    }
}

/// Builds the source prefill and decode plans from installed topology.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plan;
impl Plan {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Validates one bound block in one dispatch.
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

/// Accumulates audit facts for one block in one dispatch.
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

/// Finalizes one of exactly fourteen stage audits.
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

/// Visits the completed immutable contract descriptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Immutable block descriptor returned by [`BlockVisit`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BlockTensorSlot {
    AttentionNorm = 0,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionQNorm,
    AttentionKNorm,
    AttentionOutput,
    ShortconvConv,
    ShortconvIn,
    ShortconvOut,
    FeedForwardNorm,
    FeedForwardGate,
    FeedForwardDown,
    FeedForwardUp,
}

/// Immutable block descriptor returned by [`BlockVisit`].
#[derive(Clone, Copy, Debug)]
pub struct BlockDescriptor {
    index: i32,
    uses_attention: bool,
    layer: LayerExecution,
    tensors: [TensorId; 14],
    tensor_mask: u16,
}
impl BlockDescriptor {
    #[must_use]
    pub const fn index(self) -> i32 {
        self.index
    }
    #[must_use]
    pub const fn uses_attention(self) -> bool {
        self.uses_attention
    }
    #[must_use]
    pub const fn layer(self) -> LayerExecution {
        self.layer
    }
    /// Returns one opaque bound tensor identity by named source block-view slot.
    #[must_use]
    pub const fn tensor(self, slot: BlockTensorSlot) -> Option<TensorId> {
        let index = slot as usize;
        if self.tensor_mask & (1u16 << index) != 0 {
            Some(self.tensors[index])
        } else {
            None
        }
    }
    #[allow(clippy::large_types_passed_by_value)]
    pub(super) const fn new(
        index: i32,
        uses_attention: bool,
        layer: LayerExecution,
        tensors: [TensorId; 14],
        tensor_mask: u16,
    ) -> Self {
        Self {
            index,
            uses_attention,
            layer,
            tensors,
            tensor_mask,
        }
    }
}

/// Looks up one validated block without exposing actor storage.
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

/// Resets the active contract while retaining storage.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContractReset;
impl ContractReset {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Releases reset storage to the caller.
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
        impl Event<crate::catalog::Catalog, crate::generation::quantized_path::Resolver>
            for $event
        {
            type Output = $output;
            fn dispatch(
                self,
                actor: &mut Builder<
                    crate::catalog::Catalog,
                    crate::generation::quantized_path::Resolver,
                >,
            ) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_event!(StorageBind, Result<(), StorageBindError>, storage_bind);
impl_event!(ContractBegin, Result<(), Error>, contract_begin);
impl_event!(GlobalBindings<'_>, Result<(), Error>, global_bindings);
impl_event!(AttentionBlock, Result<(), Error>, attention_block);
impl_event!(ShortconvBlock, Result<(), Error>, shortconv_block);
impl_event!(RejectBlockTensors, Result<(), Error>, reject_block_tensors);
impl_event!(Topology, Result<(), Error>, topology);
impl_event!(Plan, Result<(), Error>, plan);
impl_event!(BlockValidation, Result<(), Error>, block_validation);
impl_event!(BlockAudit, Result<(), Error>, block_audit);
impl_event!(StageAudit, Result<(), Error>, stage_audit);
impl_event!(ContractVisit, Result<ContractDescriptor, Error>, contract_visit);
impl_event!(BlockVisit, Result<BlockDescriptor, Error>, block_visit);
impl_event!(ContractReset, Result<(), Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, Error>, storage_release);

pub(super) fn attention_request_valid(event: AttentionBlock) -> bool {
    event.layer.residual_route() == ResidualRoute::Attention
}
pub(super) fn shortconv_request_valid(event: ShortconvBlock) -> bool {
    event.layer.residual_route() == ResidualRoute::Shortconv
        && event.layer.qk_norm_route() == AttentionQkNormRoute::None
        && event.layer.value_route() == AttentionValueRoute::DedicatedValue
}
