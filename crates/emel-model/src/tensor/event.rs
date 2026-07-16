//! Typed requests and outcomes for the tensor store actor.

use core::fmt;

use emel_io::{mmap, read, staged_read};

use super::Store;
use super::dependency::TensorDependencies;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Store`].
///
/// The dependency type is part of the static dispatch contract. The trait is
/// sealed so consumers cannot bypass the actor's public event surface.
pub trait Event<D = ()>: sealed::Sealed {
    /// Result produced before the top-level dispatch returns.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Store<D>) -> Self::Output;
}

/// Tensor-store failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::enum_variant_names,
    reason = "BackendError is the pinned tensor protocol classification"
)]
#[non_exhaustive]
pub enum Error {
    /// A tensor identifier, byte payload, or metadata field is invalid.
    InvalidRequest,
    /// Construction-time slot capacity is invalid or unavailable.
    Capacity,
    /// A bind targeted a slot that already owns resident bytes.
    TensorAlreadyResident,
    /// The requested tensor has no resident bytes.
    TensorUnbound,
    /// Mapped residency must be managed by the mapped-load lifecycle.
    MappedTensorRequiresRelease,
    /// The actor is awaiting a previously planned effect batch.
    Busy,
    /// The requested load strategy is not supported.
    UnsupportedStrategy,
    /// No mapped-I/O actor is present in the static dependency set.
    MmapUnavailable,
    /// No read actor is present in the static dependency set.
    ReadUnavailable,
    /// No staged-read actor is present in the static dependency set.
    StagerUnavailable,
    /// A mapped-I/O child actor returned a classified failure.
    Mmap(mmap::event::Error),
    /// A read child actor returned a classified failure.
    Read(read::event::Error),
    /// A staged-read child actor returned a classified failure.
    Staged(staged_read::event::Error),
    /// A replacement child actor returned a success value that violates the dispatched contract.
    DependencyContract,
    /// A malformed mapped success could not be released and remains actor-owned for retry.
    DependencyContractCleanup {
        /// Live mapping handle retained by the tensor actor.
        mapping_handle: u32,
        /// Typed failure returned by the immediate cleanup attempt.
        error: mmap::event::Error,
    },
    /// A planned effect failed in its owning backend.
    BackendError,
    /// The actor encountered an internal or unexpected event error.
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid tensor request",
            Self::Capacity => "tensor store capacity unavailable",
            Self::TensorAlreadyResident => "tensor is already resident",
            Self::TensorUnbound => "tensor is not resident",
            Self::MappedTensorRequiresRelease => "mapped tensor requires mapped release",
            Self::Busy => "tensor store is awaiting effect results",
            Self::UnsupportedStrategy => "unsupported tensor load strategy",
            Self::MmapUnavailable => "tensor mapper capability unavailable",
            Self::ReadUnavailable => "tensor reader capability unavailable",
            Self::StagerUnavailable => "tensor stager capability unavailable",
            Self::Mmap(_) => "tensor mapped I/O failed",
            Self::Read(_) => "tensor read I/O failed",
            Self::Staged(_) => "tensor staged I/O failed",
            Self::DependencyContract => "tensor dependency returned an invalid success value",
            Self::DependencyContractCleanup { .. } => {
                "tensor dependency returned an invalid mapping and cleanup failed"
            }
            Self::BackendError => "tensor effect backend failed",
            Self::Internal => "internal tensor actor error",
        })
    }
}

impl std::error::Error for Error {}

/// Persistent metadata copied into a resident tensor slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorMetadata {
    file_offset: u64,
    data_size: u64,
    file_index: u16,
    tensor_type: i32,
}

impl TensorMetadata {
    /// Creates metadata for one tensor record.
    #[must_use]
    pub const fn new(file_offset: u64, data_size: u64, file_index: u16, tensor_type: i32) -> Self {
        Self {
            file_offset,
            data_size,
            file_index,
            tensor_type,
        }
    }

    /// Returns the byte offset in the owning model file.
    #[must_use]
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    /// Returns the tensor's logical data size.
    #[must_use]
    pub const fn data_size(self) -> u64 {
        self.data_size
    }

    /// Returns the split-file index.
    #[must_use]
    pub const fn file_index(self) -> u16 {
        self.file_index
    }

    /// Returns the serialized tensor type identifier.
    #[must_use]
    pub const fn tensor_type(self) -> i32 {
        self.tensor_type
    }
}

/// One setup-allocated tensor record transferred into the actor.
#[derive(Debug, Eq, PartialEq)]
pub struct StorageEntry {
    pub(crate) metadata: TensorMetadata,
    pub(crate) bytes: Option<Box<[u8]>>,
}

impl StorageEntry {
    /// Creates one owned storage record.
    #[must_use]
    pub const fn new(metadata: TensorMetadata, bytes: Option<Box<[u8]>>) -> Self {
        Self { metadata, bytes }
    }

    /// Returns the tensor metadata.
    #[must_use]
    pub const fn metadata(&self) -> TensorMetadata {
        self.metadata
    }

    /// Returns the optional setup-allocated byte count.
    #[must_use]
    pub fn initial_bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }
}

/// One setup-allocated batch transferred by [`BindStorage`].
#[derive(Debug, Eq, PartialEq)]
pub struct StorageBatch(pub(crate) Box<[StorageEntry]>);

impl StorageBatch {
    /// Creates a batch from setup-allocated entries.
    #[must_use]
    pub const fn new(entries: Box<[StorageEntry]>) -> Self {
        Self(entries)
    }

    /// Returns the number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the batch contains no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the owned entries.
    #[must_use]
    pub fn into_entries(self) -> Box<[StorageEntry]> {
        self.0
    }
}

/// Replace the actor's active tensor-storage batch.
#[derive(Debug, Eq, PartialEq)]
pub struct BindStorage {
    pub(crate) storage: StorageBatch,
}

impl BindStorage {
    /// Creates an owned storage-bind request.
    #[must_use]
    pub const fn new(storage: StorageBatch) -> Self {
        Self { storage }
    }
}

/// Successful bulk storage binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindStorageDone {
    active_extent: usize,
}

impl BindStorageDone {
    pub(crate) const fn new(active_extent: usize) -> Self {
        Self { active_extent }
    }

    /// Returns the number of active tensor records.
    #[must_use]
    pub const fn active_extent(self) -> usize {
        self.active_extent
    }
}

/// Failed bulk binding with ownership returned unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct BindStorageError {
    pub(crate) error: Error,
    pub(crate) storage: StorageBatch,
}

impl BindStorageError {
    pub(crate) const fn new(error: Error, storage: StorageBatch) -> Self {
        Self { error, storage }
    }

    /// Returns the failure classification.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns the unchanged caller allocation.
    #[must_use]
    pub fn into_storage(self) -> StorageBatch {
        self.storage
    }
}

/// Load-planning strategy selected before dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategyKind {
    /// Use bytes already supplied by the storage batch.
    None,
    /// Request file-backed mapped residency.
    MappedFile,
    /// Request an owned read-and-copy result.
    ReadCopy,
    /// Request bytes from an external buffer backend.
    ExternalBuffer,
    /// Request an owned staged-read result.
    StagedRead,
    /// Preserve an unknown wire value for explicit rejection.
    Unknown(u8),
}

/// One preallocated effect slot filled by [`PlanLoad`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EffectRequest {
    /// Caller-provided empty slot.
    #[default]
    Empty,
    /// Actor-owned initial-byte residency request.
    None {
        /// Ordered tensor identifier.
        tensor_id: i32,
        /// Source split-file index.
        file_index: u16,
        /// Source byte offset.
        offset: u64,
        /// Required byte count.
        size: u64,
    },
    /// File-backed mapping request.
    MappedFile {
        /// Ordered tensor identifier.
        tensor_id: i32,
        /// Source split-file index.
        file_index: u16,
        /// Source byte offset.
        offset: u64,
        /// Required byte count.
        size: u64,
    },
    /// Read-and-copy request.
    ReadCopy {
        /// Ordered tensor identifier.
        tensor_id: i32,
        /// Source split-file index.
        file_index: u16,
        /// Source byte offset.
        offset: u64,
        /// Required byte count.
        size: u64,
    },
    /// External-buffer request.
    ExternalBuffer {
        /// Ordered tensor identifier.
        tensor_id: i32,
        /// Source split-file index.
        file_index: u16,
        /// Source byte offset.
        offset: u64,
        /// Required byte count.
        size: u64,
    },
    /// Staged-read request.
    StagedRead {
        /// Ordered tensor identifier.
        tensor_id: i32,
        /// Source split-file index.
        file_index: u16,
        /// Source byte offset.
        offset: u64,
        /// Required byte count.
        size: u64,
    },
}

/// Setup-allocated effect slots moved through planning unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct EffectBuffer(pub(crate) Box<[EffectRequest]>);

impl EffectBuffer {
    /// Creates a buffer whose slots must all be [`EffectRequest::Empty`].
    #[must_use]
    pub const fn new(effects: Box<[EffectRequest]>) -> Self {
        Self(effects)
    }

    /// Returns the number of preallocated slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the buffer contains no slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the planned requests.
    #[must_use]
    pub const fn effects(&self) -> &[EffectRequest] {
        &self.0
    }

    /// Resets every slot for allocation-free reuse by another plan.
    pub fn reset(&mut self) {
        self.0.fill(EffectRequest::Empty);
    }

    /// Returns the setup allocation.
    #[must_use]
    pub fn into_effects(self) -> Box<[EffectRequest]> {
        self.0
    }
}

/// Plan one ordered effect per active tensor.
#[derive(Debug, Eq, PartialEq)]
pub struct PlanLoad {
    pub(crate) strategy: StrategyKind,
    pub(crate) effects: EffectBuffer,
}

impl PlanLoad {
    /// Creates a load-plan request.
    #[must_use]
    pub const fn new(strategy: StrategyKind, effects: EffectBuffer) -> Self {
        Self { strategy, effects }
    }
}

/// Successful planning with the caller allocation returned.
#[derive(Debug, Eq, PartialEq)]
pub struct PlanLoadDone {
    effects: EffectBuffer,
    effect_count: usize,
}

impl PlanLoadDone {
    pub(crate) const fn new(effects: EffectBuffer, effect_count: usize) -> Self {
        Self {
            effects,
            effect_count,
        }
    }

    /// Returns the number of filled effect slots.
    #[must_use]
    pub const fn effect_count(&self) -> usize {
        self.effect_count
    }

    /// Returns the planned effect allocation.
    #[must_use]
    pub fn into_effects(self) -> EffectBuffer {
        self.effects
    }
}

/// Failed planning with the caller allocation returned unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct PlanLoadError {
    pub(crate) error: Error,
    pub(crate) effects: EffectBuffer,
}

impl PlanLoadError {
    pub(crate) const fn new(error: Error, effects: EffectBuffer) -> Self {
        Self { error, effects }
    }

    /// Returns the failure classification.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns the unchanged effect allocation.
    #[must_use]
    pub fn into_effects(self) -> EffectBuffer {
        self.effects
    }
}

/// Apply the actor-owned initial bytes for an ordered tensor-id batch.
#[derive(Debug, Eq, PartialEq)]
pub struct ApplyBoundEffectResults {
    pub(crate) tensor_ids: Box<[i32]>,
}

impl ApplyBoundEffectResults {
    /// Creates a bound-result batch.
    #[must_use]
    pub const fn new(tensor_ids: Box<[i32]>) -> Self {
        Self { tensor_ids }
    }
}

/// Failed bound-result application with input returned unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct ApplyBoundEffectResultsError {
    pub(crate) error: Error,
    pub(crate) tensor_ids: Box<[i32]>,
}

impl ApplyBoundEffectResultsError {
    pub(crate) const fn new(error: Error, tensor_ids: Box<[i32]>) -> Self {
        Self { error, tensor_ids }
    }

    /// Returns the failure classification.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns the unchanged result allocation.
    #[must_use]
    pub fn into_tensor_ids(self) -> Box<[i32]> {
        self.tensor_ids
    }
}

/// One owned backend result transferred into tensor residency.
#[derive(Debug, Eq, PartialEq)]
pub struct OwnedEffectResult {
    pub(crate) tensor_id: i32,
    pub(crate) bytes: Box<[u8]>,
}

impl OwnedEffectResult {
    /// Creates one owned backend result.
    #[must_use]
    pub const fn new(tensor_id: i32, bytes: Box<[u8]>) -> Self {
        Self { tensor_id, bytes }
    }

    /// Returns the tensor identifier.
    #[must_use]
    pub const fn tensor_id(&self) -> i32 {
        self.tensor_id
    }

    /// Returns the result bytes.
    #[must_use]
    pub const fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Apply an ordered batch of owned backend results.
#[derive(Debug, Eq, PartialEq)]
pub struct ApplyOwnedEffectResults {
    pub(crate) results: Box<[OwnedEffectResult]>,
}

impl ApplyOwnedEffectResults {
    /// Creates an owned-result batch.
    #[must_use]
    pub const fn new(results: Box<[OwnedEffectResult]>) -> Self {
        Self { results }
    }
}

/// Failed owned-result application with input returned unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct ApplyOwnedEffectResultsError {
    pub(crate) error: Error,
    pub(crate) results: Box<[OwnedEffectResult]>,
}

impl ApplyOwnedEffectResultsError {
    pub(crate) const fn new(error: Error, results: Box<[OwnedEffectResult]>) -> Self {
        Self { error, results }
    }

    /// Returns the failure classification.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns the unchanged result allocation.
    #[must_use]
    pub fn into_results(self) -> Box<[OwnedEffectResult]> {
        self.results
    }
}

/// Backend effect failure details preserved for typed dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EffectError {
    /// Backend rejected the request.
    Backend,
    /// Backend could not allocate required storage.
    OutOfMemory,
    /// Backend returned malformed or insufficient data.
    InvalidData,
    /// Backend-specific error code.
    Unknown(u16),
}

/// Report one failed planned effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApplyEffectError {
    pub(crate) tensor_id: i32,
    pub(crate) error: EffectError,
}

impl ApplyEffectError {
    /// Creates a backend-error result.
    #[must_use]
    pub const fn new(tensor_id: i32, error: EffectError) -> Self {
        Self { tensor_id, error }
    }
}

/// Observable tensor residency lifecycle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Lifecycle {
    /// No storage has been bound to the slot.
    #[default]
    Unbound,
    /// The actor owns immutable resident bytes.
    Resident,
    /// Previously resident bytes have been evicted.
    Evicted,
    /// A future mapped-load slice owns native mapped residency.
    MappedResident,
    /// A malformed mapped success remains owned only for release retry.
    MappedCleanupPending,
    /// The slot encountered an unrecoverable internal lifecycle failure.
    InternalError,
}

/// Metadata-only snapshot of one tensor slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorState {
    lifecycle: Lifecycle,
    buffer_bytes: u64,
    metadata: TensorMetadata,
}

impl TensorState {
    pub(crate) const fn new(
        lifecycle: Lifecycle,
        buffer_bytes: u64,
        metadata: TensorMetadata,
    ) -> Self {
        Self {
            lifecycle,
            buffer_bytes,
            metadata,
        }
    }

    /// Returns the slot lifecycle.
    #[must_use]
    pub const fn lifecycle(self) -> Lifecycle {
        self.lifecycle
    }

    /// Returns the resident byte count without exposing a pointer or view.
    #[must_use]
    pub const fn buffer_bytes(self) -> u64 {
        self.buffer_bytes
    }

    /// Returns the captured tensor metadata.
    #[must_use]
    pub const fn metadata(self) -> TensorMetadata {
        self.metadata
    }
}

/// Move setup-allocated bytes into actor-owned tensor residency.
#[derive(Debug)]
pub struct BindTensor {
    pub(crate) tensor_id: i32,
    pub(crate) metadata: TensorMetadata,
    pub(crate) bytes: Box<[u8]>,
}

impl BindTensor {
    /// Creates a bind request that transfers `bytes` into the actor.
    #[must_use]
    pub const fn new(tensor_id: i32, metadata: TensorMetadata, bytes: Box<[u8]>) -> Self {
        Self {
            tensor_id,
            metadata,
            bytes,
        }
    }
}

/// Successful outcome of [`BindTensor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindTensorDone {
    tensor_id: i32,
    buffer_bytes: u64,
}

impl BindTensorDone {
    pub(crate) const fn new(tensor_id: i32, buffer_bytes: u64) -> Self {
        Self {
            tensor_id,
            buffer_bytes,
        }
    }

    /// Returns the bound tensor identifier.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    /// Returns the actor-owned resident byte count.
    #[must_use]
    pub const fn buffer_bytes(self) -> u64 {
        self.buffer_bytes
    }
}

/// Remove actor-owned resident bytes from one tensor slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvictTensor {
    pub(crate) tensor_id: i32,
}

impl EvictTensor {
    /// Creates an eviction request.
    #[must_use]
    pub const fn new(tensor_id: i32) -> Self {
        Self { tensor_id }
    }
}

/// Successful eviction with ownership returned to the caller.
#[derive(Debug, Eq, PartialEq)]
pub struct EvictTensorDone {
    tensor_id: i32,
    bytes: Box<[u8]>,
}

impl EvictTensorDone {
    pub(crate) const fn new(tensor_id: i32, bytes: Box<[u8]>) -> Self {
        Self { tensor_id, bytes }
    }

    /// Returns the evicted tensor identifier.
    #[must_use]
    pub const fn tensor_id(&self) -> i32 {
        self.tensor_id
    }

    /// Returns the bytes whose ownership left the actor.
    #[must_use]
    pub fn into_bytes(self) -> Box<[u8]> {
        self.bytes
    }
}

/// Capture metadata-only state for one tensor slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureTensorState {
    pub(crate) tensor_id: i32,
}

/// Successful mapped residency outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedLoadDone {
    tensor_id: i32,
    mapping_handle: u32,
    mapped_bytes: u64,
}

impl MappedLoadDone {
    pub(crate) const fn new(tensor_id: i32, mapping_handle: u32, mapped_bytes: u64) -> Self {
        Self {
            tensor_id,
            mapping_handle,
            mapped_bytes,
        }
    }

    /// Returns the tensor identifier.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    /// Returns the release token owned by the mapper actor.
    #[must_use]
    pub const fn mapping_handle(self) -> u32 {
        self.mapping_handle
    }

    /// Returns the mapped byte length.
    #[must_use]
    pub const fn mapped_bytes(self) -> u64 {
        self.mapped_bytes
    }
}

/// Request file-backed residency through a caller-created mapping capability.
#[derive(Debug)]
pub struct MappedLoad {
    pub(crate) tensor_id: i32,
    pub(crate) file_index: u16,
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) source: mmap::event::MmapSource,
}

impl MappedLoad {
    /// Creates a mapped-load request without opening or recreating a file.
    #[must_use]
    pub const fn new(
        tensor_id: i32,
        source: mmap::event::MmapSource,
        offset: u64,
        len: u64,
    ) -> Self {
        Self {
            tensor_id,
            file_index: 0,
            offset,
            len,
            source,
        }
    }

    /// Sets the split-file index.
    #[must_use]
    pub const fn with_file_index(mut self, file_index: u16) -> Self {
        self.file_index = file_index;
        self
    }
}

/// Successful owned read or staged-copy residency outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedLoadDone {
    tensor_id: i32,
    bytes_copied: u64,
}

impl OwnedLoadDone {
    pub(crate) const fn new(tensor_id: i32, bytes_copied: u64) -> Self {
        Self {
            tensor_id,
            bytes_copied,
        }
    }

    /// Returns the tensor identifier.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    /// Returns the resident byte count.
    #[must_use]
    pub const fn bytes_copied(self) -> u64 {
        self.bytes_copied
    }
}

/// Request an owned read/copy into setup-allocated slot storage.
#[derive(Clone, Copy, Debug)]
pub struct ReadLoad<'source> {
    pub(crate) tensor_id: i32,
    pub(crate) file_index: u16,
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) file_path: &'source str,
    pub(crate) source: Option<&'source [u8]>,
    pub(crate) source_error: Option<read::event::SourceError>,
}

impl<'source> ReadLoad<'source> {
    /// Creates an owned read/copy request.
    #[must_use]
    pub const fn new(
        tensor_id: i32,
        file_path: &'source str,
        source: Option<&'source [u8]>,
        offset: u64,
        len: u64,
    ) -> Self {
        Self {
            tensor_id,
            file_index: 0,
            offset,
            len,
            file_path,
            source,
            source_error: None,
        }
    }

    /// Sets the split-file index.
    #[must_use]
    pub const fn with_file_index(mut self, file_index: u16) -> Self {
        self.file_index = file_index;
        self
    }

    /// Preserves a setup-time source acquisition failure for the child actor.
    #[must_use]
    pub const fn with_source_error(mut self, error: read::event::SourceError) -> Self {
        self.source_error = Some(error);
        self
    }
}

/// Request an owned staged copy into setup-allocated slot storage.
#[derive(Clone, Copy, Debug)]
pub struct StagedLoad<'source> {
    pub(crate) tensor_id: i32,
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) stage_chunk_bytes: u64,
    pub(crate) source: Option<&'source [u8]>,
}

impl<'source> StagedLoad<'source> {
    /// Creates an owned staged-copy request.
    #[must_use]
    pub const fn new(
        tensor_id: i32,
        source: Option<&'source [u8]>,
        offset: u64,
        len: u64,
        stage_chunk_bytes: u64,
    ) -> Self {
        Self {
            tensor_id,
            offset,
            len,
            stage_chunk_bytes,
            source,
        }
    }
}

/// Successful mapped release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseMappedDone {
    tensor_id: i32,
}

impl ReleaseMappedDone {
    pub(crate) const fn new(tensor_id: i32) -> Self {
        Self { tensor_id }
    }

    /// Returns the released tensor identifier.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }
}

/// Release file-backed residency through the owning mapper actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseMapped {
    pub(crate) tensor_id: i32,
    pub(crate) mapping_handle: u32,
}

impl ReleaseMapped {
    /// Creates a mapped-release request.
    #[must_use]
    pub const fn new(tensor_id: i32, mapping_handle: u32) -> Self {
        Self {
            tensor_id,
            mapping_handle,
        }
    }
}

/// A statically dispatched synchronous tensor operation.
pub trait TensorOperation {
    /// Result returned before dispatch completes.
    type Output;

    /// Applies the operation to immutable resident bytes.
    fn apply(&mut self, bytes: &[u8]) -> Self::Output;
}

impl<Output, Operation> TensorOperation for Operation
where
    Operation: FnMut(&[u8]) -> Output,
{
    type Output = Output;

    fn apply(&mut self, bytes: &[u8]) -> Self::Output {
        self(bytes)
    }
}

/// Invoke a concrete operation with resident bytes during the same RTC chain.
pub struct WithTensor<'operation, Operation> {
    pub(crate) tensor_id: i32,
    pub(crate) operation: &'operation mut Operation,
}

impl<'operation, Operation> WithTensor<'operation, Operation> {
    /// Creates a synchronous tensor-access request.
    #[must_use]
    pub const fn new(tensor_id: i32, operation: &'operation mut Operation) -> Self {
        Self {
            tensor_id,
            operation,
        }
    }
}

impl<Operation> fmt::Debug for WithTensor<'_, Operation> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithTensor")
            .field("tensor_id", &self.tensor_id)
            .finish_non_exhaustive()
    }
}

impl CaptureTensorState {
    /// Creates a state-capture request.
    #[must_use]
    pub const fn new(tensor_id: i32) -> Self {
        Self { tensor_id }
    }
}

impl sealed::Sealed for BindTensor {}
impl<D> Event<D> for BindTensor {
    type Output = Result<BindTensorDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.bind_tensor(self)
    }
}

impl sealed::Sealed for EvictTensor {}
impl<D> Event<D> for EvictTensor {
    type Output = Result<EvictTensorDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.evict_tensor(self)
    }
}

impl sealed::Sealed for CaptureTensorState {}
impl<D> Event<D> for CaptureTensorState {
    type Output = Result<TensorState, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.capture_tensor_state(self)
    }
}

impl sealed::Sealed for BindStorage {}
impl<D> Event<D> for BindStorage {
    type Output = Result<BindStorageDone, BindStorageError>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.bind_storage(self)
    }
}

impl sealed::Sealed for PlanLoad {}
impl<D> Event<D> for PlanLoad {
    type Output = Result<PlanLoadDone, PlanLoadError>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.plan_load(self)
    }
}

impl sealed::Sealed for ApplyBoundEffectResults {}
impl<D> Event<D> for ApplyBoundEffectResults {
    type Output = Result<(), ApplyBoundEffectResultsError>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.apply_bound_effect_results(self)
    }
}

impl sealed::Sealed for ApplyOwnedEffectResults {}
impl<D> Event<D> for ApplyOwnedEffectResults {
    type Output = Result<(), ApplyOwnedEffectResultsError>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.apply_owned_effect_results(self)
    }
}

impl sealed::Sealed for ApplyEffectError {}
impl<D> Event<D> for ApplyEffectError {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.apply_effect_error(self)
    }
}

impl sealed::Sealed for MappedLoad {}
impl<D> Event<D> for MappedLoad
where
    D: TensorDependencies,
{
    type Output = Result<MappedLoadDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.mapped_load(self)
    }
}

impl sealed::Sealed for ReadLoad<'_> {}
impl<D> Event<D> for ReadLoad<'_>
where
    D: TensorDependencies,
{
    type Output = Result<OwnedLoadDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.read_load(self)
    }
}

impl sealed::Sealed for StagedLoad<'_> {}
impl<D> Event<D> for StagedLoad<'_>
where
    D: TensorDependencies,
{
    type Output = Result<OwnedLoadDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.staged_load(self)
    }
}

impl sealed::Sealed for ReleaseMapped {}
impl<D> Event<D> for ReleaseMapped
where
    D: TensorDependencies,
{
    type Output = Result<ReleaseMappedDone, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.release_mapped(self)
    }
}

impl<Operation> sealed::Sealed for WithTensor<'_, Operation> where Operation: TensorOperation {}
impl<D, Operation> Event<D> for WithTensor<'_, Operation>
where
    D: TensorDependencies,
    Operation: TensorOperation,
{
    type Output = Result<Operation::Output, Error>;

    fn dispatch(self, actor: &mut Store<D>) -> Self::Output {
        actor.with_tensor(self)
    }
}
