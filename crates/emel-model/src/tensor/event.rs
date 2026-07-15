//! Typed requests and outcomes for the tensor store actor.

use core::fmt;

use super::Store;

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
