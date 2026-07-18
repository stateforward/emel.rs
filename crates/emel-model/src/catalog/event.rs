//! Typed public requests, outcomes, and opaque catalog storage.

use core::fmt;
use core::marker::PhantomData;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::ParseDone;
use emel_tensor::dtype::SerializedType;

use super::Catalog;
use super::storage::{EMPTY_INDEX, Record};

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Catalog`].
pub trait Event: sealed::Sealed {
    /// Result produced before the dispatch returns.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Catalog) -> Self::Output;
}

/// Catalog protocol or model-validation failure.
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
    WrongTensorIdentity,
    StaleTensorIdentity,
    WrongNameIdentity,
    StaleNameIdentity,
    UnexpectedEvent,
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid catalog request",
            Self::ModelInvalid => "model catalog input is invalid",
            Self::Capacity => "catalog storage capacity is unavailable",
            Self::StorageUnavailable => "catalog storage is unavailable",
            Self::Busy => "catalog is busy",
            Self::WrongModelIdentity => "model identity belongs to another catalog",
            Self::StaleModelIdentity => "model identity is stale",
            Self::WrongTensorIdentity => "tensor identity belongs to another catalog",
            Self::StaleTensorIdentity => "tensor identity is stale",
            Self::WrongNameIdentity => "name identity belongs to another catalog",
            Self::StaleNameIdentity => "name identity is stale",
            Self::UnexpectedEvent => "unexpected catalog event",
            Self::Internal => "internal catalog error",
        })
    }
}

impl std::error::Error for Error {}

/// Semantic pre-dispatch input for one tensor record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorInput<'a> {
    name: &'a [u8],
    wire_type: u32,
    dimension_count: i32,
    dimensions: [i64; 4],
    data_size: u64,
    binding_present: bool,
}

impl<'a> TensorInput<'a> {
    /// Creates one raw serialized record. Validation occurs during `SealModel`.
    #[must_use]
    pub const fn new(
        name: &'a [u8],
        wire_type: u32,
        dimension_count: i32,
        dimensions: [i64; 4],
        data_size: u64,
        binding_present: bool,
    ) -> Self {
        Self {
            name,
            wire_type,
            dimension_count,
            dimensions,
            data_size,
            binding_present,
        }
    }

    /// Returns the caller-owned name length for exact storage sizing.
    #[must_use]
    pub const fn name_len(self) -> usize {
        self.name.len()
    }
}

/// Opaque caller-owned storage populated before catalog dispatch.
pub struct Storage {
    pub(super) names: Vec<u8>,
    pub(super) records: Vec<Record>,
    pub(super) index: Vec<u32>,
    pub(super) tensor_count: usize,
    pub(super) name_bytes_used: usize,
}

impl Storage {
    /// Allocates all bounded catalog regions before actor dispatch.
    ///
    /// The index capacity must cover every tensor slot so construction leaves
    /// no seal-time ambiguity about duplicate-name cardinality.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] for invalid bounds or allocation failure.
    pub fn with_capacity(
        tensor_capacity: usize,
        name_capacity: usize,
        index_capacity: usize,
    ) -> Result<Self, Error> {
        if tensor_capacity == 0
            || tensor_capacity > super::MAX_TENSORS
            || name_capacity > super::MAX_NAME_BYTES
            || index_capacity != tensor_capacity
        {
            return Err(Error::Capacity);
        }
        Ok(Self {
            names: zeroed_vec(name_capacity)?,
            records: zeroed_vec(tensor_capacity)?,
            index: filled_vec(index_capacity, EMPTY_INDEX)?,
            tensor_count: 0,
            name_bytes_used: 0,
        })
    }

    /// Builds storage through the maintained public GGUF observer boundary.
    ///
    /// The parsed loader is traversed twice: once to size the name arena and
    /// once to copy semantic records into already-allocated catalog storage.
    ///
    /// # Errors
    ///
    /// Returns a typed catalog error for unavailable or malformed GGUF data,
    /// checked conversion failure, or storage capacity failure.
    pub fn from_gguf(loader: &mut GgufLoader, parsed: ParseDone) -> Result<Self, Error> {
        super::ingest::from_gguf(loader, parsed)
    }

    /// Appends one semantic record without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] if a fixed record or name region is full.
    pub fn push_tensor(&mut self, input: TensorInput<'_>) -> Result<(), Error> {
        if self.tensor_count >= self.records.len() {
            return Err(Error::Capacity);
        }
        let end = self
            .name_bytes_used
            .checked_add(input.name.len())
            .ok_or(Error::Capacity)?;
        if end > self.names.len() {
            return Err(Error::Capacity);
        }
        let name_offset = u32::try_from(self.name_bytes_used).map_err(|_| Error::Capacity)?;
        let name_length = u32::try_from(input.name.len()).map_err(|_| Error::Capacity)?;
        self.names[self.name_bytes_used..end].copy_from_slice(input.name);
        self.records[self.tensor_count] = Record {
            name_offset,
            name_length,
            wire_type: input.wire_type,
            dimension_count: input.dimension_count,
            dimensions: input.dimensions,
            data_size: input.data_size,
            binding_present: input.binding_present,
        };
        self.tensor_count += 1;
        self.name_bytes_used = end;
        Ok(())
    }

    /// Clears populated records for allocation-free caller-side reuse.
    pub fn clear(&mut self) {
        self.names[..self.name_bytes_used].fill(0);
        self.records[..self.tensor_count].fill(Record::default());
        self.index.fill(EMPTY_INDEX);
        self.tensor_count = 0;
        self.name_bytes_used = 0;
    }

    /// Returns the fixed tensor capacity without exposing records.
    #[must_use]
    pub const fn tensor_capacity(&self) -> usize {
        self.records.len()
    }

    /// Returns the fixed name-arena capacity without exposing bytes.
    #[must_use]
    pub const fn name_capacity(&self) -> usize {
        self.names.len()
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Storage")
            .field("tensor_capacity", &self.records.len())
            .field("name_capacity", &self.names.len())
            .field("index_capacity", &self.index.len())
            .field("tensor_count", &self.tensor_count)
            .finish_non_exhaustive()
    }
}

fn zeroed_vec<T: Clone + Default>(length: usize) -> Result<Vec<T>, Error> {
    filled_vec(length, T::default())
}

fn filled_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Capacity)?;
    values.resize(length, value);
    Ok(values)
}

/// Opaque identity for one sealed catalog generation.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ModelIdentity {
    pub(super) owner: u64,
    pub(super) generation: u64,
}

impl fmt::Debug for ModelIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelIdentity")
            .finish_non_exhaustive()
    }
}

macro_rules! opaque_item_identity {
    ($name:ident) => {
        #[derive(Clone, Copy, Eq, Hash, PartialEq)]
        pub struct $name {
            pub(super) owner: u64,
            pub(super) generation: u64,
            pub(super) ordinal: u32,
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .finish_non_exhaustive()
            }
        }
    };
}

opaque_item_identity!(TensorId);
opaque_item_identity!(NameId);

/// Immutable scalar catalog facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelDescriptor {
    tensor_count: u32,
}

impl ModelDescriptor {
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }

    pub(super) const fn new(tensor_count: u32) -> Self {
        Self { tensor_count }
    }
}

/// Pinned source-compatible tensor storage outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorBindingStatus {
    Bound,
    Unbound,
}

/// Public semantic tensor facts with opaque stable identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    tensor_id: TensorId,
    name_id: NameId,
    tensor_type: SerializedType,
    dimension_count: i32,
    dimensions: [i64; 4],
    data_size: u64,
    binding_present: bool,
    binding_status: TensorBindingStatus,
}

#[derive(Clone, Copy)]
pub(super) struct TensorDescriptorFacts {
    pub(super) tensor_type: SerializedType,
    pub(super) dimension_count: i32,
    pub(super) dimensions: [i64; 4],
    pub(super) data_size: u64,
    pub(super) binding_present: bool,
    pub(super) binding_status: TensorBindingStatus,
}

impl TensorDescriptor {
    #[must_use]
    pub const fn tensor_id(self) -> TensorId {
        self.tensor_id
    }
    #[must_use]
    pub const fn name_id(self) -> NameId {
        self.name_id
    }
    #[must_use]
    pub const fn tensor_type(self) -> SerializedType {
        self.tensor_type
    }
    #[must_use]
    pub const fn dimension_count(self) -> i32 {
        self.dimension_count
    }
    #[must_use]
    pub const fn dimensions(self) -> [i64; 4] {
        self.dimensions
    }
    #[must_use]
    pub const fn data_size(self) -> u64 {
        self.data_size
    }
    #[must_use]
    pub const fn binding_present(self) -> bool {
        self.binding_present
    }
    #[must_use]
    pub const fn binding_status(self) -> TensorBindingStatus {
        self.binding_status
    }

    pub(super) const fn new(
        tensor_id: TensorId,
        name_id: NameId,
        facts: TensorDescriptorFacts,
    ) -> Self {
        Self {
            tensor_id,
            name_id,
            tensor_type: facts.tensor_type,
            dimension_count: facts.dimension_count,
            dimensions: facts.dimensions,
            data_size: facts.data_size,
            binding_present: facts.binding_present,
            binding_status: facts.binding_status,
        }
    }
}

/// Moves caller-populated storage into an empty catalog.
#[derive(Debug)]
pub struct BindStorage {
    pub(super) storage: Storage,
}
impl BindStorage {
    #[must_use]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }
}

/// Rejected bind that returns ownership of storage.
#[derive(Debug)]
pub struct BindStorageError {
    error: Error,
    storage: Storage,
}
impl BindStorageError {
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
impl fmt::Display for BindStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for BindStorageError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct SealModel;
impl SealModel {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DescribeModel {
    pub(super) identity: ModelIdentity,
}
impl DescribeModel {
    #[must_use]
    pub const fn new(identity: ModelIdentity) -> Self {
        Self { identity }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FindTensor<'a> {
    pub(super) identity: ModelIdentity,
    pub(super) name: &'a [u8],
}
impl<'a> FindTensor<'a> {
    #[must_use]
    pub const fn new(identity: ModelIdentity, name: &'a [u8]) -> Self {
        Self { identity, name }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DescribeTensor {
    pub(super) tensor_id: TensorId,
}
impl DescribeTensor {
    #[must_use]
    pub const fn new(tensor_id: TensorId) -> Self {
        Self { tensor_id }
    }
}

/// Invokes a callback over one canonical actor-owned name.
///
/// The callback runs synchronously in the catalog's query state-machine action.
/// It must not allocate, retain the borrowed name, or re-enter this catalog.
pub struct WithTensorName<F, R> {
    pub(super) name_id: NameId,
    pub(super) callback: F,
    pub(super) result: Result<Option<R>, Error>,
    marker: PhantomData<fn() -> R>,
}
impl<F, R> WithTensorName<F, R> {
    #[must_use]
    pub const fn new(name_id: NameId, callback: F) -> Self {
        Self {
            name_id,
            callback,
            result: Err(Error::Internal),
            marker: PhantomData,
        }
    }
}
impl<F, R> fmt::Debug for WithTensorName<F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithTensorName").finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Reset;
impl Reset {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReleaseStorage;
impl ReleaseStorage {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

macro_rules! simple_event {
    ($event:ty, $output:ty, $method:ident) => {
        impl sealed::Sealed for $event {}
        impl Event for $event {
            type Output = $output;
            fn dispatch(self, actor: &mut Catalog) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

simple_event!(BindStorage, Result<(), BindStorageError>, bind_storage);
simple_event!(SealModel, Result<ModelIdentity, Error>, seal_model);
simple_event!(DescribeModel, Result<ModelDescriptor, Error>, describe_model);
simple_event!(
    FindTensor<'_>,
    Result<Option<TensorDescriptor>, Error>,
    find_tensor
);
simple_event!(DescribeTensor, Result<TensorDescriptor, Error>, describe_tensor);
simple_event!(Reset, Result<(), Error>, reset);
simple_event!(ReleaseStorage, Result<Storage, Error>, release_storage);

impl<F, R> sealed::Sealed for WithTensorName<F, R> where F: for<'name> FnMut(&'name [u8]) -> R {}
impl<F, R> Event for WithTensorName<F, R>
where
    F: for<'name> FnMut(&'name [u8]) -> R,
{
    type Output = Result<Option<R>, Error>;
    fn dispatch(mut self, actor: &mut Catalog) -> Self::Output {
        actor.with_tensor_name(&mut self);
        self.result
    }
}
