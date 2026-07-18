//! Typed GGUF actor requests and outcomes.

use core::fmt;
use core::marker::PhantomData;
use std::sync::Arc;

use crate::Loader;
use crate::loader::{KvEntry, Requirements, TensorInfo};

pub use crate::loader::Error;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Loader`].
///
/// This trait is sealed; external crates dispatch the concrete events in this
/// module and cannot add events that bypass the actor boundary.
pub trait Event: sealed::Sealed {
    /// Result produced after the event runs to completion.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Loader) -> Self::Output;
}

/// Request validation and storage requirements for a GGUF image.
#[derive(Clone)]
pub struct Probe {
    source: Arc<[u8]>,
}

impl fmt::Debug for Probe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Probe")
            .field("source_bytes", &self.source.len())
            .finish_non_exhaustive()
    }
}

impl Probe {
    /// Creates a probe event owning an immutable GGUF source capability.
    #[must_use]
    pub const fn new(source: Arc<[u8]>) -> Self {
        Self { source }
    }

    pub(crate) fn into_source(self) -> Arc<[u8]> {
        self.source
    }
}

impl sealed::Sealed for Probe {}

impl Event for Probe {
    type Output = Result<ProbeDone, Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.probe(self)
    }
}

/// Successful outcome of a [`Probe`] event.
#[derive(Clone)]
pub struct ProbeDone {
    requirements: Requirements,
    source: Arc<[u8]>,
}

impl fmt::Debug for ProbeDone {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProbeDone")
            .field("tensor_count", &self.tensor_count())
            .field("metadata_count", &self.metadata_count())
            .field("max_key_bytes", &self.max_key_bytes())
            .field("max_value_bytes", &self.max_value_bytes())
            .field("tensor_data_bytes", &self.tensor_data_bytes())
            .field("source_bytes", &self.source.len())
            .finish_non_exhaustive()
    }
}

impl ProbeDone {
    pub(crate) const fn new(requirements: Requirements, source: Arc<[u8]>) -> Self {
        Self {
            requirements,
            source,
        }
    }

    /// Returns the number of tensor descriptors.
    #[must_use]
    pub const fn tensor_count(&self) -> u32 {
        self.requirements.tensor_count
    }

    /// Returns the number of key-value metadata entries.
    #[must_use]
    pub const fn metadata_count(&self) -> u32 {
        self.requirements.kv_count
    }

    /// Returns the longest serialized metadata key.
    #[must_use]
    pub const fn max_key_bytes(&self) -> u32 {
        self.requirements.max_key_bytes
    }

    /// Returns the longest serialized metadata value.
    #[must_use]
    pub const fn max_value_bytes(&self) -> u32 {
        self.requirements.max_value_bytes
    }

    /// Returns the total unpadded tensor payload bytes.
    #[must_use]
    pub const fn tensor_data_bytes(&self) -> u64 {
        self.requirements.tensor_data_bytes
    }

    /// Returns the conservative metadata arena size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the size cannot fit in `usize`.
    pub fn required_metadata_bytes(&self) -> Result<usize, Error> {
        self.requirements.required_kv_arena_bytes()
    }
}

/// Caller-allocated storage moved into a loader by [`Bind`].
///
/// Construction may allocate. Dispatching this value never allocates.
pub struct Storage {
    pub(crate) source: Arc<[u8]>,
    pub(crate) kv_arena: Vec<u8>,
    pub(crate) kv_entries: Vec<KvEntry>,
    pub(crate) tensors: Vec<TensorInfo>,
}

impl Storage {
    /// Allocates exactly the capacities described by a successful probe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] if a size cannot fit or allocation fails.
    pub fn exact(probe: ProbeDone) -> Result<Self, Error> {
        let metadata_bytes = probe.required_metadata_bytes()?;
        let metadata_entries =
            usize::try_from(probe.metadata_count()).map_err(|_| Error::Capacity)?;
        let tensors = usize::try_from(probe.tensor_count()).map_err(|_| Error::Capacity)?;
        Self::with_capacity(probe, metadata_bytes, metadata_entries, tensors)
    }

    /// Allocates explicit capacities before actor dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] if allocation fails.
    pub fn with_capacity(
        probe: ProbeDone,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            source: probe.source,
            kv_arena: zeroed_vec(metadata_bytes)?,
            kv_entries: zeroed_vec(metadata_entries)?,
            tensors: zeroed_vec(tensors)?,
        })
    }

    pub(crate) const fn capacities(&self) -> (usize, usize, usize) {
        (
            self.kv_arena.len(),
            self.kv_entries.len(),
            self.tensors.len(),
        )
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Storage")
            .field("metadata_bytes", &self.kv_arena.len())
            .field("metadata_entries", &self.kv_entries.len())
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

fn zeroed_vec<T: Clone + Default>(length: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Capacity)?;
    values.resize(length, T::default());
    Ok(values)
}

/// Move preallocated storage into the loader actor.
#[derive(Debug)]
pub struct Bind {
    storage: Storage,
}

impl Bind {
    /// Creates a bind event from storage allocated before dispatch.
    #[must_use]
    pub const fn new(storage: Storage) -> Self {
        Self { storage }
    }

    pub(crate) fn into_storage(self) -> Storage {
        self.storage
    }
}

/// Failed bind outcome returning ownership of the rejected storage.
pub struct BindError {
    error: Error,
    storage: Storage,
}

impl BindError {
    pub(crate) const fn new(error: Error, storage: Storage) -> Self {
        Self { error, storage }
    }

    /// Returns the classified loader error.
    #[must_use]
    pub const fn error(&self) -> Error {
        self.error
    }

    /// Returns ownership of the rejected source and preallocated buffers.
    #[must_use]
    pub fn into_storage(self) -> Storage {
        self.storage
    }
}

impl fmt::Debug for BindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BindError")
            .field("error", &self.error)
            .field("storage", &self.storage)
            .finish()
    }
}

impl fmt::Display for BindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for BindError {}

impl sealed::Sealed for Bind {}

impl Event for Bind {
    type Output = Result<(), BindError>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.bind(self)
    }
}

/// Request parsing into the most recently bound storage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parse;

impl Parse {
    /// Creates a parse event for the actor-owned bound source.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl sealed::Sealed for Parse {}

impl Event for Parse {
    type Output = Result<ParseDone, Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.parse(self)
    }
}

/// Successful allocation-free parse outcome.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ParseDone {
    requirements: Requirements,
}

impl ParseDone {
    pub(crate) const fn new(requirements: Requirements) -> Self {
        Self { requirements }
    }

    /// Returns the number of tensor descriptors.
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.requirements.tensor_count
    }

    /// Returns the number of metadata entries.
    #[must_use]
    pub const fn metadata_count(self) -> u32 {
        self.requirements.kv_count
    }
}

/// Typed metadata query failure.
///
/// A missing key is represented by `Ok(None)` and is never collapsed into an
/// error variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum QueryError {
    /// Metadata has not been parsed into the loader.
    NotParsed,
    /// The key exists with a wire kind incompatible with the requested view.
    TypeMismatch,
    /// The requested array index is outside the parsed array.
    IndexOutOfBounds,
    /// A numeric value cannot be represented by the requested semantic type.
    Range,
    /// Actor-owned parsed bytes violate the GGUF encoding invariant.
    Malformed,
    /// An internal query-machine invariant failed.
    Internal,
}

impl fmt::Display for QueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotParsed => "GGUF metadata has not been parsed",
            Self::TypeMismatch => "GGUF metadata type mismatch",
            Self::IndexOutOfBounds => "GGUF metadata array index is out of bounds",
            Self::Range => "GGUF metadata value is out of range",
            Self::Malformed => "malformed actor-owned GGUF metadata",
            Self::Internal => "internal GGUF metadata query error",
        })
    }
}

impl std::error::Error for QueryError {}

macro_rules! scalar_query {
    ($name:ident, $value:ty, $dispatch:ident) => {
        #[doc = concat!("Read an optional `", stringify!($value), "` metadata value.")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name<'a> {
            pub(crate) key: &'a [u8],
            pub(crate) result: Result<Option<$value>, QueryError>,
        }

        impl<'a> $name<'a> {
            /// Creates a query using an arbitrary byte key.
            #[must_use]
            pub const fn new(key: &'a [u8]) -> Self {
                Self {
                    key,
                    result: Err(QueryError::Internal),
                }
            }
        }

        impl sealed::Sealed for $name<'_> {}

        impl Event for $name<'_> {
            type Output = Result<Option<$value>, QueryError>;

            fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
                actor.$dispatch(&mut self);
                self.result
            }
        }
    };
}

scalar_query!(ReadUnsigned, u64, query);
scalar_query!(ReadSigned, i64, query);
scalar_query!(ReadF32, f32, query);
scalar_query!(ReadF64, f64, query);
scalar_query!(ReadBool, bool, query);

/// Invoke a statically dispatched callback with an optional byte string.
///
/// The callback runs synchronously inside the loader's run-to-completion
/// boundary. It must not allocate, retain the borrowed value, or re-enter the
/// loader actor.
pub struct WithString<'a, F, R> {
    pub(crate) key: &'a [u8],
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, QueryError>,
}

impl<'a, F, R> WithString<'a, F, R> {
    /// Creates a synchronous string query using an arbitrary byte key.
    #[must_use]
    pub const fn new(key: &'a [u8], callback: F) -> Self {
        Self {
            key,
            callback,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F, R> fmt::Debug for WithString<'_, F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithString")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<F, R> sealed::Sealed for WithString<'_, F, R> where F: for<'value> FnMut(&'value [u8]) -> R {}

impl<F, R> Event for WithString<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    type Output = Result<Option<R>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Semantic GGUF array element kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementKind {
    /// Unsigned 8-bit integer.
    Uint8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Signed 16-bit integer.
    Int16,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Signed 32-bit integer.
    Int32,
    /// IEEE-754 binary32.
    Float32,
    /// GGUF Boolean byte.
    Bool,
    /// Length-prefixed byte string.
    String,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Signed 64-bit integer.
    Int64,
    /// IEEE-754 binary64.
    Float64,
}

/// Stable semantic kind for one GGUF metadata value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataKind {
    /// Unsigned 8-bit integer.
    Uint8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Signed 16-bit integer.
    Int16,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Signed 32-bit integer.
    Int32,
    /// IEEE-754 binary32.
    Float32,
    /// GGUF Boolean byte.
    Bool,
    /// Length-prefixed byte string.
    String,
    /// Homogeneous GGUF array.
    Array,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Signed 64-bit integer.
    Int64,
    /// IEEE-754 binary64.
    Float64,
}

/// Stable semantic schema for one GGUF metadata entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetadataDescriptor {
    kind: MetadataKind,
    array_element_kind: Option<ElementKind>,
    array_length: Option<u64>,
}

impl MetadataDescriptor {
    pub(crate) const fn scalar(kind: MetadataKind) -> Self {
        Self {
            kind,
            array_element_kind: None,
            array_length: None,
        }
    }

    pub(crate) const fn array(element_kind: ElementKind, length: u64) -> Self {
        Self {
            kind: MetadataKind::Array,
            array_element_kind: Some(element_kind),
            array_length: Some(length),
        }
    }

    /// Returns the semantic metadata value kind.
    #[must_use]
    pub const fn kind(self) -> MetadataKind {
        self.kind
    }

    /// Returns the semantic element kind for an array.
    #[must_use]
    pub const fn array_element_kind(self) -> Option<ElementKind> {
        self.array_element_kind
    }

    /// Returns the element count for an array.
    #[must_use]
    pub const fn array_length(self) -> Option<u64> {
        self.array_length
    }
}

/// Invoke a callback with one indexed metadata key and semantic descriptor.
///
/// No value bytes, storage offsets, parsed records, or loader state cross this
/// boundary. A caller that needs a concrete typed value copies the key into
/// caller-owned preallocated storage, lets this callback return, and only then
/// dispatches the matching typed value event. The callback must not allocate,
/// retain the borrowed key, or re-enter the loader actor.
pub struct WithMetadataDescriptor<F, R> {
    pub(crate) index: u32,
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, QueryError>,
}

impl<F, R> WithMetadataDescriptor<F, R> {
    /// Creates a synchronous indexed metadata descriptor query.
    #[must_use]
    pub const fn new(index: u32, callback: F) -> Self {
        Self {
            index,
            callback,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F, R> fmt::Debug for WithMetadataDescriptor<F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithMetadataDescriptor")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<F, R> sealed::Sealed for WithMetadataDescriptor<F, R> where
    F: for<'value> FnMut(&'value [u8], MetadataDescriptor) -> R
{
}

impl<F, R> Event for WithMetadataDescriptor<F, R>
where
    F: for<'value> FnMut(&'value [u8], MetadataDescriptor) -> R,
{
    type Output = Result<Option<R>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.with_metadata_descriptor(&mut self);
        self.result
    }
}

/// Semantic descriptor for one tensor payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    tensor_type: u32,
    dimension_count: u32,
    dimensions: [u64; 4],
    data_offset: u64,
    data_size: u64,
    file_index: u16,
}

impl TensorDescriptor {
    pub(crate) const fn new(
        tensor_type: u32,
        dimension_count: u32,
        dimensions: [u64; 4],
        data_offset: u64,
        data_size: u64,
        file_index: u16,
    ) -> Self {
        Self {
            tensor_type,
            dimension_count,
            dimensions,
            data_offset,
            data_size,
            file_index,
        }
    }

    /// Returns the GGML tensor type identifier.
    #[must_use]
    pub const fn tensor_type(self) -> u32 {
        self.tensor_type
    }

    /// Returns the number of active dimensions.
    #[must_use]
    pub const fn dimension_count(self) -> u32 {
        self.dimension_count
    }

    /// Returns all four dimensions, with unused dimensions set to one.
    #[must_use]
    pub const fn dimensions(self) -> [u64; 4] {
        self.dimensions
    }

    /// Returns the tensor offset relative to the GGUF data section.
    #[must_use]
    pub const fn data_offset(self) -> u64 {
        self.data_offset
    }

    /// Returns the unpadded tensor payload size.
    #[must_use]
    pub const fn data_size(self) -> u64 {
        self.data_size
    }

    /// Returns the split-file index.
    #[must_use]
    pub const fn file_index(self) -> u16 {
        self.file_index
    }
}

/// Invoke a callback with one semantic tensor descriptor and payload.
///
/// The callback borrows the actor-owned bound source only for this synchronous
/// run-to-completion dispatch. It must not allocate, retain either borrowed
/// view, or re-enter the loader actor.
pub struct WithTensor<F, R> {
    pub(crate) index: u32,
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, QueryError>,
}

impl<F, R> WithTensor<F, R> {
    /// Creates a synchronous tensor query over the actor-owned source.
    #[must_use]
    pub const fn new(index: u32, callback: F) -> Self {
        Self {
            index,
            callback,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F, R> fmt::Debug for WithTensor<F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithTensor")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<F, R> sealed::Sealed for WithTensor<F, R> where
    F: for<'value> FnMut(&'value [u8], TensorDescriptor, &'value [u8]) -> R
{
}

impl<F, R> Event for WithTensor<F, R>
where
    F: for<'value> FnMut(&'value [u8], TensorDescriptor, &'value [u8]) -> R,
{
    type Output = Result<Option<R>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.with_tensor(&mut self);
        self.result
    }
}

/// Read the length of an array with an exact element kind.
#[derive(Clone, Copy, Debug)]
pub struct ReadArrayLength<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) element_kind: ElementKind,
    pub(crate) result: Result<Option<u64>, QueryError>,
}

impl<'a> ReadArrayLength<'a> {
    /// Creates an exact-kind array-length query.
    #[must_use]
    pub const fn new(key: &'a [u8], element_kind: ElementKind) -> Self {
        Self {
            key,
            element_kind,
            result: Err(QueryError::Internal),
        }
    }
}

impl sealed::Sealed for ReadArrayLength<'_> {}

impl Event for ReadArrayLength<'_> {
    type Output = Result<Option<u64>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Exact aggregate metrics for a validated GGUF string array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StringArrayMetrics {
    element_count: u64,
    total_string_bytes: u64,
}

impl StringArrayMetrics {
    pub(crate) const fn new(element_count: u64, total_string_bytes: u64) -> Self {
        Self {
            element_count,
            total_string_bytes,
        }
    }

    /// Returns the number of string elements.
    #[must_use]
    pub const fn element_count(self) -> u64 {
        self.element_count
    }

    /// Returns the exact sum of element payload bytes, excluding length prefixes.
    #[must_use]
    pub const fn total_string_bytes(self) -> u64 {
        self.total_string_bytes
    }
}

/// Read exact aggregate metrics for a string array.
#[derive(Clone, Copy, Debug)]
pub struct ReadStringArrayMetrics<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) result: Result<Option<StringArrayMetrics>, QueryError>,
}

impl<'a> ReadStringArrayMetrics<'a> {
    /// Creates a string-array aggregate query.
    #[must_use]
    pub const fn new(key: &'a [u8]) -> Self {
        Self {
            key,
            result: Err(QueryError::Internal),
        }
    }
}

impl sealed::Sealed for ReadStringArrayMetrics<'_> {}

impl Event for ReadStringArrayMetrics<'_> {
    type Output = Result<Option<StringArrayMetrics>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Exact aggregate metrics for an integer array decoded with raw unsigned rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsignedArrayMetrics {
    element_count: u64,
    maximum: u64,
}

impl UnsignedArrayMetrics {
    pub(crate) const fn new(element_count: u64, maximum: u64) -> Self {
        Self {
            element_count,
            maximum,
        }
    }
    /// Returns the number of elements.
    #[must_use]
    pub const fn element_count(self) -> u64 {
        self.element_count
    }
    /// Returns the maximum raw-unsigned decoded value, or zero for an empty array.
    #[must_use]
    pub const fn maximum(self) -> u64 {
        self.maximum
    }
}

/// Measure an integer array using raw unsigned decoding semantics.
#[derive(Clone, Copy, Debug)]
pub struct ReadUnsignedArrayMetrics<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) result: Result<Option<UnsignedArrayMetrics>, QueryError>,
}

impl<'a> ReadUnsignedArrayMetrics<'a> {
    /// Creates an integer-array aggregate query.
    #[must_use]
    pub const fn new(key: &'a [u8]) -> Self {
        Self {
            key,
            result: Err(QueryError::Internal),
        }
    }
}

impl sealed::Sealed for ReadUnsignedArrayMetrics<'_> {}
impl Event for ReadUnsignedArrayMetrics<'_> {
    type Output = Result<Option<UnsignedArrayMetrics>, QueryError>;
    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

macro_rules! array_element_query {
    ($name:ident, $value:ty) => {
        #[doc = concat!("Read one optional array element as `", stringify!($value), "`.")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name<'a> {
            pub(crate) key: &'a [u8],
            pub(crate) index: u64,
            pub(crate) result: Result<Option<$value>, QueryError>,
        }

        impl<'a> $name<'a> {
            /// Creates an array-element query.
            #[must_use]
            pub const fn new(key: &'a [u8], index: u64) -> Self {
                Self {
                    key,
                    index,
                    result: Err(QueryError::Internal),
                }
            }
        }

        impl sealed::Sealed for $name<'_> {}

        impl Event for $name<'_> {
            type Output = Result<Option<$value>, QueryError>;

            fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
                actor.query(&mut self);
                self.result
            }
        }
    };
}

array_element_query!(ReadUnsignedArrayElement, u64);
array_element_query!(ReadSignedArrayElement, i64);
array_element_query!(ReadF32ArrayElement, f32);
array_element_query!(ReadF64ArrayElement, f64);
array_element_query!(ReadBoolArrayElement, bool);

/// Invoke a callback with one optional string-array element.
///
/// The callback runs synchronously inside the loader's run-to-completion
/// boundary. It must not allocate, retain the borrowed value, or re-enter the
/// loader actor.
pub struct WithStringArrayElement<'a, F, R> {
    pub(crate) key: &'a [u8],
    pub(crate) index: u64,
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, QueryError>,
}

impl<'a, F, R> WithStringArrayElement<'a, F, R> {
    /// Creates a synchronous string-array element query.
    #[must_use]
    pub const fn new(key: &'a [u8], index: u64, callback: F) -> Self {
        Self {
            key,
            index,
            callback,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F, R> fmt::Debug for WithStringArrayElement<'_, F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithStringArrayElement")
            .field("key", &self.key)
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<F, R> sealed::Sealed for WithStringArrayElement<'_, F, R> where
    F: for<'value> FnMut(&'value [u8]) -> R
{
}

impl<F, R> Event for WithStringArrayElement<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    type Output = Result<Option<R>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Visit every string-array element synchronously in serialized order.
///
/// The visitor runs inside one loader run-to-completion boundary. It must not
/// allocate, retain a borrowed element, or re-enter the loader actor.
pub struct VisitStringArray<'a, F> {
    pub(crate) key: &'a [u8],
    pub(crate) visitor: F,
    pub(crate) result: Result<Option<u64>, QueryError>,
}

impl<'a, F> VisitStringArray<'a, F> {
    /// Creates a string-array visitor.
    #[must_use]
    pub const fn new(key: &'a [u8], visitor: F) -> Self {
        Self {
            key,
            visitor,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F> fmt::Debug for VisitStringArray<'_, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisitStringArray")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<F> sealed::Sealed for VisitStringArray<'_, F> where F: for<'value> FnMut(u32, &'value [u8]) {}

impl<F> Event for VisitStringArray<'_, F>
where
    F: for<'value> FnMut(u32, &'value [u8]),
{
    type Output = Result<Option<u64>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Visit every float-array element as `f32`, including exact `f64` coercion.
///
/// The visitor runs synchronously, must not allocate or retain data, and must
/// not re-enter the loader actor.
pub struct VisitF32Array<'a, F> {
    pub(crate) key: &'a [u8],
    pub(crate) visitor: F,
    pub(crate) result: Result<Option<u64>, QueryError>,
}

impl<'a, F> VisitF32Array<'a, F> {
    /// Creates a bulk float-array visitor.
    #[must_use]
    pub const fn new(key: &'a [u8], visitor: F) -> Self {
        Self {
            key,
            visitor,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F> fmt::Debug for VisitF32Array<'_, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisitF32Array")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}
impl<F> sealed::Sealed for VisitF32Array<'_, F> where F: FnMut(u32, f32) {}
impl<F> Event for VisitF32Array<'_, F>
where
    F: FnMut(u32, f32),
{
    type Output = Result<Option<u64>, QueryError>;
    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Visit every integer-array element with raw unsigned decoding semantics.
///
/// The visitor runs synchronously, must not allocate or retain data, and must
/// not re-enter the loader actor.
pub struct VisitUnsignedArray<'a, F> {
    pub(crate) key: &'a [u8],
    pub(crate) visitor: F,
    pub(crate) result: Result<Option<u64>, QueryError>,
}

impl<'a, F> VisitUnsignedArray<'a, F> {
    /// Creates a bulk integer-array visitor.
    #[must_use]
    pub const fn new(key: &'a [u8], visitor: F) -> Self {
        Self {
            key,
            visitor,
            result: Err(QueryError::Internal),
        }
    }
}

impl<F> fmt::Debug for VisitUnsignedArray<'_, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisitUnsignedArray")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}
impl<F> sealed::Sealed for VisitUnsignedArray<'_, F> where F: FnMut(u32, u64) {}
impl<F> Event for VisitUnsignedArray<'_, F>
where
    F: FnMut(u32, u64),
{
    type Output = Result<Option<u64>, QueryError>;
    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

/// Invoke a callback with an optional packed `u8`/`i8` array payload.
///
/// The callback runs synchronously inside the loader's run-to-completion
/// boundary. It must not allocate, retain the borrowed payload, or re-enter the
/// loader actor.
pub struct WithByteArray<'a, F, R> {
    pub(crate) key: &'a [u8],
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, QueryError>,
    pub(crate) marker: PhantomData<fn() -> R>,
}

impl<'a, F, R> WithByteArray<'a, F, R> {
    /// Creates a synchronous byte-array query.
    #[must_use]
    pub const fn new(key: &'a [u8], callback: F) -> Self {
        Self {
            key,
            callback,
            result: Err(QueryError::Internal),
            marker: PhantomData,
        }
    }
}

impl<F, R> fmt::Debug for WithByteArray<'_, F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithByteArray")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<F, R> sealed::Sealed for WithByteArray<'_, F, R> where F: for<'value> FnMut(&'value [u8]) -> R {}

impl<F, R> Event for WithByteArray<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    type Output = Result<Option<R>, QueryError>;

    fn dispatch(mut self, actor: &mut Loader) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}
