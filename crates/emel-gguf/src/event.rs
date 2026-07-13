//! Typed GGUF actor requests and outcomes.

use crate::Loader;
use crate::loader::{Gguf, Requirements};

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
#[derive(Clone, Copy, Debug)]
pub struct Probe<'a> {
    file_image: &'a [u8],
}

impl<'a> Probe<'a> {
    /// Creates a probe event borrowing the complete GGUF image.
    #[must_use]
    pub const fn new(file_image: &'a [u8]) -> Self {
        Self { file_image }
    }

    pub(crate) const fn file_image(self) -> &'a [u8] {
        self.file_image
    }
}

impl sealed::Sealed for Probe<'_> {}

impl Event for Probe<'_> {
    type Output = Result<ProbeDone, Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.probe(self)
    }
}

/// Successful outcome of a [`Probe`] event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProbeDone {
    requirements: Requirements,
}

impl ProbeDone {
    pub(crate) const fn new(requirements: Requirements) -> Self {
        Self { requirements }
    }

    /// Returns the number of tensor descriptors.
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.requirements.tensor_count
    }

    /// Returns the number of key-value metadata entries.
    #[must_use]
    pub const fn metadata_count(self) -> u32 {
        self.requirements.kv_count
    }

    /// Returns the longest serialized metadata key.
    #[must_use]
    pub const fn max_key_bytes(self) -> u32 {
        self.requirements.max_key_bytes
    }

    /// Returns the longest serialized metadata value.
    #[must_use]
    pub const fn max_value_bytes(self) -> u32 {
        self.requirements.max_value_bytes
    }

    /// Returns the total unpadded tensor payload bytes.
    #[must_use]
    pub const fn tensor_data_bytes(self) -> u64 {
        self.requirements.tensor_data_bytes
    }

    /// Returns the conservative metadata arena size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the size cannot fit in `usize`.
    pub fn required_metadata_bytes(self) -> Result<usize, Error> {
        self.requirements.required_kv_arena_bytes()
    }
}

/// Request binding of loader-owned storage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bind {
    capacity: Option<StorageCapacity>,
}

impl Bind {
    /// Uses the exact capacity discovered by the latest successful [`Probe`].
    #[must_use]
    pub const fn exact() -> Self {
        Self { capacity: None }
    }

    /// Uses explicit capacities, primarily for caller-controlled arenas and
    /// lifecycle testing.
    #[must_use]
    pub const fn with_capacity(
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Self {
        Self {
            capacity: Some(StorageCapacity {
                metadata_bytes,
                metadata_entries,
                tensors,
            }),
        }
    }

    pub(crate) const fn capacity(self) -> Option<(usize, usize, usize)> {
        match self.capacity {
            Some(capacity) => Some((
                capacity.metadata_bytes,
                capacity.metadata_entries,
                capacity.tensors,
            )),
            None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StorageCapacity {
    metadata_bytes: usize,
    metadata_entries: usize,
    tensors: usize,
}

impl sealed::Sealed for Bind {}

impl Event for Bind {
    type Output = Result<(), Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.bind(self)
    }
}

/// Request parsing into the most recently bound storage.
#[derive(Clone, Copy, Debug)]
pub struct Parse<'a> {
    file_image: &'a [u8],
}

impl<'a> Parse<'a> {
    /// Creates a parse event borrowing the complete GGUF image.
    #[must_use]
    pub const fn new(file_image: &'a [u8]) -> Self {
        Self { file_image }
    }

    pub(crate) const fn file_image(self) -> &'a [u8] {
        self.file_image
    }
}

impl sealed::Sealed for Parse<'_> {}

impl<'a> Event for Parse<'a> {
    type Output = Result<ParseDone<'a>, Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.parse(self)
    }
}

/// Request the complete probe, bind, and parse lifecycle.
#[derive(Clone, Copy, Debug)]
pub struct Load<'a> {
    file_image: &'a [u8],
}

impl<'a> Load<'a> {
    /// Creates a one-shot load event borrowing the complete GGUF image.
    #[must_use]
    pub const fn new(file_image: &'a [u8]) -> Self {
        Self { file_image }
    }

    pub(crate) const fn file_image(self) -> &'a [u8] {
        self.file_image
    }
}

impl sealed::Sealed for Load<'_> {}

impl<'a> Event for Load<'a> {
    type Output = Result<ParseDone<'a>, Error>;

    fn dispatch(self, actor: &mut Loader) -> Self::Output {
        actor.load(self)
    }
}

/// Successful outcome of a [`Parse`] or [`Load`] event.
#[derive(Clone, Debug)]
pub struct ParseDone<'a> {
    model: Gguf<'a>,
}

impl<'a> ParseDone<'a> {
    pub(crate) const fn new(model: Gguf<'a>) -> Self {
        Self { model }
    }

    /// Returns the requirements recorded during probing.
    #[must_use]
    pub const fn probe(&self) -> ProbeDone {
        ProbeDone::new(self.model.requirements())
    }

    /// Iterates over validated metadata views.
    ///
    /// # Panics
    ///
    /// Panics only if the private parsed storage violates invariants established
    /// by the loader actor.
    pub fn metadata(&self) -> impl ExactSizeIterator<Item = Metadata<'_>> + '_ {
        self.model.kv_entries().iter().map(|entry| Metadata {
            key: self.model.key(entry).expect("validated metadata key"),
            value: self.model.value(entry).expect("validated metadata value"),
            value_type: entry.value_type,
        })
    }

    /// Iterates over validated tensor views.
    ///
    /// # Panics
    ///
    /// Panics only if the private parsed storage violates invariants established
    /// by the loader actor.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = Tensor<'_>> + '_ {
        self.model.tensors().iter().map(|tensor| Tensor {
            name: self
                .model
                .tensor_name(tensor)
                .expect("validated tensor name"),
            data: self
                .model
                .tensor_data(tensor)
                .expect("validated tensor payload"),
            kind: tensor.tensor_type,
            dimension_count: tensor.dimension_count,
            dimensions: tensor.dimensions,
            data_offset: tensor.data_offset,
            file_offset: tensor.file_offset,
            data_size: tensor.data_size,
            file_index: tensor.file_index,
        })
    }
}

/// Read-only metadata view produced by [`ParseDone::metadata`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Metadata<'a> {
    key: &'a [u8],
    value: &'a [u8],
    value_type: u32,
}

impl<'a> Metadata<'a> {
    /// Returns the raw metadata key.
    #[must_use]
    pub const fn key(self) -> &'a [u8] {
        self.key
    }

    /// Returns the raw GGUF value-type tag.
    #[must_use]
    pub const fn value_type(self) -> u32 {
        self.value_type
    }

    /// Returns the serialized metadata value.
    #[must_use]
    pub const fn value(self) -> &'a [u8] {
        self.value
    }
}

/// Read-only tensor view produced by [`ParseDone::tensors`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor<'a> {
    name: &'a [u8],
    data: &'a [u8],
    kind: u32,
    dimension_count: u32,
    dimensions: [u64; 4],
    data_offset: u64,
    file_offset: u64,
    data_size: u64,
    file_index: u16,
}

impl<'a> Tensor<'a> {
    /// Returns the tensor name.
    #[must_use]
    pub const fn name(self) -> &'a [u8] {
        self.name
    }

    /// Returns the raw GGML tensor-type tag.
    #[must_use]
    pub const fn tensor_type(self) -> u32 {
        self.kind
    }

    /// Returns the number of active dimensions.
    #[must_use]
    pub const fn dimension_count(self) -> u32 {
        self.dimension_count
    }

    /// Returns all dimensions, with unused entries set to one.
    #[must_use]
    pub const fn dimensions(self) -> [u64; 4] {
        self.dimensions
    }

    /// Returns the offset relative to the tensor-data section.
    #[must_use]
    pub const fn data_offset(self) -> u64 {
        self.data_offset
    }

    /// Returns the absolute offset in the file image.
    #[must_use]
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    /// Returns the split-file index.
    #[must_use]
    pub const fn file_index(self) -> u16 {
        self.file_index
    }

    /// Returns the tensor payload without copying it.
    #[must_use]
    pub const fn data(self) -> &'a [u8] {
        self.data
    }

    /// Returns the unpadded tensor payload size.
    #[must_use]
    pub const fn data_size(self) -> u64 {
        self.data_size
    }
}
