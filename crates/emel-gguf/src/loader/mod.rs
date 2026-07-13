//! Stateful GGUF loader and its parsed records.

mod detail;
mod sm;

use core::fmt;

use self::sm::{
    BoundStorage, EventBindRuntime, GgufLoaderContext, GgufLoaderEvents, GgufLoaderStateMachine,
    GgufLoaderStates,
};

/// Storage requirements discovered by probing a GGUF image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Requirements {
    /// Number of tensor descriptors.
    pub tensor_count: u32,
    /// Number of key-value metadata entries.
    pub kv_count: u32,
    /// Longest serialized metadata key.
    pub max_key_bytes: u32,
    /// Longest serialized metadata value.
    pub max_value_bytes: u32,
    /// Total unpadded tensor payload bytes.
    pub tensor_data_bytes: u64,
}

impl Requirements {
    /// Returns the conservative metadata arena size used by the C++ loader.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the size cannot fit in `usize`.
    pub fn required_kv_arena_bytes(self) -> Result<usize, Error> {
        detail::required_kv_arena_bytes(self)
    }
}

/// Location of a serialized GGUF metadata entry in the bound arena.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvEntry {
    /// Byte offset of the key in the arena.
    pub key_offset: u32,
    /// Key length in bytes.
    pub key_length: u32,
    /// Byte offset of the serialized value in the arena.
    pub value_offset: u32,
    /// Serialized value length in bytes.
    pub value_length: u32,
    /// Raw GGUF value-type tag.
    pub value_type: u32,
}

/// Metadata describing one tensor payload in the GGUF image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorInfo {
    /// Byte offset of the tensor name in the original file image.
    pub name_offset: u32,
    /// Tensor name length in bytes.
    pub name_length: u32,
    /// Raw GGML tensor-type tag.
    pub tensor_type: u32,
    /// Number of active entries in [`Self::dimensions`].
    pub dimension_count: u32,
    /// Tensor dimensions, with unused entries set to one.
    pub dimensions: [u64; 4],
    /// Tensor offset relative to the tensor-data section.
    pub data_offset: u64,
    /// Absolute tensor offset in the original file image.
    pub file_offset: u64,
    /// Unpadded tensor payload size.
    pub data_size: u64,
    /// Split-file index. The current loader supports the primary image only.
    pub file_index: u16,
}

/// A probed and parsed GGUF image.
#[derive(Clone, Debug)]
pub struct Gguf<'a> {
    file_image: &'a [u8],
    requirements: Requirements,
    kv_arena: Vec<u8>,
    kv_entries: Vec<KvEntry>,
    tensors: Vec<TensorInfo>,
}

impl<'a> Gguf<'a> {
    /// Returns the immutable input image backing tensor payloads and names.
    #[must_use]
    pub const fn file_image(&self) -> &'a [u8] {
        self.file_image
    }

    /// Returns the requirements computed during probing.
    #[must_use]
    pub const fn requirements(&self) -> Requirements {
        self.requirements
    }

    /// Returns parsed metadata-entry locations.
    #[must_use]
    pub fn kv_entries(&self) -> &[KvEntry] {
        &self.kv_entries
    }

    /// Returns parsed tensor descriptors.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns the raw key bytes for an entry.
    #[must_use]
    pub fn key(&self, entry: &KvEntry) -> Option<&[u8]> {
        range(&self.kv_arena, entry.key_offset, entry.key_length)
    }

    /// Returns the serialized value bytes for an entry.
    #[must_use]
    pub fn value(&self, entry: &KvEntry) -> Option<&[u8]> {
        range(&self.kv_arena, entry.value_offset, entry.value_length)
    }

    /// Returns a tensor name as raw bytes.
    #[must_use]
    pub fn tensor_name(&self, tensor: &TensorInfo) -> Option<&'a [u8]> {
        range(self.file_image, tensor.name_offset, tensor.name_length)
    }

    /// Returns a tensor payload without copying it.
    #[must_use]
    pub fn tensor_data(&self, tensor: &TensorInfo) -> Option<&'a [u8]> {
        let offset = usize::try_from(tensor.file_offset).ok()?;
        let length = usize::try_from(tensor.data_size).ok()?;
        self.file_image.get(offset..offset.checked_add(length)?)
    }
}

fn range(bytes: &[u8], offset: u32, length: u32) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    bytes.get(start..start.checked_add(length)?)
}

/// Stable lifecycle states exposed by [`Loader`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoaderState {
    /// No successful probe has completed.
    Uninitialized,
    /// Requirements have been discovered.
    Probed,
    /// Metadata and tensor-record storage has been bound.
    Bound,
    /// The image has been parsed into the bound storage.
    Parsed,
    /// The most recent operation failed.
    Errored,
    /// The machine is processing an internal decision state.
    Processing,
}

/// GGUF loader failures, matching the C++ loader classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The operation is invalid for the current state or request shape.
    InvalidRequest,
    /// The image contains an unsupported or invalid GGUF/GGML construct.
    ModelInvalid,
    /// A count, size, allocation, or caller-supplied capacity is insufficient.
    Capacity,
    /// The image is truncated or has inconsistent serialized offsets.
    ParseFailed,
    /// An invariant inside the loader failed.
    Internal,
    /// An upstream error was not classified.
    Untracked,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid GGUF loader request",
            Self::ModelInvalid => "invalid or unsupported GGUF model",
            Self::Capacity => "GGUF loader capacity exceeded",
            Self::ParseFailed => "failed to parse GGUF image",
            Self::Internal => "internal GGUF loader error",
            Self::Untracked => "untracked GGUF loader error",
        })
    }
}

impl std::error::Error for Error {}

/// Stateful `probe → bind → parse` GGUF loader.
pub struct Loader {
    machine: GgufLoaderStateMachine<GgufLoaderContext>,
}

impl fmt::Debug for Loader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Loader")
            .field("state", &self.state())
            .field("requirements", &self.machine.context().probed)
            .finish_non_exhaustive()
    }
}

impl Default for Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader {
    /// Creates an uninitialized loader.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: GgufLoaderStateMachine::new(GgufLoaderContext::new()),
        }
    }

    /// Returns the current stable lifecycle state.
    #[must_use]
    pub fn state(&self) -> LoaderState {
        match self.machine.state() {
            GgufLoaderStates::Uninitialized => LoaderState::Uninitialized,
            GgufLoaderStates::Probed => LoaderState::Probed,
            GgufLoaderStates::Bound => LoaderState::Bound,
            GgufLoaderStates::Parsed => LoaderState::Parsed,
            GgufLoaderStates::Errored => LoaderState::Errored,
            _ => LoaderState::Processing,
        }
    }

    /// Validates an image and computes storage requirements.
    ///
    /// A probe may be used to recover a loader from any stable state.
    ///
    /// # Errors
    ///
    /// Returns the same error classification as the C++ GGUF loader.
    pub fn probe(&mut self, file_image: &[u8]) -> Result<Requirements, Error> {
        self.machine
            .process_event(GgufLoaderEvents::ProbeRequest(file_image))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        let outcome = detail::probe(file_image);
        self.machine
            .process_event(GgufLoaderEvents::ProbeResult(outcome))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        Ok(self.machine.context().probed)
    }

    /// Allocates exactly the storage required by the latest successful probe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] before a successful probe and
    /// [`Error::Capacity`] if allocation fails.
    pub fn bind(&mut self) -> Result<(), Error> {
        let requirements = self.machine.context().probed;
        self.bind_with_capacity(
            requirements.required_kv_arena_bytes()?,
            usize::try_from(requirements.kv_count).map_err(|_| Error::Capacity)?,
            usize::try_from(requirements.tensor_count).map_err(|_| Error::Capacity)?,
        )
    }

    /// Binds loader-owned storage with explicit capacities.
    ///
    /// This mirrors the C++ bind phase while retaining memory safely in Rust.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] in an invalid lifecycle state and
    /// [`Error::Capacity`] for insufficient capacities or allocation failure.
    pub fn bind_with_capacity(
        &mut self,
        kv_arena_bytes: usize,
        kv_entry_capacity: usize,
        tensor_capacity: usize,
    ) -> Result<(), Error> {
        let event = EventBindRuntime {
            kv_arena_bytes,
            kv_entry_capacity,
            tensor_capacity,
        };
        self.machine
            .process_event(GgufLoaderEvents::BindRequest(event))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        let outcome = BoundStorage::allocate(event);
        self.machine
            .process_event(GgufLoaderEvents::BindResult(outcome))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()
    }

    /// Parses an image into the latest bound storage.
    ///
    /// The returned structure owns its metadata records and borrows tensor
    /// names and payloads directly from `file_image`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] unless storage has been bound, or a
    /// format/capacity error discovered while parsing.
    pub fn parse<'a>(&mut self, file_image: &'a [u8]) -> Result<Gguf<'a>, Error> {
        self.machine
            .process_event(GgufLoaderEvents::ParseRequest(file_image))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        let outcome = self.machine.context_mut().execute_parse(file_image);
        self.machine
            .process_event(GgufLoaderEvents::ParseResult(outcome))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        self.machine.context().parsed(file_image)
    }
}

/// Probes, binds, and parses a GGUF image in one operation.
///
/// # Errors
///
/// Returns the first probe, allocation, or parse failure.
pub fn load(file_image: &[u8]) -> Result<Gguf<'_>, Error> {
    let mut loader = Loader::new();
    loader.probe(file_image)?;
    loader.bind()?;
    loader.parse(file_image)
}
