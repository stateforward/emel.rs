//! Stateful GGUF loader and its parsed records.

mod detail;
pub mod metadata;
pub mod query;
mod sm;
pub mod tensor;

use core::fmt;

use self::sm::{GgufLoaderContext, GgufLoaderEvents, GgufLoaderStateMachine, GgufLoaderStates};

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

/// Internal stable lifecycle states used for loader diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoaderState {
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
    fn state(&self) -> LoaderState {
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

    /// Moves caller-allocated storage into the loader.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] in an invalid lifecycle state and
    /// [`Error::Capacity`] for insufficient capacities.
    pub fn bind(
        &mut self,
        storage: crate::event::Storage,
    ) -> Result<(), (Error, crate::event::Storage)> {
        let mut storage = Some(storage);
        self.machine
            .process_event(GgufLoaderEvents::BindRequest(&mut storage))
            .map_err(|_| {
                (
                    Error::Internal,
                    storage.take().expect("failed dispatch preserves storage"),
                )
            })?;
        self.machine.context().result().map_err(|error| {
            (
                error,
                storage.take().expect("rejected bind preserves storage"),
            )
        })
    }

    /// Parses an image into the latest bound storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] unless storage has been bound, or a
    /// format/capacity error discovered while parsing.
    pub fn parse(&mut self) -> Result<Requirements, Error> {
        self.machine
            .process_event(GgufLoaderEvents::ParseRequest(()))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        let outcome = self.machine.context_mut().execute_parse();
        self.machine
            .process_event(GgufLoaderEvents::ParseResult(outcome))
            .map_err(|_| Error::Internal)?;
        self.machine.context().result()?;
        Ok(self.machine.context().probed)
    }

    pub(super) fn query<O: query::Operation>(&self, operation: &mut O) {
        let parsed = *self.machine.state() == GgufLoaderStates::Parsed;
        let context = self.machine.context();
        query::process(
            parsed,
            context.bound.as_ref(),
            context.probed.kv_count,
            operation,
        );
    }

    pub(super) fn with_metadata_descriptor<O: metadata::Operation>(&self, operation: &mut O) {
        let parsed = *self.machine.state() == GgufLoaderStates::Parsed;
        let context = self.machine.context();
        metadata::process(
            parsed,
            context.bound.as_ref(),
            context.probed.kv_count,
            operation,
        );
    }

    pub(super) fn with_tensor<O: tensor::Operation>(&self, operation: &mut O) {
        let parsed = *self.machine.state() == GgufLoaderStates::Parsed;
        let context = self.machine.context();
        tensor::process(
            parsed,
            context.bound.as_ref(),
            context.probed.tensor_count,
            operation,
        );
    }
}
