//! Typed requests and outcomes for the read actor.

use core::cell::Cell;
use core::fmt;

pub use crate::tensor::{SourceError, Target, TensorLoadSpan as TensorRead};

use super::Reader;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Reader`].
///
/// This trait is sealed so external crates cannot bypass the actor boundary.
pub trait Event: sealed::Sealed {
    /// Result produced after the event runs to completion.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Reader) -> Self::Output;
}

/// An allocation-free synchronous callback.
///
/// The callback borrows a caller-owned slot and invokes a plain function
/// pointer before dispatch returns. It is never retained by the actor.
#[derive(Clone, Copy)]
pub struct Callback<'a, T: Copy> {
    slot: &'a Cell<Option<T>>,
    handler: fn(&Cell<Option<T>>, T),
}

impl<'a, T: Copy> Callback<'a, T> {
    /// Creates a callback with caller-defined synchronous handling.
    #[must_use]
    pub const fn new(slot: &'a Cell<Option<T>>, handler: fn(&Cell<Option<T>>, T)) -> Self {
        Self { slot, handler }
    }

    /// Creates a callback that stores the latest outcome in `slot`.
    #[must_use]
    pub const fn store(slot: &'a Cell<Option<T>>) -> Self {
        Self::new(slot, store_outcome::<T>)
    }

    /// Publishes an outcome synchronously through this callback capability.
    pub fn publish(self, outcome: T) {
        (self.handler)(self.slot, outcome);
    }
}

impl<T: Copy> fmt::Debug for Callback<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Callback").finish_non_exhaustive()
    }
}

fn store_outcome<T: Copy>(slot: &Cell<Option<T>>, outcome: T) {
    slot.set(Some(outcome));
}

/// Read actor failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A required request field or target buffer is invalid.
    InvalidRequest,
    /// The current target platform cannot execute the read copy.
    UnsupportedPlatform,
    /// A path, file index, length, or offset is unsupported.
    UnsupportedResource,
    /// The external source could not be opened.
    FileOpenFailed,
    /// The external source could not seek to the requested offset.
    FileSeekFailed,
    /// The external source failed while reading.
    FileReadFailed,
    /// The source contains fewer bytes than requested.
    ShortRead,
    /// The actor encountered an internal or unexpected event error.
    InternalError,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid read request",
            Self::UnsupportedPlatform => "unsupported read platform",
            Self::UnsupportedResource => "unsupported read resource",
            Self::FileOpenFailed => "read source open failed",
            Self::FileSeekFailed => "read source seek failed",
            Self::FileReadFailed => "read source failed",
            Self::ShortRead => "read source was shorter than requested",
            Self::InternalError => "internal read actor error",
        })
    }
}

impl std::error::Error for Error {}

/// Successful outcome of a [`ReadTensor`] event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadTensorDone {
    tensor_id: i32,
    bytes_copied: u64,
}

impl ReadTensorDone {
    pub const fn new(tensor_id: i32, bytes_copied: u64) -> Self {
        Self {
            tensor_id,
            bytes_copied,
        }
    }

    /// Returns the caller-provided tensor identity.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    /// Returns the bytes copied into the caller-owned target.
    #[must_use]
    pub const fn bytes_copied(self) -> u64 {
        self.bytes_copied
    }
}

/// Error callback payload for a [`ReadTensor`] event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadTensorError {
    tensor_id: i32,
    error: Error,
}

impl ReadTensorError {
    pub const fn new(tensor_id: i32, error: Error) -> Self {
        Self { tensor_id, error }
    }

    /// Returns the caller-provided tensor identity.
    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    /// Returns the classified failure.
    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
}

/// Request to copy one tensor range from immutable source bytes.
#[derive(Debug)]
pub struct ReadTensor<'a> {
    pub(crate) tensor_id: i32,
    pub(crate) file_index: u16,
    pub(crate) file_offset: u64,
    pub(crate) byte_size: u64,
    pub(crate) file_path: &'a str,
    pub(crate) source: Option<&'a [u8]>,
    pub(crate) source_error: Option<SourceError>,
    pub(crate) target: &'a Target<'a>,
    pub(crate) on_done: Option<Callback<'a, ReadTensorDone>>,
    pub(crate) on_error: Option<Callback<'a, ReadTensorError>>,
}

impl<'a> ReadTensor<'a> {
    /// Creates a read request. Filesystem work must already be complete.
    #[must_use]
    pub fn new(
        tensor_id: i32,
        file_path: &'a str,
        source: Option<&'a [u8]>,
        target: &'a Target<'a>,
    ) -> Self {
        let byte_size = target.len() as u64;
        Self {
            tensor_id,
            file_index: 0,
            file_offset: 0,
            byte_size,
            file_path,
            source,
            source_error: None,
            target,
            on_done: None,
            on_error: None,
        }
    }

    /// Sets the split-file index.
    #[must_use]
    pub const fn with_file_index(mut self, file_index: u16) -> Self {
        self.file_index = file_index;
        self
    }

    /// Sets the byte range within the immutable source.
    #[must_use]
    pub const fn with_range(mut self, file_offset: u64, byte_size: u64) -> Self {
        self.file_offset = file_offset;
        self.byte_size = byte_size;
        self
    }

    /// Sets an externally produced source error.
    #[must_use]
    pub const fn with_source_error(mut self, error: SourceError) -> Self {
        self.source_error = Some(error);
        self
    }

    /// Sets the optional setup-time source outcome without runtime routing.
    #[must_use]
    pub const fn with_source_error_option(mut self, error: Option<SourceError>) -> Self {
        self.source_error = error;
        self
    }

    /// Installs a synchronous success callback.
    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, ReadTensorDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    /// Installs a synchronous error callback.
    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, ReadTensorError>) -> Self {
        self.on_error = Some(callback);
        self
    }

    /// Returns the tensor identifier.
    #[must_use]
    pub const fn tensor_id(&self) -> i32 {
        self.tensor_id
    }

    /// Returns the split-file index.
    #[must_use]
    pub const fn file_index(&self) -> u16 {
        self.file_index
    }

    /// Returns the byte offset within the source.
    #[must_use]
    pub const fn file_offset(&self) -> u64 {
        self.file_offset
    }

    /// Returns the requested byte count.
    #[must_use]
    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }

    /// Returns the source path metadata.
    #[must_use]
    pub const fn file_path(&self) -> &'a str {
        self.file_path
    }

    /// Returns the immutable source capability, when setup succeeded.
    #[must_use]
    pub const fn source(&self) -> Option<&'a [u8]> {
        self.source
    }

    /// Returns the setup-time source failure, when one was supplied.
    #[must_use]
    pub const fn source_error(&self) -> Option<SourceError> {
        self.source_error
    }

    /// Returns the caller-owned safe target capability.
    #[must_use]
    pub const fn target(&self) -> &'a Target<'a> {
        self.target
    }

    /// Returns the optional synchronous success callback capability.
    #[must_use]
    pub const fn done_callback(&self) -> Option<Callback<'a, ReadTensorDone>> {
        self.on_done
    }

    /// Returns the optional synchronous error callback capability.
    #[must_use]
    pub const fn error_callback(&self) -> Option<Callback<'a, ReadTensorError>> {
        self.on_error
    }
}

impl sealed::Sealed for ReadTensor<'_> {}

impl Event for ReadTensor<'_> {
    type Output = Result<ReadTensorDone, Error>;

    fn dispatch(self, actor: &mut Reader) -> Self::Output {
        actor.read_tensor(self)
    }
}

/// Request to read one shared tensor-span capability.
#[derive(Clone, Copy, Debug)]
pub struct ReadSpan<'a> {
    pub(crate) tensor: TensorRead<'a>,
}

impl<'a> ReadSpan<'a> {
    #[must_use]
    pub const fn new(tensor: TensorRead<'a>) -> Self {
        Self { tensor }
    }

    #[must_use]
    pub const fn tensor(self) -> TensorRead<'a> {
        self.tensor
    }
}

impl sealed::Sealed for ReadSpan<'_> {}
impl Event for ReadSpan<'_> {
    type Output = Result<ReadTensorDone, Error>;

    fn dispatch(self, actor: &mut Reader) -> Self::Output {
        actor.read_span(self)
    }
}

/// Successful outcome of a [`ReadTensorBatch`] event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadTensorBatchDone {
    done_count: u32,
    bytes_copied: u64,
}

impl ReadTensorBatchDone {
    pub const fn new(done_count: u32, bytes_copied: u64) -> Self {
        Self {
            done_count,
            bytes_copied,
        }
    }

    /// Returns the number of copied tensor spans.
    #[must_use]
    pub const fn done_count(self) -> u32 {
        self.done_count
    }

    /// Returns the total copied byte count.
    #[must_use]
    pub const fn bytes_copied(self) -> u64 {
        self.bytes_copied
    }
}

/// Error callback payload for a [`ReadTensorBatch`] event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadTensorBatchError {
    error: Error,
    failed_index: u32,
}

impl ReadTensorBatchError {
    pub const fn new(error: Error, failed_index: u32) -> Self {
        Self {
            error,
            failed_index,
        }
    }

    /// Returns the classified failure.
    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }

    /// Returns the first span with the classified failure.
    #[must_use]
    pub const fn failed_index(self) -> u32 {
        self.failed_index
    }
}

/// Request to copy a caller-built batch of tensor spans.
#[derive(Clone, Copy, Debug)]
pub struct ReadTensorBatch<'a> {
    pub(crate) tensors: &'a [TensorRead<'a>],
    pub(crate) on_done: Option<Callback<'a, ReadTensorBatchDone>>,
    pub(crate) on_error: Option<Callback<'a, ReadTensorBatchError>>,
}

impl<'a> ReadTensorBatch<'a> {
    /// Creates a batch request over caller-owned spans.
    #[must_use]
    pub const fn new(tensors: &'a [TensorRead<'a>]) -> Self {
        Self {
            tensors,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn tensors(self) -> &'a [TensorRead<'a>] {
        self.tensors
    }

    /// Installs a synchronous success callback.
    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, ReadTensorBatchDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    /// Installs a synchronous error callback.
    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, ReadTensorBatchError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

impl sealed::Sealed for ReadTensorBatch<'_> {}

impl Event for ReadTensorBatch<'_> {
    type Output = Result<ReadTensorBatchDone, ReadTensorBatchError>;

    fn dispatch(self, actor: &mut Reader) -> Self::Output {
        actor.read_tensor_batch(self)
    }
}
