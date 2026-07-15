//! Public typed events and statically dispatched child-actor contracts.

use core::cell::Cell;
use core::fmt;

pub use crate::tensor::{SourceError, Target, TargetWriteError, TensorLoadSpan};

use super::Loader;
use crate::{read, staged_read};

/// Maximum number of tensor spans accepted by one [`LoadTensorBatch`] event.
///
/// This Rust-only bound keeps one run-to-completion validation phase statically
/// bounded. The pinned C++ reference does not impose an equivalent ceiling.
pub const MAX_BATCH_TENSORS: usize = 65_536;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Loader`].
pub trait Event: sealed::Sealed {
    type Output;

    #[doc(hidden)]
    fn dispatch<R: ReadActor, S: StagedReadActor>(self, actor: &mut Loader<R, S>) -> Self::Output;
}

/// Synchronous allocation-free outcome storage.
#[derive(Clone, Copy)]
pub struct Callback<'a, T: Copy> {
    slot: &'a Cell<Option<T>>,
}

impl<'a, T: Copy> Callback<'a, T> {
    #[must_use]
    pub const fn store(slot: &'a Cell<Option<T>>) -> Self {
        Self { slot }
    }

    pub(crate) fn publish(self, value: T) {
        self.slot.set(Some(value));
    }
}

impl<T: Copy> fmt::Debug for Callback<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Callback").finish_non_exhaustive()
    }
}

/// Explicit loader strategy selected by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StrategyKind {
    None,
    MappedFile,
    ReadCopy,
    ExternalBuffer,
    StagedRead,
    Unknown(u8),
}

/// Per-dispatch strategy configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrategyPolicy {
    strategy: StrategyKind,
    staged_chunk_bytes: u64,
}

impl StrategyPolicy {
    pub const DEFAULT_STAGED_CHUNK_BYTES: u64 = 64 * 1_024;

    #[must_use]
    pub const fn new(strategy: StrategyKind) -> Self {
        Self {
            strategy,
            staged_chunk_bytes: Self::DEFAULT_STAGED_CHUNK_BYTES,
        }
    }

    #[must_use]
    pub const fn with_staged_chunk_bytes(mut self, staged_chunk_bytes: u64) -> Self {
        self.staged_chunk_bytes = staged_chunk_bytes;
        self
    }

    #[must_use]
    pub const fn strategy(self) -> StrategyKind {
        self.strategy
    }

    #[must_use]
    pub const fn staged_chunk_bytes(self) -> u64 {
        self.staged_chunk_bytes
    }
}

/// Loader-owned failure class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidRequest,
    UnsupportedStrategy,
    Unavailable,
    InternalError,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid loader request",
            Self::UnsupportedStrategy => "unsupported loader strategy",
            Self::Unavailable => "loader strategy actor unavailable",
            Self::InternalError => "internal loader actor error",
        })
    }
}

impl std::error::Error for Error {}

/// Typed child-actor failure retained across the loader boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StrategyError {
    None,
    Read(read::event::Error),
    StagedRead(staged_read::event::Error),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadTensorDone {
    strategy: StrategyKind,
    bytes_loaded: u64,
}

impl LoadTensorDone {
    pub(crate) const fn new(strategy: StrategyKind, bytes_loaded: u64) -> Self {
        Self {
            strategy,
            bytes_loaded,
        }
    }

    #[must_use]
    pub const fn strategy(self) -> StrategyKind {
        self.strategy
    }

    #[must_use]
    pub const fn bytes_loaded(self) -> u64 {
        self.bytes_loaded
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadTensorError {
    error: Error,
    strategy_error: StrategyError,
}

impl LoadTensorError {
    pub(crate) const fn new(error: Error, strategy_error: StrategyError) -> Self {
        Self {
            error,
            strategy_error,
        }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }

    #[must_use]
    pub const fn strategy_error(self) -> StrategyError {
        self.strategy_error
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadTensorBatchDone {
    strategy: StrategyKind,
    done_count: u32,
    bytes_loaded: u64,
}

impl LoadTensorBatchDone {
    pub(crate) const fn new(strategy: StrategyKind, done_count: u32, bytes_loaded: u64) -> Self {
        Self {
            strategy,
            done_count,
            bytes_loaded,
        }
    }

    #[must_use]
    pub const fn strategy(self) -> StrategyKind {
        self.strategy
    }

    #[must_use]
    pub const fn done_count(self) -> u32 {
        self.done_count
    }

    #[must_use]
    pub const fn bytes_loaded(self) -> u64 {
        self.bytes_loaded
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadTensorBatchError {
    error: Error,
    strategy_error: StrategyError,
    failed_index: u32,
}

impl LoadTensorBatchError {
    pub(crate) const fn new(
        error: Error,
        strategy_error: StrategyError,
        failed_index: u32,
    ) -> Self {
        Self {
            error,
            strategy_error,
            failed_index,
        }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }

    #[must_use]
    pub const fn strategy_error(self) -> StrategyError {
        self.strategy_error
    }

    #[must_use]
    pub const fn failed_index(self) -> u32 {
        self.failed_index
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoadTensor<'a> {
    pub(crate) tensor: TensorLoadSpan<'a>,
    pub(crate) policy: StrategyPolicy,
    pub(crate) on_done: Option<Callback<'a, LoadTensorDone>>,
    pub(crate) on_error: Option<Callback<'a, LoadTensorError>>,
}

impl<'a> LoadTensor<'a> {
    #[must_use]
    pub const fn new(tensor: TensorLoadSpan<'a>, policy: StrategyPolicy) -> Self {
        Self {
            tensor,
            policy,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, LoadTensorDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, LoadTensorError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

impl sealed::Sealed for LoadTensor<'_> {}
impl Event for LoadTensor<'_> {
    type Output = Result<LoadTensorDone, LoadTensorError>;

    fn dispatch<R: ReadActor, S: StagedReadActor>(self, actor: &mut Loader<R, S>) -> Self::Output {
        actor.load_tensor(self)
    }
}

/// Request to load a caller-built batch through one selected strategy actor.
///
/// The batch must contain between one and [`MAX_BATCH_TENSORS`] spans,
/// inclusive. The loader reports [`Error::InvalidRequest`] before dispatching a
/// child actor or mutating a target when the count is outside that range.
#[derive(Clone, Copy, Debug)]
pub struct LoadTensorBatch<'a> {
    pub(crate) tensors: &'a [TensorLoadSpan<'a>],
    pub(crate) policy: StrategyPolicy,
    pub(crate) on_done: Option<Callback<'a, LoadTensorBatchDone>>,
    pub(crate) on_error: Option<Callback<'a, LoadTensorBatchError>>,
}

impl<'a> LoadTensorBatch<'a> {
    /// Creates a batch request over caller-owned spans.
    ///
    /// Dispatch rejects an empty slice or a slice longer than
    /// [`MAX_BATCH_TENSORS`] with [`Error::InvalidRequest`].
    #[must_use]
    pub const fn new(tensors: &'a [TensorLoadSpan<'a>], policy: StrategyPolicy) -> Self {
        Self {
            tensors,
            policy,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, LoadTensorBatchDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, LoadTensorBatchError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

impl sealed::Sealed for LoadTensorBatch<'_> {}
impl Event for LoadTensorBatch<'_> {
    type Output = Result<LoadTensorBatchDone, LoadTensorBatchError>;

    fn dispatch<R: ReadActor, S: StagedReadActor>(self, actor: &mut Loader<R, S>) -> Self::Output {
        actor.load_tensor_batch(self)
    }
}

/// Static dependency contract for a replaceable read actor.
pub trait ReadActor {
    const AVAILABLE: bool = true;

    /// Processes one read span synchronously.
    ///
    /// # Errors
    ///
    /// Returns the replacement actor's typed read failure.
    fn process_read(
        &mut self,
        event: read::event::ReadSpan<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error>;

    /// Processes one read batch synchronously.
    ///
    /// # Errors
    ///
    /// Returns the replacement actor's typed batch failure.
    fn process_read_batch(
        &mut self,
        event: read::event::ReadTensorBatch<'_>,
    ) -> Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError>;
}

impl ReadActor for read::Reader {
    fn process_read(
        &mut self,
        event: read::event::ReadSpan<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        self.process_event(event)
    }

    fn process_read_batch(
        &mut self,
        event: read::event::ReadTensorBatch<'_>,
    ) -> Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError> {
        self.process_event(event)
    }
}

/// Static dependency contract for a replaceable staged-read actor.
pub trait StagedReadActor {
    const AVAILABLE: bool = true;

    /// Processes one staged span synchronously.
    ///
    /// # Errors
    ///
    /// Returns the replacement actor's typed staged-read failure.
    fn process_staged_read(
        &mut self,
        event: staged_read::event::StageTensor<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error>;

    /// Processes one staged batch synchronously.
    ///
    /// # Errors
    ///
    /// Returns the replacement actor's typed batch failure.
    fn process_staged_read_batch(
        &mut self,
        event: staged_read::event::StageTensorBatch<'_>,
    ) -> Result<staged_read::event::StageWindowBatchDone, staged_read::event::StageWindowBatchError>;
}

impl StagedReadActor for staged_read::Stager {
    fn process_staged_read(
        &mut self,
        event: staged_read::event::StageTensor<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        self.process_event(event)
    }

    fn process_staged_read_batch(
        &mut self,
        event: staged_read::event::StageTensorBatch<'_>,
    ) -> Result<staged_read::event::StageWindowBatchDone, staged_read::event::StageWindowBatchError>
    {
        self.process_event(event)
    }
}

/// Compile-time absent child used by the zero-dependency loader.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NoActor;

impl ReadActor for NoActor {
    const AVAILABLE: bool = false;

    fn process_read(
        &mut self,
        _: read::event::ReadSpan<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        Err(read::event::Error::InternalError)
    }

    fn process_read_batch(
        &mut self,
        _: read::event::ReadTensorBatch<'_>,
    ) -> Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError> {
        Err(read::event::ReadTensorBatchError::new(
            read::event::Error::InternalError,
            0,
        ))
    }
}

impl StagedReadActor for NoActor {
    const AVAILABLE: bool = false;

    fn process_staged_read(
        &mut self,
        _: staged_read::event::StageTensor<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        Err(staged_read::event::Error::InternalError)
    }

    fn process_staged_read_batch(
        &mut self,
        _: staged_read::event::StageTensorBatch<'_>,
    ) -> Result<staged_read::event::StageWindowBatchDone, staged_read::event::StageWindowBatchError>
    {
        Err(staged_read::event::StageWindowBatchError::new(
            staged_read::event::Error::InternalError,
            0,
        ))
    }
}
