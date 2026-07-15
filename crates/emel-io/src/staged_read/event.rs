//! Public typed events for bounded staged copying.

use core::cell::{Cell, RefCell};
use core::fmt;

use super::Stager;

mod sealed {
    pub trait Sealed {}
}

pub trait Event: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Stager) -> Self::Output;
}

/// A synchronous allocation-free outcome slot.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidCallbacks,
    InvalidStageContract,
    InvalidTargetWindow,
    UnsupportedPlatform,
    NullSourceSpan,
    SourceSpanSizeMismatch,
    InsufficientSourceSpan,
    InternalError,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidCallbacks => "invalid staged-read callbacks",
            Self::InvalidStageContract => "invalid staged-read contract",
            Self::InvalidTargetWindow => "invalid staged-read target window",
            Self::UnsupportedPlatform => "unsupported staged-read platform",
            Self::NullSourceSpan => "missing staged-read source span",
            Self::SourceSpanSizeMismatch => "staged-read source span is larger than requested",
            Self::InsufficientSourceSpan => "staged-read source span is smaller than requested",
            Self::InternalError => "internal staged-read actor error",
        })
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageWindowDone {
    bytes_committed: u64,
}

impl StageWindowDone {
    pub(crate) const fn new(bytes_committed: u64) -> Self {
        Self { bytes_committed }
    }

    #[must_use]
    pub const fn bytes_committed(self) -> u64 {
        self.bytes_committed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageWindowError {
    error: Error,
}

impl StageWindowError {
    pub(crate) const fn new(error: Error) -> Self {
        Self { error }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
}

#[derive(Debug)]
pub struct StageWindow<'a> {
    pub(crate) file_offset: u64,
    pub(crate) logical_byte_length: u64,
    pub(crate) stage_chunk_bytes: u64,
    pub(crate) source: Option<&'a [u8]>,
    pub(crate) target: &'a mut [u8],
    pub(crate) on_done: Option<Callback<'a, StageWindowDone>>,
    pub(crate) on_error: Option<Callback<'a, StageWindowError>>,
}

impl<'a> StageWindow<'a> {
    #[must_use]
    pub const fn new(
        file_offset: u64,
        logical_byte_length: u64,
        stage_chunk_bytes: u64,
        source: Option<&'a [u8]>,
        target: &'a mut [u8],
    ) -> Self {
        Self {
            file_offset,
            logical_byte_length,
            stage_chunk_bytes,
            source,
            target,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, StageWindowDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, StageWindowError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

impl sealed::Sealed for StageWindow<'_> {}
impl Event for StageWindow<'_> {
    type Output = Result<StageWindowDone, Error>;
    fn dispatch(self, actor: &mut Stager) -> Self::Output {
        actor.stage_window(self)
    }
}

#[derive(Debug)]
pub struct Target<'a> {
    pub(crate) bytes: RefCell<&'a mut [u8]>,
}

impl<'a> Target<'a> {
    #[must_use]
    pub const fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes: RefCell::new(bytes),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StageSpan<'a> {
    pub(crate) file_offset: u64,
    pub(crate) byte_size: u64,
    pub(crate) source: Option<&'a [u8]>,
    pub(crate) target: &'a Target<'a>,
}

impl<'a> StageSpan<'a> {
    #[must_use]
    pub const fn new(
        file_offset: u64,
        byte_size: u64,
        source: Option<&'a [u8]>,
        target: &'a Target<'a>,
    ) -> Self {
        Self {
            file_offset,
            byte_size,
            source,
            target,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageWindowBatchDone {
    done_count: u32,
    bytes_committed: u64,
}

impl StageWindowBatchDone {
    pub(crate) const fn new(done_count: u32, bytes_committed: u64) -> Self {
        Self {
            done_count,
            bytes_committed,
        }
    }
    #[must_use]
    pub const fn done_count(self) -> u32 {
        self.done_count
    }
    #[must_use]
    pub const fn bytes_committed(self) -> u64 {
        self.bytes_committed
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageWindowBatchError {
    error: Error,
    failed_index: u32,
}

impl StageWindowBatchError {
    pub(crate) const fn new(error: Error, failed_index: u32) -> Self {
        Self {
            error,
            failed_index,
        }
    }
    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
    #[must_use]
    pub const fn failed_index(self) -> u32 {
        self.failed_index
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StageWindowBatch<'a> {
    pub(crate) tensors: &'a [StageSpan<'a>],
    pub(crate) stage_chunk_bytes: u64,
    pub(crate) on_done: Option<Callback<'a, StageWindowBatchDone>>,
    pub(crate) on_error: Option<Callback<'a, StageWindowBatchError>>,
}

impl<'a> StageWindowBatch<'a> {
    #[must_use]
    pub const fn new(tensors: &'a [StageSpan<'a>], stage_chunk_bytes: u64) -> Self {
        Self {
            tensors,
            stage_chunk_bytes,
            on_done: None,
            on_error: None,
        }
    }
    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, StageWindowBatchDone>) -> Self {
        self.on_done = Some(callback);
        self
    }
    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, StageWindowBatchError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

impl sealed::Sealed for StageWindowBatch<'_> {}
impl Event for StageWindowBatch<'_> {
    type Output = Result<StageWindowBatchDone, StageWindowBatchError>;
    fn dispatch(self, actor: &mut Stager) -> Self::Output {
        actor.stage_window_batch(self)
    }
}
