//! Shared caller-owned tensor span capability for synchronous I/O actors.

use core::cell::RefCell;
use core::fmt;

/// Error reported by setup-time source acquisition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SourceError {
    /// The external source could not be opened.
    FileOpenFailed,
    /// The external source could not seek to the requested offset.
    FileSeekFailed,
    /// The external source failed while reading.
    FileReadFailed,
    /// The external source returned fewer bytes than requested.
    ShortRead,
    /// An external error not otherwise classified by an I/O actor.
    Other,
}

/// Safe target-capability write failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TargetWriteError {
    Borrowed,
    InsufficientCapacity,
}

impl fmt::Display for TargetWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Borrowed => "target is already borrowed",
            Self::InsufficientCapacity => "target capacity is insufficient",
        })
    }
}

impl std::error::Error for TargetWriteError {}

/// Caller-owned mutable target shared by synchronous I/O event descriptors.
#[derive(Debug)]
pub struct Target<'a> {
    pub(crate) bytes: RefCell<&'a mut [u8]>,
}

impl<'a> Target<'a> {
    /// Wraps a caller-owned target for one or more synchronous dispatches.
    #[must_use]
    pub const fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes: RefCell::new(bytes),
        }
    }

    /// Returns the target capacity.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.borrow().len()
    }

    /// Returns whether the target has zero capacity.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies bytes into this target without exposing a retainable mutable borrow.
    ///
    /// # Errors
    ///
    /// Returns [`TargetWriteError::Borrowed`] when another synchronous operation
    /// holds the target, or [`TargetWriteError::InsufficientCapacity`] when the
    /// source does not fit.
    pub fn try_copy_from(&self, source: &[u8]) -> Result<(), TargetWriteError> {
        let mut target = self
            .bytes
            .try_borrow_mut()
            .map_err(|_| TargetWriteError::Borrowed)?;
        if target.len() < source.len() {
            return Err(TargetWriteError::InsufficientCapacity);
        }
        target[..source.len()].copy_from_slice(source);
        Ok(())
    }

    /// Compares the complete target with caller-provided bytes without exposing its storage.
    ///
    /// # Errors
    ///
    /// Returns [`TargetWriteError::Borrowed`] when another synchronous operation
    /// holds the target.
    pub fn try_matches(&self, expected: &[u8]) -> Result<bool, TargetWriteError> {
        self.bytes
            .try_borrow()
            .map(|target| **target == *expected)
            .map_err(|_| TargetWriteError::Borrowed)
    }

    pub(crate) fn copy_from(&self, source: &[u8]) {
        self.bytes.borrow_mut()[..source.len()].copy_from_slice(source);
    }
}

/// Immutable tensor metadata plus a caller-owned synchronous target capability.
#[derive(Clone, Copy, Debug)]
pub struct TensorLoadSpan<'a> {
    pub(crate) tensor_id: i32,
    pub(crate) file_index: u16,
    pub(crate) file_offset: u64,
    pub(crate) byte_size: u64,
    pub(crate) file_path: &'a str,
    pub(crate) source: Option<&'a [u8]>,
    pub(crate) source_error: Option<SourceError>,
    pub(crate) target: &'a Target<'a>,
    pub(crate) target_bytes: u64,
}

impl<'a> TensorLoadSpan<'a> {
    /// Creates a read-oriented tensor span over caller-owned storage.
    #[must_use]
    pub fn new(
        tensor_id: i32,
        file_path: &'a str,
        source: Option<&'a [u8]>,
        target: &'a Target<'a>,
    ) -> Self {
        let target_bytes = target.len() as u64;
        Self {
            tensor_id,
            file_index: 0,
            file_offset: 0,
            byte_size: target_bytes,
            file_path,
            source,
            source_error: None,
            target,
            target_bytes,
        }
    }

    /// Creates a staged-copy span over caller-owned storage.
    #[must_use]
    pub fn staged(
        file_offset: u64,
        byte_size: u64,
        source: Option<&'a [u8]>,
        target: &'a Target<'a>,
    ) -> Self {
        Self::new(0, "", source, target).with_range(file_offset, byte_size)
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

    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    #[must_use]
    pub const fn file_index(self) -> u16 {
        self.file_index
    }

    #[must_use]
    pub const fn file_offset(self) -> u64 {
        self.file_offset
    }

    #[must_use]
    pub const fn byte_size(self) -> u64 {
        self.byte_size
    }

    #[must_use]
    pub const fn file_path(self) -> &'a str {
        self.file_path
    }

    #[must_use]
    pub const fn source(self) -> Option<&'a [u8]> {
        self.source
    }

    #[must_use]
    pub const fn source_error(self) -> Option<SourceError> {
        self.source_error
    }

    #[must_use]
    pub const fn target(self) -> &'a Target<'a> {
        self.target
    }

    #[must_use]
    pub const fn target_bytes(self) -> u64 {
        self.target_bytes
    }
}
