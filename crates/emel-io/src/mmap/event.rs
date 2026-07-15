//! Safe public events and the explicit backing-file stability capability.

use core::cell::RefCell;
use core::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::Mapper;

mod sealed {
    pub trait Sealed {}
}

const MAX_SOURCE_PATH_BYTES: usize = 4_095;

/// Event accepted by [`Mapper`].
pub trait Event: sealed::Sealed {
    /// Result produced before the synchronous dispatch returns.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, mapper: &mut Mapper) -> Self::Output;
}

/// Mapping failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidRequest,
    UnsupportedPlatform,
    UnsupportedResource,
    ResourceExhausted,
    FileOpenFailed,
    MappingFailed,
    UnmapFailed,
    InvalidAdviceRange,
    AdviceFailed,
    CallbackPanicked,
    InternalError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

/// Capability asserting that a file is stable for native mapping.
///
/// Clones share the exact safely opened [`File`]. Each map event owns one clone,
/// so the actor retains the capability for the complete native mapping lifetime.
/// Construction requires an explicit unsafe assertion:
///
/// ```compile_fail
/// use std::path::Path;
/// use emel_io::mmap::event::MmapSource;
///
/// let _ = MmapSource::open(Path::new("model.gguf"));
/// ```
pub struct MmapSource {
    file: Arc<File>,
    len: u64,
}

impl MmapSource {
    /// Opens one regular file and asserts the backing-file stability contract.
    ///
    /// # Safety
    ///
    /// Before invoking this function, the caller must establish that no handle,
    /// path, process, or filesystem operation can change the target file's length
    /// or contents. That guarantee must hold throughout this function and until
    /// every returned clone and every mapper mapping created from it has been
    /// released or dropped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileOpenFailed`] when the path cannot be opened and
    /// [`Error::UnsupportedResource`] when it is not a regular file or its
    /// metadata cannot be read.
    #[allow(
        unsafe_code,
        reason = "the user-approved MmapSource declaration records the unprovable external file-stability precondition; its body contains only safe Rust"
    )]
    pub unsafe fn open(path: &Path) -> Result<Self, Error> {
        let path_bytes = path.as_os_str().as_encoded_bytes();
        if path_bytes.is_empty()
            || path_bytes.len() > MAX_SOURCE_PATH_BYTES
            || path_bytes.contains(&0)
        {
            return Err(Error::InvalidRequest);
        }
        let file = File::open(path).map_err(|_| Error::FileOpenFailed)?;
        let metadata = file.metadata().map_err(|_| Error::UnsupportedResource)?;
        if !metadata.is_file() {
            return Err(Error::UnsupportedResource);
        }
        Ok(Self {
            file: Arc::new(file),
            len: metadata.len(),
        })
    }

    pub(super) fn file(&self) -> &File {
        &self.file
    }

    pub(super) const fn len(&self) -> u64 {
        self.len
    }
}

impl Clone for MmapSource {
    fn clone(&self) -> Self {
        Self {
            file: Arc::clone(&self.file),
            len: self.len,
        }
    }
}

impl fmt::Debug for MmapSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapSource")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Map a stable file range for one tensor.
///
/// The capability's shared allocation is completed before dispatch; moving it
/// into this event does not allocate.
#[derive(Clone, Debug)]
pub struct MapTensor {
    pub(super) tensor_id: i32,
    pub(super) file_index: u16,
    pub(super) offset: u64,
    pub(super) len: u64,
    pub(super) file: MmapSource,
}

impl MapTensor {
    #[must_use]
    pub const fn new(tensor_id: i32, file: MmapSource, offset: u64, len: u64) -> Self {
        Self {
            tensor_id,
            file_index: 0,
            offset,
            len,
            file,
        }
    }

    #[must_use]
    pub const fn with_file_index(mut self, file_index: u16) -> Self {
        self.file_index = file_index;
        self
    }
}

/// Successful committed mapping descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MapDone {
    handle: u32,
    tensor_id: i32,
    len: u64,
}

impl MapDone {
    pub(super) const fn new(handle: u32, tensor_id: i32, len: u64) -> Self {
        Self {
            handle,
            tensor_id,
            len,
        }
    }

    #[must_use]
    pub const fn handle(self) -> u32 {
        self.handle
    }

    #[must_use]
    pub const fn tensor_id(self) -> i32 {
        self.tensor_id
    }

    #[must_use]
    pub const fn len(self) -> u64 {
        self.len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Release a mapping owned by `tensor_id`.
#[derive(Clone, Copy, Debug)]
pub struct ReleaseMapping {
    pub(super) tensor_id: i32,
    pub(super) handle: u32,
}

impl ReleaseMapping {
    #[must_use]
    pub const fn new(tensor_id: i32, handle: u32) -> Self {
        Self { tensor_id, handle }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AdviceRequest {
    pub(super) tensor_id: i32,
    pub(super) handle: u32,
    pub(super) offset: u64,
    pub(super) len: u64,
}

/// Apply sequential access advice to a live mapping range.
#[derive(Clone, Copy, Debug)]
pub struct AdviseSequential(pub(super) AdviceRequest);

impl AdviseSequential {
    #[must_use]
    pub const fn new(tensor_id: i32, handle: u32, offset: u64, len: u64) -> Self {
        Self(AdviceRequest {
            tensor_id,
            handle,
            offset,
            len,
        })
    }
}

/// Apply will-need advice to a live mapping range.
#[derive(Clone, Copy, Debug)]
pub struct AdviseWillNeed(pub(super) AdviceRequest);

impl AdviseWillNeed {
    #[must_use]
    pub const fn new(tensor_id: i32, handle: u32, offset: u64, len: u64) -> Self {
        Self(AdviceRequest {
            tensor_id,
            handle,
            offset,
            len,
        })
    }
}

/// Apply don't-need advice to a live mapping range.
#[derive(Clone, Copy, Debug)]
pub struct AdviseDontNeed(pub(super) AdviceRequest);

impl AdviseDontNeed {
    #[must_use]
    pub const fn new(tensor_id: i32, handle: u32, offset: u64, len: u64) -> Self {
        Self(AdviceRequest {
            tensor_id,
            handle,
            offset,
            len,
        })
    }
}

/// Allocation-free borrowed callback invoked before dispatch returns.
type MappingFn<'state> = dyn for<'view> FnMut(&'view [u8]) + 'state;

pub struct MappingCallback<'state> {
    callback: RefCell<&'state mut MappingFn<'state>>,
}

impl<'state> MappingCallback<'state> {
    #[must_use]
    pub fn new<Callback>(callback: &'state mut Callback) -> Self
    where
        Callback: for<'view> FnMut(&'view [u8]),
    {
        Self {
            callback: RefCell::new(callback),
        }
    }

    pub(super) fn invoke(&self, bytes: &[u8]) -> Result<(), ()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.callback.borrow_mut())(bytes);
        }))
        .map_err(|_| ())
    }
}

impl fmt::Debug for MappingCallback<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappingCallback").finish_non_exhaustive()
    }
}

/// Invoke a callback synchronously with immutable access to a live mapping.
#[derive(Clone, Copy, Debug)]
pub struct WithMapping<'state> {
    pub(super) tensor_id: i32,
    pub(super) handle: u32,
    pub(super) callback: &'state MappingCallback<'state>,
}

impl<'state> WithMapping<'state> {
    #[must_use]
    pub const fn new(
        tensor_id: i32,
        handle: u32,
        callback: &'state MappingCallback<'state>,
    ) -> Self {
        Self {
            tensor_id,
            handle,
            callback,
        }
    }
}

impl sealed::Sealed for MapTensor {}
impl Event for MapTensor {
    type Output = Result<MapDone, Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.map(&self)
    }
}

impl sealed::Sealed for ReleaseMapping {}
impl Event for ReleaseMapping {
    type Output = Result<(), Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.release(self)
    }
}

impl sealed::Sealed for AdviseSequential {}
impl Event for AdviseSequential {
    type Output = Result<(), Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.advise_sequential(self.0)
    }
}

impl sealed::Sealed for AdviseWillNeed {}
impl Event for AdviseWillNeed {
    type Output = Result<(), Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.advise_will_need(self.0)
    }
}

impl sealed::Sealed for AdviseDontNeed {}
impl Event for AdviseDontNeed {
    type Output = Result<(), Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.advise_dont_need(self.0)
    }
}

impl sealed::Sealed for WithMapping<'_> {}
impl Event for WithMapping<'_> {
    type Output = Result<(), Error>;

    fn dispatch(self, mapper: &mut Mapper) -> Self::Output {
        mapper.with_mapping(self)
    }
}
