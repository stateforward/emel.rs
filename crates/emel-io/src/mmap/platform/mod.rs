//! Sole approved native mapping boundary.

#![allow(unsafe_code)]

use super::event::MmapSource;

#[cfg(unix)]
mod unix;
#[cfg(not(any(unix, windows)))]
mod unsupported;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
use self::unix as imp;
#[cfg(not(any(unix, windows)))]
use self::unsupported as imp;
#[cfg(windows)]
use self::windows as imp;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SetupError {
    MappingFailed,
    UnsupportedPlatform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PlatformError {
    AdviseFailed,
    UnmapFailed,
    UnsupportedPlatform,
}

pub(super) fn abort_on_release_failure(result: Result<(), PlatformError>) {
    if result.is_err() {
        std::process::abort();
    }
}

/// Private owner of one mapped view and its stabilizing file handle.
pub(super) struct Region {
    inner: imp::Region,
}

impl Region {
    pub(super) const fn len(&self) -> usize {
        self.inner.len()
    }

    pub(super) fn bytes(&self) -> &[u8] {
        let base = self.inner.base().expect("live region has a base");
        // SAFETY: `base` and `len` describe the immutable mapping owned by
        // this Region. Actor ownership prevents release or mutation while the
        // returned borrow is live, and no mutable view is ever constructed.
        unsafe { core::slice::from_raw_parts(base.as_ptr(), self.inner.len()) }
    }

    pub(super) fn representable(&self) -> bool {
        self.inner.base().is_some()
    }
}

#[cfg(all(test, unix))]
pub(super) fn synthetic_null_region(file: &MmapSource, len: usize) -> Region {
    Region {
        inner: imp::synthetic_null_region(file, len),
    }
}

#[cfg(all(test, unix))]
pub(super) const fn owns_mapping(region: &Region) -> bool {
    imp::owns_mapping(&region.inner)
}

#[cfg(all(test, unix))]
pub(super) fn release_synthetic_success(region: &mut Region) -> Result<(), PlatformError> {
    imp::release_synthetic_success(&mut region.inner)
}

#[cfg(all(test, unix))]
pub(super) fn release_synthetic_failure(region: &mut Region) -> Result<(), PlatformError> {
    imp::release_synthetic_failure(&mut region.inner)
}

pub(super) trait Platform: Copy {
    fn supported(self) -> bool;
    fn required_alignment(self) -> u64;
    fn offset_supported(self, offset: u64) -> bool;
    fn prepare(self, file: &MmapSource, offset: u64, len: usize) -> Result<Region, SetupError>;
    fn advise_sequential(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError>;
    fn advise_will_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError>;
    fn advise_dont_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError>;
    fn release(self, region: &mut Region) -> Result<(), PlatformError>;
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Native {
    required_alignment: u64,
}

impl Native {
    pub(super) fn new() -> Self {
        Self {
            required_alignment: imp::required_alignment(),
        }
    }
}

impl Default for Native {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for Native {
    fn supported(self) -> bool {
        imp::supported()
    }

    fn required_alignment(self) -> u64 {
        self.required_alignment
    }

    fn offset_supported(self, offset: u64) -> bool {
        imp::offset_supported(offset)
    }

    fn prepare(self, file: &MmapSource, offset: u64, len: usize) -> Result<Region, SetupError> {
        imp::prepare(file, offset, len).map(|inner| Region { inner })
    }

    fn advise_sequential(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        imp::advise_sequential(&region.inner, offset, len)
    }

    fn advise_will_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        imp::advise_will_need(&region.inner, offset, len)
    }

    fn advise_dont_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        imp::advise_dont_need(&region.inner, offset, len)
    }

    fn release(self, region: &mut Region) -> Result<(), PlatformError> {
        imp::release(&mut region.inner)
    }
}
