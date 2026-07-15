use core::ptr::NonNull;
use std::os::fd::AsRawFd;

use super::super::event::MmapSource;
use super::{PlatformError, SetupError, abort_on_release_failure};

pub(super) struct Region {
    base: Option<*mut u8>,
    len: usize,
    _file: MmapSource,
}

impl Region {
    pub(super) fn base(&self) -> Option<NonNull<u8>> {
        self.base.and_then(NonNull::new)
    }

    pub(super) const fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        abort_on_release_failure(release(self));
    }
}

pub(super) const fn supported() -> bool {
    true
}

pub(super) fn required_alignment() -> u64 {
    u64::try_from(rustix::param::page_size()).expect("Unix page size must fit the mmap wire type")
}

pub(super) fn offset_supported(offset: u64) -> bool {
    libc::off_t::try_from(offset).is_ok()
}

pub(super) fn prepare(file: &MmapSource, offset: u64, len: usize) -> Result<Region, SetupError> {
    let offset = libc::off_t::try_from(offset)
        .expect("SML resource guard selected a Unix-supported mapping offset");
    // SAFETY: offset and length were checked against the stable open file;
    // null requests an OS-selected address, the mapping is read-only/private,
    // and the returned allocation is immediately placed under Region ownership.
    let raw = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.file().as_raw_fd(),
            offset,
        )
    };
    if raw == libc::MAP_FAILED {
        return Err(SetupError::MappingFailed);
    }
    Ok(Region {
        base: Some(raw.cast()),
        len,
        _file: file.clone(),
    })
}

pub(super) fn advise_sequential(
    region: &Region,
    offset: usize,
    len: usize,
) -> Result<(), PlatformError> {
    advise_expanded::<{ libc::POSIX_MADV_SEQUENTIAL }>(region, offset, len)
}

pub(super) fn advise_will_need(
    region: &Region,
    offset: usize,
    len: usize,
) -> Result<(), PlatformError> {
    advise_expanded::<{ libc::POSIX_MADV_WILLNEED }>(region, offset, len)
}

pub(super) fn advise_dont_need(
    region: &Region,
    offset: usize,
    len: usize,
) -> Result<(), PlatformError> {
    let page = usize::try_from(required_alignment()).map_err(|_| PlatformError::AdviseFailed)?;
    let requested_end = offset.checked_add(len).ok_or(PlatformError::AdviseFailed)?;
    let start = offset
        .checked_add(page - 1)
        .ok_or(PlatformError::AdviseFailed)?
        / page
        * page;
    let end = requested_end / page * page;
    call_advice::<{ libc::POSIX_MADV_DONTNEED }>(region, start, end)
}

fn advise_expanded<const CODE: i32>(
    region: &Region,
    offset: usize,
    len: usize,
) -> Result<(), PlatformError> {
    let page = usize::try_from(required_alignment()).map_err(|_| PlatformError::AdviseFailed)?;
    let requested_end = offset.checked_add(len).ok_or(PlatformError::AdviseFailed)?;
    let start = offset / page * page;
    let end = requested_end
        .checked_add(page - 1)
        .ok_or(PlatformError::AdviseFailed)?
        / page
        * page;
    call_advice::<CODE>(region, start, end)
}

fn call_advice<const CODE: i32>(
    region: &Region,
    start: usize,
    end: usize,
) -> Result<(), PlatformError> {
    let base = region.base().ok_or(PlatformError::UnmapFailed)?;
    let end = end.min(region.len);
    if start >= end {
        return Ok(());
    }
    let advised_len = end.checked_sub(start).ok_or(PlatformError::AdviseFailed)?;
    // SAFETY: the actor validated the range within the live region; checked
    // addition below preserves allocation provenance and no release can race.
    let address = unsafe { base.as_ptr().add(start) };
    // SAFETY: address/length designate the owned live mapping, and CODE is a
    // compile-time POSIX advice constant selected by the typed entry point.
    let result = unsafe { libc::posix_madvise(address.cast(), advised_len, CODE) };
    if result == 0 {
        Ok(())
    } else {
        Err(PlatformError::AdviseFailed)
    }
}

pub(super) fn release(region: &mut Region) -> Result<(), PlatformError> {
    release_with::<NativeRelease>(region)
}

trait ReleaseOps {
    fn unmap(base: *mut u8, len: usize) -> bool;
}

struct NativeRelease;

impl ReleaseOps for NativeRelease {
    fn unmap(base: *mut u8, len: usize) -> bool {
        // SAFETY: base/len are the exact still-owned mmap allocation, including
        // a POSIX-successful address-zero mapping; actor ownership proves no
        // Rust reference exists while the synchronous release executes.
        unsafe { libc::munmap(base.cast(), len) == 0 }
    }
}

fn release_with<Ops: ReleaseOps>(region: &mut Region) -> Result<(), PlatformError> {
    let Some(base) = region.base else {
        return Ok(());
    };
    if Ops::unmap(base, region.len) {
        region.base = None;
        region.len = 0;
        Ok(())
    } else {
        Err(PlatformError::UnmapFailed)
    }
}

#[cfg(test)]
pub(super) fn synthetic_null_region(file: &MmapSource, len: usize) -> Region {
    Region {
        base: Some(core::ptr::null_mut()),
        len,
        _file: file.clone(),
    }
}

#[cfg(test)]
pub(super) const fn owns_mapping(region: &Region) -> bool {
    region.base.is_some()
}

#[cfg(test)]
pub(super) fn release_synthetic_success(region: &mut Region) -> Result<(), PlatformError> {
    struct Succeeds;
    impl ReleaseOps for Succeeds {
        fn unmap(base: *mut u8, _: usize) -> bool {
            assert!(base.is_null());
            true
        }
    }
    release_with::<Succeeds>(region)
}

#[cfg(test)]
pub(super) fn release_synthetic_failure(region: &mut Region) -> Result<(), PlatformError> {
    struct Fails;
    impl ReleaseOps for Fails {
        fn unmap(base: *mut u8, _: usize) -> bool {
            assert!(base.is_null());
            false
        }
    }
    release_with::<Fails>(region)
}
