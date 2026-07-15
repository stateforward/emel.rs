use core::ptr::NonNull;
use std::os::windows::io::AsRawHandle;

use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, PrefetchVirtualMemory,
    UnmapViewOfFile, WIN32_MEMORY_RANGE_ENTRY,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::super::event::MmapSource;
use super::{PlatformError, SetupError, abort_on_release_failure};

pub(super) struct Region {
    base: Option<NonNull<u8>>,
    mapping: Option<windows_sys::Win32::Foundation::HANDLE>,
    len: usize,
    _file: MmapSource,
}

impl Region {
    pub(super) const fn base(&self) -> Option<NonNull<u8>> {
        self.base
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
    let mut information = SYSTEM_INFO::default();
    // SAFETY: `information` is valid writable storage for one SYSTEM_INFO;
    // GetSystemInfo initializes it synchronously and retains no pointer.
    unsafe { GetSystemInfo(&mut information) };
    u64::from(information.dwAllocationGranularity)
}

pub(super) const fn offset_supported(_: u64) -> bool {
    true
}

pub(super) fn prepare(file: &MmapSource, offset: u64, len: usize) -> Result<Region, SetupError> {
    // SAFETY: the borrowed std File handle stays owned by Region; parameters
    // request a read-only file mapping and no raw handle escapes this module.
    let mapping = unsafe {
        CreateFileMappingW(
            file.file().as_raw_handle(),
            core::ptr::null(),
            PAGE_READONLY,
            0,
            0,
            core::ptr::null(),
        )
    };
    if mapping.is_null() {
        return Err(SetupError::MappingFailed);
    }
    // SAFETY: mapping is a live read-only mapping handle, offset is allocation-
    // granularity aligned by SML validation, and len is within the open file.
    let view = unsafe {
        MapViewOfFile(
            mapping,
            FILE_MAP_READ,
            (offset >> 32) as u32,
            offset as u32,
            len,
        )
    };
    let region = Region {
        base: NonNull::new(view.Value.cast()),
        mapping: Some(mapping),
        len,
        _file: file.clone(),
    };
    if region.base.is_none() {
        return Err(SetupError::MappingFailed);
    }
    Ok(region)
}

pub(super) fn advise_sequential(_: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
    Ok(())
}

pub(super) fn advise_will_need(
    region: &Region,
    offset: usize,
    len: usize,
) -> Result<(), PlatformError> {
    let base = region.base.ok_or(PlatformError::UnmapFailed)?;
    // SAFETY: SML range validation proves offset remains within the live view;
    // the immutable view cannot be released while this function borrows it.
    let address = unsafe { base.as_ptr().add(offset) };
    let range = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: address.cast(),
        NumberOfBytes: len,
    };
    // SAFETY: the process pseudo-handle is valid and range describes the owned
    // live immutable view; PrefetchVirtualMemory retains no pointer.
    if unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) } != 0 {
        Ok(())
    } else {
        Err(PlatformError::AdviseFailed)
    }
}

pub(super) fn advise_dont_need(_: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
    Ok(())
}

pub(super) fn release(region: &mut Region) -> Result<(), PlatformError> {
    release_with::<NativeRelease>(region)
}

trait ReleaseOps {
    fn close(mapping: windows_sys::Win32::Foundation::HANDLE) -> bool;
    fn unmap(base: NonNull<u8>) -> bool;
}

struct NativeRelease;

impl ReleaseOps for NativeRelease {
    fn close(mapping: windows_sys::Win32::Foundation::HANDLE) -> bool {
        // SAFETY: mapping is the exact live file-mapping handle owned only by
        // this Region; the state update remains in `release_with`.
        unsafe { CloseHandle(mapping) != 0 }
    }

    fn unmap(base: NonNull<u8>) -> bool {
        // SAFETY: base is the exact live view base; exclusive actor ownership
        // proves no Rust reference exists during this synchronous call.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: base.as_ptr().cast(),
                },
            ) != 0
        }
    }
}

fn release_with<Ops: ReleaseOps>(region: &mut Region) -> Result<(), PlatformError> {
    if let Some(mapping) = region.mapping {
        if !Ops::close(mapping) {
            return Err(PlatformError::UnmapFailed);
        }
        region.mapping = None;
    }
    let Some(base) = region.base else {
        return Ok(());
    };
    if Ops::unmap(base) {
        region.base = None;
        region.len = 0;
        Ok(())
    } else {
        Err(PlatformError::UnmapFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CloseFails;
    struct UnmapFails;
    struct Succeeds;

    impl ReleaseOps for CloseFails {
        fn close(_: windows_sys::Win32::Foundation::HANDLE) -> bool {
            false
        }
        fn unmap(_: NonNull<u8>) -> bool {
            panic!("unmap must not run after close failure")
        }
    }

    impl ReleaseOps for UnmapFails {
        fn close(_: windows_sys::Win32::Foundation::HANDLE) -> bool {
            true
        }
        fn unmap(_: NonNull<u8>) -> bool {
            false
        }
    }

    impl ReleaseOps for Succeeds {
        fn close(_: windows_sys::Win32::Foundation::HANDLE) -> bool {
            true
        }
        fn unmap(_: NonNull<u8>) -> bool {
            true
        }
    }

    fn synthetic_region() -> Region {
        let path = std::env::temp_dir().join(format!(
            "emel-io-mmap-windows-release-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [0_u8]).expect("write private stable source");
        // SAFETY: this private test path has already been created, no other
        // handle can mutate it, and the file outlives the synthetic Region.
        let source = unsafe { MmapSource::open(&path) }.expect("open stable test source");
        std::fs::remove_file(path).expect("unlink private test source");
        Region {
            base: Some(NonNull::dangling()),
            mapping: Some(1_usize as windows_sys::Win32::Foundation::HANDLE),
            len: 1,
            _file: source,
        }
    }

    #[test]
    fn close_failure_retains_both_owners_for_retry() {
        let mut region = synthetic_region();
        assert_eq!(
            release_with::<CloseFails>(&mut region),
            Err(PlatformError::UnmapFailed)
        );
        assert!(region.mapping.is_some());
        assert!(region.base.is_some());

        assert_eq!(release_with::<Succeeds>(&mut region), Ok(()));
        assert!(region.mapping.is_none());
        assert!(region.base.is_none());
    }

    #[test]
    fn unmap_failure_retains_only_view_owner_for_retry() {
        let mut region = synthetic_region();
        assert_eq!(
            release_with::<UnmapFails>(&mut region),
            Err(PlatformError::UnmapFailed)
        );
        assert!(region.mapping.is_none());
        assert!(region.base.is_some());

        assert_eq!(release_with::<Succeeds>(&mut region), Ok(()));
        assert!(region.mapping.is_none());
        assert!(region.base.is_none());
    }
}
