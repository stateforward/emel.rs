use super::super::event::MmapSource;
use super::{PlatformError, SetupError};
use core::ptr::NonNull;

pub(super) struct Region;

impl Region {
    pub(super) const fn base(&self) -> Option<NonNull<u8>> {
        None
    }
    pub(super) const fn len(&self) -> usize {
        0
    }
}

pub(super) const fn supported() -> bool {
    false
}
pub(super) const fn required_alignment() -> u64 {
    4096
}
pub(super) const fn offset_supported(_: u64) -> bool {
    true
}
pub(super) fn prepare(_: &MmapSource, _: u64, _: usize) -> Result<Region, SetupError> {
    Err(SetupError::UnsupportedPlatform)
}
pub(super) fn advise_sequential(_: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
    Err(PlatformError::UnsupportedPlatform)
}
pub(super) fn advise_will_need(_: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
    Err(PlatformError::UnsupportedPlatform)
}
pub(super) fn advise_dont_need(_: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
    Err(PlatformError::UnsupportedPlatform)
}
pub(super) fn release(_: &mut Region) -> Result<(), PlatformError> {
    Err(PlatformError::UnsupportedPlatform)
}
