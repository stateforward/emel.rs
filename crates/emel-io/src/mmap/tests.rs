use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::event::{AdviceRequest, Error, MapTensor, MmapSource, ReleaseMapping, WithMapping};
use super::platform::{
    Native, Platform, PlatformError, Region, SetupError, abort_on_release_failure,
};
use super::sm::MapperCore;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
#[test]
fn native_alignment_comes_from_the_safe_platform_primitive() {
    assert_eq!(
        Native::new().required_alignment(),
        u64::try_from(rustix::param::page_size()).expect("page size fits u64"),
    );
}

#[test]
fn rust_slice_length_boundary_rejects_first_unrepresentable_view() {
    assert!(super::sm::mapping_view_len_supported(isize::MAX as u64));
    assert!(!super::sm::mapping_view_len_supported(
        (isize::MAX as u64) + 1
    ));
}

#[test]
fn failed_destructor_release_aborts_instead_of_losing_native_ownership() {
    const CHILD_ENV: &str = "EMEL_IO_MMAP_DROP_FAILURE_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        abort_on_release_failure(Err(PlatformError::UnmapFailed));
        unreachable!("release failure policy must abort");
    }
    let status = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .args([
            "--exact",
            "mmap::tests::failed_destructor_release_aborts_instead_of_losing_native_ownership",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .status()
        .expect("run destructor failure child");
    assert!(!status.success(), "native release failure must fail closed");
}

#[derive(Clone, Copy)]
enum Mode {
    Native,
    Unsupported,
    SetupMappingFailed,
    SetupPlatformFailed,
    AdviceFailed,
    ReleaseFailed,
}

#[derive(Clone, Copy)]
struct FaultPlatform {
    mode: Mode,
    native: Native,
}

impl FaultPlatform {
    fn new(mode: Mode) -> Self {
        Self {
            mode,
            native: Native::new(),
        }
    }
}

impl Platform for FaultPlatform {
    fn supported(self) -> bool {
        !matches!(self.mode, Mode::Unsupported)
    }

    fn required_alignment(self) -> u64 {
        self.native.required_alignment()
    }

    fn offset_supported(self, offset: u64) -> bool {
        self.native.offset_supported(offset)
    }

    fn prepare(self, file: &MmapSource, offset: u64, len: usize) -> Result<Region, SetupError> {
        match self.mode {
            Mode::SetupMappingFailed => Err(SetupError::MappingFailed),
            Mode::SetupPlatformFailed => Err(SetupError::UnsupportedPlatform),
            _ => self.native.prepare(file, offset, len),
        }
    }

    fn advise_sequential(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        if matches!(self.mode, Mode::AdviceFailed) {
            Err(PlatformError::AdviseFailed)
        } else {
            self.native.advise_sequential(region, offset, len)
        }
    }

    fn advise_will_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        self.native.advise_will_need(region, offset, len)
    }

    fn advise_dont_need(
        self,
        region: &Region,
        offset: usize,
        len: usize,
    ) -> Result<(), PlatformError> {
        self.native.advise_dont_need(region, offset, len)
    }

    fn release(self, region: &mut Region) -> Result<(), PlatformError> {
        if matches!(self.mode, Mode::ReleaseFailed) {
            Err(PlatformError::UnmapFailed)
        } else {
            self.native.release(region)
        }
    }
}

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "emel-io-mmap-unit-{}-{sequence}.bin",
            std::process::id()
        ));
        std::fs::write(&path, vec![0xa5; 4_096]).expect("write mmap unit fixture");
        Self(path)
    }

    #[allow(
        unsafe_code,
        reason = "the fixture path is private and remains unchanged until all test mappings are released"
    )]
    fn mmap_source(&self) -> MmapSource {
        // SAFETY: this fixture exclusively owns an already-created private path,
        // never mutates it during or after this call, and is dropped after every
        // mapper/core.
        unsafe { MmapSource::open(&self.0) }.expect("open mmap unit fixture source")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(unix)]
trait NullCleanup {
    fn release(region: &mut Region) -> Result<(), PlatformError>;
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct CleanupSucceeds;

#[cfg(unix)]
impl NullCleanup for CleanupSucceeds {
    fn release(region: &mut Region) -> Result<(), PlatformError> {
        super::platform::release_synthetic_success(region)
    }
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct CleanupFails;

#[cfg(unix)]
impl NullCleanup for CleanupFails {
    fn release(region: &mut Region) -> Result<(), PlatformError> {
        super::platform::release_synthetic_failure(region)
    }
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct NullSetupPlatform<Cleanup>(core::marker::PhantomData<Cleanup>);

#[cfg(unix)]
impl<Cleanup> NullSetupPlatform<Cleanup> {
    const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

#[cfg(unix)]
impl<Cleanup: NullCleanup + Copy> Platform for NullSetupPlatform<Cleanup> {
    fn supported(self) -> bool {
        true
    }

    fn required_alignment(self) -> u64 {
        1
    }

    fn offset_supported(self, _: u64) -> bool {
        true
    }

    fn prepare(self, file: &MmapSource, _: u64, len: usize) -> Result<Region, SetupError> {
        Ok(super::platform::synthetic_null_region(file, len))
    }

    fn advise_sequential(self, _: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
        Ok(())
    }

    fn advise_will_need(self, _: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
        Ok(())
    }

    fn advise_dont_need(self, _: &Region, _: usize, _: usize) -> Result<(), PlatformError> {
        Ok(())
    }

    fn release(self, region: &mut Region) -> Result<(), PlatformError> {
        Cleanup::release(region)
    }
}

#[cfg(unix)]
#[test]
fn successful_null_mapping_is_released_and_never_consumes_capacity() {
    let fixture = Fixture::new();
    let request = MapTensor::new(11, fixture.mmap_source(), 0, 4_096);
    let mut core = MapperCore::new(NullSetupPlatform::<CleanupSucceeds>::new());

    for _ in 0..=256 {
        assert_eq!(core.map(&request), Err(Error::MappingFailed));
    }
}

#[cfg(unix)]
#[test]
fn failed_null_mapping_cleanup_retains_owner_for_retry() {
    let fixture = Fixture::new();
    let mut region = super::platform::synthetic_null_region(&fixture.mmap_source(), 4_096);

    assert!(super::platform::owns_mapping(&region));
    assert_eq!(
        super::platform::release_synthetic_failure(&mut region),
        Err(PlatformError::UnmapFailed)
    );
    assert!(super::platform::owns_mapping(&region));
    assert_eq!(
        super::platform::release_synthetic_success(&mut region),
        Ok(())
    );
    assert!(!super::platform::owns_mapping(&region));
}

#[cfg(unix)]
#[test]
fn failed_null_mapping_cleanup_aborts_the_dispatch() {
    const CHILD_ENV: &str = "EMEL_IO_MMAP_NULL_CLEANUP_FAILURE_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        let fixture = Fixture::new();
        let request = MapTensor::new(12, fixture.mmap_source(), 0, 4_096);
        let mut core = MapperCore::new(NullSetupPlatform::<CleanupFails>::new());
        let _ = core.map(&request);
        unreachable!("failed null cleanup must abort");
    }

    let status = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .args([
            "--exact",
            "mmap::tests::failed_null_mapping_cleanup_aborts_the_dispatch",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .status()
        .expect("run null cleanup failure child");
    assert!(!status.success(), "failed null cleanup must fail closed");
}

#[test]
fn injected_platform_and_setup_failures_are_classified_and_recover() {
    let fixture = Fixture::new();
    let request = MapTensor::new(1, fixture.mmap_source(), 0, 4_096);

    let mut unsupported = MapperCore::new(FaultPlatform::new(Mode::Unsupported));
    assert_eq!(unsupported.map(&request), Err(Error::UnsupportedPlatform));

    for (mode, expected) in [
        (Mode::SetupMappingFailed, Error::MappingFailed),
        (Mode::SetupPlatformFailed, Error::UnsupportedPlatform),
    ] {
        let mut core = MapperCore::new(FaultPlatform::new(mode));
        assert_eq!(core.map(&request), Err(expected));
        assert_eq!(core.map(&request), Err(expected));
    }
}

#[test]
fn injected_advice_and_release_failures_restore_exact_mapping_ownership() {
    let fixture = Fixture::new();
    let request = MapTensor::new(2, fixture.mmap_source(), 0, 4_096);
    let mut advice_core = MapperCore::new(FaultPlatform::new(Mode::AdviceFailed));
    let handle = map_core(&mut advice_core, &request);
    let advice = AdviceRequest {
        tensor_id: 2,
        handle,
        offset: 0,
        len: 4_096,
    };
    assert_eq!(
        advice_core.advise_sequential(advice),
        Err(Error::AdviceFailed)
    );
    let mut observe = |bytes: &[u8]| bytes.len();
    let observed = advice_core
        .with_mapping(WithMapping::new(2, handle, &mut observe))
        .expect("failed advice restores region");
    assert_eq!(observed, 4_096);
    advice_core
        .release(ReleaseMapping::new(2, handle))
        .expect("release advice test mapping");

    let mut release_core = MapperCore::new(FaultPlatform::new(Mode::ReleaseFailed));
    let handle = map_core(&mut release_core, &request);
    assert_eq!(
        release_core.release(ReleaseMapping::new(2, handle)),
        Err(Error::UnmapFailed)
    );
    let mut observe = |bytes: &[u8]| bytes.len();
    let observed = release_core
        .with_mapping(WithMapping::new(2, handle, &mut observe))
        .expect("failed release restores region");
    assert_eq!(observed, 4_096);
}

#[test]
fn full_native_slot_pool_reports_resource_exhaustion_then_recovers() {
    let fixture = Fixture::new();
    let request = MapTensor::new(3, fixture.mmap_source(), 0, 4_096);
    let mut core = MapperCore::new(FaultPlatform::new(Mode::Native));
    let mut handles = Vec::with_capacity(256);
    for _ in 0..256 {
        handles.push(map_core(&mut core, &request));
    }
    assert_eq!(core.map(&request), Err(Error::ResourceExhausted));
    for handle in handles {
        core.release(ReleaseMapping::new(3, handle))
            .expect("release capacity mapping");
    }
    let handle = core.map(&request).expect("capacity recovered").handle();
    assert_eq!(handle, 255);
    core.release(ReleaseMapping::new(3, handle))
        .expect("release recovered capacity mapping");
}

fn map_core(core: &mut MapperCore<FaultPlatform>, request: &MapTensor) -> u32 {
    core.map(request)
        .expect("map through complete SML dispatch")
        .handle()
}
