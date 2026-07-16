//! Public mmap actor lifecycle coverage.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use emel_io::mmap::Mapper;
use emel_io::mmap::event::{
    AdviseDontNeed, AdviseSequential, AdviseWillNeed, Error, MapDone, MapTensor, MmapSource,
    ReleaseMapping, WithMapping,
};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "emel-io-mmap-{}-{sequence}.bin",
            std::process::id()
        ));
        let bytes: Vec<u8> = (0_u8..=255).cycle().take(32_768).collect();
        std::fs::write(&path, bytes).expect("write mmap fixture");
        Self { path }
    }

    #[allow(
        unsafe_code,
        reason = "the fixture remains unchanged until every mapping in the test is released"
    )]
    fn mmap_source(&self) -> MmapSource {
        // SAFETY: the test exclusively owns this already-created private path,
        // never mutates it during or after this call, and releases every mapping
        // before Fixture::drop.
        unsafe { MmapSource::open(&self.path) }.expect("open mmap fixture source")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn map(mapper: &mut Mapper, file: &MmapSource, tensor_id: i32, len: u64) -> MapDone {
    mapper
        .process_event(MapTensor::new(tensor_id, file.clone(), 0, len))
        .expect("map validates, maps, and commits before dispatch returns")
}

#[test]
fn replacement_protocol_exposes_immutable_metadata_and_synchronous_apply() {
    let fixture = Fixture::new();
    let request = MapTensor::new(12, fixture.mmap_source(), 64, 128).with_file_index(3);
    assert_eq!(request.tensor_id(), 12);
    assert_eq!(request.file_index(), 3);
    assert_eq!(request.offset(), 64);
    assert_eq!(request.len(), 128);
    assert!(!request.is_empty());

    let release = ReleaseMapping::new(99, 7);
    assert_eq!(release.tensor_id(), 99);
    assert_eq!(release.handle(), 7);

    let mut checksum = |bytes: &[u8]| bytes.iter().copied().map(u16::from).sum::<u16>();
    let access = WithMapping::new(99, 7, &mut checksum);
    assert_eq!(access.tensor_id(), 99);
    assert_eq!(access.handle(), 7);
    assert_eq!(access.apply(&[1, 2, 3]), 6);
}

#[test]
fn map_access_advice_and_release_run_through_public_events() {
    let fixture = Fixture::new();
    let file = fixture.mmap_source();
    let mut mapper = Mapper::new();
    let done = map(&mut mapper, &file, 17, 4_096);

    let mut parse =
        |bytes: &[u8]| u32::from_le_bytes(bytes[..4].try_into().expect("four mapped bytes"));
    let parsed = mapper
        .process_event(WithMapping::new(17, done.handle(), &mut parse))
        .expect("mapping access");
    assert_eq!(
        parsed, 0x0302_0100,
        "callback completed before dispatch returned"
    );

    mapper
        .process_event(AdviseSequential::new(17, done.handle(), 1, 1_024))
        .expect("sequential advice completes during dispatch");
    mapper
        .process_event(AdviseWillNeed::new(17, done.handle(), 1, 1_024))
        .expect("will-need advice completes during dispatch");
    mapper
        .process_event(AdviseDontNeed::new(17, done.handle(), 1, 1_024))
        .expect("don't-need advice completes during dispatch");

    mapper
        .process_event(ReleaseMapping::new(17, done.handle()))
        .expect("native release completes during dispatch");
    assert!(matches!(
        mapper.process_event(ReleaseMapping::new(17, done.handle())),
        Err(Error::InvalidRequest)
    ));
}

#[test]
fn consecutive_dispatches_never_expose_a_waiting_phase() {
    let fixture = Fixture::new();
    let file = fixture.mmap_source();
    let mut mapper = Mapper::new();
    let first = map(&mut mapper, &file, 1, 4_096);
    let second = map(&mut mapper, &file, 2, 4_096);
    assert_eq!(first.handle(), 0);
    assert_eq!(second.handle(), 1);

    mapper
        .process_event(AdviseSequential::new(2, second.handle(), 0, 64))
        .expect("advice restores ownership before returning");

    let mut observe = |bytes: &[u8]| bytes.len();
    let observed_len = mapper
        .process_event(WithMapping::new(2, second.handle(), &mut observe))
        .expect("completed advice left the mapping ready");
    assert_eq!(observed_len, 4_096);

    mapper
        .process_event(ReleaseMapping::new(2, second.handle()))
        .expect("second release completes during dispatch");
    mapper
        .process_event(ReleaseMapping::new(1, first.handle()))
        .expect("first release completes during dispatch");
}

#[test]
fn failed_setup_frees_reservation_without_consuming_capacity() {
    let fixture = Fixture::new();
    let file = fixture.mmap_source();
    let mut mapper = Mapper::new();
    let missing = fixture.path.with_extension("missing");

    #[allow(
        unsafe_code,
        reason = "the missing path cannot yield a capability or a live mapping"
    )]
    let error = {
        // SAFETY: this process-private path is kept absent before and throughout
        // the call, so there is no target file that another operation can mutate.
        unsafe { MmapSource::open(&missing) }.expect_err("missing file must fail setup")
    };
    assert_eq!(error, Error::FileOpenFailed);

    let error = mapper
        .process_event(MapTensor::new(4, file.clone(), 0, 65_536))
        .expect_err("mapping past EOF must fail");
    assert_eq!(error, Error::UnsupportedResource);

    let done = map(&mut mapper, &file, 5, 4_096);
    assert_eq!(done.handle(), 0, "failed setup returned its reservation");
    mapper
        .process_event(ReleaseMapping::new(5, done.handle()))
        .expect("native release");
}

#[test]
fn invalid_requests_do_not_consume_slots_or_invoke_callbacks() {
    let fixture = Fixture::new();
    let file = fixture.mmap_source();
    let mut mapper = Mapper::new();
    assert!(matches!(
        mapper.process_event(MapTensor::new(6, file.clone(), 0, 0)),
        Err(Error::InvalidRequest)
    ));

    let done = map(&mut mapper, &file, 7, 4_096);
    assert!(matches!(
        mapper.process_event(AdviseWillNeed::new(8, done.handle(), 0, 64)),
        Err(Error::InvalidRequest)
    ));
    assert!(matches!(
        mapper.process_event(AdviseWillNeed::new(7, done.handle(), 4_000, 1_000)),
        Err(Error::InvalidAdviceRange)
    ));

    let mut invoked = false;
    let mut observe = |_: &[u8]| invoked = true;
    assert_eq!(
        mapper.process_event(WithMapping::new(8, done.handle(), &mut observe)),
        Err(Error::InvalidRequest),
    );
    assert!(!invoked);

    mapper
        .process_event(ReleaseMapping::new(7, done.handle()))
        .expect("native release");
}

#[test]
#[should_panic(expected = "operation panic probe")]
fn operation_panic_propagates_as_a_programming_defect() {
    let fixture = Fixture::new();
    let file = fixture.mmap_source();
    let mut mapper = Mapper::new();
    let done = map(&mut mapper, &file, 9, 4_096);

    let mut panic_operation = |_: &[u8]| panic!("operation panic probe");
    let _ = mapper.process_event(WithMapping::new(9, done.handle(), &mut panic_operation));
}
