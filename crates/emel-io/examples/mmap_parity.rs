//! Source-backed golden parity cases for the pinned emel.cpp mmap actor.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

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

const FIXTURE_BYTES: usize = 32_768;
const MAPPING_BYTES: u64 = 16_384;

#[allow(
    dead_code,
    reason = "this file is also imported by the parity integration test"
)]
fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 2 && arguments[0] == "--write-fixture" {
        write_fixture(Path::new(&arguments[1])).expect("write mmap parity fixture");
        return;
    }
    if arguments.len() != 1 {
        eprintln!("usage: mmap_parity [--write-fixture] FILE");
        std::process::exit(2);
    }
    print!("{}", render_manifest(Path::new(&arguments[0])));
}

/// Writes the deterministic file image consumed by both parity processes.
///
/// # Errors
///
/// Returns the filesystem error from creating or writing `path`.
pub fn write_fixture(path: &Path) -> std::io::Result<()> {
    let bytes = (0_u8..=255).cycle().take(FIXTURE_BYTES).collect::<Vec<_>>();
    std::fs::write(path, bytes)
}

/// Renders canonical outcomes from the public Rust mmap actor.
pub fn render_manifest(path: &Path) -> String {
    let mut manifest = String::from(
        "io-mmap-parity-snapshot/v1\n\
         source_repository=stateforward/emel.cpp\n\
         source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n\
         source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n\
         source_files=src/emel/io/mmap/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,actions.cpp,sm.hpp\n\
         source_tests=tests/io/mmap/lifecycle_tests.cpp,tests/io/mmap/advise_tests.cpp\n\
         fixture_config=file_bytes=32768,pattern=incrementing_u8,map_bytes=16384,offset=0\n\
         native_semantics_complete=true\n\
         missing_native_semantics=none\n\
         contract_delta=rust_safe_access_event,rust_mmap_source_capability\n",
    );

    let file = mmap_source(path);
    render_validation_cases(&mut manifest, &file, path);
    render_lifecycle_cases(&mut manifest, &file);
    manifest
}

fn render_validation_cases(manifest: &mut String, file: &MmapSource, path: &Path) {
    render_map_error(
        manifest,
        "invalid_zero",
        MapTensor::new(1, file.clone(), 0, 0),
    );
    render_error(
        manifest,
        "map_invalid_empty_path",
        mmap_source_error(Path::new("")),
    );
    render_map_error(
        manifest,
        "unsupported_index",
        MapTensor::new(3, file.clone(), 0, MAPPING_BYTES).with_file_index(u16::MAX),
    );
    render_map_error(
        manifest,
        "unsupported_offset",
        MapTensor::new(4, file.clone(), 1, MAPPING_BYTES),
    );
    render_map_error(
        manifest,
        "unsupported_length",
        MapTensor::new(5, file.clone(), 0, (1_u64 << 40) + 1),
    );
    render_map_error(
        manifest,
        "unsupported_layout",
        MapTensor::new(6, file.clone(), u64::MAX - 16_383, MAPPING_BYTES),
    );
    let missing = missing_path(path);
    render_error(manifest, "map_file_open", mmap_source_error(&missing));
    render_map_error(
        manifest,
        "past_eof",
        MapTensor::new(8, file.clone(), 16_384, 32_768),
    );
}

fn render_map_error(manifest: &mut String, name: &str, request: MapTensor) {
    let mut mapper = Mapper::new();
    let error = mapper
        .process_event(request)
        .expect_err("pinned map error case must fail");
    render_error(manifest, &format!("map_{name}"), error);
}

fn render_error(manifest: &mut String, name: &str, error: Error) {
    writeln!(
        manifest,
        "case={name} outcome=error error={}",
        error_name(error),
    )
    .expect("writing to String is infallible");
}

fn render_lifecycle_cases(manifest: &mut String, file: &MmapSource) {
    let mut mapper = Mapper::new();
    let done = map(&mut mapper, file, 10);

    let mut observe = |bytes: &[u8]| fnv1a64(bytes);
    let checksum = mapper
        .process_event(WithMapping::new(10, done.handle(), &mut observe))
        .expect("public mapping access");
    writeln!(
        manifest,
        "case=map_success outcome=done tensor_id={} handle={} bytes={} checksum={checksum:016x}",
        done.tensor_id(),
        done.handle(),
        done.len(),
    )
    .expect("writing to String is infallible");

    mapper
        .process_event(AdviseSequential::new(10, done.handle(), 24, 1_000))
        .expect("sequential advice");
    writeln!(manifest, "case=advise_sequential outcome=done")
        .expect("writing to String is infallible");
    mapper
        .process_event(AdviseWillNeed::new(10, done.handle(), 24, 1_000))
        .expect("will-need advice");
    writeln!(manifest, "case=advise_will_need outcome=done")
        .expect("writing to String is infallible");
    mapper
        .process_event(AdviseDontNeed::new(10, done.handle(), 24, 1_000))
        .expect("don't-need advice");
    writeln!(manifest, "case=advise_dont_need outcome=done")
        .expect("writing to String is infallible");

    render_advice_error(
        manifest,
        "wrong_owner",
        mapper.process_event(AdviseWillNeed::new(11, done.handle(), 0, 64)),
    );
    render_advice_error(
        manifest,
        "invalid_range",
        mapper.process_event(AdviseWillNeed::new(10, done.handle(), MAPPING_BYTES - 1, 2)),
    );

    let wrong_owner = mapper.process_event(ReleaseMapping::new(11, done.handle()));
    render_result(manifest, "release_wrong_owner", wrong_owner);
    mapper
        .process_event(ReleaseMapping::new(10, done.handle()))
        .expect("native release");
    writeln!(manifest, "case=release_success outcome=done")
        .expect("writing to String is infallible");
    let released = mapper.process_event(ReleaseMapping::new(10, done.handle()));
    render_result(manifest, "release_again", released);
}

fn map(mapper: &mut Mapper, file: &MmapSource, tensor_id: i32) -> MapDone {
    mapper
        .process_event(MapTensor::new(tensor_id, file.clone(), 0, MAPPING_BYTES))
        .expect("native map")
}

#[allow(
    unsafe_code,
    reason = "the parity fixture is immutable until the complete mapper lifecycle finishes"
)]
fn mmap_source(path: &Path) -> MmapSource {
    // SAFETY: fixture construction is complete before this call, both parity
    // processes exclusively read it throughout construction and mapping, and all
    // mappings are released before render_manifest returns.
    unsafe { MmapSource::open(path) }.expect("open mmap parity fixture source")
}

#[allow(
    unsafe_code,
    reason = "these paths cannot produce a capability, so no stability lifetime begins"
)]
fn mmap_source_error(path: &Path) -> Error {
    // SAFETY: the harness keeps this invalid path unchanged before and throughout
    // the call; successful construction would invalidate the pinned case and
    // immediately panic before any mapping could be created.
    unsafe { MmapSource::open(path) }.expect_err("pinned mmap-source error case must fail")
}

fn render_advice_error<Token>(manifest: &mut String, name: &str, result: Result<Token, Error>) {
    render_result(manifest, &format!("advise_{name}"), result.map(|_| ()));
}

fn render_result(manifest: &mut String, name: &str, result: Result<(), Error>) {
    let error = result.expect_err("pinned error case must fail");
    writeln!(
        manifest,
        "case={name} outcome=error error={}",
        error_name(error),
    )
    .expect("writing to String is infallible");
}

const fn error_name(error: Error) -> &'static str {
    match error {
        Error::InvalidRequest => "invalid_request",
        Error::UnsupportedPlatform => "unsupported_platform",
        Error::UnsupportedResource => "unsupported_resource",
        Error::ResourceExhausted => "resource_exhausted",
        Error::FileOpenFailed => "file_open_failed",
        Error::MappingFailed => "mapping_failed",
        Error::UnmapFailed => "unmap_failed",
        Error::InvalidAdviceRange => "invalid_advise_range",
        Error::AdviceFailed => "advise_failed",
        Error::InternalError => "internal_error",
        _ => "unknown",
    }
}

fn missing_path(path: &Path) -> PathBuf {
    let mut missing = path.as_os_str().to_os_string();
    missing.push(".missing");
    PathBuf::from(missing)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
