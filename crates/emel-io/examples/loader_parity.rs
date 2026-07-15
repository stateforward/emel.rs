//! Source-backed parity cases for the pinned emel.cpp loader actor.

use core::cell::Cell;
use std::fmt::Write as _;

use emel_io::loader::Loader;
use emel_io::loader::event::{
    Callback, Error, LoadTensor, LoadTensorBatch, LoadTensorBatchDone, LoadTensorBatchError,
    LoadTensorDone, LoadTensorError, SourceError, StrategyError, StrategyKind, StrategyPolicy,
    Target, TensorLoadSpan,
};
use emel_io::{read::Reader, staged_read::Stager};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[allow(dead_code, reason = "also imported by the parity integration test")]
fn main() {
    print!("{}", render_manifest());
}

/// Renders canonical outcomes exclusively through the public loader actor.
pub fn render_manifest() -> String {
    let mut output = String::from(
        "io-loader-parity-snapshot/v1\n\
         source_repository=stateforward/emel.cpp\n\
         source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n\
         source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n\
         source_files=src/emel/io/loader/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n\
         source_tests=tests/io/loader/lifecycle_tests.cpp\n\
         fixture_config=public_loader,static_read_and_staged_dependencies,single_and_batch,typed_failures\n\
         contract_delta=rust_target_capability_replaces_raw_buffer_result;rust_loader_batch_max=65536_for_bounded_rtc_cpp_reference_unbounded\n",
    );
    render_invalid_and_unsupported(&mut output);
    render_single_successes(&mut output);
    render_single_failures(&mut output);
    render_batch_successes(&mut output);
    render_batch_failures(&mut output);
    output
}

fn render_invalid_and_unsupported(output: &mut String) {
    let source = b"abcdef";
    let mut bytes = [0_u8; 3];
    let target = Target::new(&mut bytes);
    let valid = TensorLoadSpan::new(3, "fixture.bin", Some(source), &target).with_range(1, 3);
    let invalid = valid.with_range(1, 0);
    let mut loader = Loader::new();
    render_single_result(
        output,
        "invalid",
        loader.process_event(LoadTensor::new(
            invalid,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    for (name, strategy) in [
        ("none", StrategyKind::None),
        ("mapped", StrategyKind::MappedFile),
        ("external", StrategyKind::ExternalBuffer),
        ("unknown", StrategyKind::Unknown(255)),
    ] {
        render_single_result(
            output,
            name,
            loader.process_event(LoadTensor::new(valid, StrategyPolicy::new(strategy))),
        );
    }
}

fn render_single_successes(output: &mut String) {
    for (name, strategy) in [
        ("read_success", StrategyKind::ReadCopy),
        ("staged_success", StrategyKind::StagedRead),
    ] {
        let source = b"abcdefgh";
        let mut bytes = [0_u8; 4];
        let done = Cell::new(None::<LoadTensorDone>);
        let error = Cell::new(None::<LoadTensorError>);
        let target = Target::new(&mut bytes);
        let span = TensorLoadSpan::new(17, "fixture.bin", Some(source), &target)
            .with_file_index(1)
            .with_range(2, 4);
        let policy = StrategyPolicy::new(strategy).with_staged_chunk_bytes(2);
        let result = Loader::with_dependencies(Reader::new(), Stager::new()).process_event(
            LoadTensor::new(span, policy)
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        );
        let value = result.expect("pinned success");
        writeln!(
            output,
            "case={name} outcome=done strategy={} bytes_loaded={} callback={} target={}",
            strategy_name(value.strategy()),
            value.bytes_loaded(),
            done.get().is_some(),
            hex(&bytes),
        )
        .unwrap();
    }
}

fn render_single_failures(output: &mut String) {
    let source = b"abcdef";
    let mut bytes = [0_u8; 3];
    let target = Target::new(&mut bytes);
    let span = TensorLoadSpan::new(8, "fixture.bin", Some(source), &target).with_range(1, 3);
    render_single_result(
        output,
        "read_absent",
        Loader::new().process_event(LoadTensor::new(
            span,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    let failed = span.with_source_error(SourceError::FileReadFailed);
    render_single_result(
        output,
        "read_failure",
        Loader::with_reader(Reader::new()).process_event(LoadTensor::new(
            failed,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    render_single_result(
        output,
        "staged_failure",
        Loader::with_stager(Stager::new()).process_event(LoadTensor::new(
            span,
            StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(0),
        )),
    );
}

fn render_batch_successes(output: &mut String) {
    for (name, strategy) in [
        ("batch_read", StrategyKind::ReadCopy),
        ("batch_staged", StrategyKind::StagedRead),
    ] {
        let source = b"abcdefghij";
        let mut first = [0_u8; 3];
        let mut second = [0_u8; 4];
        let first_target = Target::new(&mut first);
        let second_target = Target::new(&mut second);
        let spans = [
            TensorLoadSpan::new(10, "fixture.bin", Some(source), &first_target).with_range(1, 3),
            TensorLoadSpan::new(11, "fixture.bin", Some(source), &second_target).with_range(5, 4),
        ];
        let done = Cell::new(None::<LoadTensorBatchDone>);
        let error = Cell::new(None::<LoadTensorBatchError>);
        let result = Loader::with_dependencies(Reader::new(), Stager::new()).process_event(
            LoadTensorBatch::new(&spans, StrategyPolicy::new(strategy))
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        );
        let value = result.expect("pinned batch success");
        writeln!(
            output,
            "case={name} outcome=done strategy={} done_count={} bytes_loaded={} callback={} first={} second={}",
            strategy_name(value.strategy()), value.done_count(), value.bytes_loaded(),
            done.get().is_some(), hex(&first), hex(&second),
        ).unwrap();
    }
}

fn render_batch_failures(output: &mut String) {
    let source = b"abcdef";
    let mut bytes = [0_u8; 3];
    let target = Target::new(&mut bytes);
    let invalid = [TensorLoadSpan::new(13, "fixture.bin", Some(source), &target).with_range(1, 0)];
    render_batch_result(
        output,
        "batch_invalid",
        Loader::with_reader(Reader::new()).process_event(LoadTensorBatch::new(
            &invalid,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    let absent = [invalid[0].with_range(1, 3)];
    render_batch_result(
        output,
        "batch_absent",
        Loader::new().process_event(LoadTensorBatch::new(
            &absent,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    let read_failed = [
        absent[0],
        absent[0].with_source_error(SourceError::FileReadFailed),
    ];
    render_batch_result(
        output,
        "batch_read_failure",
        Loader::with_reader(Reader::new()).process_event(LoadTensorBatch::new(
            &read_failed,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        )),
    );
    render_batch_result(
        output,
        "batch_staged_failure",
        Loader::with_stager(Stager::new()).process_event(LoadTensorBatch::new(
            &absent,
            StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(0),
        )),
    );
}

fn render_single_result(
    output: &mut String,
    name: &str,
    result: Result<LoadTensorDone, LoadTensorError>,
) {
    let error = result.expect_err("pinned error");
    writeln!(
        output,
        "case={name} outcome=error error={} strategy_error={}",
        error_name(error.error()),
        strategy_error_name(error.strategy_error())
    )
    .unwrap();
}

fn render_batch_result(
    output: &mut String,
    name: &str,
    result: Result<LoadTensorBatchDone, LoadTensorBatchError>,
) {
    let error = result.expect_err("pinned batch error");
    writeln!(
        output,
        "case={name} outcome=error error={} strategy_error={} failed_index={}",
        error_name(error.error()),
        strategy_error_name(error.strategy_error()),
        error.failed_index()
    )
    .unwrap();
}

const fn strategy_name(value: StrategyKind) -> &'static str {
    match value {
        StrategyKind::ReadCopy => "read_copy",
        StrategyKind::StagedRead => "staged_read",
        _ => "other",
    }
}

const fn error_name(value: Error) -> &'static str {
    match value {
        Error::InvalidRequest => "invalid_request",
        Error::UnsupportedStrategy => "unsupported_strategy",
        Error::Unavailable => "unavailable",
        _ => "internal_error",
    }
}

const fn strategy_error_name(value: StrategyError) -> &'static str {
    match value {
        StrategyError::None => "none",
        StrategyError::Read(emel_io::read::event::Error::FileReadFailed) => "file_read_failed",
        StrategyError::Read(_) => "read_error",
        StrategyError::StagedRead(emel_io::staged_read::event::Error::InvalidStageContract) => {
            "invalid_stage_contract"
        }
        StrategyError::StagedRead(_) => "staged_read_error",
        _ => "strategy_error",
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").unwrap();
    }
    output
}
