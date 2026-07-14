//! Source-backed golden parity cases for the pinned emel.cpp read actor.

use core::cell::Cell;
use std::fmt::Write as _;

use emel_io::read::Reader;
use emel_io::read::event::{
    Callback, Error, ReadTensor, ReadTensorBatch, ReadTensorBatchDone, ReadTensorBatchError,
    ReadTensorDone, ReadTensorError, SourceError, Target, TensorRead,
};
use sml as _;

const MAX_READ_BATCH_TENSORS: usize = 65_536;

#[allow(
    dead_code,
    reason = "this file is also imported by the parity integration test"
)]
fn main() {
    print!("{}", render_manifest());
}

/// Renders the canonical public-actor outcomes used by the parity gate.
pub fn render_manifest() -> String {
    let mut manifest = String::from(
        "io-read-parity-snapshot/v1\n\
         source_repository=stateforward/emel.cpp\n\
         source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n\
         source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n\
         source_files=src/emel/io/read/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n\
         source_tests=tests/io/read/lifecycle_tests.cpp\n\
         fixture_config=caller_owned_memory,platform_supported,synchronous_callbacks,distinct_batch_targets,mixed_batch_phase_precedence,batch_count_boundaries\n\
         contract_delta=rust_typed_results_make_callbacks_optional\n",
    );

    render_single_success(&mut manifest);
    render_single_errors(&mut manifest);
    render_batch_success(&mut manifest);
    render_batch_errors(&mut manifest);
    render_batch_count_boundaries(&mut manifest);
    manifest
}

fn render_single_success(manifest: &mut String) {
    let mut reader = Reader::new();
    let mut target = [0_u8; 4];
    let callback = Cell::new(None::<ReadTensorDone>);
    let result = reader.process_event(
        ReadTensor::new(7, "tensor.bin", Some(b"abcdef"), &mut target)
            .with_range(1, 4)
            .on_done(Callback::store(&callback)),
    );
    let done = result.expect("pinned single-success case must succeed");
    let callback = callback
        .get()
        .expect("pinned single-success callback must run synchronously");
    writeln!(
        manifest,
        "case=single_success outcome=done tensor_id={} bytes_copied={} target={} callback_tensor_id={} callback_bytes_copied={}",
        done.tensor_id(),
        done.bytes_copied(),
        hex(&target),
        callback.tensor_id(),
        callback.bytes_copied(),
    )
    .expect("writing to String is infallible");
}

fn render_single_errors(manifest: &mut String) {
    render_single_error(manifest, "invalid_zero", |target| {
        ReadTensor::new(1, "tensor.bin", Some(b"abcd"), target).with_range(0, 0)
    });
    render_single_error(manifest, "invalid_path", |target| {
        ReadTensor::new(2, "bad\0path", Some(b"abcd"), target)
    });
    render_single_error(manifest, "unsupported_index", |target| {
        ReadTensor::new(3, "tensor.bin", Some(b"abcd"), target).with_file_index(u16::MAX)
    });
    render_single_error(manifest, "unsupported_length", |target| {
        ReadTensor::new(4, "tensor.bin", Some(b"abcd"), target).with_range(0, (1_u64 << 40) + 1)
    });
    render_single_error(manifest, "unsupported_layout", |target| {
        ReadTensor::new(5, "tensor.bin", Some(b"abcd"), target).with_range(u64::MAX - 1, 4)
    });
    render_single_error(manifest, "invalid_target", |target| {
        ReadTensor::new(6, "tensor.bin", Some(b"abcd"), target).with_range(0, 5)
    });
    render_single_error(manifest, "file_open", |target| {
        ReadTensor::new(7, "tensor.bin", None, target)
    });
    render_single_error(manifest, "file_seek", |target| {
        ReadTensor::new(8, "tensor.bin", Some(b"abcd"), target).with_range(5, 4)
    });
    render_single_error(manifest, "file_read", |target| {
        ReadTensor::new(9, "tensor.bin", Some(b"abcd"), target)
            .with_source_error(SourceError::Other)
    });
    render_single_error(manifest, "short_read", |target| {
        ReadTensor::new(10, "tensor.bin", Some(b"ab"), target)
    });
}

fn render_single_error(
    manifest: &mut String,
    name: &str,
    request: impl for<'a> FnOnce(&'a mut [u8]) -> ReadTensor<'a>,
) {
    let mut reader = Reader::new();
    let mut target = [0_u8; 4];
    let callback = Cell::new(None::<ReadTensorError>);
    let result = reader.process_event(request(&mut target).on_error(Callback::store(&callback)));
    let error = result.expect_err("pinned single-error case must fail");
    let callback = callback
        .get()
        .expect("pinned single-error callback must run synchronously");
    writeln!(
        manifest,
        "case=single_{name} outcome=error error={} callback_error={} target={}",
        error_name(error),
        error_name(callback.error()),
        hex(&target),
    )
    .expect("writing to String is infallible");
}

fn render_batch_success(manifest: &mut String) {
    let mut reader = Reader::new();
    let mut first_bytes = [0_u8; 3];
    let mut second_bytes = [0_u8; 4];
    let callback = Cell::new(None::<ReadTensorBatchDone>);
    let result = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let tensors = [
            TensorRead::new(1, "first.bin", Some(b"abcdef"), &first_target).with_range(2, 3),
            TensorRead::new(2, "second.bin", Some(b"wxyz"), &second_target),
        ];
        reader.process_event(ReadTensorBatch::new(&tensors).on_done(Callback::store(&callback)))
    };
    let done = result.expect("pinned batch-success case must succeed");
    let callback = callback
        .get()
        .expect("pinned batch-success callback must run synchronously");
    writeln!(
        manifest,
        "case=batch_success outcome=done done_count={} bytes_copied={} first_target={} second_target={} callback_done_count={} callback_bytes_copied={}",
        done.done_count(),
        done.bytes_copied(),
        hex(&first_bytes),
        hex(&second_bytes),
        callback.done_count(),
        callback.bytes_copied(),
    )
    .expect("writing to String is infallible");
}

fn render_batch_errors(manifest: &mut String) {
    render_batch_error(manifest, BatchErrorCase::InvalidRequest);
    render_batch_error(manifest, BatchErrorCase::UnsupportedResource);
    render_batch_error(manifest, BatchErrorCase::FileOpen);
    render_batch_error(manifest, BatchErrorCase::FileSeek);
    render_batch_error(manifest, BatchErrorCase::FileRead);
    render_batch_error(manifest, BatchErrorCase::ShortRead);
    render_batch_error(manifest, BatchErrorCase::MixedShortThenFileRead);
    render_batch_error(manifest, BatchErrorCase::MixedSeekThenOpen);
    render_batch_error(manifest, BatchErrorCase::MixedResourceThenInvalid);
    render_batch_error(manifest, BatchErrorCase::SamePhaseTwoFileReads);
}

#[derive(Clone, Copy)]
enum BatchErrorCase {
    InvalidRequest,
    UnsupportedResource,
    FileOpen,
    FileSeek,
    FileRead,
    ShortRead,
    MixedShortThenFileRead,
    MixedSeekThenOpen,
    MixedResourceThenInvalid,
    SamePhaseTwoFileReads,
}

fn render_batch_error(manifest: &mut String, case: BatchErrorCase) {
    let mut reader = Reader::new();
    let mut first_bytes = [0_u8; 4];
    let mut second_bytes = [0_u8; 4];
    let callback = Cell::new(None::<ReadTensorBatchError>);
    let error = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let first = TensorRead::new(1, "first.bin", Some(b"abcd"), &first_target);
        let second = TensorRead::new(2, "second.bin", Some(b"abcd"), &second_target);
        let tensors = match case {
            BatchErrorCase::InvalidRequest => [first, second.with_range(0, 5)],
            BatchErrorCase::UnsupportedResource => [
                first,
                TensorRead::new(2, "bad\0path", Some(b"abcd"), &second_target),
            ],
            BatchErrorCase::FileOpen => [
                first,
                TensorRead::new(2, "second.bin", None, &second_target),
            ],
            BatchErrorCase::FileSeek => [first, second.with_range(5, 4)],
            BatchErrorCase::FileRead => [first, second.with_source_error(SourceError::Other)],
            BatchErrorCase::ShortRead => [
                first,
                TensorRead::new(2, "second.bin", Some(b"ab"), &second_target),
            ],
            BatchErrorCase::MixedShortThenFileRead => [
                TensorRead::new(1, "first.bin", Some(b"ab"), &first_target),
                second.with_source_error(SourceError::Other),
            ],
            BatchErrorCase::MixedSeekThenOpen => [
                first.with_range(5, 4),
                TensorRead::new(2, "second.bin", None, &second_target),
            ],
            BatchErrorCase::MixedResourceThenInvalid => [
                TensorRead::new(1, "bad\0path", Some(b"abcd"), &first_target),
                second.with_range(0, 5),
            ],
            BatchErrorCase::SamePhaseTwoFileReads => [
                first.with_source_error(SourceError::Other),
                second.with_source_error(SourceError::Other),
            ],
        };
        reader
            .process_event(ReadTensorBatch::new(&tensors).on_error(Callback::store(&callback)))
            .expect_err("pinned batch-error case must fail")
    };
    let callback = callback
        .get()
        .expect("pinned batch-error callback must run synchronously");
    let name = match case {
        BatchErrorCase::InvalidRequest => "invalid_request",
        BatchErrorCase::UnsupportedResource => "unsupported_resource",
        BatchErrorCase::FileOpen => "file_open",
        BatchErrorCase::FileSeek => "file_seek",
        BatchErrorCase::FileRead => "file_read",
        BatchErrorCase::ShortRead => "short_read",
        BatchErrorCase::MixedShortThenFileRead => "mixed_short_then_file_read",
        BatchErrorCase::MixedSeekThenOpen => "mixed_seek_then_open",
        BatchErrorCase::MixedResourceThenInvalid => "mixed_resource_then_invalid",
        BatchErrorCase::SamePhaseTwoFileReads => "same_phase_two_file_reads",
    };
    writeln!(
        manifest,
        "case=batch_{name} outcome=error error={} failed_index={} callback_error={} callback_failed_index={} first_target={} second_target={}",
        error_name(error.error()),
        error.failed_index(),
        error_name(callback.error()),
        callback.failed_index(),
        hex(&first_bytes),
        hex(&second_bytes),
    )
    .expect("writing to String is infallible");
}

fn render_batch_count_boundaries(manifest: &mut String) {
    let mut reader = Reader::new();
    let mut target_bytes = [0_u8; 1];
    let callback = Cell::new(None::<ReadTensorBatchDone>);
    let done = {
        let target = Target::new(&mut target_bytes);
        let tensors: Vec<_> = (0..MAX_READ_BATCH_TENSORS)
            .map(|_| TensorRead::new(21, "batch-cap.bin", Some(b"x"), &target))
            .collect();
        reader
            .process_event(ReadTensorBatch::new(&tensors).on_done(Callback::store(&callback)))
            .expect("the pinned exact batch cap must succeed")
    };
    let callback = callback
        .get()
        .expect("the exact-cap callback must run synchronously");
    writeln!(
        manifest,
        "case=batch_count_exact_cap outcome=done done_count={} bytes_copied={} target={} callback_done_count={} callback_bytes_copied={}",
        done.done_count(),
        done.bytes_copied(),
        hex(&target_bytes),
        callback.done_count(),
        callback.bytes_copied(),
    )
    .expect("writing to String is infallible");

    let mut target_bytes = [0_u8; 1];
    let callback = Cell::new(None::<ReadTensorBatchError>);
    let error = {
        let target = Target::new(&mut target_bytes);
        let tensors: Vec<_> = (0..=MAX_READ_BATCH_TENSORS)
            .map(|_| TensorRead::new(22, "batch-over-cap.bin", Some(b"x"), &target))
            .collect();
        reader
            .process_event(ReadTensorBatch::new(&tensors).on_error(Callback::store(&callback)))
            .expect_err("the pinned over-cap batch must fail")
    };
    let callback = callback
        .get()
        .expect("the over-cap callback must run synchronously");
    writeln!(
        manifest,
        "case=batch_count_over_cap outcome=error error={} failed_index={} callback_error={} callback_failed_index={}",
        error_name(error.error()),
        error.failed_index(),
        error_name(callback.error()),
        callback.failed_index(),
    )
    .expect("writing to String is infallible");
}

const fn error_name(error: Error) -> &'static str {
    match error {
        Error::InvalidRequest => "invalid_request",
        Error::UnsupportedPlatform => "unsupported_platform",
        Error::UnsupportedResource => "unsupported_resource",
        Error::FileOpenFailed => "file_open_failed",
        Error::FileSeekFailed => "file_seek_failed",
        Error::FileReadFailed => "file_read_failed",
        Error::ShortRead => "short_read",
        Error::InternalError => "internal_error",
        _ => "unknown",
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}
