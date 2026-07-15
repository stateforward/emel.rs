//! Source-backed golden parity cases for the pinned emel.cpp staged-read actor.

use core::cell::Cell;
use std::fmt::Write as _;

use emel_io::staged_read::Stager;
use emel_io::staged_read::event::{
    Callback, Error, StageSpan, StageWindow, StageWindowBatch, StageWindowBatchDone,
    StageWindowBatchError, StageWindowDone, StageWindowError, Target,
};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[allow(
    dead_code,
    reason = "this file is also imported by the parity integration test"
)]
fn main() {
    print!("{}", render_manifest());
}

/// Renders canonical outcomes obtained exclusively through the public actor.
pub fn render_manifest() -> String {
    let mut manifest = String::from(
        "io-staged-read-parity-snapshot/v1\n\
         source_repository=stateforward/emel.cpp\n\
         source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n\
         source_tree=ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa\n\
         source_files=src/emel/io/staged_read/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp\n\
         source_tests=tests/io/staged_read/lifecycle_tests.cpp\n\
         fixture_config=caller_owned_memory,platform_supported,synchronous_callbacks,single_aligned_and_remainder,batch_offsets,validation_precedence\n\
         contract_delta=rust_borrows_prove_non_null_targets_and_source_lengths\n",
    );
    render_single_success(&mut manifest, "aligned", b"abcdefgh", 4);
    render_single_success(&mut manifest, "remainder", b"abcdefghij", 4);
    render_single_errors(&mut manifest);
    render_batch_success(&mut manifest);
    render_batch_errors(&mut manifest);
    manifest
}

fn render_single_success(manifest: &mut String, name: &str, source: &[u8], chunk: u64) {
    let mut actor = Stager::new();
    let mut target = vec![0_u8; source.len()];
    let done = Cell::new(None::<StageWindowDone>);
    let error = Cell::new(None::<StageWindowError>);
    let outcome = actor
        .process_event(
            StageWindow::new(17, source.len() as u64, chunk, Some(source), &mut target)
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        )
        .expect("pinned success case");
    writeln!(
        manifest,
        "case=single_{name} outcome=done bytes_committed={} callback_bytes_committed={} target={}",
        outcome.bytes_committed(),
        done.get().expect("synchronous callback").bytes_committed(),
        hex(&target),
    )
    .expect("writing to String is infallible");
}

fn render_single_errors(manifest: &mut String) {
    render_single_error(
        manifest,
        "missing_callbacks",
        0,
        4,
        2,
        Some(b"abcd"),
        4,
        false,
    );
    render_single_error(manifest, "invalid_contract", 0, 0, 0, Some(b""), 0, true);
    render_single_error(
        manifest,
        "offset_overflow",
        u64::MAX,
        2,
        1,
        Some(b"ab"),
        2,
        true,
    );
    render_single_error(manifest, "invalid_target", 0, 4, 2, Some(b"abcd"), 3, true);
    render_single_error(manifest, "null_source", 0, 4, 2, None, 4, true);
    render_single_error(
        manifest,
        "insufficient_source",
        0,
        4,
        2,
        Some(b"abc"),
        4,
        true,
    );
    render_single_error(
        manifest,
        "source_mismatch",
        0,
        4,
        2,
        Some(b"abcde"),
        4,
        true,
    );
}

#[allow(
    clippy::too_many_arguments,
    reason = "parity cases mirror the wire request fields"
)]
fn render_single_error(
    manifest: &mut String,
    name: &str,
    offset: u64,
    logical: u64,
    chunk: u64,
    source: Option<&[u8]>,
    target_len: usize,
    callbacks: bool,
) {
    let mut actor = Stager::new();
    let mut target = vec![0_u8; target_len];
    let done = Cell::new(None::<StageWindowDone>);
    let error = Cell::new(None::<StageWindowError>);
    let mut event = StageWindow::new(offset, logical, chunk, source, &mut target);
    if callbacks {
        event = event
            .on_done(Callback::store(&done))
            .on_error(Callback::store(&error));
    }
    let outcome = actor.process_event(event).expect_err("pinned error case");
    writeln!(
        manifest,
        "case=single_{name} outcome=error error={} callback={} target={}",
        error_name(outcome),
        error.get().is_some(),
        hex(&target),
    )
    .expect("writing to String is infallible");
}

fn render_batch_success(manifest: &mut String) {
    let source = b"0123456789abcdef";
    let mut first_bytes = [0_u8; 4];
    let mut second_bytes = [0_u8; 5];
    let done = Cell::new(None::<StageWindowBatchDone>);
    let error = Cell::new(None::<StageWindowBatchError>);
    let mut actor = Stager::new();
    let outcome = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let spans = [
            StageSpan::new(2, 4, Some(source), &first_target),
            StageSpan::new(8, 5, Some(source), &second_target),
        ];
        actor
            .process_event(
                StageWindowBatch::new(&spans, 3)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect("pinned batch success")
    };
    let callback = done.get().expect("synchronous callback");
    writeln!(
        manifest,
        "case=batch_success outcome=done done_count={} bytes_committed={} callback_done_count={} callback_bytes_committed={} first_target={} second_target={}",
        outcome.done_count(),
        outcome.bytes_committed(),
        callback.done_count(),
        callback.bytes_committed(),
        hex(&first_bytes),
        hex(&second_bytes),
    )
    .expect("writing to String is infallible");
}

fn render_batch_errors(manifest: &mut String) {
    let source = b"0123456789abcdef";
    let mut first_bytes = [0_u8; 4];
    let mut second_bytes = [0_u8; 1];
    let done = Cell::new(None::<StageWindowBatchDone>);
    let error = Cell::new(None::<StageWindowBatchError>);
    let mut actor = Stager::new();
    let outcome = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let spans = [
            StageSpan::new(0, 4, Some(source), &first_target),
            StageSpan::new(15, 2, Some(source), &second_target),
        ];
        actor
            .process_event(
                StageWindowBatch::new(&spans, 2)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect_err("pinned batch error")
    };
    writeln!(
        manifest,
        "case=batch_invalid outcome=error error={} failed_index={} callback={} first_target={} second_target={}",
        error_name(outcome.error()),
        outcome.failed_index(),
        error.get().is_some(),
        hex(&first_bytes),
        hex(&second_bytes),
    )
    .expect("writing to String is infallible");

    let empty = [];
    let outcome = actor
        .process_event(StageWindowBatch::new(&empty, 1))
        .expect_err("missing callbacks precede empty batch validation");
    writeln!(
        manifest,
        "case=batch_missing_callbacks outcome=error error={} failed_index={} callback=false",
        error_name(outcome.error()),
        outcome.failed_index(),
    )
    .expect("writing to String is infallible");
}

const fn error_name(error: Error) -> &'static str {
    match error {
        Error::InvalidCallbacks => "invalid_callbacks",
        Error::InvalidStageContract => "invalid_stage_contract",
        Error::InvalidTargetWindow => "invalid_target_window",
        Error::UnsupportedPlatform => "unsupported_platform",
        Error::NullSourceSpan => "null_source_span",
        Error::SourceSpanSizeMismatch => "source_span_size_mismatch",
        Error::InsufficientSourceSpan => "insufficient_source_span",
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
