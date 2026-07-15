//! Public staged-read lifecycle coverage.

use core::cell::Cell;

use emel_io::staged_read::Stager;
use emel_io::staged_read::event::{
    Callback, Error, StageSpan, StageWindow, StageWindowBatch, StageWindowError, Target,
};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[test]
fn aligned_and_remainder_windows_copy_and_publish_typed_outcomes() {
    let source = *b"abcdefghij";
    let done_slot = Cell::new(None);
    let error_slot = Cell::new(None);
    let mut actor = Stager::new();

    let mut aligned = [0_u8; 8];
    let result = {
        let target = Target::new(&mut aligned);
        actor
            .process_event(
                StageWindow::new(100, 8, 4, Some(&source[..8]), &target)
                    .on_done(Callback::store(&done_slot))
                    .on_error(Callback::store(&error_slot)),
            )
            .expect("aligned stage")
    };
    assert_eq!(aligned, *b"abcdefgh");
    assert_eq!(result.bytes_committed(), 8);
    assert_eq!(done_slot.get(), Some(result));
    assert_eq!(error_slot.get(), None);

    done_slot.set(None);
    let mut remainder = [0_u8; 10];
    let result = {
        let target = Target::new(&mut remainder);
        actor
            .process_event(
                StageWindow::new(200, 10, 4, Some(&source), &target)
                    .on_done(Callback::store(&done_slot))
                    .on_error(Callback::store(&error_slot)),
            )
            .expect("remainder stage")
    };
    assert_eq!(remainder, source);
    assert_eq!(result.bytes_committed(), 10);
    assert_eq!(done_slot.get(), Some(result));
}

#[test]
fn validation_precedence_and_source_size_classes_match_the_pinned_machine() {
    let source = [7_u8; 8];
    let mut actor = Stager::new();

    let mut target_bytes = [0_u8; 2];
    {
        let target = Target::new(&mut target_bytes);
        assert_eq!(
            actor.process_event(StageWindow::new(u64::MAX, 0, 0, None, &target)),
            Err(Error::InvalidCallbacks)
        );
    }

    for (source, expected) in [
        (None, Error::NullSourceSpan),
        (Some(&source[..3]), Error::InsufficientSourceSpan),
        (Some(&source[..5]), Error::SourceSpanSizeMismatch),
    ] {
        let done = Cell::new(None);
        let error = Cell::new(None);
        let mut target_bytes = [0_u8; 4];
        {
            let target = Target::new(&mut target_bytes);
            assert_eq!(
                actor.process_event(
                    StageWindow::new(0, 4, 2, source, &target)
                        .on_done(Callback::store(&done))
                        .on_error(Callback::store(&error)),
                ),
                Err(expected)
            );
        }
        assert_eq!(error.get().map(StageWindowError::error), Some(expected));
    }

    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut short_target = [0_u8; 3];
    {
        let target = Target::new(&mut short_target);
        assert_eq!(
            actor.process_event(
                StageWindow::new(0, 4, 2, Some(&source[..4]), &target)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            ),
            Err(Error::InvalidTargetWindow)
        );
    }
}

#[test]
fn batch_copies_source_ranges_and_reports_first_invalid_index() {
    let source = *b"0123456789abcdef";
    let mut first_bytes = [0_u8; 4];
    let mut second_bytes = [0_u8; 5];
    let mut invalid_bytes = [0_u8; 1];
    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut actor = Stager::new();
    let (result, batch_error) = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let spans = [
            StageSpan::staged(2, 4, Some(&source), &first_target),
            StageSpan::staged(8, 5, Some(&source), &second_target),
        ];
        let result = actor
            .process_event(
                StageWindowBatch::new(&spans, 3)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect("valid batch");

        let invalid_target = Target::new(&mut invalid_bytes);
        let invalid = [
            StageSpan::staged(0, 4, Some(&source), &first_target),
            StageSpan::staged(15, 2, Some(&source), &invalid_target),
        ];
        let batch_error = actor
            .process_event(
                StageWindowBatch::new(&invalid, 2)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect_err("invalid batch");
        (result, batch_error)
    };
    assert_eq!(result.done_count(), 2);
    assert_eq!(result.bytes_committed(), 9);
    assert_eq!(done.get(), Some(result));
    assert_eq!(first_bytes, *b"2345");
    assert_eq!(second_bytes, *b"89abc");
    assert_eq!(batch_error.error(), Error::InvalidStageContract);
    assert_eq!(batch_error.failed_index(), 1);
    assert_eq!(error.get(), Some(batch_error));
}

#[test]
fn callback_contracts_and_public_diagnostics_are_explicit() {
    let errors = [
        (Error::InvalidCallbacks, "invalid staged-read callbacks"),
        (Error::InvalidStageContract, "invalid staged-read contract"),
        (
            Error::InvalidTargetWindow,
            "invalid staged-read target window",
        ),
        (
            Error::UnsupportedPlatform,
            "unsupported staged-read platform",
        ),
        (Error::NullSourceSpan, "missing staged-read source span"),
        (
            Error::SourceSpanSizeMismatch,
            "staged-read source span is larger than requested",
        ),
        (
            Error::InsufficientSourceSpan,
            "staged-read source span is smaller than requested",
        ),
        (Error::InternalError, "internal staged-read actor error"),
    ];
    for (error, message) in errors {
        assert_eq!(error.to_string(), message);
        let source: &dyn std::error::Error = &error;
        assert_eq!(source.to_string(), message);
    }

    let done = Cell::new(None);
    let error = Cell::new(None);
    assert_eq!(format!("{:?}", Callback::store(&done)), "Callback { .. }");
    assert_eq!(format!("{:?}", Stager::default()), "Stager { .. }");

    let source = [1_u8; 2];
    let mut target_bytes = [0_u8; 2];
    let mut actor = Stager::new();
    {
        let target = Target::new(&mut target_bytes);
        assert_eq!(
            actor.process_event(
                StageWindow::new(0, 2, 1, Some(&source), &target).on_error(Callback::store(&error)),
            ),
            Err(Error::InvalidCallbacks)
        );
    }
    assert_eq!(
        error.get().map(StageWindowError::error),
        Some(Error::InvalidCallbacks)
    );

    error.set(None);
    let mut target_bytes = [0_u8; 2];
    {
        let target = Target::new(&mut target_bytes);
        assert_eq!(
            actor.process_event(
                StageWindow::new(0, 2, 1, Some(&source), &target).on_done(Callback::store(&done)),
            ),
            Err(Error::InvalidCallbacks)
        );
    }
    assert_eq!(error.get(), None);
}

#[test]
fn batch_callback_and_empty_contract_paths_recover() {
    let mut empty_bytes = [];
    let empty_target = Target::new(&mut empty_bytes);
    assert!(empty_target.is_empty());
    assert_eq!(empty_target.len(), 0);

    let spans = [];
    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut actor = Stager::new();
    let missing = actor
        .process_event(StageWindowBatch::new(&spans, 1).on_error(Callback::store(&error)))
        .expect_err("missing done callback");
    assert_eq!(missing.error(), Error::InvalidCallbacks);
    assert_eq!(missing.failed_index(), 0);
    assert_eq!(error.get(), Some(missing));

    error.set(None);
    let invalid = actor
        .process_event(
            StageWindowBatch::new(&spans, 0)
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        )
        .expect_err("empty zero-chunk batch");
    assert_eq!(invalid.error(), Error::InvalidStageContract);
    assert_eq!(invalid.failed_index(), 0);
    assert_eq!(error.get(), Some(invalid));
}
