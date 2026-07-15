use core::cell::Cell;

use super::Stager;
use super::event::{
    Callback, Error, StageSpan, StageWindow, StageWindowBatch, StageWindowError, Target,
};

#[test]
fn unsupported_platform_is_explicit_and_actor_recovers() {
    let source = [1_u8; 4];
    let mut first_bytes = [0_u8; 4];
    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut actor = Stager::unsupported();
    {
        let first = Target::new(&mut first_bytes);
        let event = StageWindow::new(0, 4, 2, Some(&source), &first)
            .on_done(Callback::store(&done))
            .on_error(Callback::store(&error));
        assert_eq!(actor.process_event(event), Err(Error::UnsupportedPlatform));
    }
    assert_eq!(
        error.get(),
        Some(StageWindowError::new(Error::UnsupportedPlatform))
    );

    let mut second_bytes = [0_u8; 4];
    let second = Target::new(&mut second_bytes);
    let event = StageWindow::new(0, 4, 2, Some(&source), &second)
        .on_done(Callback::store(&done))
        .on_error(Callback::store(&error));
    assert_eq!(actor.process_event(event), Err(Error::UnsupportedPlatform));
}

#[test]
fn unsupported_batch_platform_publishes_error_and_recovers() {
    let source = [1_u8; 4];
    let mut first_bytes = [0_u8; 4];
    let first_target = Target::new(&mut first_bytes);
    let spans = [StageSpan::staged(0, 4, Some(&source), &first_target)];
    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut actor = Stager::unsupported();

    for _ in 0..2 {
        let result = actor
            .process_event(
                StageWindowBatch::new(&spans, 2)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect_err("unsupported platform");
        assert_eq!(result.error(), Error::UnsupportedPlatform);
        assert_eq!(result.failed_index(), 0);
        assert_eq!(error.get(), Some(result));
    }
}

#[test]
fn oversized_batch_is_rejected_before_copy_and_actor_recovers() {
    let source = [7_u8];
    let mut target_bytes = [0_u8];
    let done = Cell::new(None);
    let error = Cell::new(None);
    let mut actor = Stager::new();
    {
        let target = Target::new(&mut target_bytes);
        let span = StageSpan::staged(0, 1, Some(&source), &target);
        let spans = vec![span; super::sm::MAX_STAGE_BATCH_TENSORS + 1];
        let result = actor
            .process_event(
                StageWindowBatch::new(&spans, 1)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
            .expect_err("oversized batch must be rejected");

        assert_eq!(result.error(), Error::InvalidStageContract);
        assert_eq!(result.failed_index(), 0);
        assert_eq!(error.get(), Some(result));
        assert_eq!(done.get(), None);
    }
    assert_eq!(target_bytes, [0]);

    let mut recovered_target = [0_u8];
    let recovered_done = Cell::new(None);
    let recovered_error = Cell::new(None);
    let recovered = {
        let target = Target::new(&mut recovered_target);
        actor
            .process_event(
                StageWindow::new(0, 1, 1, Some(&source), &target)
                    .on_done(Callback::store(&recovered_done))
                    .on_error(Callback::store(&recovered_error)),
            )
            .expect("actor recovers after oversized batch")
    };
    assert_eq!(recovered.bytes_committed(), 1);
    assert_eq!(recovered_target, source);
}
