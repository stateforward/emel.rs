//! Public loader actor composition coverage.

use core::cell::Cell;
use std::rc::Rc;

use emel_io::loader::Loader;
use emel_io::loader::event::{
    Callback, Error, LoadTensor, LoadTensorBatch, MAX_BATCH_TENSORS, ReadActor, StrategyError,
    StrategyKind, StrategyPolicy, Target, TensorLoadSpan,
};
use emel_io::{read, staged_read};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[test]
fn public_loader_routes_single_and_batch_events_through_owned_children() {
    let source = *b"abcdefghij";
    let done = Cell::new(None);
    let mut single_bytes = [0_u8; 3];
    let single = {
        let target = Target::new(&mut single_bytes);
        let span = TensorLoadSpan::new(1, "tensor.bin", Some(&source), &target).with_range(2, 3);
        Loader::with_reader(read::Reader::new())
            .process_event(
                LoadTensor::new(span, StrategyPolicy::new(StrategyKind::ReadCopy))
                    .on_done(Callback::store(&done)),
            )
            .expect("single read route")
    };
    assert_eq!(single.bytes_loaded(), 3);
    assert_eq!(done.get(), Some(single));
    assert_eq!(single_bytes, *b"cde");

    let mut first_bytes = [0_u8; 2];
    let mut second_bytes = [0_u8; 3];
    let batch = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let spans = [
            TensorLoadSpan::new(2, "tensor.bin", Some(&source), &first_target).with_range(1, 2),
            TensorLoadSpan::new(3, "tensor.bin", Some(&source), &second_target).with_range(5, 3),
        ];
        Loader::with_stager(staged_read::Stager::new())
            .process_event(LoadTensorBatch::new(
                &spans,
                StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(2),
            ))
            .expect("batch staged route")
    };
    assert_eq!(batch.done_count(), 2);
    assert_eq!(batch.bytes_loaded(), 5);
    assert_eq!(first_bytes, *b"bc");
    assert_eq!(second_bytes, *b"fgh");
}

#[test]
fn public_loader_preserves_absent_actor_and_child_error_classes() {
    let source = *b"abcd";
    let mut target_bytes = [0_u8; 4];
    let target = Target::new(&mut target_bytes);
    let span = TensorLoadSpan::new(4, "tensor.bin", Some(&source), &target);
    let absent = Loader::new()
        .process_event(LoadTensor::new(
            span,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect_err("absent read actor");
    assert_eq!(absent.error(), Error::UnsupportedStrategy);
    assert_eq!(absent.strategy_error(), StrategyError::None);

    let failed = TensorLoadSpan::new(5, "tensor.bin", Some(&source), &target)
        .with_source_error(read::event::SourceError::FileReadFailed);
    let child = Loader::with_reader(read::Reader::new())
        .process_event(LoadTensor::new(
            failed,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect_err("child read failure");
    assert_eq!(child.error(), Error::Unavailable);
    assert_eq!(
        child.strategy_error(),
        StrategyError::Read(read::event::Error::FileReadFailed)
    );
}

#[test]
fn public_loader_enforces_the_exact_batch_count_boundary_and_recovers() {
    struct BoundaryReader {
        batch_calls: Rc<Cell<u32>>,
    }

    impl ReadActor for BoundaryReader {
        fn process_read(
            &mut self,
            _: read::event::ReadSpan<'_>,
        ) -> Result<read::event::ReadTensorDone, read::event::Error> {
            Err(read::event::Error::InternalError)
        }

        fn process_read_batch(
            &mut self,
            event: read::event::ReadTensorBatch<'_>,
        ) -> Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError> {
            self.batch_calls.set(self.batch_calls.get() + 1);
            let count = u32::try_from(event.tensors().len()).expect("public batch count");
            Ok(read::event::ReadTensorBatchDone::new(
                count,
                u64::from(count),
            ))
        }
    }

    let calls = Rc::new(Cell::new(0));
    let mut loader = Loader::with_reader(BoundaryReader {
        batch_calls: Rc::clone(&calls),
    });
    let source = *b"x";
    let mut target_bytes = [0_u8; 1];
    let target = Target::new(&mut target_bytes);
    let span = TensorLoadSpan::new(6, "batch-cap.bin", Some(&source), &target);

    let at_limit = vec![span; MAX_BATCH_TENSORS];
    let done = loader
        .process_event(LoadTensorBatch::new(
            &at_limit,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect("maximum public batch is accepted");
    assert_eq!(done.done_count(), 65_536);
    assert_eq!(calls.get(), 1);

    let over_limit = vec![span; MAX_BATCH_TENSORS + 1];
    let error = loader
        .process_event(LoadTensorBatch::new(
            &over_limit,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect_err("over-limit public batch is rejected");
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(error.strategy_error(), StrategyError::None);
    assert_eq!(error.failed_index(), 0);
    assert_eq!(calls.get(), 1, "rejection must precede child dispatch");
    assert_eq!(target.try_matches(&[0]), Ok(true));

    let recovery = [span];
    let done = loader
        .process_event(LoadTensorBatch::new(
            &recovery,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect("loader recovers after over-limit rejection");
    assert_eq!(done.done_count(), 1);
    assert_eq!(calls.get(), 2);
}
