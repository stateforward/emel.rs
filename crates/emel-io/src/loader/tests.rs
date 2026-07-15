use core::cell::Cell;
use std::rc::Rc;

use super::Loader;
use super::event::{
    Callback, Error, LoadTensor, LoadTensorBatch, NoActor, ReadActor, StagedReadActor,
    StrategyError, StrategyKind, StrategyPolicy, Target, TargetWriteError, TensorLoadSpan,
};
use crate::{read, staged_read};

#[test]
fn invalid_and_unsupported_single_requests_fail_closed_and_recover() {
    let source = *b"abcdef";
    let mut target_bytes = [0_u8; 3];
    let error_slot = Cell::new(None);
    let mut loader = Loader::new();
    {
        let target = Target::new(&mut target_bytes);
        let invalid = TensorLoadSpan::new(1, "tensor.bin", Some(&source), &target).with_range(0, 0);
        let error = loader
            .process_event(
                LoadTensor::new(invalid, StrategyPolicy::new(StrategyKind::ReadCopy))
                    .on_error(Callback::store(&error_slot)),
            )
            .expect_err("zero-length request");
        assert_eq!(error.error(), Error::InvalidRequest);
        assert_eq!(error.strategy_error(), StrategyError::None);
        assert_eq!(error_slot.get(), Some(error));

        for strategy in [
            StrategyKind::None,
            StrategyKind::MappedFile,
            StrategyKind::ReadCopy,
            StrategyKind::ExternalBuffer,
            StrategyKind::StagedRead,
            StrategyKind::Unknown(0xff),
        ] {
            let span = TensorLoadSpan::new(2, "tensor.bin", Some(&source), &target);
            let error = loader
                .process_event(LoadTensor::new(span, StrategyPolicy::new(strategy)))
                .expect_err("strategy is absent or unsupported");
            assert_eq!(error.error(), Error::UnsupportedStrategy);
            assert_eq!(error.strategy_error(), StrategyError::None);
        }
    }
    assert_eq!(target_bytes, [0; 3]);
}

#[test]
fn real_read_and_staged_children_copy_single_spans_and_publish_done() {
    let source = *b"abcdefgh";
    let done_slot = Cell::new(None);

    let mut read_bytes = [0_u8; 3];
    let read_done = {
        let target = Target::new(&mut read_bytes);
        let span = TensorLoadSpan::new(7, "tensor.bin", Some(&source), &target)
            .with_file_index(1)
            .with_range(2, 3);
        Loader::with_reader(read::Reader::new())
            .process_event(
                LoadTensor::new(span, StrategyPolicy::new(StrategyKind::ReadCopy))
                    .on_done(Callback::store(&done_slot)),
            )
            .expect("read-copy route")
    };
    assert_eq!(read_done.strategy(), StrategyKind::ReadCopy);
    assert_eq!(read_done.bytes_loaded(), 3);
    assert_eq!(done_slot.get(), Some(read_done));
    assert_eq!(read_bytes, *b"cde");

    done_slot.set(None);
    let mut staged_bytes = [0_u8; 4];
    let staged_done = {
        let target = Target::new(&mut staged_bytes);
        let span = TensorLoadSpan::new(8, "tensor.bin", Some(&source), &target).with_range(1, 4);
        Loader::with_stager(staged_read::Stager::new())
            .process_event(
                LoadTensor::new(
                    span,
                    StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(3),
                )
                .on_done(Callback::store(&done_slot)),
            )
            .expect("staged route")
    };
    assert_eq!(staged_done.strategy(), StrategyKind::StagedRead);
    assert_eq!(staged_done.bytes_loaded(), 4);
    assert_eq!(done_slot.get(), Some(staged_done));
    assert_eq!(staged_bytes, *b"bcde");
}

#[test]
fn child_failures_preserve_typed_strategy_errors() {
    let source = *b"abcd";
    let mut target_bytes = [0_u8; 4];
    let target = Target::new(&mut target_bytes);
    let read_span = TensorLoadSpan::new(3, "tensor.bin", Some(&source), &target)
        .with_source_error(read::event::SourceError::FileReadFailed);
    let read_error = Loader::with_reader(read::Reader::new())
        .process_event(LoadTensor::new(
            read_span,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect_err("read child error");
    assert_eq!(read_error.error(), Error::Unavailable);
    assert_eq!(
        read_error.strategy_error(),
        StrategyError::Read(read::event::Error::FileReadFailed)
    );

    let staged_span =
        TensorLoadSpan::new(4, "tensor.bin", Some(&source[..2]), &target).with_range(0, 4);
    let staged_error = Loader::with_stager(staged_read::Stager::new())
        .process_event(LoadTensor::new(
            staged_span,
            StrategyPolicy::new(StrategyKind::StagedRead),
        ))
        .expect_err("loader validates staged source span");
    assert_eq!(staged_error.error(), Error::InvalidRequest);
    assert_eq!(staged_error.strategy_error(), StrategyError::None);
}

#[test]
fn single_and_batch_routes_use_statically_injected_replacement_actors() {
    #[derive(Clone)]
    struct ReplacementReader {
        calls: Rc<Cell<u32>>,
    }

    impl ReadActor for ReplacementReader {
        fn process_read(
            &mut self,
            event: read::event::ReadSpan<'_>,
        ) -> Result<read::event::ReadTensorDone, read::event::Error> {
            self.calls.set(self.calls.get() + 1);
            let span = event.tensor();
            let start = usize::try_from(span.file_offset()).expect("test offset");
            let end = start + usize::try_from(span.byte_size()).expect("test length");
            span.target()
                .try_copy_from(&span.source().expect("test source")[start..end])
                .expect("test target");
            Ok(read::event::ReadTensorDone::new(
                span.tensor_id(),
                span.byte_size(),
            ))
        }

        fn process_read_batch(
            &mut self,
            _: read::event::ReadTensorBatch<'_>,
        ) -> Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError> {
            Err(read::event::ReadTensorBatchError::new(
                read::event::Error::InternalError,
                0,
            ))
        }
    }

    let calls = Rc::new(Cell::new(0));
    let mut loader = Loader::with_reader(ReplacementReader {
        calls: Rc::clone(&calls),
    });
    let mut target_bytes = [0_u8; 2];
    {
        let target = Target::new(&mut target_bytes);
        let span = TensorLoadSpan::new(99, "replace.bin", Some(b"abcd"), &target).with_range(1, 2);
        let done = loader
            .process_event(LoadTensor::new(
                span,
                StrategyPolicy::new(StrategyKind::ReadCopy),
            ))
            .expect("replacement actor");
        assert_eq!(done.bytes_loaded(), 2);
    }
    assert_eq!(target_bytes, *b"bc");
    assert_eq!(calls.get(), 1);
}

#[test]
fn read_and_staged_batches_dispatch_once_and_preserve_child_failures() {
    let source = *b"abcdefghij";
    let mut first_bytes = [0_u8; 3];
    let mut second_bytes = [0_u8; 4];
    let done = Cell::new(None);
    let read_done = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let tensors = [
            TensorLoadSpan::new(1, "tensor.bin", Some(&source), &first_target).with_range(1, 3),
            TensorLoadSpan::new(2, "tensor.bin", Some(&source), &second_target).with_range(5, 4),
        ];
        Loader::with_reader(read::Reader::new())
            .process_event(
                LoadTensorBatch::new(&tensors, StrategyPolicy::new(StrategyKind::ReadCopy))
                    .on_done(Callback::store(&done)),
            )
            .expect("read batch")
    };
    assert_eq!(read_done.done_count(), 2);
    assert_eq!(read_done.bytes_loaded(), 7);
    assert_eq!(first_bytes, *b"bcd");
    assert_eq!(second_bytes, *b"fghi");

    first_bytes.fill(0);
    second_bytes.fill(0);
    let staged_done = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let tensors = [
            TensorLoadSpan::new(3, "tensor.bin", Some(&source), &first_target).with_range(1, 3),
            TensorLoadSpan::new(4, "tensor.bin", Some(&source), &second_target).with_range(5, 4),
        ];
        Loader::with_stager(staged_read::Stager::new())
            .process_event(LoadTensorBatch::new(
                &tensors,
                StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(2),
            ))
            .expect("staged batch")
    };
    assert_eq!(staged_done.done_count(), 2);
    assert_eq!(staged_done.bytes_loaded(), 7);
    assert_eq!(first_bytes, *b"bcd");
    assert_eq!(second_bytes, *b"fghi");

    let mut failed_bytes = [0_u8; 2];
    let failure = {
        let target = Target::new(&mut failed_bytes);
        let tensors = [TensorLoadSpan::new(5, "tensor.bin", None, &target)];
        Loader::with_reader(read::Reader::new())
            .process_event(LoadTensorBatch::new(
                &tensors,
                StrategyPolicy::new(StrategyKind::ReadCopy),
            ))
            .expect_err("read batch child failure")
    };
    assert_eq!(failure.error(), Error::Unavailable);
    assert_eq!(
        failure.strategy_error(),
        StrategyError::Read(read::event::Error::FileOpenFailed)
    );
    assert_eq!(failure.failed_index(), 0);
}

#[test]
fn explicit_staged_and_batch_failure_routes_publish_typed_outcomes() {
    #[derive(Clone, Copy)]
    struct FailingStager;
    impl StagedReadActor for FailingStager {
        fn process_staged_read(
            &mut self,
            _: staged_read::event::StageTensor<'_>,
        ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
            Err(staged_read::event::Error::InternalError)
        }

        fn process_staged_read_batch(
            &mut self,
            _: staged_read::event::StageTensorBatch<'_>,
        ) -> Result<
            staged_read::event::StageWindowBatchDone,
            staged_read::event::StageWindowBatchError,
        > {
            Err(staged_read::event::StageWindowBatchError::new(
                staged_read::event::Error::InternalError,
                1,
            ))
        }
    }

    let source = *b"abcdefgh";
    let mut bytes = [0_u8; 4];
    let target = Target::new(&mut bytes);
    let span = TensorLoadSpan::new(1, "tensor.bin", Some(&source), &target).with_range(2, 4);
    let single_slot = Cell::new(None);
    let error = Loader::with_stager(FailingStager)
        .process_event(
            LoadTensor::new(
                span,
                StrategyPolicy::new(StrategyKind::StagedRead).with_staged_chunk_bytes(4),
            )
            .on_error(Callback::store(&single_slot)),
        )
        .expect_err("replacement staged failure");
    assert_eq!(
        error.strategy_error(),
        StrategyError::StagedRead(staged_read::event::Error::InternalError)
    );
    assert_eq!(single_slot.get(), Some(error));

    let tensors = [span, span];
    let batch_slot = Cell::new(None);
    let batch_error = Loader::with_stager(FailingStager)
        .process_event(
            LoadTensorBatch::new(&tensors, StrategyPolicy::new(StrategyKind::StagedRead))
                .on_error(Callback::store(&batch_slot)),
        )
        .expect_err("replacement staged batch failure");
    assert_eq!(batch_error.failed_index(), 1);
    assert_eq!(
        batch_error.strategy_error(),
        StrategyError::StagedRead(staged_read::event::Error::InternalError)
    );
    assert_eq!(batch_slot.get(), Some(batch_error));
}

#[test]
fn every_batch_strategy_and_staged_source_validation_fails_closed() {
    let source = *b"abcd";
    let mut bytes = [0_u8; 4];
    let target = Target::new(&mut bytes);
    let span = TensorLoadSpan::new(1, "tensor.bin", Some(&source), &target);
    let tensors = [span];
    for strategy in [
        StrategyKind::None,
        StrategyKind::MappedFile,
        StrategyKind::ExternalBuffer,
        StrategyKind::Unknown(7),
        StrategyKind::StagedRead,
    ] {
        let error = Loader::new()
            .process_event(LoadTensorBatch::new(
                &tensors,
                StrategyPolicy::new(strategy),
            ))
            .expect_err("unsupported batch route");
        assert_eq!(error.error(), Error::UnsupportedStrategy);
    }

    let missing = [TensorLoadSpan::new(2, "tensor.bin", None, &target)];
    let error = Loader::with_stager(staged_read::Stager::new())
        .process_event(LoadTensorBatch::new(
            &missing,
            StrategyPolicy::new(StrategyKind::StagedRead),
        ))
        .expect_err("staged batch source validation");
    assert_eq!(error.error(), Error::InvalidRequest);
}

#[test]
fn public_value_diagnostics_and_target_capability_are_complete() {
    assert!(format!("{:?}", Loader::default()).starts_with("Loader"));
    let callback_slot = Cell::new(None::<super::event::LoadTensorDone>);
    assert!(format!("{:?}", Callback::store(&callback_slot)).starts_with("Callback"));
    for (error, text) in [
        (Error::InvalidRequest, "invalid loader request"),
        (Error::UnsupportedStrategy, "unsupported loader strategy"),
        (Error::Unavailable, "loader strategy actor unavailable"),
        (Error::InternalError, "internal loader actor error"),
    ] {
        assert_eq!(error.to_string(), text);
    }
    assert_eq!(
        TargetWriteError::Borrowed.to_string(),
        "target is already borrowed"
    );
    assert_eq!(
        TargetWriteError::InsufficientCapacity.to_string(),
        "target capacity is insufficient"
    );

    let mut bytes = [0_u8; 2];
    let target = Target::new(&mut bytes);
    assert!(!target.is_empty());
    assert_eq!(
        target.try_copy_from(b"abc"),
        Err(TargetWriteError::InsufficientCapacity)
    );
    {
        let _borrow = target.bytes.borrow_mut();
        assert_eq!(target.try_matches(b"00"), Err(TargetWriteError::Borrowed));
        assert_eq!(target.try_copy_from(b"00"), Err(TargetWriteError::Borrowed));
    }
    let span = TensorLoadSpan::new(9, "file.bin", Some(b"12"), &target)
        .with_file_index(3)
        .with_source_error(read::event::SourceError::Other);
    assert_eq!(span.file_index(), 3);
    assert_eq!(span.file_offset(), 0);
    assert_eq!(span.file_path(), "file.bin");
    assert_eq!(span.source_error(), Some(read::event::SourceError::Other));
    assert_eq!(span.target_bytes(), 2);

    let mut absent = NoActor;
    assert_eq!(
        absent.process_read(read::event::ReadSpan::new(span)),
        Err(read::event::Error::InternalError)
    );
    assert_eq!(
        absent.process_staged_read(staged_read::event::StageTensor::new(span, 1)),
        Err(staged_read::event::Error::InternalError)
    );
    let spans = [span];
    assert!(
        absent
            .process_read_batch(read::event::ReadTensorBatch::new(&spans))
            .is_err()
    );
    assert!(
        absent
            .process_staged_read_batch(staged_read::event::StageTensorBatch::new(&spans, 1))
            .is_err()
    );
}
