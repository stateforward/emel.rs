//! Allocation-free reads from caller-provided source bytes.

mod actor;
pub mod event;
mod sm;

pub use actor::Reader;

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::Reader;
    use super::event::{
        Callback, Error, ReadTensor, ReadTensorBatch, ReadTensorBatchDone, ReadTensorBatchError,
        ReadTensorDone, ReadTensorError, SourceError, Target, TensorRead,
    };

    #[test]
    fn single_read_copies_selected_range_and_publishes_done() {
        let mut reader = Reader::new();
        let source = *b"abcdef";
        let mut target = [0_u8; 4];
        let done = Cell::new(None::<ReadTensorDone>);
        let error = Cell::new(None::<ReadTensorError>);

        let result = reader.process_event(
            ReadTensor::new(7, "tensor.bin", Some(&source), &mut target)
                .with_range(1, 4)
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        );

        assert_eq!(result, Ok(ReadTensorDone::new(7, 4)));
        assert_eq!(done.get(), result.ok());
        assert_eq!(error.get(), None);
        assert_eq!(&target, b"bcde");
        assert!(reader.is_ready());
    }

    #[test]
    fn batch_read_copies_into_genuinely_distinct_caller_targets() {
        let mut reader = Reader::new();
        let first_source = *b"abcdef";
        let second_source = *b"wxyz";
        let mut first_target_bytes = [0_u8; 3];
        let mut second_target_bytes = [0_u8; 4];
        let done = Cell::new(None::<ReadTensorBatchDone>);
        let error = Cell::new(None::<ReadTensorBatchError>);
        let result = {
            let first_target = Target::new(&mut first_target_bytes);
            let second_target = Target::new(&mut second_target_bytes);
            let tensors = [
                TensorRead::new(1, "first.bin", Some(&first_source), &first_target)
                    .with_range(2, 3),
                TensorRead::new(2, "second.bin", Some(&second_source), &second_target),
            ];
            reader.process_event(
                ReadTensorBatch::new(&tensors)
                    .on_done(Callback::store(&done))
                    .on_error(Callback::store(&error)),
            )
        };

        assert_eq!(result, Ok(ReadTensorBatchDone::new(2, 7)));
        assert_eq!(done.get(), result.ok());
        assert_eq!(error.get(), None);
        assert!(reader.is_ready());
        assert_eq!(&first_target_bytes, b"cde");
        assert_eq!(&second_target_bytes, b"wxyz");
    }

    #[test]
    fn single_read_classifies_validation_errors() {
        let source = *b"abcdefgh";

        assert_single_error(
            |target| ReadTensor::new(1, "valid.bin", Some(b"abcdefgh"), target).with_range(0, 0),
            Error::InvalidRequest,
        );
        assert_single_error(
            |target| ReadTensor::new(2, "bad\0path", Some(b"abcdefgh"), target),
            Error::InvalidRequest,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(3, "valid.bin", Some(b"abcdefgh"), target).with_file_index(u16::MAX)
            },
            Error::UnsupportedResource,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(4, "valid.bin", Some(b"abcdefgh"), target)
                    .with_range(0, super::sm::MAX_READ_BYTES + 1)
            },
            Error::UnsupportedResource,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(5, "valid.bin", Some(b"abcdefgh"), target)
                    .with_range(u64::MAX - 3, 8)
            },
            Error::UnsupportedResource,
        );
        let mut reader = Reader::new();
        let mut short_target = [0_u8; 2];
        let result = reader.process_event(
            ReadTensor::new(6, "valid.bin", Some(&source), &mut short_target).with_range(0, 4),
        );
        assert_eq!(result, Err(Error::InvalidRequest));
        assert!(reader.is_ready());
    }

    #[test]
    fn single_read_classifies_platform_and_external_source_errors() {
        let source = *b"abcdefgh";
        let mut reader = Reader::new();
        let mut target = [0_u8; 4];
        let result = reader.read_tensor_on_unsupported_platform(ReadTensor::new(
            1,
            "valid.bin",
            Some(&source),
            &mut target,
        ));
        assert_eq!(result, Err(Error::UnsupportedPlatform));
        assert!(reader.is_ready());

        assert_single_error(
            |target| ReadTensor::new(2, "valid.bin", None, target),
            Error::FileOpenFailed,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(3, "valid.bin", Some(b"abcdefgh"), target)
                    .with_source_error(SourceError::FileOpenFailed)
            },
            Error::FileOpenFailed,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(4, "valid.bin", Some(b"abcdefgh"), target)
                    .with_source_error(SourceError::FileSeekFailed)
            },
            Error::FileSeekFailed,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(5, "valid.bin", Some(b"abcdefgh"), target)
                    .with_source_error(SourceError::FileReadFailed)
            },
            Error::FileReadFailed,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(6, "valid.bin", Some(b"abcdefgh"), target)
                    .with_source_error(SourceError::Other)
            },
            Error::FileReadFailed,
        );
        assert_single_error(
            |target| {
                ReadTensor::new(7, "valid.bin", Some(b"abcdefgh"), target)
                    .with_source_error(SourceError::ShortRead)
            },
            Error::ShortRead,
        );
        assert_single_error(
            |target| ReadTensor::new(8, "valid.bin", Some(b"ab"), target),
            Error::ShortRead,
        );
        assert_single_error(
            |target| ReadTensor::new(9, "valid.bin", Some(b"abcdefgh"), target).with_range(9, 4),
            Error::FileSeekFailed,
        );
    }

    #[test]
    fn callbacks_are_optional_and_error_callback_is_synchronous() {
        let mut reader = Reader::new();
        let source = *b"abcd";
        let mut target = [0_u8; 4];
        assert_eq!(
            reader.process_event(ReadTensor::new(1, "valid.bin", Some(&source), &mut target,)),
            Ok(ReadTensorDone::new(1, 4)),
        );

        let error_slot = Cell::new(None::<ReadTensorError>);
        let mut target = [0_u8; 4];
        let result = reader.process_event(
            ReadTensor::new(2, "missing.bin", None, &mut target)
                .on_error(Callback::store(&error_slot)),
        );
        assert_eq!(result, Err(Error::FileOpenFailed));
        assert_eq!(
            error_slot.get(),
            Some(ReadTensorError::new(2, Error::FileOpenFailed)),
        );
        assert!(reader.is_ready());
    }

    #[test]
    fn batch_reports_first_validation_and_resource_failure() {
        let source = *b"abcdefgh";
        let mut reader = Reader::new();

        let empty: [TensorRead<'_>; 0] = [];
        let result = reader.process_event(ReadTensorBatch::new(&empty));
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::InvalidRequest, 0))
        );

        let mut first_bytes = [0_u8; 4];
        let mut short_bytes = [0_u8; 2];
        let first_target = Target::new(&mut first_bytes);
        let short_target = Target::new(&mut short_bytes);
        let invalid = [
            TensorRead::new(1, "first.bin", Some(&source), &first_target),
            TensorRead::new(2, "second.bin", Some(&source), &short_target).with_range(0, 4),
        ];
        let result = reader.process_event(ReadTensorBatch::new(&invalid));
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::InvalidRequest, 1))
        );

        let mut second_bytes = [0_u8; 4];
        let second_target = Target::new(&mut second_bytes);
        let unsupported = [
            TensorRead::new(1, "first.bin", Some(&source), &first_target),
            TensorRead::new(2, "bad\0path", Some(&source), &second_target),
        ];
        let result = reader.process_event(ReadTensorBatch::new(&unsupported));
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::UnsupportedResource, 1)),
        );
        assert!(reader.is_ready());
    }

    #[test]
    fn batch_reports_first_external_source_failure_and_callbacks() {
        let source = *b"abcdefgh";
        for (source_error, expected) in [
            (SourceError::FileOpenFailed, Error::FileOpenFailed),
            (SourceError::FileSeekFailed, Error::FileSeekFailed),
            (SourceError::FileReadFailed, Error::FileReadFailed),
            (SourceError::Other, Error::FileReadFailed),
            (SourceError::ShortRead, Error::ShortRead),
        ] {
            let mut first_bytes = [0_u8; 4];
            let mut second_bytes = [0_u8; 4];
            let error_slot = Cell::new(None::<ReadTensorBatchError>);
            let mut reader = Reader::new();
            let result = {
                let first_target = Target::new(&mut first_bytes);
                let second_target = Target::new(&mut second_bytes);
                let tensors = [
                    TensorRead::new(1, "first.bin", Some(&source), &first_target),
                    TensorRead::new(2, "second.bin", Some(&source), &second_target)
                        .with_source_error(source_error),
                ];
                reader.process_event(
                    ReadTensorBatch::new(&tensors).on_error(Callback::store(&error_slot)),
                )
            };
            let expected = ReadTensorBatchError::new(expected, 1);
            assert_eq!(result, Err(expected));
            assert_eq!(error_slot.get(), Some(expected));
            assert!(reader.is_ready());
            assert_eq!(&first_bytes, &[0; 4]);
            assert_eq!(&second_bytes, &[0; 4]);
        }

        let mut first_bytes = [0_u8; 4];
        let mut second_bytes = [0_u8; 4];
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let tensors = [
            TensorRead::new(1, "first.bin", Some(&source), &first_target),
            TensorRead::new(2, "second.bin", Some(&source[..2]), &second_target),
        ];
        let mut reader = Reader::new();
        assert_eq!(
            reader.process_event(ReadTensorBatch::new(&tensors)),
            Err(ReadTensorBatchError::new(Error::ShortRead, 1)),
        );
        assert!(reader.is_ready());
    }

    #[test]
    fn batch_preserves_pinned_phase_precedence_across_mixed_failures() {
        let source = *b"abcdefgh";
        let short_source = *b"ab";

        let mut first_bytes = [0_u8; 4];
        let mut second_bytes = [0_u8; 4];
        let result = {
            let first_target = Target::new(&mut first_bytes);
            let second_target = Target::new(&mut second_bytes);
            let tensors = [
                TensorRead::new(1, "first.bin", Some(&short_source), &first_target),
                TensorRead::new(2, "second.bin", Some(&source), &second_target)
                    .with_source_error(SourceError::Other),
            ];
            Reader::new().process_event(ReadTensorBatch::new(&tensors))
        };
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::FileReadFailed, 1)),
        );
        assert_eq!(first_bytes, [0; 4]);
        assert_eq!(second_bytes, [0; 4]);

        let mut first_bytes = [0_u8; 4];
        let mut second_bytes = [0_u8; 4];
        let result = {
            let first_target = Target::new(&mut first_bytes);
            let second_target = Target::new(&mut second_bytes);
            let tensors = [
                TensorRead::new(1, "first.bin", Some(&source), &first_target).with_range(9, 4),
                TensorRead::new(2, "second.bin", None, &second_target),
            ];
            Reader::new().process_event(ReadTensorBatch::new(&tensors))
        };
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::FileOpenFailed, 1)),
        );
        assert_eq!(first_bytes, [0; 4]);
        assert_eq!(second_bytes, [0; 4]);

        let mut first_bytes = [0_u8; 4];
        let mut second_bytes = [0_u8; 4];
        let result = {
            let first_target = Target::new(&mut first_bytes);
            let second_target = Target::new(&mut second_bytes);
            let tensors = [
                TensorRead::new(1, "bad\0path", Some(&source), &first_target),
                TensorRead::new(2, "second.bin", Some(&source), &second_target).with_range(0, 5),
            ];
            Reader::new().process_event(ReadTensorBatch::new(&tensors))
        };
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::InvalidRequest, 1)),
        );
        assert_eq!(first_bytes, [0; 4]);
        assert_eq!(second_bytes, [0; 4]);
    }

    #[test]
    fn batch_reports_first_failure_within_the_selected_phase() {
        let source = *b"abcdefgh";
        let mut first_bytes = [0_u8; 4];
        let mut second_bytes = [0_u8; 4];
        let result = {
            let first_target = Target::new(&mut first_bytes);
            let second_target = Target::new(&mut second_bytes);
            let tensors = [
                TensorRead::new(1, "first.bin", Some(&source), &first_target)
                    .with_source_error(SourceError::Other),
                TensorRead::new(2, "second.bin", Some(&source), &second_target)
                    .with_source_error(SourceError::Other),
            ];
            Reader::new().process_event(ReadTensorBatch::new(&tensors))
        };

        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::FileReadFailed, 0)),
        );
        assert_eq!(first_bytes, [0; 4]);
        assert_eq!(second_bytes, [0; 4]);
    }

    #[test]
    fn batch_enforces_the_exact_public_count_boundary_before_classification() {
        let source = *b"x";
        let mut target_bytes = [0_u8; 1];
        let mut reader = Reader::new();
        let result = {
            let target = Target::new(&mut target_bytes);
            let tensors: Vec<_> = (0..super::sm::MAX_READ_BATCH_TENSORS)
                .map(|_| TensorRead::new(21, "batch-cap.bin", Some(&source), &target))
                .collect();
            reader.process_event(ReadTensorBatch::new(&tensors))
        };
        assert_eq!(result, Ok(ReadTensorBatchDone::new(65_536, 65_536,)),);
        assert_eq!(target_bytes, source);
        assert!(reader.is_ready());

        let mut target_bytes = [0_u8; 1];
        let result = {
            let target = Target::new(&mut target_bytes);
            let tensors: Vec<_> = (0..=super::sm::MAX_READ_BATCH_TENSORS)
                .map(|_| TensorRead::new(22, "batch-over-cap.bin", Some(&source), &target))
                .collect();
            reader.process_event(ReadTensorBatch::new(&tensors))
        };
        assert_eq!(
            result,
            Err(ReadTensorBatchError::new(Error::InvalidRequest, 0)),
        );
        assert_eq!(target_bytes, [0]);
        assert!(reader.is_ready());
    }

    #[test]
    fn reader_recovers_after_errors() {
        let mut reader = Reader::new();
        let mut target = [0_u8; 1];
        assert_eq!(
            reader.process_event(ReadTensor::new(1, "missing.bin", None, &mut target)),
            Err(Error::FileOpenFailed),
        );
        assert!(reader.is_ready());

        let source = *b"x";
        assert_eq!(
            reader.process_event(ReadTensor::new(2, "valid.bin", Some(&source), &mut target,)),
            Ok(ReadTensorDone::new(2, 1)),
        );
        assert_eq!(target, source);
        assert!(reader.is_ready());
    }

    fn assert_single_error(
        request: impl for<'a> FnOnce(&'a mut [u8]) -> ReadTensor<'a>,
        expected: Error,
    ) {
        let mut reader = Reader::new();
        let mut target = [0_u8; 4];
        let error = Cell::new(None::<ReadTensorError>);
        let result = reader.process_event(request(&mut target).on_error(Callback::store(&error)));
        assert_eq!(result, Err(expected));
        assert_eq!(error.get().map(ReadTensorError::error), Some(expected));
        assert_eq!(target, [0; 4]);
        assert!(reader.is_ready());
    }
}
