//! Public read actor boundary coverage.

use emel_io::read::Reader;
use emel_io::read::event::{ReadTensor, ReadTensorBatch, Target, TensorRead};
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[test]
fn public_reader_dispatches_typed_single_and_batch_events() {
    let mut reader = Reader::new();
    let mut single_target = [0_u8; 3];
    let single = {
        let target = Target::new(&mut single_target);
        reader
            .process_event(
                ReadTensor::new(4, "single.bin", Some(b"abcdef"), &target).with_range(1, 3),
            )
            .expect("single read should succeed")
    };
    assert_eq!(single.tensor_id(), 4);
    assert_eq!(single.bytes_copied(), 3);
    assert_eq!(&single_target, b"bcd");

    let mut first_bytes = [0_u8; 2];
    let mut second_bytes = [0_u8; 2];
    let batch = {
        let first_target = Target::new(&mut first_bytes);
        let second_target = Target::new(&mut second_bytes);
        let tensors = [
            TensorRead::new(5, "first.bin", Some(b"abcd"), &first_target),
            TensorRead::new(6, "second.bin", Some(b"wxyz"), &second_target).with_range(1, 2),
        ];
        reader
            .process_event(ReadTensorBatch::new(&tensors))
            .expect("batch read should succeed")
    };
    assert_eq!(batch.done_count(), 2);
    assert_eq!(batch.bytes_copied(), 4);
    assert_eq!(&first_bytes, b"ab");
    assert_eq!(&second_bytes, b"xy");
}
