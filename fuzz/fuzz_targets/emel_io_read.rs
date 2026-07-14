#![no_main]

use emel_io::read::Reader;
use emel_io::read::event::ReadTensor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let file_offset = read_u64(data, 0);
    let byte_size = read_u64(data, 8);
    let file_index = read_u16(data, 16);
    let target_len = data
        .get(18)
        .map_or(64, |value| usize::from(*value % 64) + 1);
    let source = data.get(19..).unwrap_or_default();
    let mut target = [0_u8; 64];
    let mut reader = Reader::new();

    let _ = reader.process_event(
        ReadTensor::new(1, "fuzz.bin", Some(source), &mut target[..target_len])
            .with_file_index(file_index)
            .with_range(file_offset, byte_size),
    );

    let mut recovery_target = [0_u8; 1];
    let recovery = reader
        .process_event(ReadTensor::new(
            2,
            "recovery.bin",
            Some(b"x"),
            &mut recovery_target,
        ))
        .expect("the actor must recover after every classified fuzz request");
    assert_eq!(recovery.tensor_id(), 2);
    assert_eq!(recovery.bytes_copied(), 1);
    assert_eq!(recovery_target, *b"x");
});

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0_u8; 8];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..8)) {
        bytes.copy_from_slice(source);
    }
    u64::from_le_bytes(bytes)
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    let mut bytes = [0_u8; 2];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..2)) {
        bytes.copy_from_slice(source);
    }
    u16::from_le_bytes(bytes)
}
