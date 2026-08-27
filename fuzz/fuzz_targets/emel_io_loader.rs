#![no_main]

use emel_io::loader::Loader;
use emel_io::loader::event::{
    LoadTensor, LoadTensorBatch, StrategyKind, StrategyPolicy, Target, TensorLoadSpan,
};
use emel_io::{read::Reader, staged_read::Stager};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let offset = read_u64(data, 0);
    let size = read_u64(data, 8);
    let chunk = read_u64(data, 16);
    let target_len = data.get(24).map_or(1, |value| usize::from(*value % 64) + 1);
    let source = data.get(25..).unwrap_or_default();
    let strategy = match data.first().copied().unwrap_or_default() % 6 {
        0 => StrategyKind::None,
        1 => StrategyKind::MappedFile,
        2 => StrategyKind::ReadCopy,
        3 => StrategyKind::ExternalBuffer,
        4 => StrategyKind::StagedRead,
        value => StrategyKind::Unknown(value),
    };
    let mut bytes = [0_u8; 64];
    let target = Target::new(&mut bytes[..target_len]);
    let span = TensorLoadSpan::new(1, "fuzz.bin", Some(source), &target).with_range(offset, size);
    let mut loader = Loader::with_dependencies(Reader::new(), Stager::new());
    let _ = loader.process_event(LoadTensor::new(
        span,
        StrategyPolicy::new(strategy).with_staged_chunk_bytes(chunk),
    ));

    let mut first_bytes = [0_u8; 32];
    let mut second_bytes = [0_u8; 32];
    let first_target = Target::new(&mut first_bytes);
    let second_target = Target::new(&mut second_bytes);
    // Exact 65,536/65,537 cardinality proof stays in the ordinary public
    // integration test so this hostile-input target does not allocate several
    // megabytes on every execution. This lane fuzzes bounded span fields and
    // proves batch-plus-single recovery after every classified request.
    let spans = [
        TensorLoadSpan::new(3, "first.bin", Some(source), &first_target)
            .with_range(read_u64(data, 25), read_u64(data, 33)),
        TensorLoadSpan::new(4, "second.bin", Some(source), &second_target)
            .with_range(read_u64(data, 41), read_u64(data, 49)),
    ];
    let _ = loader.process_event(LoadTensorBatch::new(
        &spans,
        StrategyPolicy::new(strategy).with_staged_chunk_bytes(chunk),
    ));

    let mut batch_recovery_bytes = [0_u8; 1];
    let batch_recovery_target = Target::new(&mut batch_recovery_bytes);
    let batch_recovery_spans = [TensorLoadSpan::new(
        5,
        "batch-recovery.bin",
        Some(b"y"),
        &batch_recovery_target,
    )];
    let batch_recovery = loader
        .process_event(LoadTensorBatch::new(
            &batch_recovery_spans,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect("loader batch must recover after every classified request");
    assert_eq!(batch_recovery.done_count(), 1);
    assert_eq!(batch_recovery_target.try_matches(b"y"), Ok(true));

    let mut recovery_bytes = [0_u8; 1];
    let recovery_target = Target::new(&mut recovery_bytes);
    let recovery_span = TensorLoadSpan::new(2, "recovery.bin", Some(b"x"), &recovery_target);
    let recovery = loader
        .process_event(LoadTensor::new(
            recovery_span,
            StrategyPolicy::new(StrategyKind::ReadCopy),
        ))
        .expect("loader must recover after every classified request");
    assert_eq!(recovery.bytes_loaded(), 1);
    assert_eq!(recovery_target.try_matches(b"x"), Ok(true));
});

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0_u8; 8];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..8)) {
        bytes.copy_from_slice(source);
    }
    u64::from_le_bytes(bytes)
}
