#![no_main]

use core::cell::Cell;

use emel_io::staged_read::Stager;
use emel_io::staged_read::event::{
    Callback, StageSpan, StageWindow, StageWindowBatch, StageWindowBatchDone,
    StageWindowBatchError, StageWindowDone, StageWindowError, Target,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let offset = read_u64(data, 0);
    let logical = read_u64(data, 8);
    let chunk = read_u64(data, 16);
    let target_len = data.get(24).map_or(0, |value| usize::from(*value % 65));
    let source = data.get(25..).unwrap_or_default();
    let done = Cell::new(None::<StageWindowDone>);
    let error = Cell::new(None::<StageWindowError>);
    let mut target = [0_u8; 64];
    let mut actor = Stager::new();
    {
        let target = Target::new(&mut target[..target_len]);
        let _ = actor.process_event(
            StageWindow::new(offset, logical, chunk, Some(source), &target)
                .on_done(Callback::store(&done))
                .on_error(Callback::store(&error)),
        );
    }

    let mut first_bytes = [0_u8; 32];
    let mut second_bytes = [0_u8; 32];
    let first_target = Target::new(&mut first_bytes);
    let second_target = Target::new(&mut second_bytes);
    let first_offset = read_u64(data, 25);
    let first_size = read_u64(data, 33);
    let second_offset = read_u64(data, 41);
    let second_size = read_u64(data, 49);
    let spans = [
        StageSpan::staged(first_offset, first_size, Some(source), &first_target),
        StageSpan::staged(second_offset, second_size, Some(source), &second_target),
    ];
    let batch_done = Cell::new(None::<StageWindowBatchDone>);
    let batch_error = Cell::new(None::<StageWindowBatchError>);
    let _ = actor.process_event(
        StageWindowBatch::new(&spans, chunk)
            .on_done(Callback::store(&batch_done))
            .on_error(Callback::store(&batch_error)),
    );

    let recovery_source = [42_u8];
    let mut recovery_target = [0_u8];
    let recovery_done = Cell::new(None::<StageWindowDone>);
    let recovery_error = Cell::new(None::<StageWindowError>);
    let recovery_target_capability = Target::new(&mut recovery_target);
    let recovery = actor
        .process_event(
            StageWindow::new(0, 1, 1, Some(&recovery_source), &recovery_target_capability)
                .on_done(Callback::store(&recovery_done))
                .on_error(Callback::store(&recovery_error)),
        )
        .expect("the actor must recover after every classified fuzz request");
    assert_eq!(recovery.bytes_committed(), 1);
    assert_eq!(
        recovery_target_capability.try_matches(&recovery_source),
        Ok(true)
    );
    assert_eq!(recovery_done.get(), Some(recovery));
});

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0_u8; 8];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..8)) {
        bytes.copy_from_slice(source);
    }
    u64::from_le_bytes(bytes)
}
