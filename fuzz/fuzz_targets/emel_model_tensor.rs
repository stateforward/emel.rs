#![no_main]

use emel_io::{read, staged_read};
use emel_model::tensor::Store;
use emel_model::tensor::dependency::Actors;
use emel_model::tensor::event::{
    ApplyBoundEffectResults, ApplyEffectError, ApplyOwnedEffectResults, BindStorage,
    CaptureTensorState, EffectBuffer, EffectError, EffectRequest, Lifecycle, OwnedEffectResult,
    PlanLoad, ReadLoad, ReleaseMapped, StagedLoad, StorageBatch, StorageEntry, StrategyKind,
    TensorMetadata, WithTensor,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let capacity = usize::from(byte(data, 0) % 4) + 1;
    let mut store = Store::new(capacity).expect("bounded fuzz capacity");
    let entries = (0..capacity)
        .map(|index| {
            let size = u64::from(byte(data, index + 1) % 16) + 1;
            let length = usize::try_from(size).expect("fuzz size fits usize");
            let bytes = vec![byte(data, index + 8); length].into_boxed_slice();
            StorageEntry::new(
                TensorMetadata::new(
                    u64::try_from(index).expect("fuzz index fits u64") * 4_096,
                    size,
                    u16::try_from(index).expect("fuzz index fits u16"),
                    i32::from(byte(data, index + 12)),
                ),
                Some(bytes),
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    store
        .process_event(BindStorage::new(StorageBatch::new(entries)))
        .expect("valid fuzz storage");

    let strategy = match byte(data, 4) % 6 {
        0 => StrategyKind::None,
        1 => StrategyKind::MappedFile,
        2 => StrategyKind::ReadCopy,
        3 => StrategyKind::ExternalBuffer,
        4 => StrategyKind::StagedRead,
        value => StrategyKind::Unknown(value),
    };
    let effect_count = usize::from(byte(data, 5)) % (capacity + 2);
    let mut effect_slots = vec![EffectRequest::Empty; effect_count];
    if byte(data, 6) & 1 != 0 && !effect_slots.is_empty() {
        effect_slots[0] = EffectRequest::None {
            tensor_id: 0,
            file_index: 0,
            offset: 0,
            size: 1,
        };
    }
    let planned = store.process_event(PlanLoad::new(
        strategy,
        EffectBuffer::new(effect_slots.into_boxed_slice()),
    ));
    if planned.is_ok() {
        match strategy {
            StrategyKind::None => {
                let ids = (0..capacity)
                    .map(|index| {
                        let id = i32::try_from(index).expect("fuzz index fits i32");
                        if byte(data, 7) & 1 == 0 || index != capacity - 1 {
                            id
                        } else {
                            id + 1
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let _ = store.process_event(ApplyBoundEffectResults::new(ids));
            }
            StrategyKind::ReadCopy | StrategyKind::ExternalBuffer | StrategyKind::StagedRead => {
                let results = (0..capacity)
                    .map(|index| {
                        let size = usize::from(byte(data, index + 1) % 16) + 1;
                        let length = if byte(data, 7) & 1 == 0 || index != capacity - 1 {
                            size
                        } else {
                            size.saturating_sub(1)
                        };
                        OwnedEffectResult::new(
                            i32::try_from(index).expect("fuzz index fits i32"),
                            vec![byte(data, index + 16); length].into_boxed_slice(),
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let _ = store.process_event(ApplyOwnedEffectResults::new(results));
            }
            StrategyKind::MappedFile => {
                let _ = store.process_event(ApplyEffectError::new(
                    i32::from(byte(data, 7)),
                    EffectError::Unknown(u16::from(byte(data, 8))),
                ));
            }
            StrategyKind::Unknown(_) => unreachable!("unknown strategy cannot plan"),
        }
    }

    let recovery = StorageBatch::new(
        vec![StorageEntry::new(
            TensorMetadata::new(0, 1, 0, 0),
            Some(vec![9].into_boxed_slice()),
        )]
        .into_boxed_slice(),
    );
    store
        .process_event(BindStorage::new(recovery))
        .expect("actor recovers ready after every classified plan/result path");
    store
        .process_event(PlanLoad::new(
            StrategyKind::None,
            EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice()),
        ))
        .expect("recovery plan");
    store
        .process_event(ApplyBoundEffectResults::new(vec![0].into_boxed_slice()))
        .expect("recovery apply");
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .expect("recovery capture")
            .lifecycle(),
        Lifecycle::Resident
    );

    fuzz_io_routes(data);
});

fn fuzz_io_routes(data: &[u8]) {
    const TARGET_BYTES: usize = 16;
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).expect("bounded I/O fuzz capacity");
    store
        .process_event(BindStorage::new(io_storage(TARGET_BYTES)))
        .expect("I/O fuzz target storage");

    let offset = u64::from(byte(data, 20));
    let len = u64::from(byte(data, 21) % 24);
    let source = (byte(data, 22) & 1 == 0).then_some(data);
    if byte(data, 23) & 1 == 0 {
        let path = if byte(data, 24) & 1 == 0 {
            "fuzz.bin"
        } else {
            ""
        };
        let request =
            ReadLoad::new(0, path, source, offset, len).with_file_index(u16::from(byte(data, 25)));
        let _ = store.process_event(request);
    } else {
        let chunk = u64::from(byte(data, 24));
        let _ = store.process_event(StagedLoad::new(0, source, offset, len, chunk));
    }

    let mut checksum = |bytes: &[u8]| {
        bytes.iter().fold(0_u64, |value, byte| {
            value.wrapping_mul(16_777_619) ^ u64::from(*byte)
        })
    };
    let _ = store.process_event(WithTensor::new(0, &mut checksum));
    let _ = store.process_event(ReleaseMapped::new(0, u32::from(byte(data, 26))));

    store
        .process_event(BindStorage::new(io_storage(4)))
        .expect("I/O actor recovers after every classified route");
    store
        .process_event(ReadLoad::new(0, "recovery.bin", Some(&[1, 2, 3, 4]), 0, 4))
        .expect("valid recovery read");
    let mut capture = |bytes: &[u8]| <[u8; 4]>::try_from(bytes).ok();
    assert_eq!(
        store.process_event(WithTensor::new(0, &mut capture)),
        Ok(Some([1, 2, 3, 4]))
    );

    let mut absent = Store::new(1).expect("missing-capability fuzz capacity");
    absent
        .process_event(BindStorage::new(io_storage(4)))
        .expect("missing-capability target storage");
    let _ = absent.process_event(ReadLoad::new(0, "fuzz.bin", Some(data), offset, len));
    let _ = absent.process_event(StagedLoad::new(0, Some(data), offset, len, 4));
    let mut capture = |bytes: &[u8]| bytes.len();
    let _ = absent.process_event(WithTensor::new(0, &mut capture));
}

fn io_storage(bytes: usize) -> StorageBatch {
    StorageBatch::new(
        vec![StorageEntry::new(
            TensorMetadata::new(0, bytes as u64, 0, 0),
            Some(vec![0; bytes].into_boxed_slice()),
        )]
        .into_boxed_slice(),
    )
}

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or_default()
}
