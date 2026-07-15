#![no_main]

use emel_model::model_tensor_proof::event::{
    ApplyBoundEffectResults, ApplyEffectError, ApplyOwnedEffectResults, BindStorage,
    CaptureTensorState, EffectBuffer, EffectError, EffectRequest, Lifecycle, OwnedEffectResult,
    PlanLoad, StorageBatch, StorageEntry, StrategyKind, TensorMetadata,
};
use emel_model::model_tensor_proof::Store;
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
            StrategyKind::ReadCopy
            | StrategyKind::ExternalBuffer
            | StrategyKind::StagedRead => {
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
        .process_event(PlanLoad::new(StrategyKind::None, EffectBuffer::new(
            vec![EffectRequest::Empty].into_boxed_slice(),
        )))
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
});

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or_default()
}
