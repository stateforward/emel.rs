use allocation_counter::measure;
use core::cell::RefCell;

use super::Store;
use super::event::{
    ApplyBoundEffectResults, ApplyEffectError, ApplyOwnedEffectResults, BindStorage, BindTensor,
    CaptureTensorState, EffectBuffer, EffectError, EffectRequest, Error, EvictTensor, Lifecycle,
    OwnedEffectResult, PlanLoad, StorageBatch, StorageEntry, StrategyKind, TensorMetadata,
};

fn metadata(index: u64, size: u64) -> TensorMetadata {
    TensorMetadata::new(
        4_096 * (index + 1),
        size,
        u16::try_from(index + 1).unwrap(),
        7,
    )
}

fn storage(entries: &[(u64, Option<&[u8]>)]) -> StorageBatch {
    StorageBatch::new(
        entries
            .iter()
            .enumerate()
            .map(|(index, (size, bytes))| {
                StorageEntry::new(
                    metadata(u64::try_from(index).unwrap(), *size),
                    bytes.map(|value| value.to_vec().into_boxed_slice()),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    )
}

fn effect_buffer(count: usize) -> EffectBuffer {
    EffectBuffer::new(vec![EffectRequest::Empty; count].into_boxed_slice())
}

macro_rules! dispatch_without_allocation {
    ($store:ident, $event:ident) => {{
        let output = RefCell::new(None);
        let allocation = measure(|| {
            output.replace(Some($store.process_event($event)));
        });
        assert_eq!(allocation.count_total, 0);
        output.into_inner().expect("dispatch produced an outcome")
    }};
}

#[test]
fn bulk_bind_owns_metadata_and_initial_bytes_and_clears_stale_slots() {
    let mut store = Store::new(3).unwrap();
    let done = store
        .process_event(BindStorage::new(storage(&[
            (2, Some(&[1, 2])),
            (3, Some(&[3, 4, 5])),
        ])))
        .unwrap();
    assert_eq!(done.active_extent(), 2);

    let first = store.process_event(CaptureTensorState::new(0)).unwrap();
    assert_eq!(first.lifecycle(), Lifecycle::Unbound);
    assert_eq!(first.buffer_bytes(), 0);
    assert_eq!(first.metadata(), metadata(0, 2));

    store
        .process_event(BindStorage::new(storage(&[(4, Some(&[8, 7, 6, 5]))])))
        .unwrap();
    let stale = store.process_event(CaptureTensorState::new(1)).unwrap();
    assert_eq!(stale.lifecycle(), Lifecycle::Unbound);
    assert_eq!(stale.metadata(), TensorMetadata::default());
}

#[test]
fn rejected_bulk_binding_returns_the_unchanged_allocation_and_preserves_binding() {
    let mut store = Store::new(1).unwrap();
    store
        .process_event(BindStorage::new(storage(&[(2, Some(&[1, 2]))])))
        .unwrap();

    let empty = StorageBatch::new(Vec::new().into_boxed_slice());
    let error = store.process_event(BindStorage::new(empty)).unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert!(error.into_storage().is_empty());

    let oversized = storage(&[(1, None), (2, Some(&[1, 2]))]);
    let expected = storage(&[(1, None), (2, Some(&[1, 2]))]);
    let error = store
        .process_event(BindStorage::new(oversized))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(error.into_storage(), expected);

    let prior = store.process_event(CaptureTensorState::new(0)).unwrap();
    assert_eq!(prior.metadata(), metadata(0, 2));
}

#[test]
fn none_planning_is_ordered_and_bound_results_make_initial_bytes_resident() {
    let mut store = Store::new(2).unwrap();
    store
        .process_event(BindStorage::new(storage(&[
            (2, Some(&[1, 2, 9])),
            (3, Some(&[3, 4, 5])),
        ])))
        .unwrap();

    let effects = effect_buffer(3);
    let allocation = effects.effects().as_ptr();
    let done = store
        .process_event(PlanLoad::new(StrategyKind::None, effects))
        .unwrap();
    assert_eq!(done.effect_count(), 2);
    let effects = done.into_effects();
    assert_eq!(effects.effects().as_ptr(), allocation);
    assert_eq!(
        effects.effects(),
        &[
            EffectRequest::None {
                tensor_id: 0,
                file_index: 1,
                offset: 4_096,
                size: 2,
            },
            EffectRequest::None {
                tensor_id: 1,
                file_index: 2,
                offset: 8_192,
                size: 3,
            },
            EffectRequest::Empty,
        ]
    );
    assert!(store.is_awaiting_bound_results());

    store
        .process_event(ApplyBoundEffectResults::new(vec![0, 1].into_boxed_slice()))
        .unwrap();
    assert!(store.is_ready());
    for (tensor_id, bytes) in [(0, 2), (1, 3)] {
        let state = store
            .process_event(CaptureTensorState::new(tensor_id))
            .unwrap();
        assert_eq!(state.lifecycle(), Lifecycle::Resident);
        assert_eq!(state.buffer_bytes(), bytes);
    }
}

#[test]
fn each_io_strategy_plans_its_typed_request_and_owned_results_move_into_slots() {
    for (strategy, expected) in [
        (
            StrategyKind::ReadCopy,
            EffectRequest::ReadCopy {
                tensor_id: 0,
                file_index: 1,
                offset: 4_096,
                size: 2,
            },
        ),
        (
            StrategyKind::ExternalBuffer,
            EffectRequest::ExternalBuffer {
                tensor_id: 0,
                file_index: 1,
                offset: 4_096,
                size: 2,
            },
        ),
        (
            StrategyKind::StagedRead,
            EffectRequest::StagedRead {
                tensor_id: 0,
                file_index: 1,
                offset: 4_096,
                size: 2,
            },
        ),
    ] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(2, None)])))
            .unwrap();
        let done = store
            .process_event(PlanLoad::new(strategy, effect_buffer(1)))
            .unwrap();
        assert_eq!(done.into_effects().effects(), &[expected]);
        assert!(store.is_awaiting_owned_results());

        store
            .process_event(ApplyOwnedEffectResults::new(
                vec![OwnedEffectResult::new(0, vec![9, 8, 7].into_boxed_slice())]
                    .into_boxed_slice(),
            ))
            .unwrap();
        let resident = store.process_event(CaptureTensorState::new(0)).unwrap();
        assert_eq!(resident.lifecycle(), Lifecycle::Resident);
        assert_eq!(resident.buffer_bytes(), 2);
    }
}

#[test]
fn mapped_plan_defers_success_and_backend_error_recovers_ready() {
    let mut store = Store::new(1).unwrap();
    store
        .process_event(BindStorage::new(storage(&[(4, None)])))
        .unwrap();
    let done = store
        .process_event(PlanLoad::new(StrategyKind::MappedFile, effect_buffer(1)))
        .unwrap();
    assert_eq!(
        done.into_effects().effects(),
        &[EffectRequest::MappedFile {
            tensor_id: 0,
            file_index: 1,
            offset: 4_096,
            size: 4,
        }]
    );
    assert!(store.is_awaiting_mapped_results());
    assert_eq!(
        store.process_event(ApplyEffectError::new(0, EffectError::Backend)),
        Err(Error::BackendError)
    );
    assert!(store.is_ready());
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Unbound
    );
}

#[test]
fn planning_rejects_missing_storage_capacity_dirty_slots_and_unknown_strategy() {
    let mut empty_store = Store::new(1).unwrap();
    let effects = effect_buffer(1);
    let allocation = effects.effects().as_ptr();
    let error = empty_store
        .process_event(PlanLoad::new(StrategyKind::None, effects))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(error.into_effects().effects().as_ptr(), allocation);

    let mut store = Store::new(2).unwrap();
    store
        .process_event(BindStorage::new(storage(&[(1, Some(&[1])), (1, None)])))
        .unwrap();
    let error = store
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(1)))
        .unwrap_err();
    assert_eq!(error.error(), Error::Capacity);
    assert_eq!(error.into_effects().len(), 1);

    let dirty = EffectBuffer::new(
        vec![
            EffectRequest::None {
                tensor_id: 0,
                file_index: 0,
                offset: 0,
                size: 0,
            },
            EffectRequest::Empty,
        ]
        .into_boxed_slice(),
    );
    let error = store
        .process_event(PlanLoad::new(StrategyKind::None, dirty))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);

    let error = store
        .process_event(PlanLoad::new(StrategyKind::Unknown(17), effect_buffer(2)))
        .unwrap_err();
    assert_eq!(error.error(), Error::UnsupportedStrategy);
    assert!(store.is_ready());
}

#[test]
fn complete_batch_validation_prevents_partial_mutation() {
    let mut bound_store = Store::new(2).unwrap();
    bound_store
        .process_event(BindStorage::new(storage(&[
            (1, Some(&[7])),
            (2, Some(&[8])),
        ])))
        .unwrap();
    bound_store
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(2)))
        .unwrap();
    let ids = vec![0, 1].into_boxed_slice();
    let allocation = ids.as_ptr();
    let error = bound_store
        .process_event(ApplyBoundEffectResults::new(ids))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    let returned_ids = error.into_tensor_ids();
    assert_eq!(returned_ids.as_ptr(), allocation);
    assert!(bound_store.is_ready());
    for tensor_id in [0, 1] {
        assert_eq!(
            bound_store
                .process_event(CaptureTensorState::new(tensor_id))
                .unwrap()
                .lifecycle(),
            Lifecycle::Unbound
        );
    }

    let mut owned_store = Store::new(2).unwrap();
    owned_store
        .process_event(BindStorage::new(storage(&[(1, None), (2, None)])))
        .unwrap();
    owned_store
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effect_buffer(2)))
        .unwrap();
    let results = vec![
        OwnedEffectResult::new(0, vec![1].into_boxed_slice()),
        OwnedEffectResult::new(1, vec![2].into_boxed_slice()),
    ]
    .into_boxed_slice();
    let allocation = results.as_ptr();
    let error = owned_store
        .process_event(ApplyOwnedEffectResults::new(results))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    let returned_results = error.into_results();
    assert_eq!(returned_results.as_ptr(), allocation);
    for tensor_id in [0, 1] {
        assert_eq!(
            owned_store
                .process_event(CaptureTensorState::new(tensor_id))
                .unwrap()
                .lifecycle(),
            Lifecycle::Unbound
        );
    }
}

#[test]
fn wrong_result_family_returns_input_and_recovers_ready_without_mutation() {
    let mut store = Store::new(1).unwrap();
    store
        .process_event(BindStorage::new(storage(&[(1, Some(&[4]))])))
        .unwrap();
    store
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(1)))
        .unwrap();
    let result = OwnedEffectResult::new(0, vec![9].into_boxed_slice());
    let error = store
        .process_event(ApplyOwnedEffectResults::new(
            vec![result].into_boxed_slice(),
        ))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    let returned = error.into_results();
    assert_eq!(returned[0].tensor_id(), 0);
    assert_eq!(returned[0].bytes(), &[9]);
    assert!(store.is_ready());
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Unbound
    );

    store
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effect_buffer(1)))
        .unwrap();
    let error = store
        .process_event(ApplyBoundEffectResults::new(vec![0].into_boxed_slice()))
        .unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(&*error.into_tensor_ids(), &[0]);
    assert!(store.is_ready());
}

#[test]
fn results_without_an_outstanding_plan_are_explicitly_invalid() {
    let mut store = Store::new(1).unwrap();
    let bound = store
        .process_event(ApplyBoundEffectResults::new(vec![0].into_boxed_slice()))
        .unwrap_err();
    assert_eq!(bound.error(), Error::InvalidRequest);
    assert_eq!(&*bound.into_tensor_ids(), &[0]);

    let owned = store
        .process_event(ApplyOwnedEffectResults::new(
            vec![OwnedEffectResult::new(0, vec![1].into_boxed_slice())].into_boxed_slice(),
        ))
        .unwrap_err();
    assert_eq!(owned.error(), Error::InvalidRequest);
    assert_eq!(owned.into_results()[0].bytes(), &[1]);
    assert_eq!(
        store.process_event(ApplyEffectError::new(0, EffectError::InvalidData)),
        Err(Error::InvalidRequest)
    );
    assert!(store.is_ready());
}

#[test]
fn awaiting_phase_returns_busy_for_conflicting_events_and_is_preserved() {
    let mut store = Store::new(1).unwrap();
    store
        .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
        .unwrap();
    store
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(1)))
        .unwrap();

    let error = store
        .process_event(BindStorage::new(storage(&[(2, Some(&[2, 3]))])))
        .unwrap_err();
    assert_eq!(error.error(), Error::Busy);
    assert_eq!(error.into_storage(), storage(&[(2, Some(&[2, 3]))]));
    assert!(store.is_awaiting_bound_results());

    let error = store
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effect_buffer(1)))
        .unwrap_err();
    assert_eq!(error.error(), Error::Busy);
    assert!(store.is_awaiting_bound_results());
    assert_eq!(
        store.process_event(BindTensor::new(
            0,
            metadata(0, 1),
            vec![8].into_boxed_slice(),
        )),
        Err(Error::Busy)
    );
    assert_eq!(store.process_event(EvictTensor::new(0)), Err(Error::Busy));
    assert_eq!(
        store.process_event(CaptureTensorState::new(0)),
        Err(Error::Busy)
    );
    assert!(store.is_awaiting_bound_results());
    store
        .process_event(ApplyBoundEffectResults::new(vec![0].into_boxed_slice()))
        .unwrap();
}

#[test]
fn backend_errors_from_bound_and_owned_phases_do_not_mutate_slots() {
    for strategy in [StrategyKind::None, StrategyKind::ExternalBuffer] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(1, Some(&[5]))])))
            .unwrap();
        store
            .process_event(PlanLoad::new(strategy, effect_buffer(1)))
            .unwrap();
        assert_eq!(
            store.process_event(ApplyEffectError::new(0, EffectError::OutOfMemory)),
            Err(Error::BackendError)
        );
        assert!(store.is_ready());
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            Lifecycle::Unbound
        );
    }
}

#[test]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "each distinct bulk transition family has an explicit allocation assertion"
)]
fn every_prepared_bulk_dispatch_path_is_allocation_free() {
    let mut store = Store::new(2).unwrap();
    let bind = BindStorage::new(storage(&[(2, Some(&[1, 2]))]));
    assert!(dispatch_without_allocation!(store, bind).is_ok());
    let empty_bind = BindStorage::new(StorageBatch::new(Vec::new().into_boxed_slice()));
    let invalid = dispatch_without_allocation!(store, empty_bind).unwrap_err();
    assert_eq!(invalid.error(), Error::InvalidRequest);
    assert!(invalid.into_storage().is_empty());

    for strategy in [
        StrategyKind::None,
        StrategyKind::ReadCopy,
        StrategyKind::ExternalBuffer,
        StrategyKind::StagedRead,
        StrategyKind::MappedFile,
    ] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(2, Some(&[1, 2]))])))
            .unwrap();
        let plan = PlanLoad::new(strategy, effect_buffer(1));
        let planned = dispatch_without_allocation!(store, plan).unwrap();
        assert_eq!(planned.effect_count(), 1);
    }

    let mut empty = Store::new(1).unwrap();
    let empty_plan = PlanLoad::new(StrategyKind::None, effect_buffer(1));
    let invalid = dispatch_without_allocation!(empty, empty_plan).unwrap_err();
    assert_eq!(invalid.error(), Error::InvalidRequest);
    assert_eq!(invalid.into_effects().len(), 1);

    let mut capacity = Store::new(2).unwrap();
    capacity
        .process_event(BindStorage::new(storage(&[(1, None), (1, None)])))
        .unwrap();
    let short_plan = PlanLoad::new(StrategyKind::None, effect_buffer(1));
    let error = dispatch_without_allocation!(capacity, short_plan).unwrap_err();
    assert_eq!(error.error(), Error::Capacity);
    assert_eq!(error.into_effects().len(), 1);

    let mut dirty = Store::new(1).unwrap();
    dirty
        .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
        .unwrap();
    let dirty_effects = EffectBuffer::new(
        vec![EffectRequest::None {
            tensor_id: 0,
            file_index: 0,
            offset: 0,
            size: 0,
        }]
        .into_boxed_slice(),
    );
    let dirty_plan = PlanLoad::new(StrategyKind::None, dirty_effects);
    let error = dispatch_without_allocation!(dirty, dirty_plan).unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);

    let mut unsupported = Store::new(1).unwrap();
    unsupported
        .process_event(BindStorage::new(storage(&[(1, None)])))
        .unwrap();
    let unknown_plan = PlanLoad::new(StrategyKind::Unknown(9), effect_buffer(1));
    let error = dispatch_without_allocation!(unsupported, unknown_plan).unwrap_err();
    assert_eq!(error.error(), Error::UnsupportedStrategy);

    let mut bound = Store::new(1).unwrap();
    bound
        .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
        .unwrap();
    bound
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(1)))
        .unwrap();
    let bound_results = ApplyBoundEffectResults::new(vec![0].into_boxed_slice());
    assert!(dispatch_without_allocation!(bound, bound_results).is_ok());

    let mut invalid_bound = Store::new(1).unwrap();
    invalid_bound
        .process_event(BindStorage::new(storage(&[(2, Some(&[1]))])))
        .unwrap();
    invalid_bound
        .process_event(PlanLoad::new(StrategyKind::None, effect_buffer(1)))
        .unwrap();
    let invalid_bound_results = ApplyBoundEffectResults::new(vec![0].into_boxed_slice());
    let error = dispatch_without_allocation!(invalid_bound, invalid_bound_results).unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(&*error.into_tensor_ids(), &[0]);

    let mut owned = Store::new(1).unwrap();
    owned
        .process_event(BindStorage::new(storage(&[(1, None)])))
        .unwrap();
    owned
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effect_buffer(1)))
        .unwrap();
    let owned_results = ApplyOwnedEffectResults::new(
        vec![OwnedEffectResult::new(0, vec![4].into_boxed_slice())].into_boxed_slice(),
    );
    assert!(dispatch_without_allocation!(owned, owned_results).is_ok());

    let mut invalid_owned = Store::new(1).unwrap();
    invalid_owned
        .process_event(BindStorage::new(storage(&[(2, None)])))
        .unwrap();
    invalid_owned
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effect_buffer(1)))
        .unwrap();
    let short_owned_results = ApplyOwnedEffectResults::new(
        vec![OwnedEffectResult::new(0, vec![4].into_boxed_slice())].into_boxed_slice(),
    );
    let error = dispatch_without_allocation!(invalid_owned, short_owned_results).unwrap_err();
    assert_eq!(error.error(), Error::InvalidRequest);
    assert_eq!(error.into_results()[0].bytes(), &[4]);

    for strategy in [
        StrategyKind::None,
        StrategyKind::ReadCopy,
        StrategyKind::MappedFile,
    ] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
            .unwrap();
        store
            .process_event(PlanLoad::new(strategy, effect_buffer(1)))
            .unwrap();

        let competing_bind = BindStorage::new(storage(&[(1, Some(&[2]))]));
        let bind = dispatch_without_allocation!(store, competing_bind).unwrap_err();
        assert_eq!(bind.error(), Error::Busy);
        assert_eq!(bind.into_storage(), storage(&[(1, Some(&[2]))]));

        let competing_plan = PlanLoad::new(StrategyKind::None, effect_buffer(1));
        let plan = dispatch_without_allocation!(store, competing_plan).unwrap_err();
        assert_eq!(plan.error(), Error::Busy);
        assert_eq!(plan.into_effects().len(), 1);

        let bind_tensor = BindTensor::new(0, metadata(0, 1), vec![3].into_boxed_slice());
        assert_eq!(
            dispatch_without_allocation!(store, bind_tensor),
            Err(Error::Busy)
        );
        let evict = EvictTensor::new(0);
        assert_eq!(dispatch_without_allocation!(store, evict), Err(Error::Busy));
        let capture = CaptureTensorState::new(0);
        assert_eq!(
            dispatch_without_allocation!(store, capture),
            Err(Error::Busy)
        );
        let backend_error = ApplyEffectError::new(0, EffectError::Backend);
        assert_eq!(
            dispatch_without_allocation!(store, backend_error),
            Err(Error::BackendError)
        );
    }

    for strategy in [
        StrategyKind::None,
        StrategyKind::ReadCopy,
        StrategyKind::MappedFile,
    ] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
            .unwrap();
        store
            .process_event(PlanLoad::new(strategy, effect_buffer(1)))
            .unwrap();
        let bound_results = ApplyBoundEffectResults::new(vec![0].into_boxed_slice());
        let bound = dispatch_without_allocation!(store, bound_results);
        if strategy == StrategyKind::None {
            assert!(bound.is_ok());
        } else {
            assert_eq!(bound.unwrap_err().error(), Error::InvalidRequest);
        }
    }

    for strategy in [
        StrategyKind::None,
        StrategyKind::ReadCopy,
        StrategyKind::MappedFile,
    ] {
        let mut store = Store::new(1).unwrap();
        store
            .process_event(BindStorage::new(storage(&[(1, Some(&[1]))])))
            .unwrap();
        store
            .process_event(PlanLoad::new(strategy, effect_buffer(1)))
            .unwrap();
        let owned_results = ApplyOwnedEffectResults::new(
            vec![OwnedEffectResult::new(0, vec![5].into_boxed_slice())].into_boxed_slice(),
        );
        let owned = dispatch_without_allocation!(store, owned_results);
        if strategy == StrategyKind::ReadCopy {
            assert!(owned.is_ok());
        } else {
            assert_eq!(owned.unwrap_err().error(), Error::InvalidRequest);
        }
    }

    let mut ready = Store::new(1).unwrap();
    let ready_bound = ApplyBoundEffectResults::new(vec![0].into_boxed_slice());
    let bound = dispatch_without_allocation!(ready, ready_bound).unwrap_err();
    assert_eq!(bound.error(), Error::InvalidRequest);
    let ready_owned = ApplyOwnedEffectResults::new(
        vec![OwnedEffectResult::new(0, vec![1].into_boxed_slice())].into_boxed_slice(),
    );
    let owned = dispatch_without_allocation!(ready, ready_owned).unwrap_err();
    assert_eq!(owned.error(), Error::InvalidRequest);
    let ready_error = ApplyEffectError::new(0, EffectError::InvalidData);
    assert_eq!(
        dispatch_without_allocation!(ready, ready_error),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn bulk_error_messages_and_effect_error_variants_are_stable() {
    for (error, message) in [
        (Error::Busy, "tensor store is awaiting effect results"),
        (
            Error::UnsupportedStrategy,
            "unsupported tensor load strategy",
        ),
        (Error::BackendError, "tensor effect backend failed"),
    ] {
        assert_eq!(error.to_string(), message);
    }
    assert_ne!(EffectError::InvalidData, EffectError::Unknown(1));
}
