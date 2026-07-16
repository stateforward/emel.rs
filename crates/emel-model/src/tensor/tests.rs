use super::Store;
use super::actor::MAX_TENSORS;
use super::event::{BindTensor, CaptureTensorState, Error, EvictTensor, Lifecycle, TensorMetadata};
use allocation_counter::measure;
use core::cell::{Cell, RefCell};

fn metadata(data_size: u64) -> TensorMetadata {
    TensorMetadata::new(128, data_size, 3, 7)
}

#[test]
fn construction_preallocates_bounded_slots_and_retains_dependencies() {
    #[derive(Debug, Eq, PartialEq)]
    struct Dependencies(u8);

    assert_eq!(Store::new(0).unwrap_err(), Error::Capacity);
    assert_eq!(Store::new(MAX_TENSORS + 1).unwrap_err(), Error::Capacity);

    let store = Store::with_dependencies(2, Dependencies(9)).unwrap();
    assert_eq!(store.dependencies(), &Dependencies(9));
    assert!(store.is_ready());
    assert_eq!(format!("{store:?}"), "Store { .. }");
}

#[test]
fn metadata_accessors_and_error_messages_are_stable() {
    let record = metadata(64);
    assert_eq!(record.file_offset(), 128);
    assert_eq!(record.data_size(), 64);
    assert_eq!(record.file_index(), 3);
    assert_eq!(record.tensor_type(), 7);

    for (error, message) in [
        (Error::InvalidRequest, "invalid tensor request"),
        (Error::Capacity, "tensor store capacity unavailable"),
        (Error::TensorAlreadyResident, "tensor is already resident"),
        (Error::TensorUnbound, "tensor is not resident"),
        (
            Error::MappedTensorRequiresRelease,
            "mapped tensor requires mapped release",
        ),
        (Error::Internal, "internal tensor actor error"),
    ] {
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn bind_capture_evict_and_rebind_preserve_owned_bytes_and_metadata() {
    let mut store = Store::new(2).unwrap();
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(1))
            .unwrap()
            .lifecycle(),
        Lifecycle::Unbound
    );

    let first = vec![1_u8, 2, 3, 4].into_boxed_slice();
    let bound = store
        .process_event(BindTensor::new(1, metadata(4), first))
        .unwrap();
    assert_eq!(bound.tensor_id(), 1);
    assert_eq!(bound.buffer_bytes(), 4);

    let resident = store.process_event(CaptureTensorState::new(1)).unwrap();
    assert_eq!(resident.lifecycle(), Lifecycle::Resident);
    assert_eq!(resident.buffer_bytes(), 4);
    assert_eq!(resident.metadata(), metadata(4));

    let evicted = store.process_event(EvictTensor::new(1)).unwrap();
    assert_eq!(evicted.tensor_id(), 1);
    assert_eq!(&*evicted.into_bytes(), &[1, 2, 3, 4]);

    let state = store.process_event(CaptureTensorState::new(1)).unwrap();
    assert_eq!(state.lifecycle(), Lifecycle::Evicted);
    assert_eq!(state.buffer_bytes(), 0);
    assert_eq!(state.metadata(), metadata(4));

    let second = vec![9_u8, 8].into_boxed_slice();
    assert_eq!(
        store
            .process_event(BindTensor::new(1, metadata(2), second))
            .unwrap()
            .buffer_bytes(),
        2
    );
}

#[test]
fn core_events_reject_invalid_identifiers_and_payloads() {
    let mut store = Store::new(2).unwrap();

    assert_eq!(
        store.process_event(BindTensor::new(-1, metadata(1), vec![1].into_boxed_slice(),)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(BindTensor::new(2, metadata(1), vec![1].into_boxed_slice(),)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(BindTensor::new(0, metadata(0), vec![1].into_boxed_slice(),)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(BindTensor::new(
            0,
            metadata(1),
            Vec::new().into_boxed_slice(),
        )),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(BindTensor::new(0, metadata(2), vec![1].into_boxed_slice(),)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(EvictTensor::new(-1)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store.process_event(EvictTensor::new(0)),
        Err(Error::TensorUnbound)
    );
    assert_eq!(
        store.process_event(CaptureTensorState::new(2)),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn resident_rebind_is_rejected_without_replacing_actor_owned_bytes() {
    let mut store = Store::new(1).unwrap();
    store
        .process_event(BindTensor::new(
            0,
            metadata(3),
            vec![3, 2, 1].into_boxed_slice(),
        ))
        .unwrap();

    assert_eq!(
        store.process_event(BindTensor::new(0, metadata(1), vec![9].into_boxed_slice(),)),
        Err(Error::TensorAlreadyResident)
    );
    let evicted = store.process_event(EvictTensor::new(0)).unwrap();
    assert_eq!(&*evicted.into_bytes(), &[3, 2, 1]);
}

#[test]
fn core_events_dispatch_for_non_default_dependency_types() {
    struct Dependencies;

    let mut store = Store::with_dependencies(1, Dependencies).unwrap();
    store
        .process_event(BindTensor::new(0, metadata(1), vec![5].into_boxed_slice()))
        .unwrap();
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Resident
    );
    assert_eq!(
        &*store
            .process_event(EvictTensor::new(0))
            .unwrap()
            .into_bytes(),
        &[5]
    );
}

#[test]
fn private_machine_declares_explicit_unexpected_event_recovery() {
    let source = include_str!("sm.rs");
    assert!(source.contains("unexpected_event<_> / effect_unexpected"));
    assert!(source.contains("fn effect_unexpected"));
    assert!(!source.contains("todo!"));
    assert!(!source.contains("unsafe"));
}

#[test]
fn prepared_core_dispatches_do_not_allocate() {
    let mut bind_store = Store::new(1).unwrap();
    let bind_done = Cell::new(None);
    let bind = BindTensor::new(0, metadata(1), vec![7].into_boxed_slice());
    let allocation = measure(|| bind_done.set(Some(bind_store.process_event(bind))));
    assert_eq!(allocation.count_total, 0);
    assert_eq!(bind_done.get().unwrap().unwrap().buffer_bytes(), 1);

    let mut bind_error_store = Store::new(1).unwrap();
    let bind_error = Cell::new(None);
    let invalid_bind = BindTensor::new(1, metadata(1), vec![7].into_boxed_slice());
    let allocation = measure(|| bind_error.set(Some(bind_error_store.process_event(invalid_bind))));
    assert_eq!(allocation.count_total, 0);
    assert_eq!(bind_error.get().unwrap(), Err(Error::InvalidRequest));

    let capture_done = Cell::new(None);
    let capture = CaptureTensorState::new(0);
    let allocation = measure(|| capture_done.set(Some(bind_store.process_event(capture))));
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        capture_done.get().unwrap().unwrap().lifecycle(),
        Lifecycle::Resident
    );

    let capture_error = Cell::new(None);
    let invalid_capture = CaptureTensorState::new(1);
    let allocation = measure(|| capture_error.set(Some(bind_store.process_event(invalid_capture))));
    assert_eq!(allocation.count_total, 0);
    assert_eq!(capture_error.get().unwrap(), Err(Error::InvalidRequest));

    let evict_done = RefCell::new(None);
    let evict = EvictTensor::new(0);
    let allocation = measure(|| {
        evict_done.replace(Some(bind_store.process_event(evict)));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        evict_done
            .borrow()
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .tensor_id(),
        0
    );

    let evict_error = RefCell::new(None);
    let invalid_evict = EvictTensor::new(0);
    let allocation = measure(|| {
        evict_error.replace(Some(bind_store.process_event(invalid_evict)));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        evict_error.borrow().as_ref().unwrap(),
        &Err(Error::TensorUnbound)
    );
}

#[test]
fn tensor_actor_exports_only_its_actor_surface_and_uses_generated_state_inspection() {
    let crate_root = include_str!("../lib.rs");
    let module_root = include_str!("mod.rs");
    let actor = include_str!("actor.rs");

    assert!(crate_root.contains("pub mod tensor;"));
    assert!(module_root.contains("mod actor;"));
    assert!(module_root.contains("mod sm;"));
    assert!(!module_root.contains("pub mod sm;"));
    assert!(actor.contains("self.machine.is(&ModelTensorStates::StateReady)"));

    let mut store = Store::new(1).unwrap();
    assert!(store.is_ready());
    store
        .process_event(BindTensor::new(0, metadata(1), vec![1].into_boxed_slice()))
        .unwrap();
    assert!(store.is_ready());
    store.process_event(EvictTensor::new(0)).unwrap();
    assert!(store.is_ready());
}
