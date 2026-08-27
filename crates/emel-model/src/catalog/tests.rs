use allocation_counter::measure;
use core::cell::{Cell, RefCell};
use std::sync::Arc;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, ParseDone, Probe};

use super::Catalog;
use super::event::{
    BindStorage, DescribeModel, DescribeTensor, Error, FindTensor, PrefixScanSummary,
    ReleaseStorage, Reset, ScanPrefix, SealModel, Storage, TensorBindingStatus, TensorInput,
    ValidateTensorShape, WithTensorName,
};

fn input(name: &[u8]) -> TensorInput<'_> {
    TensorInput::new(name, 0, 2, [4, 8, 1, 1], 128, true)
}

fn storage(records: &[TensorInput<'_>]) -> Storage {
    let name_bytes = records.iter().map(|record| record.name_len()).sum();
    let mut storage = Storage::with_capacity(records.len(), name_bytes, records.len()).unwrap();
    for record in records {
        storage.push_tensor(*record).unwrap();
    }
    storage
}

fn sealed_catalog(name: &[u8]) -> (super::Catalog, super::event::ModelIdentity) {
    let mut catalog = super::Catalog::try_new().unwrap();
    catalog
        .process_event(BindStorage::new(storage(&[input(name)])))
        .unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    (catalog, model)
}

#[test]
fn cross_actor_requests_classify_owner_generation_and_state() {
    let (mut first, old_model) = sealed_catalog(b"x");
    let (_, other_model) = sealed_catalog(b"other");
    assert_eq!(
        first.process_event(FindTensor::new(other_model, b"x")),
        Err(Error::WrongModelIdentity)
    );
    assert_eq!(
        first.process_event(DescribeModel::new(other_model)),
        Err(Error::WrongModelIdentity)
    );
    assert_eq!(
        first
            .process_event(ScanPrefix::new(other_model, b"x"))
            .unwrap_err(),
        Error::WrongModelIdentity
    );
    assert_eq!(
        first
            .process_event(ValidateTensorShape::new(other_model, b"x", &[4]))
            .unwrap_err(),
        Error::WrongModelIdentity
    );
    first.process_event(Reset::new()).unwrap();
    let new_model = first.process_event(SealModel::new()).unwrap();

    assert_eq!(
        first.process_event(FindTensor::new(old_model, b"x")),
        Err(Error::StaleModelIdentity)
    );
    assert_eq!(
        first.process_event(DescribeModel::new(old_model)),
        Err(Error::StaleModelIdentity)
    );
    assert_eq!(
        first
            .process_event(ScanPrefix::new(old_model, b"x"))
            .unwrap_err(),
        Error::StaleModelIdentity
    );
    assert_eq!(
        first
            .process_event(ValidateTensorShape::new(old_model, b"x", &[4]))
            .unwrap_err(),
        Error::StaleModelIdentity
    );

    let mut unbound = super::Catalog::try_new().unwrap();
    let mut bound_only = super::Catalog::try_new().unwrap();
    bound_only
        .process_event(BindStorage::new(storage(&[input(b"q")])))
        .unwrap();
    for expected_error in [
        unbound.process_event(ValidateTensorShape::new(new_model, b"x", &[4])),
        bound_only.process_event(ValidateTensorShape::new(new_model, b"x", &[4])),
    ] {
        assert_eq!(expected_error.unwrap_err(), Error::StorageUnavailable);
    }
}

fn tensor_gguf(names: &[&[u8]]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(names.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&17u64.to_le_bytes());
    bytes.extend_from_slice(b"general.alignment");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    for (index, name) in names.iter().enumerate() {
        bytes.extend_from_slice(&u64::try_from(name.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(index.checked_mul(4).unwrap())
                .unwrap()
                .to_le_bytes(),
        );
    }
    for _ in names {
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
    }
    bytes
}

fn parsed_gguf(bytes: Vec<u8>) -> (GgufLoader, ParseDone) {
    let mut loader = GgufLoader::new();
    let probe = loader.process_event(Probe::new(Arc::from(bytes))).unwrap();
    loader
        .process_event(Bind::new(emel_gguf::event::Storage::exact(probe).unwrap()))
        .unwrap();
    let parsed = loader.process_event(Parse::new()).unwrap();
    (loader, parsed)
}

#[test]
fn exact_bytes_duplicates_empty_nul_and_canonical_names() {
    let records = [
        input(b"same"),
        TensorInput::new(b"same", 1, 1, [99, 1, 1, 1], 42, true),
        input(&[0xff, 0, b'x']),
        input(b""),
    ];
    let mut catalog = Catalog::try_new().unwrap();
    assert!(catalog.is_empty());
    catalog
        .process_event(BindStorage::new(storage(&records)))
        .unwrap();
    assert!(catalog.is_bound());
    let model = catalog.process_event(SealModel::new()).unwrap();
    assert!(catalog.is_sealed());

    let first = catalog
        .process_event(FindTensor::new(model, b"same"))
        .unwrap()
        .unwrap();
    let second = catalog
        .process_event(DescribeTensor::new(first.tensor_id()))
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.tensor_type(), emel_tensor::dtype::SerializedType::F32);
    assert_eq!(first.dimensions(), [4, 8, 1, 1]);
    assert_eq!(first.data_size(), 128);
    let arbitrary = catalog
        .process_event(FindTensor::new(model, &[0xff, 0, b'x']))
        .unwrap()
        .unwrap();
    assert_eq!(arbitrary.dimensions(), [4, 8, 1, 1]);
    let empty = catalog
        .process_event(FindTensor::new(model, b""))
        .unwrap()
        .unwrap();
    let mut copied = [0u8; 4];
    let mut length = usize::MAX;
    catalog
        .process_event(WithTensorName::new(arbitrary.name_id(), |name: &[u8]| {
            copied[..name.len()].copy_from_slice(name);
            length = name.len();
        }))
        .unwrap();
    assert_eq!(&copied[..length], &[0xff, 0, b'x']);
    assert_eq!(
        catalog
            .process_event(WithTensorName::new(empty.name_id(), |name: &[u8]| name.len()))
            .unwrap(),
        Some(0)
    );
}

#[test]
fn bound_predicate_matches_pinned_generation_semantics() {
    let records = [
        TensorInput::new(b"unbound", 0, 1, [1; 4], 1, false),
        TensorInput::new(b"zero-size", 0, 1, [1; 4], 0, true),
        TensorInput::new(b"zero-dims", 0, 0, [1; 4], 1, true),
        TensorInput::new(b"negative", 0, 2, [1, -1, 1, 1], 1, true),
        TensorInput::new(b"five-good", 0, 5, [1, 2, 3, 4], 1, true),
        TensorInput::new(b"five-bad", 0, 5, [1, 2, 3, 0], 1, true),
    ];
    let mut catalog = Catalog::try_new().unwrap();
    catalog
        .process_event(BindStorage::new(storage(&records)))
        .unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    for name in [
        b"unbound".as_slice(),
        b"zero-size",
        b"zero-dims",
        b"negative",
        b"five-bad",
    ] {
        assert!(
            catalog
                .process_event(FindTensor::new(model, name))
                .unwrap()
                .is_some()
        );
    }
    assert_eq!(
        catalog.process_event(FindTensor::new(model, b"missing")),
        Ok(None)
    );
    let unbound = catalog
        .process_event(FindTensor::new(model, b"unbound"))
        .unwrap()
        .unwrap();
    assert!(!unbound.binding_present());
    assert_eq!(unbound.binding_status(), TensorBindingStatus::Unbound);
    let zero_size = catalog
        .process_event(FindTensor::new(model, b"zero-size"))
        .unwrap()
        .unwrap();
    assert_eq!(zero_size.data_size(), 0);
    assert_eq!(zero_size.binding_status(), TensorBindingStatus::Unbound);
    let five_good = catalog
        .process_event(FindTensor::new(model, b"five-good"))
        .unwrap()
        .unwrap();
    assert_eq!(five_good.dimension_count(), 5);
    assert_eq!(five_good.binding_status(), TensorBindingStatus::Bound);
    assert!(
        five_good
            .dimensions()
            .iter()
            .all(|dimension| *dimension > 0)
    );
}

#[test]
fn prefix_scan_is_typed_identity_checked_and_allocation_free() {
    let records = [
        input(b"prep.a"),
        TensorInput::new(b"prep.b", 0, 1, [1, 1, 1, 1], 0, true),
        input(b"enc.a"),
    ];
    let mut catalog = Catalog::try_new().unwrap();
    assert_eq!(
        catalog.process_event(ScanPrefix::new(
            super::event::ModelIdentity {
                owner: 0,
                generation: 0
            },
            b"prep.",
        )),
        Err(Error::StorageUnavailable)
    );
    catalog
        .process_event(BindStorage::new(storage(&records)))
        .unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    assert_eq!(
        catalog.process_event(ScanPrefix::new(model, b"prep.")),
        Ok(PrefixScanSummary::new(2, false, true))
    );
    assert_eq!(
        catalog.process_event(ScanPrefix::new(model, b"enc.")),
        Ok(PrefixScanSummary::new(1, true, true))
    );
    let wrong = Catalog::try_new().unwrap().process_event(SealModel::new());
    assert_eq!(wrong, Err(Error::StorageUnavailable));
}

#[test]
fn seal_rejects_unknown_type_bad_range_and_short_index() {
    let mut unknown = Catalog::try_new().unwrap();
    unknown
        .process_event(BindStorage::new(storage(&[TensorInput::new(
            b"x",
            u32::MAX,
            1,
            [1; 4],
            1,
            true,
        )])))
        .unwrap();
    assert_eq!(
        unknown.process_event(SealModel::new()),
        Err(Error::ModelInvalid)
    );

    let mut malformed_storage = storage(&[input(b"x")]);
    malformed_storage.records[0].name_offset = u32::MAX;
    let mut malformed = Catalog::try_new().unwrap();
    malformed
        .process_event(BindStorage::new(malformed_storage))
        .unwrap();
    assert_eq!(
        malformed.process_event(SealModel::new()),
        Err(Error::ModelInvalid)
    );

    assert_eq!(
        Storage::with_capacity(1, 1, 0).unwrap_err(),
        Error::Capacity
    );
}

#[test]
fn public_diagnostics_and_storage_boundaries_are_stable() {
    let messages = [
        (Error::InvalidRequest, "invalid catalog request"),
        (Error::ModelInvalid, "model catalog input is invalid"),
        (Error::Capacity, "catalog storage capacity is unavailable"),
        (Error::StorageUnavailable, "catalog storage is unavailable"),
        (Error::Busy, "catalog is busy"),
        (
            Error::WrongModelIdentity,
            "model identity belongs to another catalog",
        ),
        (Error::StaleModelIdentity, "model identity is stale"),
        (
            Error::WrongTensorIdentity,
            "tensor identity belongs to another catalog",
        ),
        (Error::StaleTensorIdentity, "tensor identity is stale"),
        (
            Error::WrongNameIdentity,
            "name identity belongs to another catalog",
        ),
        (Error::StaleNameIdentity, "name identity is stale"),
        (Error::UnexpectedEvent, "unexpected catalog event"),
        (Error::Internal, "internal catalog error"),
    ];
    for (error, message) in messages {
        assert_eq!(error.to_string(), message);
    }

    let mut storage = Storage::with_capacity(1, 1, 1).unwrap();
    assert_eq!(storage.tensor_capacity(), 1);
    assert_eq!(storage.name_capacity(), 1);
    assert_eq!(
        format!("{storage:?}"),
        "Storage { tensor_capacity: 1, name_capacity: 1, index_capacity: 1, tensor_count: 0, .. }"
    );
    storage.push_tensor(input(b"x")).unwrap();
    assert_eq!(storage.push_tensor(input(b"")), Err(Error::Capacity));

    let mut no_name_capacity = Storage::with_capacity(1, 0, 1).unwrap();
    assert_eq!(
        no_name_capacity.push_tensor(input(b"x")),
        Err(Error::Capacity)
    );
}

#[test]
fn identities_distinguish_owners_and_generations() {
    let mut first = Catalog::try_new().unwrap();
    first
        .process_event(BindStorage::new(storage(&[input(b"x")])))
        .unwrap();
    let old_model = first.process_event(SealModel::new()).unwrap();
    let old_tensor = first
        .process_event(FindTensor::new(old_model, b"x"))
        .unwrap()
        .unwrap();

    let mut second = Catalog::try_new().unwrap();
    second
        .process_event(BindStorage::new(storage(&[input(b"x")])))
        .unwrap();
    let second_model = second.process_event(SealModel::new()).unwrap();
    let second_tensor = second
        .process_event(FindTensor::new(second_model, b"x"))
        .unwrap()
        .unwrap();
    assert_eq!(
        second.process_event(DescribeModel::new(old_model)),
        Err(Error::WrongModelIdentity)
    );
    assert_eq!(
        second.process_event(DescribeTensor::new(old_tensor.tensor_id())),
        Err(Error::WrongTensorIdentity)
    );
    assert_eq!(
        second.process_event(WithTensorName::new(old_tensor.name_id(), |_: &[u8]| ())),
        Err(Error::WrongNameIdentity)
    );

    first.process_event(Reset::new()).unwrap();
    let new_model = first.process_event(SealModel::new()).unwrap();
    assert_eq!(
        first.process_event(DescribeModel::new(old_model)),
        Err(Error::StaleModelIdentity)
    );
    assert_eq!(
        first.process_event(DescribeTensor::new(old_tensor.tensor_id())),
        Err(Error::StaleTensorIdentity)
    );
    assert_eq!(
        first.process_event(WithTensorName::new(old_tensor.name_id(), |_: &[u8]| ())),
        Err(Error::StaleNameIdentity)
    );
    assert_eq!(
        first
            .process_event(DescribeModel::new(new_model))
            .unwrap()
            .tensor_count(),
        1
    );
    assert_eq!(format!("{old_model:?}"), "ModelIdentity { .. }");
    assert_eq!(format!("{:?}", old_tensor.tensor_id()), "TensorId { .. }");
    assert_eq!(format!("{:?}", old_tensor.name_id()), "NameId { .. }");
    assert_ne!(second_tensor.tensor_id(), old_tensor.tensor_id());
}

#[test]
fn invalid_sequences_return_storage_and_support_reuse() {
    let mut catalog = Catalog::try_new().unwrap();
    assert_eq!(
        catalog.process_event(SealModel::new()),
        Err(Error::StorageUnavailable)
    );
    assert_eq!(
        catalog.process_event(Reset::new()),
        Err(Error::StorageUnavailable)
    );
    assert!(matches!(
        catalog.process_event(ReleaseStorage::new()),
        Err(Error::StorageUnavailable)
    ));

    catalog
        .process_event(BindStorage::new(storage(&[input(b"x")])))
        .unwrap();
    let rejected = catalog
        .process_event(BindStorage::new(storage(&[input(b"y")])))
        .unwrap_err();
    assert_eq!(rejected.error(), Error::Busy);
    let mut returned = rejected.into_storage();
    returned.clear();
    returned.push_tensor(input(b"z")).unwrap();
    assert_eq!(
        catalog.process_event(Reset::new()),
        Err(Error::InvalidRequest)
    );
    let model = catalog.process_event(SealModel::new()).unwrap();
    assert!(matches!(
        catalog.process_event(ReleaseStorage::new()),
        Err(Error::Busy)
    ));
    assert_eq!(
        catalog
            .process_event(DescribeModel::new(model))
            .unwrap()
            .tensor_count(),
        1
    );
    let tensor = catalog
        .process_event(FindTensor::new(model, b"x"))
        .unwrap()
        .unwrap();
    catalog.process_event(Reset::new()).unwrap();
    assert_eq!(
        catalog.process_event(WithTensorName::new(tensor.name_id(), |_: &[u8]| ())),
        Err(Error::StorageUnavailable)
    );
    let mut released = catalog.process_event(ReleaseStorage::new()).unwrap();
    released.clear();
    released.push_tensor(input(b"q")).unwrap();
    catalog.process_event(BindStorage::new(released)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    assert!(
        catalog
            .process_event(FindTensor::new(model, b"q"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn every_public_dispatch_is_allocation_free() {
    let mut catalog = Catalog::try_new().unwrap();
    let bind = BindStorage::new(storage(&[input(b"x")]));
    let bind_result = RefCell::new(None);
    let allocation = measure(|| {
        bind_result.replace(Some(catalog.process_event(bind)));
    });
    assert_eq!(allocation.count_total, 0);
    assert!(bind_result.borrow().as_ref().unwrap().is_ok());

    let model = Cell::new(None);
    let allocation = measure(|| model.set(Some(catalog.process_event(SealModel::new()))));
    assert_eq!(allocation.count_total, 0);
    let model = model.get().unwrap().unwrap();

    let found = Cell::new(None);
    let allocation =
        measure(|| found.set(Some(catalog.process_event(FindTensor::new(model, b"x")))));
    assert_eq!(allocation.count_total, 0);
    let descriptor = found.get().unwrap().unwrap().unwrap();

    let described = Cell::new(None);
    let allocation = measure(|| {
        described.set(Some(
            catalog.process_event(DescribeTensor::new(descriptor.tensor_id())),
        ));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(described.get().unwrap().unwrap(), descriptor);

    let callback = Cell::new(0usize);
    let named = RefCell::new(None);
    let allocation = measure(|| {
        named.replace(Some(catalog.process_event(WithTensorName::new(
            descriptor.name_id(),
            |name: &[u8]| callback.set(name.len()),
        ))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(callback.get(), 1);

    let allocation = measure(|| {
        catalog.process_event(Reset::new()).unwrap();
    });
    assert_eq!(allocation.count_total, 0);
    let released = RefCell::new(None);
    let allocation = measure(|| {
        released.replace(Some(catalog.process_event(ReleaseStorage::new())));
    });
    assert_eq!(allocation.count_total, 0);
    assert!(released.borrow().as_ref().unwrap().is_ok());
}

#[test]
fn public_gguf_ingestion_allocates_only_three_setup_regions() {
    let (mut loader, parsed) = parsed_gguf(tensor_gguf(&[b"x"]));
    let result = RefCell::new(None);
    let allocation = measure(|| {
        result.replace(Some(Storage::from_gguf(&mut loader, parsed)));
    });
    assert_eq!(allocation.count_total, 3);
    let storage = result.into_inner().unwrap().unwrap();
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    assert!(
        catalog
            .process_event(FindTensor::new(model, b"x"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn gguf_ingestion_rejects_a_parse_token_with_a_different_tensor_count() {
    let (_one_loader, one_parsed) = parsed_gguf(tensor_gguf(&[b"one"]));
    let (mut two_loader, _two_parsed) = parsed_gguf(tensor_gguf(&[b"one", b"two"]));
    assert_eq!(
        Storage::from_gguf(&mut two_loader, one_parsed).unwrap_err(),
        Error::ModelInvalid
    );
}

#[test]
fn maximum_name_and_tensor_count_are_supported() {
    let maximum_name = vec![b'n'; super::MAX_NAME_BYTES];
    let mut long_storage = Storage::with_capacity(1, maximum_name.len(), 1).unwrap();
    long_storage.push_tensor(input(&maximum_name)).unwrap();
    let mut long_catalog = Catalog::try_new().unwrap();
    long_catalog
        .process_event(BindStorage::new(long_storage))
        .unwrap();
    let long_model = long_catalog.process_event(SealModel::new()).unwrap();
    assert!(
        long_catalog
            .process_event(FindTensor::new(long_model, &maximum_name))
            .unwrap()
            .is_some()
    );

    let mut maximum = Storage::with_capacity(super::MAX_TENSORS, 0, super::MAX_TENSORS).unwrap();
    for _ in 0..super::MAX_TENSORS {
        maximum.push_tensor(input(b"")).unwrap();
    }
    let mut maximum_catalog = Catalog::try_new().unwrap();
    maximum_catalog
        .process_event(BindStorage::new(maximum))
        .unwrap();
    let model = maximum_catalog.process_event(SealModel::new()).unwrap();
    assert_eq!(
        maximum_catalog
            .process_event(DescribeModel::new(model))
            .unwrap()
            .tensor_count(),
        u32::try_from(super::MAX_TENSORS).unwrap()
    );
}

#[test]
fn pinned_llama_and_lfm_fixtures_use_public_ingestion() {
    let Ok(source_dir) = std::env::var("EMEL_CPP_SOURCE_DIR") else {
        return;
    };
    for (file, required_name) in [
        (
            "Llama-68M-Chat-v1-Q2_K.gguf",
            b"output_norm.weight".as_slice(),
        ),
        ("LFM2.5-230M-Q8_0.gguf", b"token_embd.weight".as_slice()),
    ] {
        let bytes = std::fs::read(
            std::path::Path::new(&source_dir)
                .join("tests/models")
                .join(file),
        )
        .unwrap();
        let (mut loader, parsed) = parsed_gguf(bytes);
        let storage = Storage::from_gguf(&mut loader, parsed).unwrap();
        let mut catalog = Catalog::try_new().unwrap();
        catalog.process_event(BindStorage::new(storage)).unwrap();
        let model = catalog.process_event(SealModel::new()).unwrap();
        let descriptor = catalog
            .process_event(FindTensor::new(model, required_name))
            .unwrap()
            .unwrap();
        assert!(descriptor.binding_present());
        assert!(descriptor.data_size() > 0);
        assert!(descriptor.dimension_count() > 0);
    }
}

#[test]
fn source_and_api_contracts_are_explicit() {
    let machine = include_str!("sm.rs");
    let root = include_str!("mod.rs");
    assert!(machine.contains("unexpected_event<_> / effect_unexpected"));
    assert!(machine.contains("[guard_find_present_bound] / effect_find_present_bound"));
    assert!(machine.contains("[guard_find_present_unbound] / effect_find_present_unbound"));
    assert!(machine.contains("CatalogNameQuery"));
    assert!(root.contains("mod storage;"));
    assert!(!machine.contains("unsafe"));
}
