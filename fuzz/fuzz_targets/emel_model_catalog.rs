#![no_main]

use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, DescribeModel, FindTensor, ReleaseStorage, Reset, SealModel, Storage, TensorInput,
    WithTensorName,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let record_count = usize::from(data.first().copied().unwrap_or(0) % 32) + 1;
    let mut names = Vec::with_capacity(record_count);
    let mut cursor = 1usize;
    for _ in 0..record_count {
        let available = data.len().saturating_sub(cursor);
        let length = usize::from(data.get(cursor).copied().unwrap_or(0) % 17).min(available);
        cursor = cursor.saturating_add(1).min(data.len());
        let end = cursor.saturating_add(length).min(data.len());
        names.push(&data[cursor..end]);
        cursor = end;
    }
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage = Storage::with_capacity(record_count, name_bytes, record_count).unwrap();
    for (index, name) in names.iter().enumerate() {
        let seed = data.get(cursor.wrapping_add(index)).copied().unwrap_or(0);
        let dimensions = [
            i64::from(seed as i8),
            i64::from(seed.wrapping_add(1) as i8),
            1,
            1,
        ];
        storage
            .push_tensor(TensorInput::new(
                name,
                u32::from(seed) % 44,
                i32::from(seed % 7) - 1,
                dimensions,
                u64::from(seed),
                seed & 1 != 0,
            ))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let sealed = catalog.process_event(SealModel::new());
    if let Ok(model) = sealed {
        assert_eq!(
            usize::try_from(
                catalog
                    .process_event(DescribeModel::new(model))
                    .unwrap()
                    .tensor_count()
            )
            .unwrap(),
            record_count
        );
        for query in names.iter().chain(core::iter::once(&b"missing".as_slice())) {
            let expected = names.iter().position(|name| *name == *query);
            let found = catalog.process_event(FindTensor::new(model, query)).unwrap();
            assert_eq!(found.is_some(), expected.is_some());
            if let (Some(descriptor), Some(index)) = (found, expected) {
                let seed = data.get(cursor.wrapping_add(index)).copied().unwrap_or(0);
                assert_eq!(descriptor.data_size(), u64::from(seed));
                let callback_name = catalog
                    .process_event(WithTensorName::new(descriptor.name_id(), |name: &[u8]| {
                        name == *query
                    }))
                    .unwrap();
                assert_eq!(callback_name, Some(true));
            }
        }
        catalog.process_event(Reset::new()).unwrap();
    }
    let _ = catalog.process_event(ReleaseStorage::new());
});
