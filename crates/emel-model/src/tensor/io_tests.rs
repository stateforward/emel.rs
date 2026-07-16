//! Focused tensor I/O route tests.

use allocation_counter::measure;
use core::cell::Cell;

use emel_io::{mmap, read, staged_read};

use super::Store;
use super::dependency::{Actors, Mapper, Reader, Stager};
use super::event::{
    ApplyEffectError, BindStorage, BindTensor, CaptureTensorState, EffectBuffer, EffectError,
    EffectRequest, Error, Lifecycle, PlanLoad, ReadLoad, ReleaseMapped, StagedLoad, StorageBatch,
    StorageEntry, StrategyKind, TensorMetadata, WithTensor,
};

macro_rules! dispatch_without_allocation {
    ($dispatch:expr) => {{
        let outcome = Cell::new(None);
        let allocation = measure(|| outcome.set(Some($dispatch)));
        assert_eq!(allocation.count_total, 0);
        outcome.get().expect("dispatch stores one Copy outcome")
    }};
}

fn storage(bytes: usize) -> StorageBatch {
    StorageBatch::new(
        vec![StorageEntry::new(
            TensorMetadata::new(0, bytes as u64, 0, 0),
            Some(vec![0; bytes].into_boxed_slice()),
        )]
        .into_boxed_slice(),
    )
}

fn storage_with_metadata_and_capacity(metadata_bytes: u64, capacity: usize) -> StorageBatch {
    StorageBatch::new(
        vec![StorageEntry::new(
            TensorMetadata::new(0, metadata_bytes, 0, 0),
            Some(vec![0; capacity].into_boxed_slice()),
        )]
        .into_boxed_slice(),
    )
}

#[test]
fn per_tensor_bind_supports_allocation_free_typed_access_without_a_bulk_extent() {
    let mut store = Store::new(1).unwrap();
    let bytes = vec![1_u8, 2, 3, 4].into_boxed_slice();

    dispatch_without_allocation!(store.process_event(BindTensor::new(
        0,
        TensorMetadata::new(0, 4, 0, 0),
        bytes
    )))
    .unwrap();

    let mut copy = |resident: &[u8]| <[u8; 4]>::try_from(resident).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut copy))).unwrap(),
        [1, 2, 3, 4]
    );
}

#[test]
fn read_and_staged_routes_commit_actor_owned_bytes_and_support_typed_access() {
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();

    let read = dispatch_without_allocation!(store.process_event(ReadLoad::new(
        0,
        "fixture.bin",
        Some(&[9, 8, 1, 2, 3, 4]),
        2,
        4,
    )))
    .unwrap();
    assert_eq!(read.tensor_id(), 0);
    assert_eq!(read.bytes_copied(), 4);
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Resident
    );
    let mut copy = |bytes: &[u8]| <[u8; 4]>::try_from(bytes).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut copy))).unwrap(),
        [1, 2, 3, 4]
    );

    store.process_event(BindStorage::new(storage(4))).unwrap();
    let staged = dispatch_without_allocation!(store.process_event(StagedLoad::new(
        0,
        Some(&[9, 8, 7, 6, 5, 4]),
        2,
        4,
        3
    )))
    .unwrap();
    assert_eq!(staged.tensor_id(), 0);
    assert_eq!(staged.bytes_copied(), 4);
    let mut copy = |bytes: &[u8]| <[u8; 4]>::try_from(bytes).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut copy))).unwrap(),
        [7, 6, 5, 4]
    );
}

#[test]
fn capability_validation_and_child_errors_are_explicit_and_recoverable() {
    let mut absent = Store::new(1).unwrap();
    absent.process_event(BindStorage::new(storage(4))).unwrap();
    assert_eq!(
        dispatch_without_allocation!(absent.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 4]),
            0,
            4
        ))),
        Err(Error::ReadUnavailable)
    );
    assert_eq!(
        dispatch_without_allocation!(absent.process_event(StagedLoad::new(
            0,
            Some(&[1; 4]),
            0,
            4,
            2
        ))),
        Err(Error::StagerUnavailable)
    );
    let mut observe = |bytes: &[u8]| bytes.len();
    assert_eq!(
        dispatch_without_allocation!(absent.process_event(WithTensor::new(0, &mut observe))),
        Err(Error::TensorUnbound)
    );
    assert_eq!(
        dispatch_without_allocation!(absent.process_event(WithTensor::new(1, &mut observe))),
        Err(Error::InvalidRequest)
    );

    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReadLoad::new(
            0,
            "",
            Some(&[1; 4]),
            0,
            4
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 3]),
            0,
            4
        ))),
        Err(Error::Read(read::event::Error::ShortRead))
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(StagedLoad::new(
            0,
            Some(&[1; 3]),
            0,
            4,
            2
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReadLoad::new(
            1,
            "fixture.bin",
            Some(&[1; 4]),
            0,
            4
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 5]),
            0,
            5
        ))),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn staged_ranges_reject_out_of_source_and_overflow_without_allocation() {
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();

    assert_eq!(
        dispatch_without_allocation!(store.process_event(StagedLoad::new(
            0,
            Some(&[1; 6]),
            3,
            4,
            2
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(StagedLoad::new(
            0,
            Some(&[1; 4]),
            u64::MAX,
            4,
            2
        ))),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn direct_io_requires_active_bound_metadata_and_logical_size() {
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut unbound = Store::with_dependencies(2, dependencies).unwrap();
    assert_eq!(
        dispatch_without_allocation!(unbound.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 4]),
            0,
            4,
        ))),
        Err(Error::InvalidRequest)
    );

    unbound
        .process_event(BindStorage::new(storage_with_metadata_and_capacity(4, 8)))
        .unwrap();
    assert_eq!(
        dispatch_without_allocation!(unbound.process_event(ReadLoad::new(
            1,
            "fixture.bin",
            Some(&[1; 4]),
            0,
            4,
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(unbound.process_event(StagedLoad::new(
            1,
            Some(&[1; 4]),
            0,
            4,
            2,
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(unbound.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 5]),
            0,
            5,
        ))),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        dispatch_without_allocation!(unbound.process_event(StagedLoad::new(
            0,
            Some(&[1; 5]),
            0,
            5,
            2,
        ))),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn owned_access_exposes_only_the_committed_resident_prefix() {
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store
        .process_event(BindStorage::new(storage_with_metadata_and_capacity(4, 8)))
        .unwrap();
    store
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&[9, 8, 7, 6]), 0, 4))
        .unwrap();
    let mut observe = |bytes: &[u8]| (bytes.len(), bytes.iter().copied().sum::<u8>());
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut observe))),
        Ok((4, 30))
    );

    store
        .process_event(BindStorage::new(storage_with_metadata_and_capacity(4, 8)))
        .unwrap();
    store
        .process_event(StagedLoad::new(0, Some(&[4, 3, 2, 1]), 0, 4, 3))
        .unwrap();
    let mut observe = |bytes: &[u8]| (bytes.len(), bytes.iter().copied().sum::<u8>());
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut observe))),
        Ok((4, 10))
    );
}

#[derive(Clone, Copy)]
enum MalformedRead {
    WrongTensor,
    WrongBytes,
}

struct MalformedDependencies {
    read: MalformedRead,
    staged_bytes: u64,
}

impl Mapper for MalformedDependencies {
    const AVAILABLE: bool = false;

    fn map_tensor(
        &mut self,
        _: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn release_mapping(
        &mut self,
        _: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn with_mapping<Operation>(
        &mut self,
        _: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        Err(mmap::event::Error::UnsupportedPlatform)
    }
}

impl Reader for MalformedDependencies {
    const AVAILABLE: bool = true;

    fn read_tensor(
        &mut self,
        _: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        Ok(match self.read {
            MalformedRead::WrongTensor => read::event::ReadTensorDone::new(1, 4),
            MalformedRead::WrongBytes => read::event::ReadTensorDone::new(0, 3),
        })
    }
}

impl Stager for MalformedDependencies {
    const AVAILABLE: bool = true;

    fn stage_tensor(
        &mut self,
        _: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        Ok(staged_read::event::StageWindowDone::new(self.staged_bytes))
    }
}

#[test]
fn malformed_replacement_successes_fail_without_committing_residency() {
    for read in [MalformedRead::WrongTensor, MalformedRead::WrongBytes] {
        let dependencies = MalformedDependencies {
            read,
            staged_bytes: 4,
        };
        let mut store = Store::with_dependencies(1, dependencies).unwrap();
        store.process_event(BindStorage::new(storage(4))).unwrap();
        assert_eq!(
            dispatch_without_allocation!(store.process_event(ReadLoad::new(
                0,
                "fixture.bin",
                Some(&[1; 4]),
                0,
                4,
            ))),
            Err(Error::DependencyContract)
        );
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            Lifecycle::Unbound
        );
    }

    let dependencies = MalformedDependencies {
        read: MalformedRead::WrongBytes,
        staged_bytes: 3,
    };
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(StagedLoad::new(
            0,
            Some(&[1; 4]),
            0,
            4,
            2,
        ))),
        Err(Error::DependencyContract)
    );
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Unbound
    );
}

#[derive(Default)]
struct ReplacementIo {
    read_calls: usize,
    staged_calls: usize,
}

impl Mapper for ReplacementIo {
    const AVAILABLE: bool = false;

    fn map_tensor(
        &mut self,
        _: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn release_mapping(
        &mut self,
        _: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn with_mapping<Operation>(
        &mut self,
        _: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        Err(mmap::event::Error::UnsupportedPlatform)
    }
}

impl Reader for ReplacementIo {
    const AVAILABLE: bool = true;

    fn read_tensor(
        &mut self,
        event: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        self.read_calls += 1;
        assert_eq!(event.tensor_id(), 0);
        assert_eq!(event.file_index(), 0);
        assert_eq!(event.file_path(), "fixture.bin");
        assert_eq!(event.file_offset(), 1);
        assert_eq!(event.byte_size(), 4);
        assert_eq!(event.source_error(), None);
        let source = event.source().ok_or(read::event::Error::FileOpenFailed)?;
        let start = usize::try_from(event.file_offset())
            .map_err(|_| read::event::Error::UnsupportedResource)?;
        let len = usize::try_from(event.byte_size())
            .map_err(|_| read::event::Error::UnsupportedResource)?;
        let end = start
            .checked_add(len)
            .ok_or(read::event::Error::UnsupportedResource)?;
        let bytes = source
            .get(start..end)
            .ok_or(read::event::Error::ShortRead)?;
        event
            .target()
            .try_copy_from(bytes)
            .map_err(|_| read::event::Error::InvalidRequest)?;
        let done = read::event::ReadTensorDone::new(event.tensor_id(), event.byte_size());
        if let Some(callback) = event.done_callback() {
            callback.publish(done);
        }
        Ok(done)
    }
}

impl Stager for ReplacementIo {
    const AVAILABLE: bool = true;

    fn stage_tensor(
        &mut self,
        event: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        self.staged_calls += 1;
        assert_eq!(event.file_offset(), 2);
        assert_eq!(event.logical_byte_length(), 4);
        assert_eq!(event.stage_chunk_bytes(), 3);
        let source = event
            .source()
            .ok_or(staged_read::event::Error::NullSourceSpan)?;
        event
            .target()
            .try_copy_from(source)
            .map_err(|_| staged_read::event::Error::InvalidTargetWindow)?;
        let done = staged_read::event::StageWindowDone::new(event.logical_byte_length());
        if let Some(callback) = event.done_callback() {
            callback.publish(done);
        }
        Ok(done)
    }
}

#[test]
fn replacement_actor_and_prepared_dispatches_are_static_and_allocation_free() {
    let mut store = Store::with_dependencies(1, ReplacementIo::default()).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();
    let outcome = Cell::new(None);
    let allocation = measure(|| {
        outcome.set(Some(store.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[9, 4, 3, 2, 1]),
            1,
            4,
        ))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(outcome.get().unwrap().unwrap().bytes_copied(), 4);
    assert_eq!(store.dependencies().read_calls, 1);

    let result = Cell::new(None);
    let mut sum = |bytes: &[u8]| bytes.iter().copied().map(u16::from).sum::<u16>();
    let allocation = measure(|| {
        result.set(Some(store.process_event(WithTensor::new(0, &mut sum))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(result.get(), Some(Ok(10)));

    store.process_event(BindStorage::new(storage(4))).unwrap();
    let staged = Cell::new(None);
    let allocation = measure(|| {
        staged.set(Some(store.process_event(StagedLoad::new(
            0,
            Some(&[9, 8, 7, 6, 5, 4]),
            2,
            4,
            3,
        ))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(staged.get().unwrap().unwrap().bytes_copied(), 4);
    assert_eq!(store.dependencies().staged_calls, 1);
    let mut copy = |bytes: &[u8]| <[u8; 4]>::try_from(bytes).unwrap();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut copy))).unwrap(),
        [7, 6, 5, 4]
    );

    let error = Cell::new(None);
    let allocation = measure(|| {
        error.set(Some(store.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[0; 4]),
            0,
            4,
        ))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(error.get(), Some(Err(Error::TensorAlreadyResident)));
}

#[test]
fn public_io_values_and_absent_dependency_adapters_are_complete() {
    for (error, text) in [
        (
            Error::MmapUnavailable,
            "tensor mapper capability unavailable",
        ),
        (
            Error::ReadUnavailable,
            "tensor reader capability unavailable",
        ),
        (
            Error::StagerUnavailable,
            "tensor stager capability unavailable",
        ),
        (
            Error::Mmap(mmap::event::Error::InvalidRequest),
            "tensor mapped I/O failed",
        ),
        (
            Error::Read(read::event::Error::InvalidRequest),
            "tensor read I/O failed",
        ),
        (
            Error::Staged(staged_read::event::Error::InvalidStageContract),
            "tensor staged I/O failed",
        ),
        (
            Error::DependencyContractCleanup {
                mapping_handle: 7,
                error: mmap::event::Error::UnmapFailed,
            },
            "tensor dependency returned an invalid mapping and cleanup failed",
        ),
    ] {
        assert_eq!(error.to_string(), text);
    }

    let metadata = TensorMetadata::new(7, 2, 1, 3);
    let entry = StorageEntry::new(metadata, Some(vec![5, 6].into_boxed_slice()));
    assert_eq!(entry.metadata(), metadata);
    assert_eq!(entry.initial_bytes(), Some([5, 6].as_slice()));
    let storage = StorageBatch::new(vec![entry].into_boxed_slice());
    assert_eq!(storage.len(), 1);
    assert!(!storage.is_empty());
    assert_eq!(storage.into_entries().len(), 1);
    assert!(StorageBatch::new(Box::default()).is_empty());

    let mut effects = EffectBuffer::new(
        vec![EffectRequest::None {
            tensor_id: 0,
            file_index: 1,
            offset: 2,
            size: 3,
        }]
        .into_boxed_slice(),
    );
    assert!(!effects.is_empty());
    assert_eq!(effects.effects().len(), 1);
    effects.reset();
    assert_eq!(effects.into_effects().as_ref(), &[EffectRequest::Empty]);
    assert!(EffectBuffer::new(Box::default()).is_empty());

    let _ = ReadLoad::new(0, "fixture.bin", Some(&[1]), 0, 1)
        .with_file_index(2)
        .with_source_error(read::event::SourceError::FileOpenFailed);
    let mut operation = |bytes: &[u8]| bytes.len();
    assert!(format!("{:?}", WithTensor::new(7, &mut operation)).contains("tensor_id: 7"));

    let mut absent = ();
    assert_eq!(
        Mapper::release_mapping(&mut absent, mmap::event::ReleaseMapping::new(0, 1)),
        Err(mmap::event::Error::UnsupportedPlatform)
    );
    let mut operation = |bytes: &[u8]| bytes.len();
    assert_eq!(
        Mapper::with_mapping(
            &mut absent,
            mmap::event::WithMapping::new(0, 1, &mut operation)
        ),
        Err(mmap::event::Error::UnsupportedPlatform)
    );
    let mut target_bytes = [0_u8; 1];
    let target = read::event::Target::new(&mut target_bytes);
    assert_eq!(
        Reader::read_tensor(
            &mut absent,
            read::event::ReadTensor::new(0, "fixture.bin", Some(&[1]), &target)
        ),
        Err(read::event::Error::UnsupportedPlatform)
    );
    let mut target_bytes = [0_u8; 1];
    let target = staged_read::event::Target::new(&mut target_bytes);
    assert_eq!(
        Stager::stage_tensor(
            &mut absent,
            staged_read::event::StageWindow::new(0, 1, 1, Some(&[1]), &target)
        ),
        Err(staged_read::event::Error::UnsupportedPlatform)
    );
}

#[test]
fn direct_io_routes_preserve_an_outstanding_bulk_phase() {
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut store = Store::with_dependencies(1, dependencies).unwrap();
    store.process_event(BindStorage::new(storage(4))).unwrap();
    store
        .process_event(PlanLoad::new(
            StrategyKind::ReadCopy,
            EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice()),
        ))
        .unwrap();

    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReadLoad::new(
            0,
            "fixture.bin",
            Some(&[1; 4]),
            0,
            4
        ))),
        Err(Error::Busy)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(StagedLoad::new(
            0,
            Some(&[1; 4]),
            0,
            4,
            2
        ))),
        Err(Error::Busy)
    );
    assert_eq!(
        dispatch_without_allocation!(store.process_event(ReleaseMapped::new(0, 1))),
        Err(Error::Busy)
    );
    let mut access = |bytes: &[u8]| bytes.len();
    assert_eq!(
        dispatch_without_allocation!(store.process_event(WithTensor::new(0, &mut access))),
        Err(Error::Busy)
    );

    assert_eq!(
        store.process_event(ApplyEffectError::new(0, EffectError::Backend)),
        Err(Error::BackendError)
    );
    assert!(
        store
            .process_event(ReadLoad::new(0, "fixture.bin", Some(&[1; 4]), 0, 4))
            .is_ok()
    );
}
