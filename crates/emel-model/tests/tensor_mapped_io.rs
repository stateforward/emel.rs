//! Available mapper contract coverage for the public tensor actor.
use allocation_counter as _;
use emel_gguf as _;
use emel_kernels as _;
use emel_tensor as _;
use emel_token as _;
use sml as _;

use std::fs;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use emel_io::loader::event::StrategyKind;
use emel_io::mmap;
use emel_model::bridge::{
    Data, MimiDataInput, MimiHParams, MimiHParamsInput, TensorInput,
    TensorMetadata as ModelTensorMetadata, TensorMetadataInput,
};
use emel_model::loader::{Error as LoaderError, OwnedTensorLoader, TensorLoader};
use emel_model::tensor::{
    self,
    dependency::{Actors, Mapper},
    event::{
        BindStorage, CaptureTensorState, Error, Lifecycle, MappedLoad, ReleaseMapped, StorageBatch,
        StorageEntry, TensorMetadata, WithTensor,
    },
};
use emel_tensor::dtype::SerializedType;
#[derive(Clone, Copy)]
enum Mode {
    Success,
    WrongTensor,
    WrongLength,
    MapError,
    ReleaseError,
    AccessError,
}

struct MapperStub {
    mode: Mode,
    releases: Arc<AtomicU64>,
}

impl Mapper for MapperStub {
    const AVAILABLE: bool = true;
    fn map_tensor(
        &mut self,
        event: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        match self.mode {
            Mode::MapError => Err(mmap::event::Error::MappingFailed),
            Mode::WrongTensor => Ok(mmap::event::MapDone::new(
                41,
                event.tensor_id() + 1,
                event.len(),
            )),
            Mode::WrongLength => Ok(mmap::event::MapDone::new(
                41,
                event.tensor_id(),
                event.len() - 1,
            )),
            _ => Ok(mmap::event::MapDone::new(
                41,
                event.tensor_id(),
                event.len(),
            )),
        }
    }
    fn release_mapping(
        &mut self,
        event: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        self.releases.fetch_add(1, Ordering::Relaxed);
        assert_eq!(event.handle(), 41);
        if matches!(self.mode, Mode::ReleaseError) {
            Err(mmap::event::Error::UnmapFailed)
        } else {
            Ok(())
        }
    }
    fn with_mapping<Operation>(
        &mut self,
        event: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        if matches!(self.mode, Mode::AccessError) {
            Err(mmap::event::Error::MappingFailed)
        } else {
            Ok(event.apply(&[9, 8, 7, 6]))
        }
    }
}

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[allow(
    unsafe_code,
    reason = "the private fixture remains immutable for every clone and mapping lifetime"
)]
fn source() -> mmap::event::MmapSource {
    let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("emel-model-mapped-{}-{id}.bin", std::process::id()));
    fs::write(&path, [0_u8; 16]).unwrap();
    // SAFETY: this private fixture remains immutable for every clone and mapping lifetime.
    let source = unsafe { mmap::event::MmapSource::open(&path) }.unwrap();
    fs::remove_file(path).unwrap();
    source
}

fn request() -> MappedLoad {
    MappedLoad::new(0, source(), 0, 4)
}
fn storage() -> StorageBatch {
    StorageBatch::new(
        vec![StorageEntry::new(
            TensorMetadata::new(0, 4, 0, 0),
            Some(vec![0; 4].into_boxed_slice()),
        )]
        .into_boxed_slice(),
    )
}

#[test]
fn available_mapper_maps_accesses_and_releases() {
    let releases = Arc::new(AtomicU64::new(0));
    let mut store = tensor::Store::with_dependencies(
        1,
        Actors::new(
            MapperStub {
                mode: Mode::Success,
                releases: Arc::clone(&releases),
            },
            (),
            (),
        ),
    )
    .unwrap();
    store.process_event(BindStorage::new(storage())).unwrap();
    let done = store.process_event(request()).unwrap();
    assert_eq!(
        (done.tensor_id(), done.mapping_handle(), done.mapped_bytes()),
        (0, 41, 4)
    );
    let mut op = |bytes: &[u8]| bytes.len();
    assert_eq!(store.process_event(WithTensor::new(0, &mut op)), Ok(4));
    assert_eq!(
        store.process_event(ReleaseMapped::new(0, 99)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        store
            .process_event(ReleaseMapped::new(0, 41))
            .unwrap()
            .tensor_id(),
        0
    );
    assert_eq!(releases.load(Ordering::Relaxed), 1);
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::Evicted
    );
}

#[test]
fn available_mapper_classifies_malformed_and_cleanup_failures() {
    for (mode, expected) in [
        (
            Mode::MapError,
            Error::Mmap(mmap::event::Error::MappingFailed),
        ),
        (Mode::WrongTensor, Error::DependencyContract),
        (Mode::WrongLength, Error::DependencyContract),
    ] {
        let releases = Arc::new(AtomicU64::new(0));
        let mut store =
            tensor::Store::with_dependencies(1, Actors::new(MapperStub { mode, releases }, (), ()))
                .unwrap();
        store.process_event(BindStorage::new(storage())).unwrap();
        assert_eq!(store.process_event(request()), Err(expected));
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            Lifecycle::Unbound
        );
    }
    let releases = Arc::new(AtomicU64::new(0));
    let mut store = tensor::Store::with_dependencies(
        1,
        Actors::new(
            MapperStub {
                mode: Mode::ReleaseError,
                releases,
            },
            (),
            (),
        ),
    )
    .unwrap();
    store.process_event(BindStorage::new(storage())).unwrap();
    store.process_event(request()).unwrap();
    assert_eq!(
        store.process_event(ReleaseMapped::new(0, 41)),
        Err(Error::Mmap(mmap::event::Error::UnmapFailed))
    );
    assert_eq!(
        store
            .process_event(CaptureTensorState::new(0))
            .unwrap()
            .lifecycle(),
        Lifecycle::MappedResident
    );
}

#[test]
fn available_mapper_access_and_release_errors_propagate() {
    let releases = Arc::new(AtomicU64::new(0));
    let mut store = tensor::Store::with_dependencies(
        1,
        Actors::new(
            MapperStub {
                mode: Mode::AccessError,
                releases,
            },
            (),
            (),
        ),
    )
    .unwrap();
    store.process_event(BindStorage::new(storage())).unwrap();
    store.process_event(request()).unwrap();
    let mut op = |bytes: &[u8]| bytes.len();
    assert_eq!(
        store.process_event(WithTensor::new(0, &mut op)),
        Err(Error::Mmap(mmap::event::Error::MappingFailed))
    );
    assert_eq!(
        store
            .process_event(ReleaseMapped::new(0, 41))
            .unwrap()
            .tensor_id(),
        0
    );
}

#[test]
fn owned_loader_rolls_back_partial_mapped_load_and_can_be_reused() {
    let mapped_file = source();
    let mapped_files = [mapped_file];
    let mut invalid_model = mapped_model(16);
    let mut loader = OwnedTensorLoader::try_new(&invalid_model).unwrap();
    let source = emel_model::loader::event::Source::new("fixture.gguf", None)
        .with_mapped_files(&mapped_files);

    assert_eq!(
        loader.load(&mut invalid_model, source, StrategyKind::MappedFile),
        Err(LoaderError::BackendError)
    );
    assert_eq!(loader.mapping_handle(0), None);
    assert_eq!(loader.mapping_handle(1), None);
    assert_eq!(
        loader.tensor_state(0).unwrap().lifecycle(),
        Lifecycle::Evicted
    );
    assert_eq!(
        loader.tensor_state(1).unwrap().lifecycle(),
        Lifecycle::Unbound
    );

    let mut repaired_model = mapped_model(0);
    let source = emel_model::loader::event::Source::new("fixture.gguf", None)
        .with_mapped_files(&mapped_files);
    let stats = loader
        .load(&mut repaired_model, source, StrategyKind::MappedFile)
        .unwrap();
    assert_eq!(stats.bytes_total(), 8);
    assert_eq!(stats.bytes_done(), 8);
    assert!(stats.used_mmap());
    assert_eq!(loader.mapping_handle(0), Some(0));
    assert_eq!(loader.mapping_handle(1), Some(1));
    assert_eq!(
        loader.tensor_state(0).unwrap().lifecycle(),
        Lifecycle::MappedResident
    );
    assert_eq!(
        loader.tensor_state(1).unwrap().lifecycle(),
        Lifecycle::MappedResident
    );
    loader.release_all_mapped().unwrap();
}

fn valid_mimi_hparams() -> MimiHParams {
    MimiHParams::try_new(MimiHParamsInput {
        sample_rate: 24_000,
        frame_rate: 12.5,
        n_q: 2,
        card: 32,
        dim: 16,
        semantic_n_q: 1,
        codebook_dim: 8,
        transformer_num_layers: 2,
        transformer_num_heads: 2,
        transformer_context: 8,
        transformer_max_period: 1_000,
    })
    .unwrap()
}

fn mapped_model(second_file_offset: u64) -> Data {
    let metadata = |file_offset| {
        ModelTensorMetadata::new(TensorMetadataInput {
            tensor_type: SerializedType::F32,
            dimension_count: 1,
            dimensions: [1, 1, 1, 1],
            data_offset: file_offset,
            file_offset,
            data_size: 4,
            file_index: 0,
            storage: None,
        })
    };
    let tensors = [
        TensorInput::new(b"tensor.zero", metadata(0)),
        TensorInput::new(b"tensor.one", metadata(second_file_offset)),
    ];
    Data::try_from_mimi(MimiDataInput {
        hparams: valid_mimi_hparams(),
        tensors: &tensors,
    })
    .unwrap()
}
