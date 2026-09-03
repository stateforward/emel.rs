//! Owning model-loader actor and static tensor dependency boundary.

use std::cell::{Cell, RefCell};
use std::fmt;

use super::event::{Error, LoadRequest, LoadStats};
use super::sm::{EventLoadRuntime, ModelLoaderContext, ModelLoaderEvents, ModelLoaderStateMachine};
use crate::data::Data;
use crate::tensor::Store;
use crate::tensor::dependency::Actors;
use crate::tensor::event::{
    ApplyBoundEffectResults, BindStorage, CaptureTensorState, EffectBuffer, EffectRequest,
    EvictTensor, MappedLoad, PlanLoad, ReadLoad, ReleaseMapped, StorageBatch, StorageEntry,
    StrategyKind as TensorStrategy,
};

/// Replaceable tensor residency actor used by [`ModelLoader`].
pub trait TensorLoader {
    /// Loads tensor bytes into the caller-owned model.
    ///
    /// # Errors
    ///
    /// Returns the typed loader error when the requested strategy or tensor data cannot be loaded.
    fn load(
        &mut self,
        model: &mut Data,
        source: super::event::Source<'_>,
        strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<LoadStats, Error>;
}

/// Default dependency. It explicitly reports the unavailable actor path.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTensorLoader;

impl TensorLoader for NoTensorLoader {
    fn load(
        &mut self,
        _model: &mut Data,
        _source: super::event::Source<'_>,
        _strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<LoadStats, Error> {
        Err(Error::IoStrategyUnavailable)
    }
}

type ResidencyStore =
    Store<Actors<emel_io::mmap::Mapper, emel_io::read::Reader, emel_io::staged_read::Stager>>;

/// Concrete owned tensor loader backed by the public tensor and I/O actors.
///
/// Storage, planning effects, and result slots are allocated by [`Self::try_new`].
/// Dispatch only transfers those existing allocations through typed actor events;
/// it does not acquire filesystem resources or grow a collection. Read requests
/// use the in-memory source capability, while mapped requests consume caller-owned,
/// safely opened split-file capabilities from [`super::event::Source`].
#[derive(Debug)]
pub struct OwnedTensorLoader {
    store: ResidencyStore,
    storage: Option<StorageBatch>,
    effects: Option<EffectBuffer>,
    results: Box<[Option<Box<[u8]>>]>,
    bound_ids: Option<Box<[i32]>>,
    mapping_handles: Box<[Option<u32>]>,
}

impl OwnedTensorLoader {
    /// Allocates residency slots and all load buffers for the model's tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] when model metadata cannot be represented
    /// by the bounded actor allocations.
    pub fn try_new(model: &Data) -> Result<Self, Error> {
        let count = usize::try_from(model.n_tensors).map_err(|_| Error::InvalidRequest)?;
        if count == 0 || count > crate::data::MAX_TENSORS {
            return Err(Error::InvalidRequest);
        }

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| Error::InternalError)?;
        let mut result_slots = Vec::new();
        result_slots
            .try_reserve_exact(count)
            .map_err(|_| Error::InternalError)?;
        let mut effect_slots = Vec::new();
        effect_slots
            .try_reserve_exact(count)
            .map_err(|_| Error::InternalError)?;
        let mut bound_ids = Vec::new();
        bound_ids
            .try_reserve_exact(count)
            .map_err(|_| Error::InternalError)?;

        for (index, tensor) in model.tensors.iter().take(count).enumerate() {
            let size = usize::try_from(tensor.data_size).map_err(|_| Error::InvalidRequest)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(size)
                .map_err(|_| Error::InternalError)?;
            bytes.resize(size, 0);
            entries.push(StorageEntry::new(
                crate::tensor::event::TensorMetadata::new(
                    tensor.file_offset,
                    tensor.data_size,
                    tensor.file_index,
                    tensor.r#type,
                ),
                Some(bytes.into_boxed_slice()),
            ));
            result_slots.push(None);
            effect_slots.push(EffectRequest::Empty);
            bound_ids.push(i32::try_from(index).map_err(|_| Error::InvalidRequest)?);
        }

        let store = Store::with_dependencies(
            count,
            Actors::new(
                emel_io::mmap::Mapper::new(),
                emel_io::read::Reader::new(),
                emel_io::staged_read::Stager::new(),
            ),
        )
        .map_err(|_| Error::InternalError)?;
        Ok(Self {
            store,
            storage: Some(StorageBatch::new(entries.into_boxed_slice())),
            effects: Some(EffectBuffer::new(effect_slots.into_boxed_slice())),
            results: result_slots.into_boxed_slice(),
            bound_ids: Some(bound_ids.into_boxed_slice()),
            mapping_handles: vec![None; count].into_boxed_slice(),
        })
    }
    fn load_prebound(&mut self, model: &mut Data) -> Result<LoadStats, Error> {
        let count = usize::try_from(model.n_tensors).map_err(|_| Error::InvalidRequest)?;
        if model.tensors.iter().take(count).any(|tensor| {
            tensor
                .bytes
                .as_deref()
                .is_none_or(|bytes| u64::try_from(bytes.len()).ok() != Some(tensor.data_size))
        }) {
            return Err(Error::InvalidRequest);
        }

        let storage = self.storage.take().ok_or(Error::InternalError)?;
        let mut entries = storage.into_entries();
        for (index, entry) in entries.iter_mut().enumerate() {
            entry.bytes = model.tensors[index].bytes.take();
        }
        let bind = self
            .store
            .process_event(BindStorage::new(StorageBatch::new(entries)));
        if let Err(error) = bind {
            let class = error.error();
            let mut entries = error.into_storage().into_entries();
            for (index, entry) in entries.iter_mut().enumerate() {
                model.tensors[index].bytes = entry.bytes.take();
            }
            self.storage = Some(StorageBatch::new(entries));
            return Err(Self::tensor_error(class));
        }

        let effects = self.effects.take().ok_or(Error::InternalError)?;
        let effects = match self
            .store
            .process_event(PlanLoad::new(TensorStrategy::None, effects))
        {
            Ok(done) => done.into_effects(),
            Err(error) => {
                let class = error.error();
                self.effects = Some(error.into_effects());
                return Err(Self::tensor_error(class));
            }
        };
        if effects
            .effects()
            .iter()
            .take(count)
            .any(|effect| !matches!(effect, EffectRequest::None { .. }))
        {
            self.effects = Some(effects);
            return Err(Error::BackendError);
        }

        let ids = self.bound_ids.take().ok_or(Error::InternalError)?;
        if let Err(error) = self.store.process_event(ApplyBoundEffectResults::new(ids)) {
            let class = error.error();
            self.effects = Some(effects);
            self.bound_ids = Some(error.into_tensor_ids());
            return Err(Self::tensor_error(class));
        }
        self.effects = Some(EffectBuffer::new(effects.into_effects()));

        for index in 0..count {
            let tensor_id = i32::try_from(index).map_err(|_| Error::InvalidRequest)?;
            let bytes = self
                .store
                .process_event(EvictTensor::new(tensor_id))
                .map_err(Self::tensor_error)?
                .into_bytes();
            model
                .install_tensor_bytes(
                    u32::try_from(index).map_err(|_| Error::InvalidRequest)?,
                    bytes,
                )
                .map_err(|_| Error::InvalidRequest)?;
        }
        Ok(LoadStats {
            bytes_total: model.weights_size,
            bytes_done: model.weights_size,
            used_mmap: false,
            used_strategy: emel_io::loader::event::StrategyKind::None,
        })
    }

    const fn strategy(
        strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<TensorStrategy, Error> {
        match strategy {
            emel_io::loader::event::StrategyKind::ReadCopy => Ok(TensorStrategy::ReadCopy),
            emel_io::loader::event::StrategyKind::StagedRead => Ok(TensorStrategy::StagedRead),
            emel_io::loader::event::StrategyKind::MappedFile => Ok(TensorStrategy::MappedFile),
            emel_io::loader::event::StrategyKind::None => Ok(TensorStrategy::None),
            emel_io::loader::event::StrategyKind::ExternalBuffer
            | emel_io::loader::event::StrategyKind::Unknown(_)
            | _ => Err(Error::IoStrategyUnavailable),
        }
    }

    const fn tensor_error(error: crate::tensor::event::Error) -> Error {
        match error {
            crate::tensor::event::Error::InvalidRequest => Error::InvalidRequest,
            crate::tensor::event::Error::UnsupportedStrategy
            | crate::tensor::event::Error::MmapUnavailable
            | crate::tensor::event::Error::ReadUnavailable
            | crate::tensor::event::Error::StagerUnavailable => Error::IoStrategyUnavailable,
            crate::tensor::event::Error::Capacity
            | crate::tensor::event::Error::TensorAlreadyResident
            | crate::tensor::event::Error::TensorUnbound
            | crate::tensor::event::Error::MappedTensorRequiresRelease
            | crate::tensor::event::Error::Busy
            | crate::tensor::event::Error::Mmap(_)
            | crate::tensor::event::Error::Read(_)
            | crate::tensor::event::Error::Staged(_)
            | crate::tensor::event::Error::DependencyContract
            | crate::tensor::event::Error::DependencyContractCleanup { .. }
            | crate::tensor::event::Error::BackendError
            | crate::tensor::event::Error::Internal => Error::BackendError,
        }
    }
}
impl OwnedTensorLoader {
    /// Returns the mapper-owned release token for a retained tensor mapping.
    #[must_use]
    pub fn mapping_handle(&self, tensor_id: usize) -> Option<u32> {
        self.mapping_handles.get(tensor_id).copied().flatten()
    }

    /// Releases retained mapped residency through the owning mapper actor.
    pub fn release_mapped(&mut self, tensor_id: i32, mapping_handle: u32) -> Result<(), Error> {
        match self.store.process_event(ReleaseMapped::new(tensor_id, mapping_handle)) {
            Ok(_) => {
                if let Ok(index) = usize::try_from(tensor_id) {
                    if self.mapping_handles.get(index).copied().flatten() == Some(mapping_handle) {
                        self.mapping_handles[index] = None;
                    }
                }
                Ok(())
            }
            Err(crate::tensor::event::Error::Mmap(error)) => {
                Err(Error::MappedReleaseFailed(error))
            }
            Err(error) => Err(Self::tensor_error(error)),
        }
    }

    /// Captures one tensor's lifecycle for explicit mapped ownership management.
    pub fn tensor_state(
        &mut self,
        tensor_id: i32,
    ) -> Result<crate::tensor::event::TensorState, Error> {
        self.store
            .process_event(CaptureTensorState::new(tensor_id))
            .map_err(Self::tensor_error)
    }
}

impl TensorLoader for OwnedTensorLoader {
    fn load(
        &mut self,
        model: &mut Data,
        source: super::event::Source<'_>,
        strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<LoadStats, Error> {
        if strategy == emel_io::loader::event::StrategyKind::None {
            return self.load_prebound(model);
        }
        let tensor_strategy = Self::strategy(strategy)?;
        let storage = self.storage.take().ok_or(Error::InternalError)?;
        let bind = self.store.process_event(BindStorage::new(storage));
        if let Err(error) = bind {
            let class = error.error();
            self.storage = Some(error.into_storage());
            return Err(Self::tensor_error(class));
        }

        let mut total = 0_u64;
        for (index, tensor) in model.tensors.iter().take(self.results.len()).enumerate() {
            let tensor_id = i32::try_from(index).map_err(|_| Error::InvalidRequest)?;
            match tensor_strategy {
                TensorStrategy::MappedFile => {
                    let files = source.mapped_files.ok_or(Error::IoStrategyUnavailable)?;
                    let mapped_source = files
                        .get(usize::from(tensor.file_index))
                        .ok_or(Error::InvalidRequest)?
                        .clone();
                    self.mapping_handles[index] = Some(
                        self.store
                            .process_event(
                                MappedLoad::new(
                                    tensor_id,
                                    mapped_source,
                                    tensor.file_offset,
                                    tensor.data_size,
                                )
                                .with_file_index(tensor.file_index),
                            )
                            .map_err(Self::tensor_error)?
                            .mapping_handle(),
                    );
                }
                TensorStrategy::ReadCopy => {
                    self.store
                        .process_event(ReadLoad::new(
                            tensor_id,
                            source.model_path,
                            source.file_image,
                            tensor.file_offset,
                            tensor.data_size,
                        )
                        .with_file_index(tensor.file_index))
                        .map_err(Self::tensor_error)?;
                }
                TensorStrategy::StagedRead => {
                    self.store
                        .process_event(crate::tensor::event::StagedLoad::new(
                            tensor_id,
                            source.file_image,
                            tensor.file_offset,
                            tensor.data_size,
                            emel_io::loader::event::StrategyPolicy::DEFAULT_STAGED_CHUNK_BYTES,
                        ))
                        .map_err(Self::tensor_error)?;
                }
                _ => return Err(Error::IoStrategyUnavailable),
            }
            if tensor_strategy != TensorStrategy::MappedFile {
                let bytes = self
                    .store
                    .process_event(EvictTensor::new(tensor_id))
                    .map_err(Self::tensor_error)?
                    .into_bytes();
                self.results[index] = Some(bytes);
            }
            total = total
                .checked_add(tensor.data_size)
                .ok_or(Error::InvalidRequest)?;
        }

        if tensor_strategy == TensorStrategy::MappedFile {
            return Ok(LoadStats {
                bytes_total: total,
                bytes_done: total,
                used_mmap: true,
                used_strategy: strategy,
            });
        }

        for (index, result) in self.results.iter_mut().enumerate() {
            let bytes = result.take().ok_or(Error::BackendError)?;
            model
                .install_tensor_bytes(
                    u32::try_from(index).map_err(|_| Error::InvalidRequest)?,
                    bytes,
                )
                .map_err(|_| Error::InvalidRequest)?;
        }
        Ok(LoadStats {
            bytes_total: total,
            bytes_done: total,
            used_mmap: false,
            used_strategy: strategy,
        })
    }
}

/// Single-writer, run-to-completion model loader.
pub struct ModelLoader<T = NoTensorLoader>
where
    T: TensorLoader,
{
    machine: ModelLoaderStateMachine<ModelLoaderContext<T>>,
}

impl ModelLoader<NoTensorLoader> {
    /// Creates a loader using the unavailable default tensor actor.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_tensor_loader(NoTensorLoader)
    }
}
impl OwnedTensorLoader {
    /// Releases all mapped tensor residency, preserving the first typed failure.
    pub fn release_all_mapped(&mut self) -> Result<(), Error> {
        for tensor_id in 0..self.mapping_handles.len() {
            if let Some(mapping_handle) = self.mapping_handles[tensor_id] {
                self.release_mapped(
                    i32::try_from(tensor_id).map_err(|_| Error::InvalidRequest)?,
                    mapping_handle,
                )?;
            }
        }
        Ok(())
    }
}

impl<T: TensorLoader> ModelLoader<T> {
    /// Creates a loader with the supplied tensor residency actor.
    #[must_use]
    pub const fn with_tensor_loader(tensor_loader: T) -> Self {
        Self {
            machine: ModelLoaderStateMachine::new(ModelLoaderContext::new(tensor_loader)),
        }
    }

    /// Processes one caller-owned request synchronously.
    ///
    /// # Errors
    ///
    /// Returns the typed loader failure reported while validating or loading
    /// the caller-owned model request.
    pub fn process_event(&mut self, request: LoadRequest<'_>) -> Result<LoadStats, Error> {
        let request = RefCell::new(request);
        let status = Cell::new(super::event::LoadStatus::default());
        let outcome = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(ModelLoaderEvents::Load(EventLoadRuntime {
                request: &request,
                status: &status,
                outcome: &outcome,
            }))
            .map_err(|_| Error::InternalError)?;
        outcome.get()
    }

    /// Reports whether the loader state machine is ready for another request.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&super::sm::ModelLoaderStates::Ready)
    }
}

impl Default for ModelLoader<NoTensorLoader> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TensorLoader> fmt::Debug for ModelLoader<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelLoader")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{OwnedTensorLoader, TensorLoader};
    use crate::data::{Data, TensorRecord};
    use crate::loader::event::{Error, Source};

    fn model_with_one_tensor() -> Data {
        let mut model = Data::try_new().expect("bounded model allocation");
        model.n_tensors = 1;
        model.tensors[0] = TensorRecord {
            n_dims: 1,
            dims: [1, 1, 1, 1],
            file_offset: 2,
            data_size: 4,
            r#type: 0,
            ..TensorRecord::default()
        };
        model
    }

    #[test]
    fn owned_loader_reads_source_bytes_and_installs_them() {
        let mut model = model_with_one_tensor();
        let mut loader = OwnedTensorLoader::try_new(&model).expect("loader allocation");
        let stats = loader
            .load(
                &mut model,
                Source::new("fixture.gguf", Some(&[9, 8, 1, 2, 3, 4, 7])),
                emel_io::loader::event::StrategyKind::ReadCopy,
            )
            .expect("read-copy load");
        assert_eq!(stats.bytes_total, 4);
        assert_eq!(stats.bytes_done, 4);
        assert_eq!(model.tensor(0).expect("tensor view").bytes(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn owned_loader_rejects_mapped_without_capability() {
        let mut model = model_with_one_tensor();
        let mut loader = OwnedTensorLoader::try_new(&model).expect("loader allocation");
        assert_eq!(
            loader.load(
                &mut model,
                Source::new("fixture.gguf", Some(&[0, 0, 1, 2, 3, 4])),
                emel_io::loader::event::StrategyKind::MappedFile,
            ),
            Err(Error::IoStrategyUnavailable)
        );
    }

    #[test]
    fn owned_loader_none_strategy_uses_prebound_bytes() {
        let mut model = model_with_one_tensor();
        model.weights_size = 4;
        model.tensors[0].bytes = Some(Box::new([4, 3, 2, 1]));
        let mut loader = OwnedTensorLoader::try_new(&model).expect("loader allocation");
        let stats = loader
            .load(&mut model, Source::new("", None), emel_io::loader::event::StrategyKind::None)
            .expect("prebound load");
        assert_eq!(stats.bytes_total, 4);
        assert_eq!(stats.bytes_done, 4);
        assert!(!stats.used_mmap);
        assert_eq!(model.tensor(0).expect("tensor view").bytes(), Some(&[4, 3, 2, 1][..]));
    }
}
