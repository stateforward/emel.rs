//! Sortformer actor wrapper and explicit guard/action implementation.

#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps
)]

use core::cell::{Cell, RefCell};
use core::fmt;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{ParseDone, TensorDescriptor as GgufTensorDescriptor, WithTensor};

use crate::catalog::Catalog;
use crate::catalog::event::{
    BindStorage, FindTensor, ModelIdentity, SealModel, Storage as CatalogStorage, TensorDescriptor,
};

use super::event::{
    self, ContractDescriptor, Error, Family, FamilyDescriptor, LoadError, ObservationRecord,
    Parameters, Storage,
};
use super::sm::{
    SortformerMachineEvents, SortformerMachineStateMachine, SortformerMachineStateMachineContext,
    SortformerMachineStates,
};

#[derive(Clone, Copy, Debug, Default)]
struct FamilyRecord {
    tensor_count: u32,
    first: Option<TensorDescriptor>,
    name_offset: usize,
    name_length: usize,
}

#[derive(Clone, Copy)]
pub(super) struct SourceObservation {
    name_length: usize,
    descriptor: GgufTensorDescriptor,
    payload_present: bool,
}

pub(super) fn capture_source_observation(
    names: &mut [u8],
    offset: usize,
    name: &[u8],
    descriptor: GgufTensorDescriptor,
    payload: &[u8],
) -> Result<SourceObservation, LoadError> {
    let end = offset.checked_add(name.len()).ok_or(LoadError::Capacity)?;
    names
        .get_mut(offset..end)
        .ok_or(LoadError::Capacity)?
        .copy_from_slice(name);
    Ok(SourceObservation {
        name_length: name.len(),
        descriptor,
        payload_present: !payload.is_empty(),
    })
}

fn source_descriptor_matches_catalog(
    source: GgufTensorDescriptor,
    catalog: TensorDescriptor,
) -> Result<bool, LoadError> {
    let dimension_count =
        i32::try_from(source.dimension_count()).map_err(|_| LoadError::Internal)?;
    let mut dimensions = [0_i64; 4];
    for (destination, source) in dimensions.iter_mut().zip(source.dimensions()) {
        *destination = i64::try_from(source).map_err(|_| LoadError::Internal)?;
    }
    Ok(source.tensor_type() == catalog.tensor_type()
        && dimension_count == catalog.dimension_count()
        && dimensions == catalog.dimensions()
        && source.data_size() == catalog.data_size())
}

pub(super) struct Context {
    _source: GgufLoader,
    _catalog: Catalog,
    storage: Option<Storage>,
    observation_count: u32,
    used: usize,
    processed_count: u32,
    model: ModelIdentity,
    parameters: Parameters,
    families: [FamilyRecord; 4],
}

#[derive(Clone, Copy)]
pub(super) struct BeginRuntime<'a> {
    event: event::ContractBegin,
    result: &'a Cell<Result<(), Error>>,
}
#[derive(Clone, Copy)]
pub(super) struct ObserveRuntime<'a> {
    index: u32,
    result: &'a Cell<Result<(), Error>>,
}
#[derive(Clone, Copy)]
pub(super) struct FinishRuntime<'a> {
    result: &'a Cell<Result<(), Error>>,
}
#[derive(Clone, Copy)]
pub(super) struct VisitRuntime<'a> {
    result: &'a Cell<Result<ContractDescriptor, Error>>,
}
#[derive(Clone, Copy)]
pub(super) struct ResetRuntime<'a> {
    result: &'a Cell<Result<(), Error>>,
}
pub(super) struct ReleaseRuntime<'a> {
    storage: &'a RefCell<Option<Storage>>,
    result: &'a Cell<Result<(), Error>>,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct UnexpectedRuntime;

pub struct Sortformer {
    machine: SortformerMachineStateMachine<Context>,
}

impl Sortformer {
    /// Constructs a source-bound actor from one parsed GGUF loader.
    ///
    /// The actor retains this exact loader for every descriptor lifetime. It
    /// preallocates scan storage before runtime dispatch can begin.
    ///
    /// # Errors
    ///
    /// Returns a typed construction failure when metadata, catalog binding,
    /// source traversal, or preallocation fails.
    pub fn load(
        mut loader: GgufLoader,
        parsed: ParseDone,
        mut storage: Storage,
    ) -> Result<Self, LoadError> {
        let parameters = super::load_hparams(&mut loader).map_err(LoadError::Hparams)?;
        let catalog_storage =
            CatalogStorage::from_gguf(&mut loader, parsed).map_err(LoadError::Catalog)?;
        let mut catalog = Catalog::try_new().map_err(|_| LoadError::Capacity)?;
        catalog
            .process_event(BindStorage::new(catalog_storage))
            .map_err(|error| LoadError::Catalog(error.error()))?;
        let model = catalog
            .process_event(SealModel::new())
            .map_err(LoadError::Catalog)?;

        let tensor_count =
            usize::try_from(parsed.tensor_count()).map_err(|_| LoadError::Capacity)?;
        let mut total_name_bytes = 0usize;
        for index in 0..parsed.tensor_count() {
            let name_length = loader
                .process_event(WithTensor::new(
                    index,
                    |name: &[u8], _: emel_gguf::event::TensorDescriptor, _: &[u8]| name.len(),
                ))
                .map_err(LoadError::Gguf)?
                .ok_or(LoadError::Internal)?;
            total_name_bytes = total_name_bytes
                .checked_add(name_length)
                .ok_or(LoadError::Capacity)?;
        }
        storage.observation_names.clear();
        storage.observations.clear();
        storage
            .observation_names
            .try_reserve_exact(total_name_bytes)
            .map_err(|_| LoadError::Capacity)?;
        storage.observation_names.resize(total_name_bytes, 0);
        storage
            .observations
            .try_reserve_exact(tensor_count)
            .map_err(|_| LoadError::Capacity)?;

        let mut offset = 0usize;
        for index in 0..parsed.tensor_count() {
            let source = loader
                .process_event(WithTensor::new(
                    index,
                    |name: &[u8], descriptor, payload: &[u8]| {
                        capture_source_observation(
                            &mut storage.observation_names,
                            offset,
                            name,
                            descriptor,
                            payload,
                        )
                    },
                ))
                .map_err(LoadError::Gguf)?
                .ok_or(LoadError::Internal)??;
            let end = offset
                .checked_add(source.name_length)
                .ok_or(LoadError::Capacity)?;
            let name = storage
                .observation_names
                .get(offset..end)
                .ok_or(LoadError::Capacity)?;
            let tensor = catalog
                .process_event(FindTensor::new(model, name))
                .map_err(LoadError::Catalog)?
                .ok_or(LoadError::Internal)?;
            if !source_descriptor_matches_catalog(source.descriptor, tensor)? {
                return Err(LoadError::Internal);
            }
            storage.observations.push(ObservationRecord {
                name_offset: offset,
                name_length: source.name_length,
                tensor,
                payload_present: source.payload_present,
            });
            offset = end;
        }

        Ok(Self {
            machine: SortformerMachineStateMachine::new(Context {
                _source: loader,
                _catalog: catalog,
                storage: Some(storage),
                observation_count: parsed.tensor_count(),
                used: 0,
                processed_count: 0,
                model,
                parameters,
                families: [FamilyRecord::default(); 4],
            }),
        })
    }

    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn contract_begin(&mut self, event: event::ContractBegin) -> Result<(), Error> {
        let result = Cell::new(Err(Error::Internal));
        self.machine
            .process_event(SortformerMachineEvents::Begin(BeginRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        let observation_count = self.machine.context().observation_count();
        for index in 0..observation_count {
            self.machine
                .process_event(SortformerMachineEvents::Observe(ObserveRuntime {
                    index,
                    result: &result,
                }))
                .map_err(|_| Error::Internal)?;
        }
        self.machine
            .process_event(SortformerMachineEvents::Finish(FinishRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn contract_visit(
        &mut self,
        _: event::ContractVisit,
    ) -> Result<ContractDescriptor, Error> {
        let result = Cell::new(Err(Error::Internal));
        self.machine
            .process_event(SortformerMachineEvents::Visit(VisitRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn contract_reset(&mut self, _: event::ContractReset) -> Result<(), Error> {
        let result = Cell::new(Err(Error::Internal));
        self.machine
            .process_event(SortformerMachineEvents::Reset(ResetRuntime {
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()
    }
    pub(crate) fn storage_release(&mut self, _: event::StorageRelease) -> Result<Storage, Error> {
        let storage = RefCell::new(None);
        let result = Cell::new(Err(Error::Internal));
        self.machine
            .process_event(SortformerMachineEvents::Release(ReleaseRuntime {
                storage: &storage,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.get()?;
        storage.into_inner().ok_or(Error::Internal)
    }

    pub(crate) fn with_first_name<F, R>(&self, event: &mut event::WithFirstName<F, R>)
    where
        F: for<'name> FnMut(&'name [u8]) -> R,
    {
        super::query::process(
            self.machine.is(&SortformerMachineStates::StateEmpty),
            self.machine.is(&SortformerMachineStates::StateScanning),
            self.machine.is(&SortformerMachineStates::StateReady),
            self.machine.context(),
            event,
        );
    }

    #[cfg(test)]
    pub(super) fn first_name(&self, family: Family) -> &[u8] {
        self.machine.context().first_name(family)
    }
}

impl fmt::Debug for Sortformer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Sortformer").finish_non_exhaustive()
    }
}

impl Context {
    pub(super) fn storage_available(&self) -> bool {
        self.storage.is_some()
    }
    fn record(&self, family: Family) -> FamilyRecord {
        self.families[family as usize]
    }
    pub(super) fn first_name(&self, family: Family) -> &[u8] {
        let record = self.record(family);
        let end = record
            .name_offset
            .checked_add(record.name_length)
            .expect("ready-state family name range");
        self.storage
            .as_ref()
            .expect("ready-state storage invariant")
            .names
            .get(record.name_offset..end)
            .expect("ready-state family name invariant")
    }
    fn observation(&self, event: &ObserveRuntime<'_>) -> Option<ObservationRecord> {
        let index = usize::try_from(event.index).ok()?;
        self.storage.as_ref()?.observations.get(index).copied()
    }
    fn observation_count(&self) -> u32 {
        self.observation_count
    }
    fn observation_name(&self, event: &ObserveRuntime<'_>) -> Option<&[u8]> {
        let record = self.observation(event)?;
        let end = record.name_offset.checked_add(record.name_length)?;
        self.storage
            .as_ref()?
            .observation_names
            .get(record.name_offset..end)
    }
    fn observation_valid(&self, event: &ObserveRuntime<'_>) -> bool {
        let Some(record) = self.observation(event) else {
            return false;
        };
        let dimensions = record.tensor.dimensions();
        let count = usize::try_from(record.tensor.dimension_count()).unwrap_or(usize::MAX);
        record.payload_present
            && record.tensor.data_size() > 0
            && count > 0
            && count <= dimensions.len()
            && dimensions[..count].iter().all(|value| *value > 0)
    }
    fn matches(&self, event: &ObserveRuntime<'_>, family: Family) -> bool {
        self.observation_name(event)
            .is_some_and(|name| name.starts_with(family.prefix()))
            && self.observation_valid(event)
    }
    fn first_fits(&self, event: &ObserveRuntime<'_>, family: Family) -> bool {
        self.matches(event, family)
            && self.record(family).tensor_count == 0
            && self.storage.as_ref().is_some_and(|storage| {
                self.used
                    .checked_add(self.observation_name(event).map_or(usize::MAX, <[u8]>::len))
                    .is_some_and(|end| end <= storage.names.len())
            })
    }
    fn first_capacity(&self, event: &ObserveRuntime<'_>, family: Family) -> bool {
        self.matches(event, family)
            && self.record(family).tensor_count == 0
            && !self.first_fits(event, family)
    }
    fn additional(&self, event: &ObserveRuntime<'_>, family: Family) -> bool {
        self.matches(event, family) && self.record(family).tensor_count > 0
    }
    fn store_first(&mut self, event: ObserveRuntime<'_>, family: Family) -> Result<(), ()> {
        let observation = self
            .observation(&event)
            .expect("first-fits guard proves observation index");
        let name_offset = observation.name_offset;
        let name_length = observation.name_length;
        let end = self
            .used
            .checked_add(name_length)
            .expect("first-fits guard proves name offset");
        let storage = self
            .storage
            .as_mut()
            .expect("first-fits guard proves bound storage");
        let name_end = name_offset
            .checked_add(name_length)
            .expect("bound observation name range");
        storage.names[self.used..end]
            .copy_from_slice(&storage.observation_names[name_offset..name_end]);
        self.families[family as usize] = FamilyRecord {
            tensor_count: 1,
            first: Some(observation.tensor),
            name_offset: self.used,
            name_length,
        };
        self.used = end;
        self.processed_count = self
            .processed_count
            .checked_add(1)
            .expect("source tensor count fits u32");
        event.result.set(Ok(()));
        Ok(())
    }
    fn increment(&mut self, event: ObserveRuntime<'_>, family: Family) -> Result<(), ()> {
        let record = &mut self.families[family as usize];
        record.tensor_count = record
            .tensor_count
            .checked_add(1)
            .expect("GGUF tensor count fits u32");
        self.processed_count = self
            .processed_count
            .checked_add(1)
            .expect("source tensor count fits u32");
        event.result.set(Ok(()));
        Ok(())
    }
}

macro_rules! family_guards {
    ($first:ident, $capacity:ident, $additional:ident, $family:expr) => {
        fn $first(&self, event: &ObserveRuntime<'_>) -> Result<bool, ()> {
            Ok(self.first_fits(event, $family))
        }
        fn $capacity(&self, event: &ObserveRuntime<'_>) -> Result<bool, ()> {
            Ok(self.first_capacity(event, $family))
        }
        fn $additional(&self, event: &ObserveRuntime<'_>) -> Result<bool, ()> {
            Ok(self.additional(event, $family))
        }
    };
}

impl SortformerMachineStateMachineContext for Context {
    fn guard_begin_valid(&self, _event: &BeginRuntime<'_>) -> Result<bool, ()> {
        let p = self.parameters;
        Ok(self.storage.is_some()
            && p.original_tensor_count > 0
            && p.tensor_count > 0
            && p.skipped_tensor_count >= 0
            && p.sample_rate == super::SAMPLE_RATE
            && p.speaker_count == super::SPEAKER_COUNT
            && p.chunk_len == super::CHUNK_LEN
            && p.chunk_right_context == super::CHUNK_RIGHT_CONTEXT
            && p.fifo_len == super::FIFO_LEN
            && p.spkcache_update_period == super::SPKCACHE_UPDATE_PERIOD
            && p.spkcache_len == super::SPKCACHE_LEN)
    }
    fn guard_begin_invalid(&self, event: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some() && !self.guard_begin_valid(event)?)
    }
    fn guard_begin_storage_unavailable(&self, _: &BeginRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        self.used = 0;
        self.processed_count = 0;
        self.families.fill(FamilyRecord::default());
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_model_invalid(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_storage_unavailable_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_busy_begin(&mut self, event: BeginRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    family_guards!(
        guard_feature_first_fits,
        guard_feature_first_capacity,
        guard_feature_additional,
        Family::FeatureExtractor
    );
    family_guards!(
        guard_encoder_first_fits,
        guard_encoder_first_capacity,
        guard_encoder_additional,
        Family::Encoder
    );
    family_guards!(
        guard_modules_first_fits,
        guard_modules_first_capacity,
        guard_modules_additional,
        Family::Modules
    );
    family_guards!(
        guard_transformer_first_fits,
        guard_transformer_first_capacity,
        guard_transformer_additional,
        Family::TransformerEncoder
    );

    fn guard_observation_ignored(&self, event: &ObserveRuntime<'_>) -> Result<bool, ()> {
        Ok(self.observation(event).is_some()
            && !self.first_fits(event, Family::FeatureExtractor)
            && !self.first_capacity(event, Family::FeatureExtractor)
            && !self.additional(event, Family::FeatureExtractor)
            && !self.first_fits(event, Family::Encoder)
            && !self.first_capacity(event, Family::Encoder)
            && !self.additional(event, Family::Encoder)
            && !self.first_fits(event, Family::Modules)
            && !self.first_capacity(event, Family::Modules)
            && !self.additional(event, Family::Modules)
            && !self.first_fits(event, Family::TransformerEncoder)
            && !self.first_capacity(event, Family::TransformerEncoder)
            && !self.additional(event, Family::TransformerEncoder))
    }
    fn guard_observation_index_invalid(&self, event: &ObserveRuntime<'_>) -> Result<bool, ()> {
        Ok(self.observation(event).is_none())
    }
    fn effect_feature_first(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.store_first(event, Family::FeatureExtractor)
    }
    fn effect_encoder_first(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.store_first(event, Family::Encoder)
    }
    fn effect_modules_first(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.store_first(event, Family::Modules)
    }
    fn effect_transformer_first(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.store_first(event, Family::TransformerEncoder)
    }
    fn effect_feature_additional(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.increment(event, Family::FeatureExtractor)
    }
    fn effect_encoder_additional(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.increment(event, Family::Encoder)
    }
    fn effect_modules_additional(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.increment(event, Family::Modules)
    }
    fn effect_transformer_additional(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.increment(event, Family::TransformerEncoder)
    }
    fn effect_capacity(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Capacity));
        Ok(())
    }
    fn effect_observation_ignored(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        self.processed_count = self
            .processed_count
            .checked_add(1)
            .expect("source tensor count fits u32");
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_preserve_observe(&mut self, _: ObserveRuntime<'_>) -> Result<(), ()> {
        Ok(())
    }
    fn guard_storage_available_observe(&self, _: &ObserveRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some())
    }
    fn guard_storage_unavailable_observe(&self, _: &ObserveRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_invalid_request(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_storage_unavailable_observe(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_busy_observe(&mut self, event: ObserveRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_families_complete(&self, _: &FinishRuntime<'_>) -> Result<bool, ()> {
        Ok(self.processed_count == self.observation_count()
            && self
                .families
                .iter()
                .all(|family| family.tensor_count > 0 && family.first.is_some()))
    }
    fn guard_families_incomplete(&self, event: &FinishRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_families_complete(event)?)
    }
    fn effect_finish(&mut self, event: FinishRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_model_invalid_finish(&mut self, event: FinishRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_preserve_finish(&mut self, _: FinishRuntime<'_>) -> Result<(), ()> {
        Ok(())
    }
    fn guard_storage_available_finish(&self, _: &FinishRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some())
    }
    fn guard_storage_unavailable_finish(&self, _: &FinishRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_invalid_finish(&mut self, event: FinishRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_storage_unavailable_finish(&mut self, event: FinishRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_busy_finish(&mut self, event: FinishRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn effect_visit(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        let feature = self.record(Family::FeatureExtractor);
        let encoder = self.record(Family::Encoder);
        let modules = self.record(Family::Modules);
        let transformer = self.record(Family::TransformerEncoder);
        let descriptors = [
            FamilyDescriptor::new(
                Family::FeatureExtractor,
                feature.tensor_count,
                feature.first.expect("ready feature tensor"),
            ),
            FamilyDescriptor::new(
                Family::Encoder,
                encoder.tensor_count,
                encoder.first.expect("ready encoder tensor"),
            ),
            FamilyDescriptor::new(
                Family::Modules,
                modules.tensor_count,
                modules.first.expect("ready modules tensor"),
            ),
            FamilyDescriptor::new(
                Family::TransformerEncoder,
                transformer.tensor_count,
                transformer.first.expect("ready transformer tensor"),
            ),
        ];
        event.result.set(Ok(ContractDescriptor::new(
            self.model,
            self.parameters,
            &descriptors,
        )));
        Ok(())
    }
    fn guard_storage_available_visit(&self, _: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some())
    }
    fn guard_storage_unavailable_visit(&self, _: &VisitRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_invalid_visit(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_storage_unavailable(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_busy_visit(&mut self, event: VisitRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_storage_available_reset(&self, _: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some())
    }
    fn guard_storage_unavailable_reset(&self, _: &ResetRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_reset_storage(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        self.used = 0;
        self.processed_count = 0;
        self.families.fill(FamilyRecord::default());
        self.storage
            .as_mut()
            .expect("storage-available reset transition")
            .names
            .fill(0);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_reset_without_storage(&mut self, event: ResetRuntime<'_>) -> Result<(), ()> {
        self.used = 0;
        self.processed_count = 0;
        self.families.fill(FamilyRecord::default());
        event.result.set(Ok(()));
        Ok(())
    }
    fn guard_storage_available_release(&self, _: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_some())
    }
    fn guard_storage_unavailable_release(&self, _: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(self.storage.is_none())
    }
    fn effect_release_storage(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        *event.storage.borrow_mut() = Some(
            self.storage
                .take()
                .expect("storage-available release transition"),
        );
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_storage_unavailable_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_busy_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }
    fn guard_never(&self, _: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
