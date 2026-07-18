//! Public catalog actor and private dispatch runtimes.

use core::cell::{Cell, RefCell};
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use emel_tensor::dtype::SerializedType;

use super::event::{
    self, BindStorageError, Error, ModelDescriptor, ModelIdentity, Storage, TensorBindingStatus,
    TensorDescriptor, TensorId,
};
use super::sm::{
    CatalogMachineEvents, CatalogMachineStateMachine, CatalogMachineStateMachineContext,
    CatalogMachineStates,
};
use super::storage::{self, EMPTY_INDEX};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub(super) struct Context {
    owner: u64,
    generation: u64,
    storage: Option<Storage>,
}

#[derive(Clone, Copy)]
pub(super) struct BindRuntime<'a> {
    storage: &'a RefCell<Option<Storage>>,
    result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct SealRuntime<'a> {
    result: &'a Cell<Result<ModelIdentity, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct DescribeModelRuntime<'a> {
    identity: ModelIdentity,
    result: &'a Cell<Result<ModelDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct FindRuntime<'a> {
    identity: ModelIdentity,
    name: &'a [u8],
    result: &'a Cell<Result<Option<TensorDescriptor>, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct DescribeTensorRuntime<'a> {
    tensor_id: TensorId,
    result: &'a Cell<Result<TensorDescriptor, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ResetRuntime<'a> {
    result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ReleaseRuntime<'a> {
    storage: &'a RefCell<Option<Storage>>,
    result: &'a Cell<Result<(), Error>>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UnexpectedRuntime;

/// Single-writer, run-to-completion model catalog actor.
pub struct Catalog {
    machine: CatalogMachineStateMachine<Context>,
}

impl Catalog {
    /// Constructs a catalog with a process-unique checked owner identity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] if the owner nonce space is exhausted.
    pub fn try_new() -> Result<Self, Error> {
        let owner = next_owner()?;
        Ok(Self {
            machine: CatalogMachineStateMachine::new(Context {
                owner,
                generation: 0,
                storage: None,
            }),
        })
    }

    /// Dispatches one public event synchronously.
    pub fn process_event<E: event::Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn bind_storage(
        &mut self,
        event: event::BindStorage,
    ) -> Result<(), BindStorageError> {
        let storage = RefCell::new(Some(event.storage));
        let result = Cell::new(Err(Error::Internal));
        let runtime = BindRuntime {
            storage: &storage,
            result: &result,
        };
        if self
            .machine
            .process_event(CatalogMachineEvents::Bind(runtime))
            .is_err()
        {
            result.set(Err(Error::Internal));
        }
        match result.get() {
            Ok(()) => Ok(()),
            Err(error) => Err(BindStorageError::new(
                error,
                storage
                    .into_inner()
                    .expect("a rejected bind retains caller storage"),
            )),
        }
    }

    pub(crate) fn seal_model(&mut self, _event: event::SealModel) -> Result<ModelIdentity, Error> {
        let result = Cell::new(Err(Error::Internal));
        let runtime = SealRuntime { result: &result };
        self.machine
            .process_event(CatalogMachineEvents::Seal(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn describe_model(
        &mut self,
        event: event::DescribeModel,
    ) -> Result<ModelDescriptor, Error> {
        let result = Cell::new(Err(Error::Internal));
        let runtime = DescribeModelRuntime {
            identity: event.identity,
            result: &result,
        };
        self.machine
            .process_event(CatalogMachineEvents::DescribeModel(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn find_tensor(
        &mut self,
        event: event::FindTensor<'_>,
    ) -> Result<Option<TensorDescriptor>, Error> {
        let result = Cell::new(Err(Error::Internal));
        let runtime = FindRuntime {
            identity: event.identity,
            name: event.name,
            result: &result,
        };
        self.machine
            .process_event(CatalogMachineEvents::Find(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn describe_tensor(
        &mut self,
        event: event::DescribeTensor,
    ) -> Result<TensorDescriptor, Error> {
        let result = Cell::new(Err(Error::Internal));
        let runtime = DescribeTensorRuntime {
            tensor_id: event.tensor_id,
            result: &result,
        };
        self.machine
            .process_event(CatalogMachineEvents::DescribeTensor(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn with_tensor_name<F, R>(&self, event: &mut event::WithTensorName<F, R>)
    where
        F: for<'name> FnMut(&'name [u8]) -> R,
    {
        super::name_query::process(
            self.machine.is(&CatalogMachineStates::StateSealed),
            self.machine.context(),
            event,
        );
    }

    pub(crate) fn reset(&mut self, _event: event::Reset) -> Result<(), Error> {
        let result = Cell::new(Err(Error::Internal));
        let runtime = ResetRuntime { result: &result };
        self.machine
            .process_event(CatalogMachineEvents::Reset(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()
    }

    pub(crate) fn release_storage(
        &mut self,
        _event: event::ReleaseStorage,
    ) -> Result<Storage, Error> {
        let storage = RefCell::new(None);
        let result = Cell::new(Err(Error::Internal));
        let runtime = ReleaseRuntime {
            storage: &storage,
            result: &result,
        };
        self.machine
            .process_event(CatalogMachineEvents::Release(runtime))
            .map_err(|_| Error::Internal)?;
        result.get()?;
        storage.into_inner().ok_or(Error::Internal)
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.machine.is(&CatalogMachineStates::StateEmpty)
    }
    #[cfg(test)]
    pub(super) fn is_bound(&self) -> bool {
        self.machine.is(&CatalogMachineStates::StateBound)
    }
    #[cfg(test)]
    pub(super) fn is_sealed(&self) -> bool {
        self.machine.is(&CatalogMachineStates::StateSealed)
    }
}

fn next_owner() -> Result<u64, Error> {
    let mut owner = NEXT_OWNER.load(Ordering::Relaxed);
    loop {
        let next = owner.checked_add(1).ok_or(Error::Capacity)?;
        match NEXT_OWNER.compare_exchange_weak(owner, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(owner),
            Err(observed) => owner = observed,
        }
    }
}

impl fmt::Debug for Catalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Catalog").finish_non_exhaustive()
    }
}

impl CatalogMachineStateMachineContext for Context {
    fn guard_bind_valid(&self, event: &BindRuntime) -> Result<bool, ()> {
        Ok(event.storage.borrow().as_ref().is_some_and(|storage| {
            storage.tensor_count <= storage.records.len()
                && storage.name_bytes_used <= storage.names.len()
        }))
    }

    fn guard_bind_invalid(&self, event: &BindRuntime) -> Result<bool, ()> {
        Ok(!self.guard_bind_valid(event)?)
    }

    fn guard_seal_valid(&self, _event: &SealRuntime) -> Result<bool, ()> {
        Ok(!self.seal_capacity() && !self.seal_model_invalid())
    }

    fn guard_seal_capacity(&self, _event: &SealRuntime) -> Result<bool, ()> {
        Ok(self.seal_capacity())
    }

    fn guard_seal_model_invalid(&self, _event: &SealRuntime) -> Result<bool, ()> {
        Ok(!self.seal_capacity() && self.seal_model_invalid())
    }

    fn guard_model_wrong(&self, event: &DescribeModelRuntime) -> Result<bool, ()> {
        Ok(event.identity.owner != self.owner)
    }

    fn guard_model_stale(&self, event: &DescribeModelRuntime) -> Result<bool, ()> {
        Ok(event.identity.owner == self.owner && event.identity.generation != self.generation)
    }

    fn guard_model_valid(&self, event: &DescribeModelRuntime) -> Result<bool, ()> {
        Ok(event.identity.owner == self.owner && event.identity.generation == self.generation)
    }

    fn guard_find_wrong(&self, event: &FindRuntime<'_>) -> Result<bool, ()> {
        Ok(event.identity.owner != self.owner)
    }

    fn guard_find_stale(&self, event: &FindRuntime<'_>) -> Result<bool, ()> {
        Ok(event.identity.owner == self.owner && event.identity.generation != self.generation)
    }

    fn guard_find_present_bound(&self, event: &FindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.find_identity_valid(event)
            && self
                .find_ordinal(event.name)
                .is_some_and(|ordinal| self.ordinal_bound(ordinal)))
    }

    fn guard_find_present_unbound(&self, event: &FindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.find_identity_valid(event)
            && self
                .find_ordinal(event.name)
                .is_some_and(|ordinal| !self.ordinal_bound(ordinal)))
    }

    fn guard_find_absent(&self, event: &FindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.find_identity_valid(event) && self.find_ordinal(event.name).is_none())
    }

    fn guard_tensor_wrong(&self, event: &DescribeTensorRuntime) -> Result<bool, ()> {
        Ok(event.tensor_id.owner != self.owner)
    }

    fn guard_tensor_stale(&self, event: &DescribeTensorRuntime) -> Result<bool, ()> {
        Ok(event.tensor_id.owner == self.owner
            && (event.tensor_id.generation != self.generation
                || !self.ordinal_valid(event.tensor_id.ordinal)))
    }

    fn guard_tensor_valid_bound(&self, event: &DescribeTensorRuntime) -> Result<bool, ()> {
        Ok(event.tensor_id.owner == self.owner
            && event.tensor_id.generation == self.generation
            && self.ordinal_valid(event.tensor_id.ordinal)
            && self.ordinal_bound(event.tensor_id.ordinal))
    }

    fn guard_tensor_valid_unbound(&self, event: &DescribeTensorRuntime) -> Result<bool, ()> {
        Ok(event.tensor_id.owner == self.owner
            && event.tensor_id.generation == self.generation
            && self.ordinal_valid(event.tensor_id.ordinal)
            && !self.ordinal_bound(event.tensor_id.ordinal))
    }

    fn guard_never(&self, _event: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }

    fn effect_bind(&mut self, event: BindRuntime) -> Result<(), ()> {
        self.storage = event.storage.take();
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_bind_invalid(&mut self, event: BindRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_bind_busy(&mut self, event: BindRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn effect_seal(&mut self, event: SealRuntime) -> Result<(), ()> {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("capacity guard validates generation");
        let storage = self.storage.as_mut().expect("bound state owns storage");
        storage.index.fill(EMPTY_INDEX);
        for ordinal in 0..storage.tensor_count {
            let record = storage.records[ordinal];
            let name = storage::range(&storage.names, record.name_offset, record.name_length)
                .expect("seal guard validates range");
            let mut slot = usize::try_from(
                storage::hash(name)
                    % u64::try_from(storage.index.len()).expect("capacity guard requires index"),
            )
            .expect("slot fits usize");
            for _ in 0..storage.index.len() {
                let existing = storage.index[slot];
                if existing == EMPTY_INDEX {
                    storage.index[slot] = u32::try_from(ordinal).expect("catalog maximum fits u32");
                    break;
                }
                let existing_record =
                    storage.records[usize::try_from(existing).expect("index fits usize")];
                let existing_name = storage::range(
                    &storage.names,
                    existing_record.name_offset,
                    existing_record.name_length,
                )
                .expect("seal guard validates range");
                if existing_name == name {
                    break;
                }
                slot = (slot + 1) % storage.index.len();
            }
        }
        event.result.set(Ok(ModelIdentity {
            owner: self.owner,
            generation: self.generation,
        }));
        Ok(())
    }

    fn effect_seal_capacity(&mut self, event: SealRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::Capacity));
        Ok(())
    }
    fn effect_seal_model_invalid(&mut self, event: SealRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::ModelInvalid));
        Ok(())
    }
    fn effect_seal_storage_unavailable(&mut self, event: SealRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_seal_busy(&mut self, event: SealRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }
    fn effect_model_storage_unavailable(&mut self, event: DescribeModelRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_find_storage_unavailable(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_tensor_storage_unavailable(
        &mut self,
        event: DescribeTensorRuntime,
    ) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_reset_storage_unavailable(&mut self, event: ResetRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_reset_invalid_request(&mut self, event: ResetRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }
    fn effect_release_storage_unavailable(&mut self, event: ReleaseRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StorageUnavailable));
        Ok(())
    }
    fn effect_release_busy(&mut self, event: ReleaseRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn effect_model_wrong(&mut self, event: DescribeModelRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::WrongModelIdentity));
        Ok(())
    }
    fn effect_model_stale(&mut self, event: DescribeModelRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StaleModelIdentity));
        Ok(())
    }
    fn effect_describe_model(&mut self, event: DescribeModelRuntime) -> Result<(), ()> {
        event.result.set(Ok(ModelDescriptor::new(
            u32::try_from(self.storage.as_ref().expect("sealed storage").tensor_count)
                .expect("maximum fits u32"),
        )));
        Ok(())
    }

    fn effect_find_wrong(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::WrongModelIdentity));
        Ok(())
    }
    fn effect_find_stale(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::StaleModelIdentity));
        Ok(())
    }
    fn effect_find_present_bound(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        let ordinal = self
            .find_ordinal(event.name)
            .expect("present guard found record");
        event.result.set(Ok(Some(
            self.make_descriptor(ordinal, TensorBindingStatus::Bound)
                .expect("seal validates serialized type"),
        )));
        Ok(())
    }
    fn effect_find_present_unbound(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        let ordinal = self
            .find_ordinal(event.name)
            .expect("present guard found record");
        event.result.set(Ok(Some(
            self.make_descriptor(ordinal, TensorBindingStatus::Unbound)
                .expect("seal validates serialized type"),
        )));
        Ok(())
    }
    fn effect_find_absent(&mut self, event: FindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(None));
        Ok(())
    }

    fn effect_tensor_wrong(&mut self, event: DescribeTensorRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::WrongTensorIdentity));
        Ok(())
    }
    fn effect_tensor_stale(&mut self, event: DescribeTensorRuntime) -> Result<(), ()> {
        event.result.set(Err(Error::StaleTensorIdentity));
        Ok(())
    }
    fn effect_describe_tensor_bound(&mut self, event: DescribeTensorRuntime) -> Result<(), ()> {
        event
            .result
            .set(self.make_descriptor(event.tensor_id.ordinal, TensorBindingStatus::Bound));
        Ok(())
    }
    fn effect_describe_tensor_unbound(&mut self, event: DescribeTensorRuntime) -> Result<(), ()> {
        event
            .result
            .set(self.make_descriptor(event.tensor_id.ordinal, TensorBindingStatus::Unbound));
        Ok(())
    }

    fn effect_reset(&mut self, event: ResetRuntime) -> Result<(), ()> {
        self.storage
            .as_mut()
            .expect("sealed storage")
            .index
            .fill(EMPTY_INDEX);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_release(&mut self, event: ReleaseRuntime) -> Result<(), ()> {
        event.storage.replace(self.storage.take());
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl Context {
    pub(super) const fn owner(&self) -> u64 {
        self.owner
    }

    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn name(&self, ordinal: u32) -> Option<&[u8]> {
        let storage = self.storage.as_ref()?;
        let record = storage.records.get(usize::try_from(ordinal).ok()?)?;
        storage::range(&storage.names, record.name_offset, record.name_length)
    }

    fn seal_capacity(&self) -> bool {
        self.generation == u64::MAX
            || self
                .storage
                .as_ref()
                .is_none_or(|storage| storage.tensor_count > storage.index.len())
    }

    fn seal_model_invalid(&self) -> bool {
        let Some(storage) = self.storage.as_ref() else {
            return true;
        };
        if storage.tensor_count == 0
            || storage.tensor_count > storage.records.len()
            || storage.name_bytes_used > storage.names.len()
        {
            return true;
        }
        storage.records[..storage.tensor_count]
            .iter()
            .any(|record| {
                storage::range(
                    &storage.names[..storage.name_bytes_used],
                    record.name_offset,
                    record.name_length,
                )
                .is_none()
                    || SerializedType::try_from(record.wire_type).is_err()
            })
    }

    const fn find_identity_valid(&self, event: &FindRuntime<'_>) -> bool {
        event.identity.owner == self.owner && event.identity.generation == self.generation
    }
    pub(super) fn ordinal_valid(&self, ordinal: u32) -> bool {
        usize::try_from(ordinal).is_ok_and(|value| {
            value
                < self
                    .storage
                    .as_ref()
                    .map_or(0, |storage| storage.tensor_count)
        })
    }

    fn ordinal_bound(&self, ordinal: u32) -> bool {
        let Some(record) = usize::try_from(ordinal).ok().and_then(|ordinal| {
            self.storage
                .as_ref()
                .and_then(|storage| storage.records.get(ordinal))
        }) else {
            return false;
        };
        record.binding_present
            && record.data_size > 0
            && record.dimension_count > 0
            && record.dimensions
                [..usize::min(usize::try_from(record.dimension_count).unwrap_or(0), 4)]
                .iter()
                .all(|dimension| *dimension > 0)
    }

    fn find_ordinal(&self, name: &[u8]) -> Option<u32> {
        let storage = self.storage.as_ref()?;
        if storage.index.is_empty() {
            return None;
        }
        let mut slot =
            usize::try_from(storage::hash(name) % u64::try_from(storage.index.len()).ok()?).ok()?;
        for _ in 0..storage.index.len() {
            let ordinal = storage.index[slot];
            if ordinal == EMPTY_INDEX {
                return None;
            }
            let record = storage.records[usize::try_from(ordinal).ok()?];
            if storage::range(&storage.names, record.name_offset, record.name_length)? == name {
                return Some(ordinal);
            }
            slot = (slot + 1) % storage.index.len();
        }
        None
    }

    fn make_descriptor(
        &self,
        ordinal: u32,
        binding_status: TensorBindingStatus,
    ) -> Result<TensorDescriptor, Error> {
        let record = self
            .storage
            .as_ref()
            .ok_or(Error::StorageUnavailable)?
            .records[usize::try_from(ordinal).map_err(|_| Error::Internal)?];
        storage::descriptor(self.owner, self.generation, ordinal, record, binding_status)
    }
}
