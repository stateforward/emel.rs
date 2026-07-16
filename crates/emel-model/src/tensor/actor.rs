//! Owned actor boundary for tensor lifecycle dispatch.

use core::cell::{Cell, RefCell};
use core::fmt;

use super::dependency::TensorDependencies;
use super::event::{self, Event};
use super::sm::{
    AccessRequest, ApplyBoundRuntime, ApplyEffectErrorRuntime, ApplyOwnedRuntime, BindRuntime,
    BindStorageRuntime, CaptureRuntime, Context, EvictRuntime, IoDispatcher, ModelTensorEvents,
    ModelTensorStateMachine, ModelTensorStates, PlanLoadRuntime, TensorAccessDispatcher,
};

/// Maximum tensor slots supported by one store.
pub const MAX_TENSORS: usize = 65_536;

/// Stateful owner of tensor residency and injected dependencies.
pub struct Store<D = ()> {
    machine: ModelTensorStateMachine<Context>,
    dependencies: D,
}

impl Store<()> {
    /// Creates a tensor store with no injected dependencies.
    ///
    /// All slot storage is allocated before this function returns. Event
    /// dispatch never grows the slot table.
    ///
    /// # Errors
    ///
    /// Returns [`event::Error::Capacity`] when `tensor_capacity` is zero,
    /// exceeds [`MAX_TENSORS`], or cannot be allocated.
    pub fn new(tensor_capacity: usize) -> Result<Self, event::Error> {
        Self::with_dependencies(tensor_capacity, ())
    }
}

impl<D> Store<D> {
    /// Creates a tensor store with statically dispatched dependencies.
    ///
    /// # Errors
    ///
    /// Returns [`event::Error::Capacity`] when `tensor_capacity` is zero,
    /// exceeds [`MAX_TENSORS`], or cannot be allocated.
    pub fn with_dependencies(
        tensor_capacity: usize,
        dependencies: D,
    ) -> Result<Self, event::Error> {
        let context = Context::allocate(tensor_capacity)?;
        Ok(Self {
            machine: ModelTensorStateMachine::new(context),
            dependencies,
        })
    }

    /// Dispatches one typed event run-to-completion.
    pub fn process_event<E: Event<D>>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn bind_tensor(
        &mut self,
        event: event::BindTensor,
    ) -> Result<event::BindTensorDone, event::Error> {
        let event::BindTensor {
            tensor_id,
            metadata,
            bytes,
        } = event;
        let bytes = RefCell::new(Some(bytes));
        let result = Cell::new(Err(event::Error::Internal));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::Bind(BindRuntime {
                tensor_id,
                metadata,
                bytes: &bytes,
                result: &result,
            }));
        result.get()
    }

    pub(crate) fn bind_storage(
        &mut self,
        event: event::BindStorage,
    ) -> Result<event::BindStorageDone, event::BindStorageError> {
        let result = RefCell::new(Err(event::BindStorageError::new(
            event::Error::Internal,
            event.storage,
        )));
        let _state =
            self.machine
                .process_event(ModelTensorEvents::BindStorage(BindStorageRuntime {
                    result: &result,
                }));
        result.into_inner()
    }

    pub(crate) fn plan_load(
        &mut self,
        event: event::PlanLoad,
    ) -> Result<event::PlanLoadDone, event::PlanLoadError> {
        let result = RefCell::new(Err(event::PlanLoadError::new(
            event::Error::Internal,
            event.effects,
        )));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::PlanLoad(PlanLoadRuntime {
                strategy: event.strategy,
                result: &result,
            }));
        result.into_inner()
    }

    pub(crate) fn apply_bound_effect_results(
        &mut self,
        event: event::ApplyBoundEffectResults,
    ) -> Result<(), event::ApplyBoundEffectResultsError> {
        let result = RefCell::new(Err(event::ApplyBoundEffectResultsError::new(
            event::Error::Internal,
            event.tensor_ids,
        )));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::ApplyBound(ApplyBoundRuntime {
                result: &result,
            }));
        result.into_inner()
    }

    pub(crate) fn apply_owned_effect_results(
        &mut self,
        event: event::ApplyOwnedEffectResults,
    ) -> Result<(), event::ApplyOwnedEffectResultsError> {
        let result = RefCell::new(Err(event::ApplyOwnedEffectResultsError::new(
            event::Error::Internal,
            event.results,
        )));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::ApplyOwned(ApplyOwnedRuntime {
                result: &result,
            }));
        result.into_inner()
    }

    pub(crate) fn apply_effect_error(
        &mut self,
        event: event::ApplyEffectError,
    ) -> Result<(), event::Error> {
        let result = Cell::new(Err(event::Error::Internal));
        let _state =
            self.machine
                .process_event(ModelTensorEvents::ApplyError(ApplyEffectErrorRuntime {
                    tensor_id: event.tensor_id,
                    error: event.error,
                    result: &result,
                }));
        result.get()
    }

    pub(crate) fn evict_tensor(
        &mut self,
        event: event::EvictTensor,
    ) -> Result<event::EvictTensorDone, event::Error> {
        let result = RefCell::new(Err(event::Error::Internal));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::Evict(EvictRuntime {
                tensor_id: event.tensor_id,
                result: &result,
            }));
        result.into_inner()
    }

    pub(crate) fn capture_tensor_state(
        &mut self,
        event: event::CaptureTensorState,
    ) -> Result<event::TensorState, event::Error> {
        let result = Cell::new(Err(event::Error::Internal));
        let _state = self
            .machine
            .process_event(ModelTensorEvents::Capture(CaptureRuntime {
                tensor_id: event.tensor_id,
                result: &result,
            }));
        result.get()
    }

    pub(crate) const fn dependencies(&self) -> &D {
        &self.dependencies
    }

    #[cfg(test)]
    pub(crate) fn is_ready(&self) -> bool {
        self.machine.is(&ModelTensorStates::StateReady)
    }

    #[cfg(test)]
    pub(crate) fn is_awaiting_bound_results(&self) -> bool {
        self.machine
            .is(&ModelTensorStates::StateAwaitingBoundResults)
    }

    #[cfg(test)]
    pub(crate) fn is_awaiting_owned_results(&self) -> bool {
        self.machine
            .is(&ModelTensorStates::StateAwaitingOwnedResults)
    }

    #[cfg(test)]
    pub(crate) fn is_awaiting_mapped_results(&self) -> bool {
        self.machine
            .is(&ModelTensorStates::StateAwaitingMappedResults)
    }
}

impl<D> Store<D>
where
    D: TensorDependencies,
{
    pub(crate) fn mapped_load(
        &mut self,
        event: event::MappedLoad,
    ) -> Result<event::MappedLoadDone, event::Error> {
        let Self {
            machine,
            dependencies,
        } = self;
        let owner_ready = machine.is(&ModelTensorStates::StateReady);
        IoDispatcher::new(machine.context_mut(), dependencies).process_mapped(event, owner_ready)
    }

    pub(crate) fn read_load(
        &mut self,
        event: event::ReadLoad<'_>,
    ) -> Result<event::OwnedLoadDone, event::Error> {
        let Self {
            machine,
            dependencies,
        } = self;
        let owner_ready = machine.is(&ModelTensorStates::StateReady);
        IoDispatcher::new(machine.context_mut(), dependencies).process_read(event, owner_ready)
    }

    pub(crate) fn staged_load(
        &mut self,
        event: event::StagedLoad<'_>,
    ) -> Result<event::OwnedLoadDone, event::Error> {
        let Self {
            machine,
            dependencies,
        } = self;
        let owner_ready = machine.is(&ModelTensorStates::StateReady);
        IoDispatcher::new(machine.context_mut(), dependencies).process_staged(event, owner_ready)
    }

    pub(crate) fn release_mapped(
        &mut self,
        event: event::ReleaseMapped,
    ) -> Result<event::ReleaseMappedDone, event::Error> {
        let Self {
            machine,
            dependencies,
        } = self;
        let owner_ready = machine.is(&ModelTensorStates::StateReady);
        IoDispatcher::new(machine.context_mut(), dependencies).process_release(event, owner_ready)
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "typed events are consumed at the public actor dispatch boundary"
    )]
    pub(crate) fn with_tensor<Operation>(
        &mut self,
        event: event::WithTensor<'_, Operation>,
    ) -> Result<Operation::Output, event::Error>
    where
        Operation: event::TensorOperation,
    {
        let Self {
            machine,
            dependencies,
        } = self;
        let owner_ready = machine.is(&ModelTensorStates::StateReady);
        TensorAccessDispatcher::new(machine.context_mut(), dependencies).process_event(
            AccessRequest {
                tensor_id: event.tensor_id,
            },
            event.operation,
            owner_ready,
        )
    }
}

impl<D> fmt::Debug for Store<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Store").finish_non_exhaustive()
    }
}
