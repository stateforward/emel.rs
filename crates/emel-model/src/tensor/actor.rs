//! Owned actor boundary for tensor lifecycle dispatch.

use core::cell::{Cell, RefCell};
use core::fmt;

use super::event::{self, Event};
use super::sm::{
    BindRuntime, CaptureRuntime, Context, EvictRuntime, ModelTensorEvents, ModelTensorStateMachine,
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
        use super::sm::ModelTensorStates;

        self.machine.is(&ModelTensorStates::StateReady)
    }
}

impl<D> fmt::Debug for Store<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Store").finish_non_exhaustive()
    }
}
