//! Owned static loader composition boundary.

use core::cell::Cell;
use core::fmt;

use super::event::{self, Event, NoActor, ReadActor, StagedReadActor};
use super::sm::{
    BatchRuntime, BatchStatus, Context, IoLoaderEvents, IoLoaderStateMachine, SingleRuntime,
    SingleStatus,
};

/// Loader actor with statically dispatched replaceable child actors.
pub struct Loader<R = NoActor, S = NoActor>
where
    R: ReadActor,
    S: StagedReadActor,
{
    machine: IoLoaderStateMachine<Context<R, S>>,
}

impl Loader<NoActor, NoActor> {
    /// Creates a loader with no installed strategy actors.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_dependencies(NoActor, NoActor)
    }

    /// Creates a loader with one replaceable read actor.
    #[must_use]
    pub const fn with_reader<R: ReadActor>(reader: R) -> Loader<R, NoActor> {
        Loader::with_dependencies(reader, NoActor)
    }

    /// Creates a loader with one replaceable staged-read actor.
    #[must_use]
    pub const fn with_stager<S: StagedReadActor>(stager: S) -> Loader<NoActor, S> {
        Loader::with_dependencies(NoActor, stager)
    }
}

impl<R: ReadActor, S: StagedReadActor> Loader<R, S> {
    /// Creates a loader owning both injected strategy actors.
    #[must_use]
    pub const fn with_dependencies(reader: R, stager: S) -> Self {
        Self {
            machine: IoLoaderStateMachine::new(Context { reader, stager }),
        }
    }

    /// Dispatches one typed loader event run-to-completion.
    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn load_tensor(
        &mut self,
        event: event::LoadTensor<'_>,
    ) -> Result<event::LoadTensorDone, event::LoadTensorError> {
        let status = Cell::new(SingleStatus::new());
        let read_result = Cell::new(Err(crate::read::event::Error::InternalError));
        let staged_result = Cell::new(Err(crate::staged_read::event::Error::InternalError));
        self.machine
            .process_event(IoLoaderEvents::Single(SingleRuntime {
                tensor: event.tensor,
                policy: event.policy,
                on_done: event.on_done,
                on_error: event.on_error,
                status: &status,
                read_result: &read_result,
                staged_result: &staged_result,
            }))
            .map_err(|_| {
                event::LoadTensorError::new(event::Error::InternalError, event::StrategyError::None)
            })?;
        status.get().result
    }

    pub(crate) fn load_tensor_batch(
        &mut self,
        event: event::LoadTensorBatch<'_>,
    ) -> Result<event::LoadTensorBatchDone, event::LoadTensorBatchError> {
        let status = Cell::new(BatchStatus::new());
        let read_result = Cell::new(Err(crate::read::event::ReadTensorBatchError::new(
            crate::read::event::Error::InternalError,
            0,
        )));
        let staged_result = Cell::new(Err(crate::staged_read::event::StageWindowBatchError::new(
            crate::staged_read::event::Error::InternalError,
            0,
        )));
        self.machine
            .process_event(IoLoaderEvents::Batch(BatchRuntime {
                tensors: event.tensors,
                policy: event.policy,
                on_done: event.on_done,
                on_error: event.on_error,
                status: &status,
                read_result: &read_result,
                staged_result: &staged_result,
            }))
            .map_err(|_| {
                event::LoadTensorBatchError::new(
                    event::Error::InternalError,
                    event::StrategyError::None,
                    0,
                )
            })?;
        status.get().result
    }
}

impl Default for Loader<NoActor, NoActor> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: ReadActor, S: StagedReadActor> fmt::Debug for Loader<R, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Loader").finish_non_exhaustive()
    }
}
