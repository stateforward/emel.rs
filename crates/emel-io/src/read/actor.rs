//! Owned actor boundary for read dispatch.

use core::cell::{Cell, RefCell};
use core::fmt;

use super::event::{self, Event};
#[cfg(test)]
use super::sm::IoReadStates;
use super::sm::{
    BatchAnalysis, BatchStatus, Context, IoReadEvents, IoReadStateMachine, ReadRuntime, ReadStatus,
    TensorBatchRuntime,
};

/// Allocation-free reader over caller-provided immutable source bytes.
pub struct Reader {
    machine: IoReadStateMachine<Context>,
}

impl Reader {
    /// Creates a reader in its ready state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: IoReadStateMachine::new(Context::new()),
        }
    }

    /// Dispatches one typed event run-to-completion.
    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn read_tensor(
        &mut self,
        event: event::ReadTensor<'_>,
    ) -> Result<event::ReadTensorDone, event::Error> {
        self.dispatch_read_tensor(event, super::sm::PLATFORM_SUPPORTED)
    }

    fn dispatch_read_tensor(
        &mut self,
        event: event::ReadTensor<'_>,
        platform_supported: bool,
    ) -> Result<event::ReadTensorDone, event::Error> {
        let event::ReadTensor {
            tensor_id,
            file_index,
            file_offset,
            byte_size,
            file_path,
            source,
            source_error,
            target: target_bytes,
            on_done,
            on_error,
        } = event;
        let status = Cell::new(ReadStatus::new());
        let target = RefCell::new(&mut *target_bytes);
        let runtime = ReadRuntime {
            tensor_id,
            file_index,
            file_offset,
            byte_size,
            file_path,
            source,
            source_error,
            target: &target,
            on_done,
            on_error,
            platform_supported,
            status: &status,
        };
        self.machine
            .process_event(IoReadEvents::ReadTensor(runtime))
            .map_err(|_| event::Error::InternalError)?;
        status.get().result
    }

    #[cfg(test)]
    pub(super) fn read_tensor_on_unsupported_platform(
        &mut self,
        event: event::ReadTensor<'_>,
    ) -> Result<event::ReadTensorDone, event::Error> {
        self.dispatch_read_tensor(event, false)
    }

    pub(crate) fn read_tensor_batch(
        &mut self,
        event: event::ReadTensorBatch<'_>,
    ) -> Result<event::ReadTensorBatchDone, event::ReadTensorBatchError> {
        let event::ReadTensorBatch {
            tensors,
            on_done,
            on_error,
        } = event;
        let status = Cell::new(BatchStatus::new());
        let analysis = Cell::new(BatchAnalysis::default());
        let runtime = TensorBatchRuntime {
            tensors,
            on_done,
            on_error,
            status: &status,
            analysis: &analysis,
        };
        self.machine
            .process_event(IoReadEvents::ReadTensorBatch(runtime))
            .map_err(|_| event::ReadTensorBatchError::new(event::Error::InternalError, 0))?;
        status.get().result
    }

    #[cfg(test)]
    pub(super) fn is_ready(&self) -> bool {
        matches!(self.machine.state(), IoReadStates::StateReady)
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Reader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Reader").finish_non_exhaustive()
    }
}
