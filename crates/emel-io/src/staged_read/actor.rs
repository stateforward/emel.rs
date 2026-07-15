//! Owned staged-copy actor boundary.

use core::cell::{Cell, RefCell};
use core::fmt;

use super::event::{self, Event};
use super::sm::{
    BatchAssessment, BatchRuntime, BatchStatus, Context, IoStagedReadEvents,
    IoStagedReadStateMachine, SingleRuntime, SingleStatus,
};

pub struct Stager {
    machine: IoStagedReadStateMachine<Context>,
}

impl Stager {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_platform_support(cfg!(any(unix, windows)))
    }

    const fn with_platform_support(platform_supported: bool) -> Self {
        Self {
            machine: IoStagedReadStateMachine::new(Context { platform_supported }),
        }
    }

    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn stage_window(
        &mut self,
        event: event::StageWindow<'_>,
    ) -> Result<event::StageWindowDone, event::Error> {
        let event::StageWindow {
            file_offset,
            logical_byte_length,
            stage_chunk_bytes,
            source,
            target: target_bytes,
            on_done,
            on_error,
        } = event;
        let target = RefCell::new(&mut *target_bytes);
        let status = Cell::new(SingleStatus::new());
        self.machine
            .process_event(IoStagedReadEvents::Single(SingleRuntime {
                file_offset,
                logical_byte_length,
                stage_chunk_bytes,
                source,
                target: &target,
                on_done,
                on_error,
                status: &status,
            }))
            .expect("staged-read SML callbacks are infallible");
        status.get().result
    }

    pub(crate) fn stage_window_batch(
        &mut self,
        event: event::StageWindowBatch<'_>,
    ) -> Result<event::StageWindowBatchDone, event::StageWindowBatchError> {
        let status = Cell::new(BatchStatus::new());
        let assessment = Cell::new(BatchAssessment::pending());
        self.machine
            .process_event(IoStagedReadEvents::Batch(BatchRuntime {
                tensors: event.tensors,
                stage_chunk_bytes: event.stage_chunk_bytes,
                on_done: event.on_done,
                on_error: event.on_error,
                assessment: &assessment,
                status: &status,
            }))
            .expect("staged-read SML callbacks are infallible");
        status.get().result
    }

    #[cfg(test)]
    pub(super) const fn unsupported() -> Self {
        Self::with_platform_support(false)
    }
}

impl Default for Stager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Stager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Stager").finish_non_exhaustive()
    }
}
