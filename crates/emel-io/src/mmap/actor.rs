//! Owned mapper actor and synchronous native phase boundary.

use core::fmt;

use super::event::{self, Event};
use super::platform::Native;
use super::sm::MapperCore;

/// Safe owner of native file-backed mappings.
pub struct Mapper {
    core: MapperCore<Native>,
}

impl Mapper {
    #[must_use]
    pub fn new() -> Self {
        Self {
            core: MapperCore::new(Native::new()),
        }
    }

    /// Dispatches one event to completion before returning.
    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(super) fn map(
        &mut self,
        request: &event::MapTensor,
    ) -> Result<event::MapDone, event::Error> {
        self.core.map(request)
    }

    pub(super) fn release(&mut self, request: event::ReleaseMapping) -> Result<(), event::Error> {
        self.core.release(request)
    }

    pub(super) fn advise_sequential(
        &mut self,
        request: event::AdviceRequest,
    ) -> Result<(), event::Error> {
        self.core.advise_sequential(request)
    }

    pub(super) fn advise_will_need(
        &mut self,
        request: event::AdviceRequest,
    ) -> Result<(), event::Error> {
        self.core.advise_will_need(request)
    }

    pub(super) fn advise_dont_need(
        &mut self,
        request: event::AdviceRequest,
    ) -> Result<(), event::Error> {
        self.core.advise_dont_need(request)
    }

    pub(super) fn with_mapping(
        &mut self,
        request: event::WithMapping<'_>,
    ) -> Result<(), event::Error> {
        self.core.with_mapping(request)
    }
}

impl Default for Mapper {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Mapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mapper").finish_non_exhaustive()
    }
}
