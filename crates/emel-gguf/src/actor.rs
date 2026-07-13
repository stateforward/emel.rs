//! Public actor boundary for GGUF loading.

use core::fmt;

use crate::event::{self, Event};
use crate::loader;

/// Stateful GGUF loader actor.
pub struct Loader {
    inner: loader::Loader,
}

impl Loader {
    /// Creates an uninitialized loader actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: loader::Loader::new(),
        }
    }

    /// Dispatches one typed event run-to-completion.
    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(crate) fn probe(
        &mut self,
        event: event::Probe<'_>,
    ) -> Result<event::ProbeDone, event::Error> {
        self.inner
            .probe(event.file_image())
            .map(event::ProbeDone::new)
    }

    pub(crate) fn bind(&mut self, event: event::Bind) -> Result<(), event::Error> {
        if let Some((kv_arena_bytes, kv_entry_capacity, tensor_capacity)) = event.capacity() {
            self.inner
                .bind_with_capacity(kv_arena_bytes, kv_entry_capacity, tensor_capacity)
        } else {
            self.inner.bind()
        }
    }

    pub(crate) fn parse<'a>(
        &mut self,
        event: event::Parse<'a>,
    ) -> Result<event::ParseDone<'a>, event::Error> {
        self.inner
            .parse(event.file_image())
            .map(event::ParseDone::new)
    }

    pub(crate) fn load<'a>(
        &mut self,
        event: event::Load<'a>,
    ) -> Result<event::ParseDone<'a>, event::Error> {
        let file_image = event.file_image();
        self.probe(event::Probe::new(file_image))?;
        self.bind(event::Bind::exact())?;
        self.parse(event::Parse::new(file_image))
    }
}

impl Default for Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Loader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Loader").finish_non_exhaustive()
    }
}
