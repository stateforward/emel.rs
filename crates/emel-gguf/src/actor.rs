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

    pub(crate) fn probe(&mut self, event: event::Probe) -> Result<event::ProbeDone, event::Error> {
        let source = event.into_source();
        self.inner
            .probe(&source)
            .map(|requirements| event::ProbeDone::new(requirements, source))
    }

    pub(crate) fn bind(&mut self, event: event::Bind) -> Result<(), event::BindError> {
        self.inner
            .bind(event.into_storage())
            .map_err(|(error, storage)| event::BindError::new(error, storage))
    }

    pub(crate) fn parse(&mut self, _event: event::Parse) -> Result<event::ParseDone, event::Error> {
        self.inner.parse().map(event::ParseDone::new)
    }

    pub(crate) fn query<O: crate::loader::query::Operation>(&self, operation: &mut O) {
        self.inner.query(operation);
    }

    pub(crate) fn with_metadata_descriptor<O: crate::loader::metadata::Operation>(
        &self,
        operation: &mut O,
    ) {
        self.inner.with_metadata_descriptor(operation);
    }

    pub(crate) fn with_tensor<O: crate::loader::tensor::Operation>(&self, operation: &mut O) {
        self.inner.with_tensor(operation);
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
