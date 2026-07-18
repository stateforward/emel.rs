//! Exact, allocation-free tokenizer profile resolver.

use core::fmt;

pub mod event;
mod sm;

use event::Event;

/// Statically dispatched profile-resolution dependency for model actors.
///
/// Implementations receive only the public typed event and return its typed
/// outcome, preserving the tokenizer actor boundary.
pub trait Dependency {
    /// Resolves one profile request run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`event::Error::Internal`] only when the replacement dependency
    /// cannot complete its typed dispatch contract.
    fn process_event(&mut self, event: event::Resolve<'_>)
    -> Result<event::Resolved, event::Error>;
}

/// Stateful single-writer tokenizer profile resolver.
pub struct Resolver {
    machine: sm::ProfileResolverStateMachine<sm::Context>,
}

impl Resolver {
    /// Creates a resolver in its ready state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: sm::ProfileResolverStateMachine::new(sm::Context),
        }
    }

    /// Dispatches one typed event run-to-completion.
    pub fn process_event<E: Event>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    pub(super) fn resolve(
        &mut self,
        event: event::Resolve<'_>,
    ) -> Result<event::Resolved, event::Error> {
        self.machine.resolve(event)
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Dependency for Resolver {
    #[allow(
        clippy::use_self,
        reason = "qualified inherent dispatch avoids recursion into this same trait method"
    )]
    fn process_event(
        &mut self,
        event: event::Resolve<'_>,
    ) -> Result<event::Resolved, event::Error> {
        Resolver::process_event(self, event)
    }
}

impl fmt::Debug for Resolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Resolver").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
