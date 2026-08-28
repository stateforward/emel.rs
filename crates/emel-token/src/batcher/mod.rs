//! Token batching request validation and sequence normalization.

mod sm;

pub use sm::{BatchError, BatchOutputs, BatchRequest, BatchResult, MAX_SEQ, MAX_TOKENS, SEQ_WORDS};

/// Stateful single-writer token batcher for the bounded validation slice.
pub struct TokenBatcher {
    machine: sm::TokenBatcherStateMachine<sm::Context>,
}

impl TokenBatcher {
    /// Creates a batcher in its ready state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: sm::TokenBatcherStateMachine::new(sm::Context::default()),
        }
    }

    /// Validates one request and normalizes its sequence inputs in place.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidRequest`] when a request contract check
    /// fails, or [`BatchError::Internal`] when the local state-machine
    /// contract cannot complete.
    pub fn process_event(&mut self, request: BatchRequest<'_>) -> Result<BatchResult, BatchError> {
        self.machine.dispatch(request)
    }

    /// Exercises the explicit unexpected-event outcome.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::UnexpectedEvent`] while the batcher is in its
    /// ready state.
    pub fn process_unexpected(&mut self) -> Result<(), BatchError> {
        self.machine.dispatch_unexpected()
    }
}

impl Default for TokenBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for TokenBatcher {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TokenBatcher")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
