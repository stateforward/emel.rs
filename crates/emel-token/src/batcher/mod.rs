//! Token batching request validation and sequence normalization.

mod sm;

pub use sm::{
    BatchDone, BatchDoneCallback, BatchError, BatchErrorCallback, BatchFailure, BatchOutputs,
    BatchRequest, BatchResult, MAX_SEQ, MAX_TOKENS, PositionSeedContext, PositionSeedError,
    PositionSeedFn, PositionSeedResolver, SEQ_WORDS,
};

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
    ///
    /// # Panics
    ///
    /// Panics if the generated state machine fails to return to its ready state.
    pub fn process_event(&mut self, request: BatchRequest<'_>) -> Result<BatchResult, BatchError> {
        let result = self.machine.dispatch(request);
        assert!(
            self.machine.is_ready(),
            "token batcher must return to ready"
        );
        result
    }

    /// Reports whether the generated state machine is in its stable ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is_ready()
    }

    /// Exercises the explicit unexpected-event outcome.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::UnexpectedEvent`] while the batcher is in its
    /// ready state.
    ///
    /// # Panics
    ///
    /// Panics if the generated state machine fails to return to its ready state.
    pub fn process_unexpected(&mut self) -> Result<(), BatchError> {
        let result = self.machine.dispatch_unexpected();
        assert!(
            self.machine.is_ready(),
            "token batcher must return to ready"
        );
        result
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
