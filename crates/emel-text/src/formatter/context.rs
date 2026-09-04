//! Caller-owned result context for one bounded formatter dispatch.

use crate::ConditionerError;

/// Mutable state published by the synchronous formatter actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Context {
    /// Exact number of bytes written on the most recent successful request.
    pub output_length: usize,
    /// Error from the most recent request, if any.
    pub error: ConditionerError,
    /// Set when an event violates the generated machine sequencing contract.
    pub unexpected: bool,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            output_length: 0,
            error: ConditionerError::None,
            unexpected: false,
        }
    }
}

impl Context {
    /// Resets the caller-owned result before a new request.
    pub(crate) const fn reset(&mut self) {
        self.output_length = 0;
        self.error = ConditionerError::None;
        self.unexpected = false;
    }

    /// Records a successful bounded write.
    pub(crate) const fn done(&mut self, output_length: usize) {
        self.output_length = output_length;
        self.error = ConditionerError::None;
    }

    /// Records a failed request and guarantees a zero published length.
    pub(crate) const fn error(&mut self, error: ConditionerError) {
        self.output_length = 0;
        self.error = error;
    }
}
