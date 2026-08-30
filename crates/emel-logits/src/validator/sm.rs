//! Run-to-completion logits candidate validation and preparation.
//!
//! The transition table follows the pinned C++ validator topology.  Request
//! output storage is borrowed for one synchronous dispatch and never retained
//! by the actor.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;

use sml::sml;

/// Errors produced while validating a candidate-build request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidatorError {
    /// No error occurred.
    None,
    /// The vocabulary or candidate storage contract is invalid.
    InvalidRequest,
    /// A backend operation failed.
    BackendError,
    /// The state-machine contract could not complete.
    InternalError,
    /// An unclassified error was reported.
    Untracked,
}

/// Borrowed request and outputs for one candidate-build dispatch.
///
/// The caller owns every input and output allocation.  The validator writes
/// only the first `vocab_size` entries of the candidate buffers.
#[derive(Debug)]
pub struct Build<'a> {
    /// Unnormalized vocabulary logits.
    pub logits: &'a [f32],
    /// Number of vocabulary entries to copy.
    pub vocab_size: i32,
    /// Candidate token identifiers output.
    pub candidate_ids: &'a mut [i32],
    /// Candidate score output.
    pub candidate_scores: &'a mut [f32],
    /// Capacity promised by the caller for candidate outputs.
    pub candidate_capacity: i32,
    /// Number of candidates written by the validator.
    pub candidate_count_out: &'a mut i32,
    /// Typed error output, reset and published during dispatch.
    pub error_out: &'a mut ValidatorError,
}

impl<'a> Build<'a> {
    /// Creates a borrowed candidate-build request.
    #[must_use]
    pub const fn new(
        logits: &'a [f32],
        vocab_size: i32,
        candidate_ids: &'a mut [i32],
        candidate_scores: &'a mut [f32],
        candidate_capacity: i32,
        candidate_count_out: &'a mut i32,
        error_out: &'a mut ValidatorError,
    ) -> Self {
        Self {
            logits,
            vocab_size,
            candidate_ids,
            candidate_scores,
            candidate_capacity,
            candidate_count_out,
            error_out,
        }
    }
}

/// Successful output from a candidate-build dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildResult {
    /// Number of candidates written.
    pub candidate_count: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchError {
    None,
    InvalidRequest,
    InternalError,
}

#[derive(Clone, Copy)]
pub(super) struct Runtime<'dispatch> {
    request: RequestView<'dispatch>,
    candidate_ids: &'dispatch [Cell<i32>],
    candidate_scores: &'dispatch [Cell<f32>],
    candidate_count_out: &'dispatch Cell<i32>,
    error_out: &'dispatch Cell<ValidatorError>,
    dispatch_error: &'dispatch Cell<DispatchError>,
}

#[derive(Clone, Copy)]
struct RequestView<'dispatch> {
    logits: &'dispatch [f32],
    vocab_size: i32,
    candidate_capacity: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Context;

sml! {
    LogitsValidator {
        "request_decision"_s <= *"ready"_s + Build(&'dispatch Runtime<'dispatch>) / begin_build,
        "done"_s <= "request_decision"_s + completion<Build>(&'dispatch Runtime<'dispatch>) [valid_request] / execute_build,
        "errored"_s <= "request_decision"_s + completion<Build>(&'dispatch Runtime<'dispatch>) [invalid_request] / mark_invalid_request,

        "ready"_s <= "done"_s + completion<Build>(&'dispatch Runtime<'dispatch>) / publish_done,
        "ready"_s <= "errored"_s + completion<Build>(&'dispatch Runtime<'dispatch>) / publish_error,

        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected,
    }
}

impl LogitsValidatorStateMachine<Context> {
    pub(super) fn dispatch(&mut self, request: Build<'_>) -> Result<BuildResult, ValidatorError> {
        let Build {
            logits,
            vocab_size,
            candidate_ids,
            candidate_scores,
            candidate_capacity,
            candidate_count_out,
            error_out,
        } = request;
        let ids = Cell::from_mut(candidate_ids).as_slice_of_cells();
        let scores = Cell::from_mut(candidate_scores).as_slice_of_cells();
        let count = Cell::from_mut(candidate_count_out);
        let error = Cell::from_mut(error_out);
        let dispatch_error = Cell::new(DispatchError::None);
        let runtime = Runtime {
            request: RequestView {
                logits,
                vocab_size,
                candidate_capacity,
            },
            candidate_ids: ids,
            candidate_scores: scores,
            candidate_count_out: count,
            error_out: error,
            dispatch_error: &dispatch_error,
        };

        if self
            .process_event(LogitsValidatorEvents::Build(&runtime))
            .is_err()
        {
            return Err(ValidatorError::InternalError);
        }
        match dispatch_error.get() {
            DispatchError::None => Ok(BuildResult {
                candidate_count: count.get(),
            }),
            DispatchError::InvalidRequest => Err(ValidatorError::InvalidRequest),
            DispatchError::InternalError => Err(ValidatorError::InternalError),
        }
    }
}

impl LogitsValidatorStateMachineContext for Context {
    fn begin_build(&mut self, event: &Runtime<'_>) -> Result<(), ()> {
        event.dispatch_error.set(DispatchError::None);
        event.candidate_count_out.set(0);
        event.error_out.set(ValidatorError::None);
        Ok(())
    }

    fn valid_request(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.request.vocab_size > 0
            && event.request.candidate_capacity >= event.request.vocab_size
            && i32::try_from(event.request.logits.len())
                .is_ok_and(|len| len >= event.request.vocab_size)
            && i32::try_from(event.candidate_ids.len())
                .is_ok_and(|len| len >= event.request.vocab_size)
            && i32::try_from(event.candidate_scores.len())
                .is_ok_and(|len| len >= event.request.vocab_size))
    }

    fn invalid_request(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_request(event)?)
    }

    fn execute_build(&mut self, event: &Runtime<'_>) -> Result<(), ()> {
        let count = usize::try_from(event.request.vocab_size).map_err(|_| ())?;
        let mut maximum = event.request.logits[0];
        for index in 0..count {
            let score = event.request.logits[index];
            event.candidate_ids[index].set(i32::try_from(index).map_err(|_| ())?);
            event.candidate_scores[index].set(score);
            if score > maximum {
                maximum = score;
            }
        }
        for index in 0..count {
            event.candidate_scores[index].set(event.candidate_scores[index].get() - maximum);
        }
        event.candidate_count_out.set(event.request.vocab_size);
        Ok(())
    }

    fn mark_invalid_request(&mut self, event: &Runtime<'_>) -> Result<(), ()> {
        event.dispatch_error.set(DispatchError::InvalidRequest);
        event.error_out.set(ValidatorError::InvalidRequest);
        Ok(())
    }

    fn publish_done(&mut self, event: &Runtime<'_>) -> Result<(), ()> {
        event.dispatch_error.set(DispatchError::None);
        event.error_out.set(ValidatorError::None);
        Ok(())
    }

    fn publish_error(&mut self, event: &Runtime<'_>) -> Result<(), ()> {
        event.candidate_count_out.set(0);
        event.error_out.set(ValidatorError::InvalidRequest);
        Ok(())
    }

    fn on_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// Stateful single-writer logits validator.
pub struct Validator {
    machine: LogitsValidatorStateMachine<Context>,
}

impl core::fmt::Debug for Validator {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("Validator").finish_non_exhaustive()
    }
}

impl Validator {
    /// Creates a validator in its ready state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: LogitsValidatorStateMachine::new(Context),
        }
    }

    /// Validates and prepares candidates synchronously.
    ///
    /// # Errors
    ///
    /// Returns [`ValidatorError::InvalidRequest`] when the vocabulary or
    /// caller-provided candidate capacity is invalid.
    pub fn process_event(&mut self, request: Build<'_>) -> Result<BuildResult, ValidatorError> {
        self.machine.dispatch(request)
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}
