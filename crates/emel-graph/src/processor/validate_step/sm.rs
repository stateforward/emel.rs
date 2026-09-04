//! Bounded graph-processor validation phase state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/processor/validate_step/{sm,context,events,actions,guards,errors}.hpp`.
//! The request is copied into the actor context before synchronous dispatch; callbacks receive
//! only that borrowed copied request and may not retain or re-enter the actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

/// Processor error values matching the pinned C++ processor errors.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProcessorError {
    #[default]
    None,
    InvalidRequest,
    KernelFailed,
    InternalError,
    Untracked,
    /// A callback supplied a non-zero error value.
    Callback(i32),
}

/// Outcome retained by the validation phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Callback used by the validation phase.
///
/// The callback receives a borrowed copied request and writes its error value to `err_out`.
/// Returning `false` with a zero error is treated as `ProcessorError::KernelFailed`, matching
/// the source action's `kernel_failed` fallback.
pub type ValidateFn = fn(&ProcessorExecuteRequest, &mut i32) -> bool;

/// Copied, bounded execution request fields needed by processor phases.
///
/// Pointer-bearing C++ handles are represented by bounded opaque integer handles. They are
/// carried for source-shape compatibility but are never dereferenced by this child actor.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessorExecuteRequest {
    pub step_plan: u64,
    pub output_out: u64,
    pub lifecycle: u64,
    pub tensor_machine: u64,
    pub step_index: i32,
    pub step_size: i32,
    pub kv_tokens: i32,
    pub memory_sm: u64,
    pub memory_view: u64,
    pub expected_outputs: i32,
    pub compute_ctx: u64,
    pub positions: u64,
    pub positions_count: i32,
    pub seq_masks: u64,
    pub seq_mask_words: i32,
    pub seq_masks_count: i32,
    pub seq_primary_ids: u64,
    pub seq_primary_ids_count: i32,
    pub validate: Option<ValidateFn>,
    /// Root callback retained by the owning processor for the typed bridge.
    pub root_validate: Option<crate::processor::sm::ValidateFn>,
}

/// Internal copied event corresponding to C++ `processor::event::execute_step`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessorEventExecuteStep {
    pub request: ProcessorExecuteRequest,
    /// Error retained by an earlier processor phase in the shared C++ execute context.
    pub err: ProcessorError,
}

impl ProcessorEventExecuteStep {
    /// Creates an event from a copied execution request with no prior phase error.
    #[must_use]
    pub const fn new(request: ProcessorExecuteRequest) -> Self {
        Self {
            request,
            err: ProcessorError::None,
        }
    }

    /// Creates an event with a previously retained processor error.
    #[must_use]
    pub const fn with_error(request: ProcessorExecuteRequest, err: ProcessorError) -> Self {
        Self { request, err }
    }

    /// Creates an event with the validation callback selected explicitly.
    #[must_use]
    pub const fn with_callback(mut request: ProcessorExecuteRequest, callback: ValidateFn) -> Self {
        request.validate = Some(callback);
        Self::new(request)
    }
}

sml! {
    GraphProcessorValidateStep {
        "execute_failed"_s <= *"deciding"_s + completion<ProcessorEventExecuteStep> [phase_prefailed] / mark_failed_existing_error,
        "callback_decision"_s <= "deciding"_s + completion<ProcessorEventExecuteStep> [phase_request_callback] / run_callback,
        "execute_failed"_s <= "deciding"_s + completion<ProcessorEventExecuteStep> [phase_missing_callback] / mark_failed_invalid_request,
        "executed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_ok] / mark_done,
        "execute_failed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_error] / mark_failed_callback_error,
        "execute_failed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_failed_without_error] / mark_failed_callback_without_error,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "callback_decision"_s + unexpected_event<_> / on_unexpected_from_callback_decision,
        "unexpected_event"_s <= "executed"_s + unexpected_event<_> / on_unexpected_from_executed,
        "unexpected_event"_s <= "execute_failed"_s + unexpected_event<_> / on_unexpected_from_execute_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "executed"_s = X,
        "execute_failed"_s = X,
    }
}

/// Persistent bounded context for `GraphProcessorValidateStep`.
#[derive(Debug, Default)]
pub struct GraphProcessorValidateStepContext {
    /// Copied request retained while the completion event is dispatched.
    pub request: ProcessorExecuteRequest,
    /// Phase outcome retained for the processor root.
    pub validate_outcome: PhaseOutcome,
    /// Processor error retained for the processor root.
    pub err: ProcessorError,
    /// Callback return value retained between callback decision transitions.
    pub phase_callback_ok: bool,
    /// Callback error value retained between callback decision transitions.
    pub phase_callback_err: i32,
}

impl GraphProcessorValidateStepContext {
    /// Copies a request into the single-writer actor context and resets transient callback state.
    pub fn set_request(&mut self, event: ProcessorEventExecuteStep) {
        self.request = event.request;
        self.validate_outcome = PhaseOutcome::Unknown;
        self.err = event.err;
        self.phase_callback_ok = false;
        self.phase_callback_err = 0;
    }

    /// Returns the retained phase outcome.
    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome {
        self.validate_outcome
    }

    /// Returns the retained processor error.
    #[must_use]
    pub const fn error(&self) -> ProcessorError {
        self.err
    }
}

impl GraphProcessorValidateStepStateMachineContext for GraphProcessorValidateStepContext {
    fn callback_error(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_err != 0)
    }

    fn callback_failed_without_error(&self) -> Result<bool, ()> {
        Ok(!self.phase_callback_ok && self.phase_callback_err == 0)
    }

    fn callback_ok(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_ok && self.phase_callback_err == 0)
    }

    fn mark_done(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Done;
        self.err = ProcessorError::None;
        Ok(())
    }

    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::Callback(self.phase_callback_err);
        Ok(())
    }

    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::KernelFailed;
        Ok(())
    }

    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        Ok(())
    }

    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InvalidRequest;
        Ok(())
    }

    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.validate_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn phase_missing_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.validate.is_none())
    }

    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }

    fn phase_request_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.validate.is_some())
    }

    fn run_callback(&mut self) -> Result<(), ()> {
        let mut callback_err = 0;
        let callback_ok = match self.request.validate {
            Some(callback) => callback(&self.request, &mut callback_err),
            None => return Err(()),
        };
        self.phase_callback_ok = callback_ok;
        self.phase_callback_err = callback_err;
        Ok(())
    }
}

/// Synchronous single-writer validation-phase actor.
pub struct GraphProcessorValidateStepActor {
    machine: GraphProcessorValidateStepStateMachine<GraphProcessorValidateStepContext>,
}

impl Default for GraphProcessorValidateStepActor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphProcessorValidateStepActor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphProcessorValidateStepStateMachine::new(
                GraphProcessorValidateStepContext::default(),
            ),
        }
    }

    /// Copies and dispatches one execution event synchronously.
    pub fn process_event(&mut self, event: ProcessorEventExecuteStep) -> bool {
        self.machine.context_mut().set_request(event);
        let origin = GraphProcessorValidateStepCompletionOrigin::ProcessorEventExecuteStep;
        if self.machine.process_completion(&origin) != Ok(true) {
            return false;
        }
        if self
            .machine
            .is(&GraphProcessorValidateStepStates::CallbackDecision)
        {
            self.machine.process_completion(&origin) == Ok(true)
        } else {
            true
        }
    }

    /// Records an explicit unexpected event and transitions to the generated unexpected state.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.context_mut().validate_outcome = PhaseOutcome::Failed;
        self.machine.context_mut().err = ProcessorError::InternalError;
        self.machine
            .set_state(GraphProcessorValidateStepStates::UnexpectedEvent);
        true
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphProcessorValidateStepStates {
        self.machine.state()
    }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: &GraphProcessorValidateStepStates) -> bool {
        self.machine.is(state)
    }

    /// Returns retained bounded context for outcome/error inspection.
    #[must_use]
    pub fn context(&self) -> &GraphProcessorValidateStepContext {
        self.machine.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(_request: &ProcessorExecuteRequest, err: &mut i32) -> bool {
        *err = 0;
        true
    }

    fn rejected(_request: &ProcessorExecuteRequest, err: &mut i32) -> bool {
        *err = 0;
        false
    }

    fn failed(_request: &ProcessorExecuteRequest, err: &mut i32) -> bool {
        *err = 17;
        false
    }

    #[test]
    fn validates_and_retains_done_outcome() {
        let request = ProcessorExecuteRequest {
            validate: Some(valid),
            ..ProcessorExecuteRequest::default()
        };
        let mut actor = GraphProcessorValidateStepActor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Done);
        assert_eq!(actor.context().error(), ProcessorError::None);
        assert!(actor.is(&GraphProcessorValidateStepStates::Executed));
    }

    #[test]
    fn missing_and_failed_callbacks_are_typed() {
        let mut actor = GraphProcessorValidateStepActor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::default()));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::InvalidRequest);
        assert!(actor.is(&GraphProcessorValidateStepStates::ExecuteFailed));

        let request = ProcessorExecuteRequest {
            validate: Some(rejected),
            ..ProcessorExecuteRequest::default()
        };
        let mut actor = GraphProcessorValidateStepActor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().error(), ProcessorError::KernelFailed);
    }

    #[test]
    fn callback_error_and_unexpected_event_are_internal_boundaries() {
        let request = ProcessorExecuteRequest {
            validate: Some(failed),
            ..ProcessorExecuteRequest::default()
        };
        let mut actor = GraphProcessorValidateStepActor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().error(), ProcessorError::Callback(17));

        let mut actor = GraphProcessorValidateStepActor::new();
        assert!(actor.process_unexpected_event());
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::InternalError);
        assert!(actor.is(&GraphProcessorValidateStepStates::UnexpectedEvent));
    }
}
