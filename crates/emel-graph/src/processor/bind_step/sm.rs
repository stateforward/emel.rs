//! Source-aligned graph processor bind-step child actor.
//!
//! The actor mirrors `bind_step/{sm,context,events,actions,guards,errors}.hpp`
//! while retaining a copied, bounded request at the synchronous dispatch
//! boundary.  Dispatch is run-to-completion and never allocates.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_braces,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs,
)]

use sml::sml;

/// Errors carried by the processor execution context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ProcessorError {
    /// No earlier phase failed.
    None = 0,
    /// The execution request is invalid or a required callback is absent.
    InvalidRequest = 1,
    /// A callback reported failure without a specific error.
    KernelFailed = 2,
    /// The state machine received an invalid event or encountered an internal failure.
    InternalError = 4,
    /// The request refers to an error not tracked by this processor.
    Untracked = 8,
}

impl Default for ProcessorError {
    fn default() -> Self { Self::None }
}

impl ProcessorError {
    /// Converts a callback's bounded integer error channel to the processor error set.
    #[must_use]
    pub const fn from_callback_error(error: i32) -> Self {
        match error {
            1 => Self::InvalidRequest,
            2 => Self::KernelFailed,
            4 => Self::InternalError,
            8 => Self::Untracked,
            _ => Self::InternalError,
        }
    }
}

/// Outcome of this phase, matching `bind_step::events::phase_outcome`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    /// No terminal phase result has been selected.
    #[default]
    Unknown = 0,
    /// Binding completed successfully.
    Done = 1,
    /// Binding failed.
    Failed = 2,
}

/// Callback for `bind_inputs_fn(const execute&, int32_t* err_out)`.
///
/// The request is copied and cannot be retained by the callback.  Callbacks
/// must complete synchronously and must not re-enter the owning actor.
pub type BindInputsFn = fn(&ExecuteRequest, &mut i32) -> bool;
/// Source spelling retained for callers porting the C++ callback declaration.
pub type BindInputsFnType = BindInputsFn;

/// Copied bounded fields from the processor `execute` request.
///
/// Opaque C++ pointers are represented as non-owning integer identities at this
/// boundary.  The bind callback itself receives this complete copied request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecuteRequest {
    /// Opaque step-plan identity.
    pub step_plan: usize,
    /// Opaque output destination identity.
    pub output_out: usize,
    /// Opaque lifecycle manifest identity.
    pub lifecycle: usize,
    /// Opaque tensor-machine identity.
    pub tensor_machine: usize,
    /// Current step index.
    pub step_index: i32,
    /// Number of items in this step.
    pub step_size: i32,
    /// KV-cache token count.
    pub kv_tokens: i32,
    /// Opaque memory state-machine identity.
    pub memory_sm: usize,
    /// Opaque memory-view identity.
    pub memory_view: usize,
    /// Expected output count.
    pub expected_outputs: i32,
    /// Opaque compute-context identity.
    pub compute_ctx: usize,
    /// Opaque positions-buffer identity.
    pub positions: usize,
    /// Number of retained positions.
    pub positions_count: i32,
    /// Opaque sequence-mask identity.
    pub seq_masks: usize,
    /// Number of words per sequence mask.
    pub seq_mask_words: i32,
    /// Number of sequence masks.
    pub seq_masks_count: i32,
    /// Opaque primary-sequence-id identity.
    pub seq_primary_ids: usize,
    /// Number of primary sequence ids.
    pub seq_primary_ids_count: i32,
    /// Input-binding callback.
    pub bind_inputs: Option<BindInputsFn>,
}

/// Runtime event consumed by the bind child state machine.
///
/// `initial_error` is the retained error from an earlier processor phase and
/// drives the source `phase_prefailed` guard before callback selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessorEventExecuteStep {
    /// Copied execute request.
    pub request: ExecuteRequest,
    /// Error retained from an earlier phase.
    pub initial_error: ProcessorError,
}

/// Explicit event used by the synchronous wrapper to exercise unexpected paths.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

// Source mapping: bind_step/sm.hpp.  Runtime selection remains in guards;
// actions only perform the bounded operation selected by a transition.
sml! {
    GraphProcessorBindStep {
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

/// Context retained by the generated bind state machine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphProcessorBindStepContext {
    /// Copied request used by the completion phases.
    pub request: ExecuteRequest,
    /// Existing processor error entering this phase or the phase error set by actions.
    pub err: ProcessorError,
    /// Retained bind callback result.
    pub phase_callback_ok: bool,
    /// Retained callback error channel.
    pub phase_callback_err: i32,
    /// Terminal bind outcome.
    pub bind_outcome: PhaseOutcome,
}

impl GraphProcessorBindStepContext {
    fn set_event(&mut self, event: ProcessorEventExecuteStep) {
        self.request = event.request;
        self.err = event.initial_error;
        self.phase_callback_ok = false;
        self.phase_callback_err = 0;
        self.bind_outcome = PhaseOutcome::Unknown;
    }

    fn mark_unexpected(&mut self) {
        self.bind_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
    }
}

impl GraphProcessorBindStepStateMachineContext for GraphProcessorBindStepContext {
    // Source mapping: bind_step/guards.hpp::callback_error.
    fn callback_error(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_err != 0)
    }

    // Source mapping: bind_step/guards.hpp::callback_failed_without_error.
    fn callback_failed_without_error(&self) -> Result<bool, ()> {
        Ok(!self.phase_callback_ok && self.phase_callback_err == 0)
    }

    // Source mapping: bind_step/guards.hpp::callback_ok.
    fn callback_ok(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_ok && self.phase_callback_err == 0)
    }

    // Source mapping: bind_step/actions.hpp::mark_done.
    fn mark_done(&mut self) -> Result<(), ()> {
        self.bind_outcome = PhaseOutcome::Done;
        self.err = ProcessorError::None;
        Ok(())
    }

    // Source mapping: bind_step/actions.hpp::mark_failed_callback_error.
    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        self.bind_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::from_callback_error(self.phase_callback_err);
        Ok(())
    }

    // Source mapping: bind_step/actions.hpp::mark_failed_callback_without_error.
    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        self.bind_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::KernelFailed;
        Ok(())
    }

    // Source mapping: bind_step/actions.hpp::mark_failed_existing_error.
    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        self.bind_outcome = PhaseOutcome::Failed;
        Ok(())
    }

    // Source mapping: bind_step/actions.hpp::mark_failed_invalid_request.
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.bind_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InvalidRequest;
        Ok(())
    }

    // Source mapping: bind_step/actions.hpp::on_unexpected.
    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }

    // Source mapping: bind_step/guards.hpp::phase_missing_callback.
    fn phase_missing_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.bind_inputs.is_none())
    }

    // Source mapping: bind_step/guards.hpp::phase_prefailed.
    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }

    // Source mapping: bind_step/guards.hpp::phase_request_callback.
    fn phase_request_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.bind_inputs.is_some())
    }

    // Source mapping: bind_step/actions.hpp::run_callback.
    fn run_callback(&mut self) -> Result<(), ()> {
        let callback = self.request.bind_inputs.expect("phase_request_callback selected");
        let mut callback_error = 0_i32;
        self.phase_callback_ok = callback(&self.request, &mut callback_error);
        self.phase_callback_err = callback_error;
        Ok(())
    }
}

/// Synchronous actor around the generated bind-step state machine.
pub struct GraphProcessorBindStepActor {
    machine: GraphProcessorBindStepStateMachine<GraphProcessorBindStepContext>,
}

impl Default for GraphProcessorBindStepActor {
    fn default() -> Self { Self::new() }
}

impl GraphProcessorBindStepActor {
    /// Creates an actor in the generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphProcessorBindStepStateMachine::new(GraphProcessorBindStepContext::default()),
        }
    }

    /// Dispatches one copied request synchronously to a terminal phase state.
    pub fn process_event(&mut self, event: ProcessorEventExecuteStep) -> bool {
        if !self.machine.is(&GraphProcessorBindStepStates::Deciding) {
            self.machine.context_mut().mark_unexpected();
            return false;
        }
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphProcessorBindStepEvents::ProcessorEventExecuteStep(event))
            .is_ok()
    }

    /// Dispatches an explicit unexpected event and records an internal failure.
    pub fn process_unexpected(&mut self) -> bool {
        self.machine.context_mut().mark_unexpected();
        self.machine
            .set_state(GraphProcessorBindStepStates::UnexpectedEvent);
        false
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GraphProcessorBindStepStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GraphProcessorBindStepStates) -> bool { self.machine.is(state) }

    /// Returns the retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphProcessorBindStepContext { self.machine.context() }

    /// Returns mutable context for orchestrator phase setup and inspection.
    #[must_use]
    pub fn context_mut(&mut self) -> &mut GraphProcessorBindStepContext { self.machine.context_mut() }
}

/// Short actor alias retained for processor callers.
pub type BindStep = GraphProcessorBindStepActor;

#[cfg(test)]
mod tests {
    use super::*;

    fn succeeds(_: &ExecuteRequest, error: &mut i32) -> bool {
        *error = 0;
        true
    }

    fn fails_without_error(_: &ExecuteRequest, error: &mut i32) -> bool {
        *error = 0;
        false
    }

    fn fails_with_error(_: &ExecuteRequest, error: &mut i32) -> bool {
        *error = 4;
        false
    }

    #[test]
    fn callback_success_is_retained() {
        let mut actor = BindStep::new();
        assert!(actor.process_event(ProcessorEventExecuteStep {
            request: ExecuteRequest { bind_inputs: Some(succeeds), ..ExecuteRequest::default() },
            initial_error: ProcessorError::None,
        }));
        assert_eq!(actor.context().bind_outcome, PhaseOutcome::Done);
        assert_eq!(actor.context().err, ProcessorError::None);
        assert!(actor.is(&GraphProcessorBindStepStates::Executed));
    }

    #[test]
    fn callback_failure_modes_are_distinct() {
        let mut actor = BindStep::new();
        assert!(actor.process_event(ProcessorEventExecuteStep {
            request: ExecuteRequest { bind_inputs: Some(fails_with_error), ..ExecuteRequest::default() },
            initial_error: ProcessorError::None,
        }));
        assert_eq!(actor.context().bind_outcome, PhaseOutcome::Failed);
        assert_eq!(actor.context().err, ProcessorError::InternalError);

        let mut actor = BindStep::new();
        assert!(actor.process_event(ProcessorEventExecuteStep {
            request: ExecuteRequest { bind_inputs: Some(fails_without_error), ..ExecuteRequest::default() },
            initial_error: ProcessorError::None,
        }));
        assert_eq!(actor.context().err, ProcessorError::KernelFailed);
    }

    #[test]
    fn missing_and_prefailed_requests_do_not_call_back() {
        let mut actor = BindStep::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::default()));
        assert_eq!(actor.context().bind_outcome, PhaseOutcome::Failed);
        assert_eq!(actor.context().err, ProcessorError::InvalidRequest);

        let mut actor = BindStep::new();
        assert!(actor.process_event(ProcessorEventExecuteStep {
            request: ExecuteRequest { bind_inputs: Some(succeeds), ..ExecuteRequest::default() },
            initial_error: ProcessorError::InternalError,
        }));
        assert_eq!(actor.context().bind_outcome, PhaseOutcome::Failed);
        assert_eq!(actor.context().err, ProcessorError::InternalError);
    }
}
