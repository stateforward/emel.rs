//! Bounded graph-processor prepare phase state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/processor/prepare_step/{sm,context,events,actions,guards,errors}.hpp`.
//! The request is copied into actor-owned context before dispatch; callbacks are
//! synchronous and processing performs no allocation.

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

/// Outcome retained by this processor phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Public snapshot of the child phase boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrepareStepOutcome {
    /// Terminal state of the prepare phase.
    pub outcome: PhaseOutcome,
    /// Whether the callback reused the existing graph.
    pub reused: bool,
    /// Error retained for processor-root integration.
    pub error: ProcessorError,
}

/// Compatibility alias used by callers that name phase outcomes directly.
pub type PrepareOutcome = PrepareStepOutcome;

/// Callback corresponding to C++ `prepare_graph_fn`.
///
/// The callback receives the copied execute request, writes both output
/// values synchronously, and returns its success status.
pub type PrepareGraphFn = fn(&ProcessorExecuteRequest, &mut bool, &mut i32) -> bool;
/// Callback naming alias for processor-root integration.
pub type PrepareGraphCallback = PrepareGraphFn;

/// Copied, bounded execution request fields needed by processor phases.
///
/// Pointer-bearing C++ handles are represented by opaque integers. They are
/// carried for source-shape compatibility and are never dereferenced here.
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
    pub prepare_graph: Option<PrepareGraphFn>,
}

/// Compatibility alias for the C++ `event::execute` request role.
pub type Execute = ProcessorExecuteRequest;

/// Internal copied event corresponding to C++ `processor::event::execute_step`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessorEventExecuteStep {
    pub request: ProcessorExecuteRequest,
    /// Error retained by an earlier processor phase in the shared context.
    pub err: ProcessorError,
}

impl ProcessorEventExecuteStep {
    /// Creates an event with no pre-existing phase error.
    #[must_use]
    pub const fn new(request: ProcessorExecuteRequest) -> Self {
        Self { request, err: ProcessorError::None }
    }

    /// Creates an event with a pre-existing phase error.
    #[must_use]
    pub const fn with_error(request: ProcessorExecuteRequest, err: ProcessorError) -> Self {
        Self { request, err }
    }

    /// Creates an event selecting the prepare callback.
    #[must_use]
    pub const fn with_callback(mut request: ProcessorExecuteRequest, callback: PrepareGraphFn) -> Self {
        request.prepare_graph = Some(callback);
        Self::new(request)
    }
}

sml! {
    GraphProcessorPrepareStep {
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

/// Persistent bounded context for `GraphProcessorPrepareStep`.
#[derive(Debug, Default)]
pub struct GraphProcessorPrepareStepContext {
    /// Copied request retained for callback invocation.
    pub request: ProcessorExecuteRequest,
    /// Phase outcome retained for processor-root integration.
    pub prepare_outcome: PhaseOutcome,
    /// Error retained for processor-root integration.
    pub err: ProcessorError,
    /// Callback return value retained between callback decision transitions.
    pub phase_callback_ok: bool,
    /// Callback error value retained between callback decision transitions.
    pub phase_callback_err: i32,
    /// Reuse result returned by `prepare_graph_fn`.
    pub graph_reused: bool,
}

impl GraphProcessorPrepareStepContext {
    /// Copies an event into the single-writer context and resets transient state.
    pub fn set_request(&mut self, event: ProcessorEventExecuteStep) {
        self.request = event.request;
        self.prepare_outcome = PhaseOutcome::Unknown;
        self.err = event.err;
        self.phase_callback_ok = false;
        self.phase_callback_err = 0;
        self.graph_reused = false;
    }

    /// Returns the retained phase outcome.
    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.prepare_outcome }

    /// Returns the retained processor error.
    #[must_use]
    pub const fn error(&self) -> ProcessorError { self.err }

    /// Returns whether preparation reused an existing graph.
    #[must_use]
    pub const fn reused(&self) -> bool { self.graph_reused }

    /// Returns a public snapshot of outcome, reuse, and error.
    #[must_use]
    pub const fn outcome_snapshot(&self) -> PrepareStepOutcome {
        PrepareStepOutcome {
            outcome: self.prepare_outcome,
            reused: self.graph_reused,
            error: self.err,
        }
    }
}

impl GraphProcessorPrepareStepStateMachineContext for GraphProcessorPrepareStepContext {
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
        self.prepare_outcome = PhaseOutcome::Done;
        self.err = ProcessorError::None;
        Ok(())
    }

    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::Callback(self.phase_callback_err);
        Ok(())
    }

    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::KernelFailed;
        Ok(())
    }

    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        Ok(())
    }

    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InvalidRequest;
        Ok(())
    }

    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.prepare_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }

    fn phase_missing_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.prepare_graph.is_none())
    }

    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }

    fn phase_request_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.prepare_graph.is_some())
    }

    fn run_callback(&mut self) -> Result<(), ()> {
        let mut graph_reused = false;
        let mut callback_err = 0;
        let callback_ok = match self.request.prepare_graph {
            Some(callback) => callback(&self.request, &mut graph_reused, &mut callback_err),
            None => return Err(()),
        };
        self.graph_reused = graph_reused;
        self.phase_callback_ok = callback_ok;
        self.phase_callback_err = callback_err;
        Ok(())
    }
}

/// Synchronous single-writer prepare-phase actor.
pub struct Processor {
    machine: GraphProcessorPrepareStepStateMachine<GraphProcessorPrepareStepContext>,
}

impl Default for Processor {
    fn default() -> Self { Self::new() }
}

impl Processor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphProcessorPrepareStepStateMachine::new(GraphProcessorPrepareStepContext::default()),
        }
    }

    /// Copies and dispatches one execute event synchronously.
    pub fn process_event(&mut self, event: ProcessorEventExecuteStep) -> bool {
        self.machine.context_mut().set_request(event);
        self.machine
            .process_event(GraphProcessorPrepareStepEvents::ProcessorEventExecuteStep(event))
            .is_ok()
    }

    /// Records an explicit unexpected event and transitions to its generated state.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine
            .process_event(GraphProcessorPrepareStepEvents::UnexpectedEvent)
            .is_ok()
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphProcessorPrepareStepStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphProcessorPrepareStepStates) -> bool { self.machine.is(&state) }

    /// Returns retained bounded context for outcome, reuse, and error inspection.
    #[must_use]
    pub fn context(&self) -> &GraphProcessorPrepareStepContext { self.machine.context() }

    /// Returns a public snapshot of the retained child outcome.
    #[must_use]
    pub fn outcome(&self) -> PrepareStepOutcome { self.machine.context().outcome_snapshot() }
}

/// Short actor alias used by processor-root integration.
pub type PrepareStep = Processor;

#[cfg(test)]
mod tests {
    use super::*;

    fn success(request: &ProcessorExecuteRequest, reused: &mut bool, error: &mut i32) -> bool {
        *reused = request.step_index == 7;
        *error = 0;
        true
    }

    fn callback_error(_: &ProcessorExecuteRequest, reused: &mut bool, error: &mut i32) -> bool {
        *reused = true;
        *error = 37;
        false
    }

    fn callback_failed(_: &ProcessorExecuteRequest, reused: &mut bool, error: &mut i32) -> bool {
        *reused = false;
        *error = 0;
        false
    }

    #[test]
    fn success_preserves_reuse_and_reaches_executed() {
        let mut request = ProcessorExecuteRequest::default();
        request.step_index = 7;
        request.prepare_graph = Some(success);
        let mut actor = Processor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Done);
        assert!(actor.context().reused());
        assert_eq!(actor.context().error(), ProcessorError::None);
        assert!(actor.is(GraphProcessorPrepareStepStates::Executed));
    }

    #[test]
    fn callback_error_preserves_error_and_reuse() {
        let request = ProcessorExecuteRequest { prepare_graph: Some(callback_error), ..Default::default() };
        let mut actor = Processor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::Callback(37));
        assert!(actor.context().reused());
        assert!(actor.is(GraphProcessorPrepareStepStates::ExecuteFailed));
    }

    #[test]
    fn callback_without_error_maps_to_kernel_failure() {
        let request = ProcessorExecuteRequest { prepare_graph: Some(callback_failed), ..Default::default() };
        let mut actor = Processor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(request)));
        assert_eq!(actor.context().error(), ProcessorError::KernelFailed);
    }

    #[test]
    fn missing_and_prefailed_requests_are_rejected() {
        let mut actor = Processor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::new(ProcessorExecuteRequest::default())));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::InvalidRequest);

        let mut actor = Processor::new();
        assert!(actor.process_event(ProcessorEventExecuteStep::with_error(
            ProcessorExecuteRequest::default(),
            ProcessorError::Callback(99),
        )));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::Callback(99));
    }

    #[test]
    fn unexpected_event_is_explicit_and_inspectable() {
        let mut actor = Processor::new();
        assert!(actor.process_unexpected_event());
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), ProcessorError::InternalError);
        assert!(actor.is(GraphProcessorPrepareStepStates::UnexpectedEvent));
    }
}
