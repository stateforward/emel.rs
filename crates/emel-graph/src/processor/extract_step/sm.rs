//! Bounded graph-processor extract phase state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/processor/extract_step/{sm,context,events,actions,guards}.hpp`.
//! The request is copied into the actor context before synchronous dispatch;
//! callbacks receive only that borrowed request and may not retain or re-enter
//! the actor.

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

/// Outcome retained by a processor phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Extract callback corresponding to C++ `extract_outputs_fn`.
pub type ExtractOutputsFn = fn(&ProcessorExecuteRequest, &mut i32, &mut i32) -> bool;

/// Copied, bounded execution request fields consumed by the extract phase.
///
/// Pointer-bearing C++ handles are represented by opaque integer handles. They
/// are retained for source-shape compatibility but are never dereferenced by
/// this child actor.
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
    pub extract_outputs: Option<ExtractOutputsFn>,
}

/// Internal copied event corresponding to C++ `processor::event::execute_step`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessorEventExecuteStep {
    pub request: ProcessorExecuteRequest,
    /// Error retained by an earlier processor phase in the shared execute context.
    pub err: ProcessorError,
}

impl ProcessorEventExecuteStep {
    /// Creates an event from a copied execution request with no prior phase error.
    #[must_use]
    pub const fn new(request: ProcessorExecuteRequest) -> Self {
        Self { request, err: ProcessorError::None }
    }

    /// Creates an event with a previously retained processor error.
    #[must_use]
    pub const fn with_error(request: ProcessorExecuteRequest, err: ProcessorError) -> Self {
        Self { request, err }
    }

    /// Creates an event with the extract callback selected explicitly.
    #[must_use]
    pub const fn with_callback(mut request: ProcessorExecuteRequest, callback: ExtractOutputsFn) -> Self {
        request.extract_outputs = Some(callback);
        Self::new(request)
    }
}

sml! {
    GraphProcessorExtractStep {
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

/// Persistent bounded context for `GraphProcessorExtractStep`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphProcessorExtractStepContext {
    /// Copied request retained while completion transitions are dispatched.
    pub request: ProcessorExecuteRequest,
    /// Number of outputs reported by the callback.
    pub outputs_produced: i32,
    /// Extract phase outcome retained for processor-root integration.
    pub extract_outcome: PhaseOutcome,
    /// Processor error retained for processor-root integration.
    pub err: ProcessorError,
    /// Callback return value retained between callback decision transitions.
    pub phase_callback_ok: bool,
    /// Callback error value retained between callback decision transitions.
    pub phase_callback_err: i32,
}

impl GraphProcessorExtractStepContext {
    /// Copies a request into the single-writer actor context and resets transient state.
    pub fn set_request(&mut self, event: ProcessorEventExecuteStep) {
        self.request = event.request;
        self.outputs_produced = 0;
        self.extract_outcome = PhaseOutcome::Unknown;
        self.err = event.err;
        self.phase_callback_ok = false;
        self.phase_callback_err = 0;
    }

    /// Returns the retained phase outcome.
    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.extract_outcome }

    /// Returns the retained processor error.
    #[must_use]
    pub const fn error(&self) -> ProcessorError { self.err }
}

impl GraphProcessorExtractStepStateMachineContext for GraphProcessorExtractStepContext {
    fn callback_error(&self) -> Result<bool, ()> { Ok(self.phase_callback_err != 0) }
    fn callback_failed_without_error(&self) -> Result<bool, ()> {
        Ok(!self.phase_callback_ok && self.phase_callback_err == 0)
    }
    fn callback_ok(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_ok && self.phase_callback_err == 0)
    }
    fn mark_done(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Done;
        self.err = ProcessorError::None;
        Ok(())
    }
    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::Callback(self.phase_callback_err);
        Ok(())
    }
    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::KernelFailed;
        Ok(())
    }
    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        Ok(())
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InvalidRequest;
        Ok(())
    }
    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.extract_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn phase_missing_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.extract_outputs.is_none())
    }
    fn phase_prefailed(&self) -> Result<bool, ()> { Ok(self.err != ProcessorError::None) }
    fn phase_request_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None && self.request.extract_outputs.is_some())
    }
    fn run_callback(&mut self) -> Result<(), ()> {
        let mut outputs_produced = 0;
        let mut callback_err = 0;
        let callback_ok = match self.request.extract_outputs {
            Some(callback) => callback(&self.request, &mut outputs_produced, &mut callback_err),
            None => return Err(()),
        };
        self.outputs_produced = outputs_produced;
        self.phase_callback_ok = callback_ok;
        self.phase_callback_err = callback_err;
        Ok(())
    }
}

/// Synchronous single-writer extract-phase actor.
pub struct Processor {
    machine: GraphProcessorExtractStepStateMachine<GraphProcessorExtractStepContext>,
}

impl Default for Processor {
    fn default() -> Self { Self::new() }
}

impl Processor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphProcessorExtractStepStateMachine::new(GraphProcessorExtractStepContext::default()),
        }
    }

    /// Copies and dispatches one execution event synchronously.
    pub fn process_event(&mut self, event: ProcessorEventExecuteStep) -> bool {
        self.machine.context_mut().set_request(event);
        self.machine
            .process_event(GraphProcessorExtractStepEvents::ProcessorEventExecuteStep(event))
            .is_ok()
    }

    /// Records an explicit unexpected event and transitions to the generated unexpected state.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine
            .process_event(GraphProcessorExtractStepEvents::UnexpectedEvent)
            .is_ok()
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphProcessorExtractStepStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphProcessorExtractStepStates) -> bool { self.machine.is(state) }

    /// Returns retained bounded context for outcome/error inspection.
    #[must_use]
    pub fn context(&self) -> &GraphProcessorExtractStepContext { self.machine.context() }

    /// Returns callback-reported output count.
    #[must_use]
    pub fn outputs_produced(&self) -> i32 { self.machine.context().outputs_produced }
}
