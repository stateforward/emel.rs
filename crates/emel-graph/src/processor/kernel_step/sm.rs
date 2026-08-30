//! Source-aligned, allocation-free graph processor kernel-step actor.
//!
//! The public request is a bounded projection of the pinned C++ `execute`
//! event. Pointer-bearing C++ fields are represented as caller-owned opaque
//! identities; the kernel callback remains a synchronous function pointer.

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
    missing_docs,
    private_interfaces
)]

use core::fmt;

use sml::sml;

/// Numeric processor error retained by the phase context.
///
/// The newtype preserves callback-provided error codes instead of narrowing
/// them to the processor's built-in values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct ProcessorError(pub i32);

impl ProcessorError {
    /// No error.
    pub const NONE: Self = Self(0);
    /// The request did not provide the required kernel callback.
    pub const INVALID_REQUEST: Self = Self(1);
    /// The kernel callback returned false without an error code.
    pub const KERNEL_FAILED: Self = Self(2);
    /// An event was delivered outside the child contract.
    pub const INTERNAL_ERROR: Self = Self(4);
    /// An error could not be associated with a phase.
    pub const UNTRACKED: Self = Self(8);

    /// Retains a callback's raw error code.
    #[must_use]
    pub const fn from_code(code: i32) -> Self { Self(code) }
}

/// Outcome retained for the kernel phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    /// No kernel decision has been made.
    #[default]
    Unknown = 0,
    /// The kernel callback completed successfully.
    Done = 1,
    /// The kernel callback or request failed.
    Failed = 2,
}

/// Caller-owned opaque identity for pointer-bearing source fields.
pub type OpaqueIdentity = usize;

/// Bounded lifecycle tensor binding projection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecycleTensorBinding {
    pub tensor_id: i32,
    pub buffer: OpaqueIdentity,
    pub buffer_bytes: u64,
    pub consumer_refs: i32,
    pub is_leaf: bool,
}

/// Bounded lifecycle phase projection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecyclePhase {
    pub required_filled_ids: OpaqueIdentity,
    pub required_filled_count: i32,
    pub publish_ids: OpaqueIdentity,
    pub publish_count: i32,
    pub release_ids: OpaqueIdentity,
    pub release_count: i32,
}

/// Bounded lifecycle manifest projection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecycleManifest {
    pub tensors: OpaqueIdentity,
    pub tensor_count: i32,
    pub phase: OpaqueIdentity,
}

/// Bounded output projection used by completion callbacks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub outputs_produced: i32,
    pub graph_reused: u8,
    pub lifecycle: OpaqueIdentity,
}

/// Synchronous validation callback corresponding to C++ `validate_fn`.
pub type ValidateFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Synchronous graph-preparation callback corresponding to C++ `prepare_graph_fn`.
pub type PrepareGraphFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Synchronous graph-allocation callback corresponding to C++ `alloc_graph_fn`.
pub type AllocGraphFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Synchronous input-binding callback corresponding to C++ `bind_inputs_fn`.
pub type BindInputsFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Synchronous kernel callback corresponding to C++ `run_kernel_fn`.
pub type RunKernelFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Synchronous output-extraction callback corresponding to C++ `extract_outputs_fn`.
pub type ExtractOutputsFn = fn(&ExecuteRequest, &mut i32) -> bool;

/// Optional callback used by the complete processor root.
pub type DispatchDoneFn = fn(&ExecutionOutput) -> bool;

/// Optional callback used by the complete processor root for errors.
pub type DispatchErrorFn = fn(&ExecutionOutput, i32) -> bool;

/// Copied, bounded projection of the pinned processor `event::execute`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecuteRequest {
    pub step_plan: OpaqueIdentity,
    pub output_out: OpaqueIdentity,
    pub lifecycle: OpaqueIdentity,
    pub tensor_machine: OpaqueIdentity,
    pub step_index: i32,
    pub step_size: i32,
    pub kv_tokens: i32,
    pub memory_sm: OpaqueIdentity,
    pub memory_view: OpaqueIdentity,
    pub expected_outputs: i32,
    pub compute_ctx: OpaqueIdentity,
    pub positions: OpaqueIdentity,
    pub positions_count: i32,
    pub seq_masks: OpaqueIdentity,
    pub seq_mask_words: i32,
    pub seq_masks_count: i32,
    pub seq_primary_ids: OpaqueIdentity,
    pub seq_primary_ids_count: i32,
    pub validate: Option<ValidateFn>,
    pub prepare_graph: Option<PrepareGraphFn>,
    pub alloc_graph: Option<AllocGraphFn>,
    pub bind_inputs: Option<BindInputsFn>,
    pub run_kernel: Option<RunKernelFn>,
    pub extract_outputs: Option<ExtractOutputsFn>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Runtime event consumed by the generated child machine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessorEventExecuteStep {
    /// Copied execute request for this synchronous dispatch.
    pub request: ExecuteRequest,
    /// Error already produced by an earlier processor phase.
    pub initial_error: ProcessorError,
}
/// Explicit event used to exercise unexpected-event handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

sml! {
    GraphProcessorKernelStep {
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

/// Context retained by `GraphProcessorKernelStep`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphProcessorKernelStepContext {
    /// Request copied before generated dispatch.
    pub request: ExecuteRequest,
    /// Result of the last kernel callback.
    pub phase_callback_ok: bool,
    /// Error code written by the last kernel callback.
    pub phase_callback_err: i32,
    /// Kernel phase result.
    pub kernel_outcome: PhaseOutcome,
    /// Error retained for the processor root.
    pub err: ProcessorError,
}

impl GraphProcessorKernelStepContext {
    fn set_request(&mut self, event: ProcessorEventExecuteStep) {
        self.request = event.request;
        self.phase_callback_ok = false;
        self.phase_callback_err = 0;
        self.kernel_outcome = PhaseOutcome::Unknown;
        self.err = event.initial_error;
    }

    fn mark_unexpected(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::INTERNAL_ERROR;
        Ok(())
    }
}

impl GraphProcessorKernelStepStateMachineContext for GraphProcessorKernelStepContext {
    fn callback_error(&self) -> Result<bool, ()> { Ok(self.phase_callback_err != 0) }

    fn callback_failed_without_error(&self) -> Result<bool, ()> {
        Ok(!self.phase_callback_ok && self.phase_callback_err == 0)
    }

    fn callback_ok(&self) -> Result<bool, ()> {
        Ok(self.phase_callback_ok && self.phase_callback_err == 0)
    }

    fn mark_done(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Done;
        self.err = ProcessorError::NONE;
        Ok(())
    }

    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::from_code(self.phase_callback_err);
        Ok(())
    }

    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::KERNEL_FAILED;
        Ok(())
    }

    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Failed;
        Ok(())
    }

    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.kernel_outcome = PhaseOutcome::Failed;
        self.err = ProcessorError::INVALID_REQUEST;
        Ok(())
    }

    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> { self.mark_unexpected() }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.mark_unexpected() }
    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> { self.mark_unexpected() }
    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> { self.mark_unexpected() }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.mark_unexpected() }

    fn phase_missing_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::NONE && self.request.run_kernel.is_none())
    }

    fn phase_prefailed(&self) -> Result<bool, ()> { Ok(self.err != ProcessorError::NONE) }

    fn phase_request_callback(&self) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::NONE && self.request.run_kernel.is_some())
    }

    fn run_callback(&mut self) -> Result<(), ()> {
        self.phase_callback_err = 0;
        self.phase_callback_ok = self
            .request
            .run_kernel
            .map_or(false, |callback| callback(&self.request, &mut self.phase_callback_err));
        Ok(())
    }
}

/// Synchronous, single-writer kernel-step actor.
pub struct GraphProcessorKernelStepActor {
    machine: GraphProcessorKernelStepStateMachine<GraphProcessorKernelStepContext>,
}

impl fmt::Debug for GraphProcessorKernelStepActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GraphProcessorKernelStepActor")
            .field("state", self.machine.state())
            .field("context", self.machine.context())
            .finish()
    }
}

impl Default for GraphProcessorKernelStepActor {
    fn default() -> Self { Self::new() }
}

impl GraphProcessorKernelStepActor {
    /// Constructs an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: GraphProcessorKernelStepStateMachine::new(Default::default()) }
    }

    /// Processes one copied request synchronously through the child machine.
    pub fn process_event(
        &mut self,
        event: ProcessorEventExecuteStep,
    ) -> Result<PhaseOutcome, ProcessorError> {
        self.machine.context_mut().set_request(event);
        if self
            .machine
            .process_event(GraphProcessorKernelStepEvents::ProcessorEventExecuteStep(event))
            .is_err()
        {
            self.machine.context_mut().mark_unexpected().ok();
        }
        let context = self.machine.context();
        match context.err {
            ProcessorError::NONE => Ok(context.kernel_outcome),
            error => Err(error),
        }
    }

    /// Processes an explicit unexpected event synchronously.
    pub fn process_unexpected(&mut self) -> Result<PhaseOutcome, ProcessorError> {
        self.machine.context_mut().mark_unexpected().ok();
        self.machine.set_state(GraphProcessorKernelStepStates::UnexpectedEvent);
        Err(ProcessorError::INTERNAL_ERROR)
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GraphProcessorKernelStepStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GraphProcessorKernelStepStates) -> bool { self.machine.is(state) }

    /// Returns retained phase context for root integration.
    #[must_use]
    pub fn context(&self) -> &GraphProcessorKernelStepContext { self.machine.context() }
}

/// Short actor alias for processor callers.
pub type KernelStep = GraphProcessorKernelStepActor;
