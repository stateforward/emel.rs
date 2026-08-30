//! Source-aligned, bounded graph-processor root actor.
//!
//! Requests and phase results are retained in a single-writer context. Phase
//! callbacks and lifecycle operations are synchronous and bounded.

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

use core::fmt;
use sml::sml;

use super::{alloc_step, bind_step, extract_step, kernel_step, prepare_step, validate_step};

/// Processor root error values from the pinned source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ProcessorError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    KernelFailed = 2,
    InternalError = 4,
    Untracked = 8,
    Callback = 16,
    Unknown = 255,
}

/// Result of a lifecycle operation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum LifecycleOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Bounded output committed at the end of an execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub outputs_produced: i32,
    pub graph_reused: u8,
    pub lifecycle: usize,
}

/// Synchronous lifecycle callback.
pub type LifecycleFn = fn(&ExecuteRequest) -> bool;
/// Synchronous completion callback.
pub type DispatchDoneFn = fn(&ExecutionOutput) -> bool;
/// Synchronous failure callback.
pub type DispatchErrorFn = fn(&ExecutionOutput, i32) -> bool;

/// Bounded projection of the pinned `event::execute` request.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecuteRequest {
    pub step_plan: usize,
    pub output_out: usize,
    pub lifecycle: usize,
    pub tensor_machine: usize,
    pub step_index: i32,
    pub step_size: i32,
    pub kv_tokens: i32,
    pub memory_sm: usize,
    pub memory_view: usize,
    pub expected_outputs: i32,
    pub compute_ctx: usize,
    pub positions: usize,
    pub positions_count: i32,
    pub seq_masks: usize,
    pub seq_mask_words: i32,
    pub seq_masks_count: i32,
    pub seq_primary_ids: usize,
    pub seq_primary_ids_count: i32,
    pub validate: Option<validate_step::ValidateFn>,
    pub prepare_graph: Option<prepare_step::PrepareGraphFn>,
    pub alloc_graph: Option<alloc_step::AllocGraphFn>,
    pub bind_inputs: Option<bind_step::BindInputsFn>,
    pub run_kernel: Option<kernel_step::RunKernelFn>,
    pub extract_outputs: Option<extract_step::ExtractOutputsFn>,
    pub lifecycle_gate: Option<LifecycleFn>,
    pub lifecycle_publish: Option<LifecycleFn>,
    pub lifecycle_release: Option<LifecycleFn>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Public execute event. The request is copied by the actor before dispatch.
#[derive(Clone, Copy, Debug, Default)]
pub struct EventExecuteStep {
    pub request: ExecuteRequest,
}

impl EventExecuteStep {
    #[must_use]
    pub const fn new(request: ExecuteRequest) -> Self { Self { request } }
}

/// Explicit event for exercising unexpected-event recovery.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

fn valid_request(request: &ExecuteRequest) -> bool {
    request.step_plan != 0
        && request.output_out != 0
        && request.lifecycle != 0
        && request.tensor_machine != 0
        && request.step_index >= 0
        && request.step_size > 0
        && request.kv_tokens >= 0
        && request.expected_outputs >= 0
        && request.positions_count >= 0
        && (request.positions_count == 0 || request.positions != 0)
        && request.seq_mask_words > 0
        && request.seq_masks_count >= 0
        && (request.seq_masks_count == 0 || request.seq_masks != 0)
        && request.seq_primary_ids_count >= 0
        && (request.seq_primary_ids_count == 0 || request.seq_primary_ids != 0)
        && (request.seq_primary_ids_count == 0 || request.positions_count > 0)
        && request.validate.is_some()
        && request.prepare_graph.is_some()
        && request.alloc_graph.is_some()
        && request.bind_inputs.is_some()
        && request.run_kernel.is_some()
        && request.extract_outputs.is_some()
        && request.lifecycle_gate.is_some()
        && request.lifecycle_publish.is_some()
        && request.lifecycle_release.is_some()
        && request.dispatch_done.is_some()
        && request.dispatch_error.is_some()
}

sml! {
    GraphProcessor {
        "validate_step_model"_s <= *"ready"_s + event<EventExecuteStep> [valid_execute] / begin_execute,
        "ready"_s <= "ready"_s + event<EventExecuteStep> [invalid_execute_with_dispatchable_output] / reject_invalid_execute_with_dispatch,
        "ready"_s <= "ready"_s + event<EventExecuteStep> [invalid_execute_with_output_only] / reject_invalid_execute_with_output_only,
        "ready"_s <= "ready"_s + event<EventExecuteStep> [invalid_execute_without_output] / reject_invalid_execute_without_output,
        "validate_decision"_s <= "validate_step_model"_s + completion<EventExecuteStep> / run_validate,
        "prepare_step_model"_s <= "validate_decision"_s + completion<EventExecuteStep> [validate_done],
        "execution_decision"_s <= "validate_decision"_s + completion<EventExecuteStep> [validate_failed],
        "prepare_decision"_s <= "prepare_step_model"_s + completion<EventExecuteStep> / run_prepare,
        "bind_step_model"_s <= "prepare_decision"_s + completion<EventExecuteStep> [prepare_done_reused],
        "alloc_step_model"_s <= "prepare_decision"_s + completion<EventExecuteStep> [prepare_done_needs_allocation],
        "execution_decision"_s <= "prepare_decision"_s + completion<EventExecuteStep> [prepare_failed],
        "alloc_decision"_s <= "alloc_step_model"_s + completion<EventExecuteStep> / run_alloc,
        "bind_step_model"_s <= "alloc_decision"_s + completion<EventExecuteStep> [alloc_done],
        "execution_decision"_s <= "alloc_decision"_s + completion<EventExecuteStep> [alloc_failed],
        "bind_decision"_s <= "bind_step_model"_s + completion<EventExecuteStep> / run_bind,
        "lifecycle_gate_decision"_s <= "bind_decision"_s + completion<EventExecuteStep> [bind_done] / request_lifecycle_gate,
        "execution_decision"_s <= "bind_decision"_s + completion<EventExecuteStep> [bind_failed],
        "kernel_step_model"_s <= "lifecycle_gate_decision"_s + completion<EventExecuteStep> [lifecycle_gate_done],
        "execution_decision"_s <= "lifecycle_gate_decision"_s + completion<EventExecuteStep> [lifecycle_gate_failed],
        "kernel_decision"_s <= "kernel_step_model"_s + completion<EventExecuteStep> / run_kernel,
        "publish_decision"_s <= "kernel_decision"_s + completion<EventExecuteStep> [kernel_done] / request_lifecycle_publish,
        "execution_decision"_s <= "kernel_decision"_s + completion<EventExecuteStep> [kernel_failed],
        "extract_step_model"_s <= "publish_decision"_s + completion<EventExecuteStep> [publish_done],
        "execution_decision"_s <= "publish_decision"_s + completion<EventExecuteStep> [publish_failed],
        "extract_decision"_s <= "extract_step_model"_s + completion<EventExecuteStep> / run_extract,
        "release_decision"_s <= "extract_decision"_s + completion<EventExecuteStep> [extract_done] / request_lifecycle_release,
        "execution_decision"_s <= "extract_decision"_s + completion<EventExecuteStep> [extract_failed],
        "execution_decision"_s <= "release_decision"_s + completion<EventExecuteStep> [release_done] / commit_output,
        "execution_decision"_s <= "release_decision"_s + completion<EventExecuteStep> [release_failed],
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_none] / dispatch_done,
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_invalid_request] / dispatch_error_from_execution_decision,
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_kernel_failed] / dispatch_error_from_execution_decision,
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_internal_error] / dispatch_error_from_execution_decision,
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_untracked] / dispatch_error_from_execution_decision,
        "ready"_s <= "execution_decision"_s + completion<EventExecuteStep> [execution_error_unknown] / dispatch_error_from_execution_decision,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "execution_decision"_s <= "validate_decision"_s + unexpected_event<_> / on_unexpected_from_validate_decision,
        "execution_decision"_s <= "prepare_decision"_s + unexpected_event<_> / on_unexpected_from_prepare_decision,
        "execution_decision"_s <= "alloc_decision"_s + unexpected_event<_> / on_unexpected_from_alloc_decision,
        "execution_decision"_s <= "bind_decision"_s + unexpected_event<_> / on_unexpected_from_bind_decision,
        "execution_decision"_s <= "lifecycle_gate_decision"_s + unexpected_event<_> / on_unexpected_from_lifecycle_gate_decision,
        "execution_decision"_s <= "kernel_decision"_s + unexpected_event<_> / on_unexpected_from_kernel_decision,
        "execution_decision"_s <= "publish_decision"_s + unexpected_event<_> / on_unexpected_from_publish_decision,
        "execution_decision"_s <= "extract_decision"_s + unexpected_event<_> / on_unexpected_from_extract_decision,
        "execution_decision"_s <= "release_decision"_s + unexpected_event<_> / on_unexpected_from_release_decision,
        "ready"_s <= "execution_decision"_s + unexpected_event<_> / on_unexpected_from_execution_decision,
    }
}

/// Actor-owned state and bounded phase results.
#[derive(Debug, Default)]
pub struct GraphProcessorContext {
    pub request: ExecuteRequest,
    pub output: ExecutionOutput,
    pub err: ProcessorError,
    pub validate_outcome: validate_step::PhaseOutcome,
    pub prepare_outcome: prepare_step::PhaseOutcome,
    pub alloc_outcome: alloc_step::PhaseOutcome,
    pub bind_outcome: bind_step::PhaseOutcome,
    pub kernel_outcome: kernel_step::PhaseOutcome,
    pub extract_outcome: extract_step::PhaseOutcome,
    pub graph_reused: u8,
    pub outputs_produced: i32,
    pub gate_outcome: LifecycleOutcome,
    pub publish_outcome: LifecycleOutcome,
    pub release_outcome: LifecycleOutcome,
    pub phase_error: i32,
    pub dispatch_generation: u64,
}

impl GraphProcessorStateMachineContext for GraphProcessorContext {
    fn valid_execute(&self, e: &EventExecuteStep) -> Result<bool, ()> { Ok(valid_request(&e.request)) }
    fn invalid_execute_with_dispatchable_output(&self, e: &EventExecuteStep) -> Result<bool, ()> { Ok(!valid_request(&e.request) && e.request.output_out != 0 && e.request.dispatch_error.is_some()) }
    fn invalid_execute_with_output_only(&self, e: &EventExecuteStep) -> Result<bool, ()> { Ok(!valid_request(&e.request) && e.request.output_out != 0 && e.request.dispatch_error.is_none()) }
    fn invalid_execute_without_output(&self, e: &EventExecuteStep) -> Result<bool, ()> { Ok(!valid_request(&e.request) && e.request.output_out == 0) }
    fn begin_execute(&mut self, e: &EventExecuteStep) -> Result<(), ()> {
        self.request = e.request; self.output = ExecutionOutput::default(); self.err = ProcessorError::None;
        self.validate_outcome = validate_step::PhaseOutcome::Unknown; self.prepare_outcome = prepare_step::PhaseOutcome::Unknown;
        self.alloc_outcome = alloc_step::PhaseOutcome::Unknown; self.bind_outcome = bind_step::PhaseOutcome::Unknown;
        self.kernel_outcome = kernel_step::PhaseOutcome::Unknown; self.extract_outcome = extract_step::PhaseOutcome::Unknown;
        self.gate_outcome = LifecycleOutcome::Unknown; self.publish_outcome = LifecycleOutcome::Unknown; self.release_outcome = LifecycleOutcome::Unknown;
        self.graph_reused = 0; self.outputs_produced = 0; self.phase_error = 0; self.dispatch_generation = self.dispatch_generation.wrapping_add(1); Ok(())
    }
    fn run_validate(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.validate.ok_or(())?; self.phase_error = 0; let ok = f(&self.request, &mut self.phase_error); self.validate_outcome = if ok && self.phase_error == 0 { validate_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; validate_step::PhaseOutcome::Failed }; Ok(()) }
    fn run_prepare(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.prepare_graph.ok_or(())?; self.phase_error = 0; let mut reused = false; let ok = f(&self.request, &mut reused, &mut self.phase_error); self.graph_reused = reused as u8; self.prepare_outcome = if ok && self.phase_error == 0 { prepare_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; prepare_step::PhaseOutcome::Failed }; Ok(()) }
    fn run_alloc(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.alloc_graph.ok_or(())?; self.phase_error = 0; let ok = f(&self.request, &mut self.phase_error); self.alloc_outcome = if ok && self.phase_error == 0 { alloc_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; alloc_step::PhaseOutcome::Failed }; Ok(()) }
    fn run_bind(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.bind_inputs.ok_or(())?; self.phase_error = 0; let ok = f(&self.request, &mut self.phase_error); self.bind_outcome = if ok && self.phase_error == 0 { bind_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; bind_step::PhaseOutcome::Failed }; Ok(()) }
    fn run_kernel(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.run_kernel.ok_or(())?; self.phase_error = 0; let ok = f(&self.request, &mut self.phase_error); self.kernel_outcome = if ok && self.phase_error == 0 { kernel_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; kernel_step::PhaseOutcome::Failed }; Ok(()) }
    fn run_extract(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let f = self.request.extract_outputs.ok_or(())?; self.phase_error = 0; let mut count = 0; let ok = f(&self.request, &mut count, &mut self.phase_error); self.outputs_produced = count; self.extract_outcome = if ok && self.phase_error == 0 { extract_step::PhaseOutcome::Done } else { self.err = if self.phase_error == 0 { ProcessorError::KernelFailed } else { ProcessorError::Callback }; extract_step::PhaseOutcome::Failed }; Ok(()) }
    fn request_lifecycle_gate(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let ok = self.request.lifecycle_gate.map_or(false, |f| f(&self.request)); self.gate_outcome = if ok { LifecycleOutcome::Done } else { LifecycleOutcome::Failed }; if !ok { self.err = ProcessorError::InternalError; } Ok(()) }
    fn request_lifecycle_publish(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let ok = self.request.lifecycle_publish.map_or(false, |f| f(&self.request)); self.publish_outcome = if ok { LifecycleOutcome::Done } else { LifecycleOutcome::Failed }; if !ok { self.err = ProcessorError::InternalError; } Ok(()) }
    fn request_lifecycle_release(&mut self, _: &EventExecuteStep) -> Result<(), ()> { let ok = self.request.lifecycle_release.map_or(false, |f| f(&self.request)); self.release_outcome = if ok { LifecycleOutcome::Done } else { LifecycleOutcome::Failed }; if !ok { self.err = ProcessorError::InternalError; } Ok(()) }
    fn commit_output(&mut self, _: &EventExecuteStep) -> Result<(), ()> { self.output = ExecutionOutput { outputs_produced: self.outputs_produced, graph_reused: self.graph_reused, lifecycle: self.request.lifecycle }; Ok(()) }
    fn dispatch_done(&mut self, _: &EventExecuteStep) -> Result<(), ()> { if let Some(f) = self.request.dispatch_done { let _ = f(&self.output); } Ok(()) }
    fn dispatch_error_from_execution_decision(&mut self, _: &EventExecuteStep) -> Result<(), ()> { if let Some(f) = self.request.dispatch_error { let _ = f(&self.output, self.err as i32); } Ok(()) }
    fn reject_invalid_execute_with_dispatch(&mut self, e: &EventExecuteStep) -> Result<(), ()> { self.request = e.request; self.err = ProcessorError::InvalidRequest; self.output = ExecutionOutput::default(); if let Some(f) = self.request.dispatch_error { let _ = f(&self.output, self.err as i32); } Ok(()) }
    fn reject_invalid_execute_with_output_only(&mut self, e: &EventExecuteStep) -> Result<(), ()> { self.request = e.request; self.err = ProcessorError::InvalidRequest; self.output = ExecutionOutput::default(); Ok(()) }
    fn reject_invalid_execute_without_output(&mut self, e: &EventExecuteStep) -> Result<(), ()> { self.request = e.request; self.err = ProcessorError::InvalidRequest; Ok(()) }
    fn validate_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.validate_outcome == validate_step::PhaseOutcome::Done) }
    fn validate_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.validate_outcome == validate_step::PhaseOutcome::Failed) }
    fn prepare_done_reused(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.prepare_outcome == prepare_step::PhaseOutcome::Done && self.graph_reused == 1) }
    fn prepare_done_needs_allocation(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.prepare_outcome == prepare_step::PhaseOutcome::Done && self.graph_reused == 0) }
    fn prepare_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.prepare_outcome == prepare_step::PhaseOutcome::Failed) }
    fn alloc_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.alloc_outcome == alloc_step::PhaseOutcome::Done) }
    fn alloc_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.alloc_outcome == alloc_step::PhaseOutcome::Failed) }
    fn bind_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.bind_outcome == bind_step::PhaseOutcome::Done) }
    fn bind_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.bind_outcome == bind_step::PhaseOutcome::Failed) }
    fn lifecycle_gate_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.gate_outcome == LifecycleOutcome::Done) }
    fn lifecycle_gate_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.gate_outcome == LifecycleOutcome::Failed) }
    fn kernel_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.kernel_outcome == kernel_step::PhaseOutcome::Done) }
    fn kernel_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.kernel_outcome == kernel_step::PhaseOutcome::Failed) }
    fn publish_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.publish_outcome == LifecycleOutcome::Done) }
    fn publish_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.publish_outcome == LifecycleOutcome::Failed) }
    fn extract_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.extract_outcome == extract_step::PhaseOutcome::Done) }
    fn extract_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.extract_outcome == extract_step::PhaseOutcome::Failed) }
    fn release_done(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None && self.release_outcome == LifecycleOutcome::Done) }
    fn release_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err != ProcessorError::None || self.release_outcome == LifecycleOutcome::Failed) }
    fn execution_error_none(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::None) }
    fn execution_error_invalid_request(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::InvalidRequest) }
    fn execution_error_kernel_failed(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::KernelFailed) }
    fn execution_error_internal_error(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::InternalError) }
    fn execution_error_untracked(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(self.err == ProcessorError::Untracked) }
    fn execution_error_unknown(&self, _: &EventExecuteStep) -> Result<bool, ()> { Ok(matches!(self.err, ProcessorError::Callback | ProcessorError::Unknown)) }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_validate_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_prepare_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_alloc_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_bind_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_lifecycle_gate_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_kernel_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_publish_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_extract_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_release_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
    fn on_unexpected_from_execution_decision(&mut self) -> Result<(), ()> { self.err = ProcessorError::InternalError; Ok(()) }
}

/// Synchronous graph processor actor.
pub struct GraphProcessor { machine: GraphProcessorStateMachine<GraphProcessorContext> }
impl fmt::Debug for GraphProcessor { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("GraphProcessor").field("state", self.machine.state()).field("context", self.machine.context()).finish() } }
impl Default for GraphProcessor { fn default() -> Self { Self::new() } }
impl GraphProcessor {
    #[must_use]
    pub fn new() -> Self { Self { machine: GraphProcessorStateMachine::new(GraphProcessorContext::default()) } }
    pub fn process_event(&mut self, event: EventExecuteStep) -> bool { if self.machine.process_event(GraphProcessorEvents::EventExecuteStep).is_err() { self.machine.context_mut().err = ProcessorError::InternalError; return false; } self.machine.context().err == ProcessorError::None }
    pub fn process_unexpected_event(&mut self) -> bool { self.machine.process_event(GraphProcessorEvents::UnexpectedEvent).is_ok() }
    #[must_use]
    pub fn state(&self) -> &GraphProcessorStates { self.machine.state() }
    #[must_use]
    pub fn is(&self, state: &GraphProcessorStates) -> bool { self.machine.is(state) }
    #[must_use]
    pub fn context(&self) -> &GraphProcessorContext { self.machine.context() }
    #[must_use]
    pub fn output(&self) -> ExecutionOutput { self.machine.context().output }
    #[must_use]
    pub fn error(&self) -> ProcessorError { self.machine.context().err }
}

/// Compatibility alias matching the pinned C++ actor name.
pub type Processor = GraphProcessor;
