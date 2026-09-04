//! Source-aligned, bounded graph-processor root actor.
//!
//! Requests and phase results are transported through the synchronous generated
//! dispatch; only child actor ownership is retained in the persistent context.
//! Phase callbacks and lifecycle operations are synchronous and bounded.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::redundant_field_names,
    clippy::struct_field_names,
    clippy::unnecessary_map_or,
    dead_code,
    unused_imports,
    missing_docs
)]
use core::fmt;
use sml::sml;

use super::{alloc_step, bind_step, extract_step, kernel_step, prepare_step, validate_step};
use alloc_step::sm::GraphProcessorAllocStepActor as AllocActor;
use bind_step::sm::GraphProcessorBindStepActor as BindActor;
use extract_step::sm::GraphProcessorExtractStepActor as ExtractActor;
use kernel_step::sm::GraphProcessorKernelStepActor as KernelActor;
use prepare_step::sm::GraphProcessorPrepareStepActor as PrepareActor;
use validate_step::sm::GraphProcessorValidateStepActor as ValidateActor;

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

/// Typed lifecycle phase IDs projected from the graph manifest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecyclePhase<'a> {
    pub required_filled_ids: &'a [usize],
    pub publish_ids: &'a [usize],
    pub release_ids: &'a [usize],
}

/// Borrowed lifecycle manifest supplied by the graph root.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecycleManifest<'a> {
    pub phase: LifecyclePhase<'a>,
}

/// Synchronous borrowed lifecycle driver over actor-owned graph tensors.
#[derive(Debug)]
pub struct LifecycleDriver<'tensor, 'manifest> {
    tensor: &'tensor mut crate::tensor::sm::GraphTensor,
    manifest: LifecycleManifest<'manifest>,
}

impl<'tensor, 'manifest> LifecycleDriver<'tensor, 'manifest> {
    pub const fn new(
        tensor: &'tensor mut crate::tensor::sm::GraphTensor,
        manifest: LifecycleManifest<'manifest>,
    ) -> Self {
        Self { tensor, manifest }
    }

    pub fn gate(&mut self) -> bool {
        let mut ok = true;
        for &tensor_id in self.manifest.phase.required_filled_ids {
            let captured = self.tensor.process_event(crate::tensor::sm::Event::Capture(
                crate::tensor::sm::CaptureTensorState { tensor_id },
            ));
            ok &= matches!(
                captured,
                Ok(crate::tensor::sm::Outcome::State(
                    crate::tensor::sm::TensorState {
                        lifecycle: crate::tensor::sm::Lifecycle::Filled
                            | crate::tensor::sm::Lifecycle::LeafFilled,
                        ..
                    }
                ))
            );
        }
        for &tensor_id in self.manifest.phase.publish_ids {
            let captured = self.tensor.process_event(crate::tensor::sm::Event::Capture(
                crate::tensor::sm::CaptureTensorState { tensor_id },
            ));
            ok &= matches!(
                captured,
                Ok(crate::tensor::sm::Outcome::State(
                    crate::tensor::sm::TensorState {
                        lifecycle: crate::tensor::sm::Lifecycle::Empty,
                        ..
                    }
                ))
            );
        }
        ok
    }

    pub fn publish(&mut self) -> bool {
        let mut ok = true;
        for &tensor_id in self.manifest.phase.publish_ids {
            ok &= self
                .tensor
                .process_event(crate::tensor::sm::Event::PublishFilled(
                    crate::tensor::sm::PublishFilledTensor { tensor_id },
                ))
                .is_ok();
        }
        ok
    }

    pub fn release(&mut self) -> bool {
        let mut ok = true;
        for &tensor_id in self.manifest.phase.release_ids {
            ok &= self
                .tensor
                .process_event(crate::tensor::sm::Event::Release(
                    crate::tensor::sm::ReleaseTensorRef { tensor_id },
                ))
                .is_ok();
        }
        ok
    }
}

/// Bounded output committed at the end of an execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionOutput {
    pub outputs_produced: i32,
    pub graph_reused: u8,
    pub lifecycle: usize,
}

/// Synchronous phase callback used by the owned child actor bridges.
pub type ValidateFn = fn(&ExecuteRequest, &mut i32) -> bool;
pub type PrepareGraphFn = fn(&ExecuteRequest, &mut bool, &mut i32) -> bool;
pub type AllocGraphFn = fn(&ExecuteRequest, &mut i32) -> bool;
pub type BindInputsFn = fn(&ExecuteRequest, &mut i32) -> bool;
pub type RunKernelFn = fn(&ExecuteRequest, &mut i32) -> bool;
pub type ExtractOutputsFn = fn(&ExecuteRequest, &mut i32, &mut i32) -> bool;
/// Synchronous lifecycle operation callback.
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
    pub validate: Option<ValidateFn>,
    pub prepare_graph: Option<PrepareGraphFn>,
    pub alloc_graph: Option<AllocGraphFn>,
    pub bind_inputs: Option<BindInputsFn>,
    pub run_kernel: Option<RunKernelFn>,
    pub extract_outputs: Option<ExtractOutputsFn>,
    pub lifecycle_gate: Option<LifecycleFn>,
    pub lifecycle_publish: Option<LifecycleFn>,
    pub lifecycle_release: Option<LifecycleFn>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Public execute event. The request is copied by the actor before dispatch.
#[derive(Clone, Copy, Debug)]
pub struct EventExecuteStep {
    pub request: ExecuteRequest,
}

impl EventExecuteStep {
    #[must_use]
    pub const fn new(request: ExecuteRequest) -> Self {
        Self { request }
    }
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
        && (request.seq_primary_ids_count == 0 && request.positions_count == 0
            || request.seq_primary_ids_count > 0 && request.positions_count > 0)
        && request.extract_outputs.is_some()
        && request.lifecycle_gate.is_some()
        && request.lifecycle_publish.is_some()
        && request.lifecycle_release.is_some()
        && request.dispatch_done.is_some()
        && request.dispatch_error.is_some()
}
fn handle_usize(value: u64) -> usize {
    usize::try_from(value).expect("opaque handle must fit the platform usize")
}

fn handle_u64(value: usize) -> u64 {
    u64::try_from(value).expect("opaque handle must fit in u64")
}

#[allow(clippy::too_many_arguments)]
fn bridge_request(
    step_plan: u64,
    output_out: u64,
    lifecycle: u64,
    tensor_machine: u64,
    step_index: i32,
    step_size: i32,
    kv_tokens: i32,
    memory_sm: u64,
    memory_view: u64,
    expected_outputs: i32,
    compute_ctx: u64,
    positions: u64,
    positions_count: i32,
    seq_masks: u64,
    seq_mask_words: i32,
    seq_masks_count: i32,
    seq_primary_ids: u64,
    seq_primary_ids_count: i32,
) -> ExecuteRequest {
    ExecuteRequest {
        step_plan: handle_usize(step_plan),
        output_out: handle_usize(output_out),
        lifecycle: handle_usize(lifecycle),
        tensor_machine: handle_usize(tensor_machine),
        step_index,
        step_size,
        kv_tokens,
        memory_sm: handle_usize(memory_sm),
        memory_view: handle_usize(memory_view),
        expected_outputs,
        compute_ctx: handle_usize(compute_ctx),
        positions: handle_usize(positions),
        positions_count,
        seq_masks: handle_usize(seq_masks),
        seq_mask_words,
        seq_masks_count,
        seq_primary_ids: handle_usize(seq_primary_ids),
        seq_primary_ids_count,
        ..ExecuteRequest::default()
    }
}
fn bridge_validate(r: &validate_step::sm::ProcessorExecuteRequest, e: &mut i32) -> bool {
    r.root_validate.map_or(false, |f| {
        f(
            &bridge_request(
                r.step_plan,
                r.output_out,
                r.lifecycle,
                r.tensor_machine,
                r.step_index,
                r.step_size,
                r.kv_tokens,
                r.memory_sm,
                r.memory_view,
                r.expected_outputs,
                r.compute_ctx,
                r.positions,
                r.positions_count,
                r.seq_masks,
                r.seq_mask_words,
                r.seq_masks_count,
                r.seq_primary_ids,
                r.seq_primary_ids_count,
            ),
            e,
        )
    })
}
fn bridge_prepare(
    r: &prepare_step::sm::ProcessorExecuteRequest,
    reused: &mut bool,
    e: &mut i32,
) -> bool {
    r.root_prepare_graph.map_or(false, |f| {
        f(
            &bridge_request(
                r.step_plan,
                r.output_out,
                r.lifecycle,
                r.tensor_machine,
                r.step_index,
                r.step_size,
                r.kv_tokens,
                r.memory_sm,
                r.memory_view,
                r.expected_outputs,
                r.compute_ctx,
                r.positions,
                r.positions_count,
                r.seq_masks,
                r.seq_mask_words,
                r.seq_masks_count,
                r.seq_primary_ids,
                r.seq_primary_ids_count,
            ),
            reused,
            e,
        )
    })
}
fn bridge_alloc(r: &alloc_step::sm::ProcessorExecuteRequest, e: &mut i32) -> bool {
    r.root_alloc_graph.map_or(false, |f| {
        f(
            &bridge_request(
                r.step_plan,
                r.output_out,
                r.lifecycle,
                r.tensor_machine,
                r.step_index,
                r.step_size,
                r.kv_tokens,
                r.memory_sm,
                r.memory_view,
                r.expected_outputs,
                r.compute_ctx,
                r.positions,
                r.positions_count,
                r.seq_masks,
                r.seq_mask_words,
                r.seq_masks_count,
                r.seq_primary_ids,
                r.seq_primary_ids_count,
            ),
            e,
        )
    })
}
fn bridge_bind(r: &bind_step::sm::ExecuteRequest, e: &mut i32) -> bool {
    r.root_bind_inputs.map_or(false, |f| {
        f(
            &bridge_request(
                handle_u64(r.step_plan),
                handle_u64(r.output_out),
                handle_u64(r.lifecycle),
                handle_u64(r.tensor_machine),
                r.step_index,
                r.step_size,
                r.kv_tokens,
                handle_u64(r.memory_sm),
                handle_u64(r.memory_view),
                r.expected_outputs,
                handle_u64(r.compute_ctx),
                handle_u64(r.positions),
                r.positions_count,
                handle_u64(r.seq_masks),
                r.seq_mask_words,
                r.seq_masks_count,
                handle_u64(r.seq_primary_ids),
                r.seq_primary_ids_count,
            ),
            e,
        )
    })
}
fn bridge_kernel(r: &kernel_step::sm::ExecuteRequest, e: &mut i32) -> bool {
    r.root_run_kernel.map_or(false, |f| {
        f(
            &bridge_request(
                handle_u64(r.step_plan),
                handle_u64(r.output_out),
                handle_u64(r.lifecycle),
                handle_u64(r.tensor_machine),
                r.step_index,
                r.step_size,
                r.kv_tokens,
                handle_u64(r.memory_sm),
                handle_u64(r.memory_view),
                r.expected_outputs,
                handle_u64(r.compute_ctx),
                handle_u64(r.positions),
                r.positions_count,
                handle_u64(r.seq_masks),
                r.seq_mask_words,
                r.seq_masks_count,
                handle_u64(r.seq_primary_ids),
                r.seq_primary_ids_count,
            ),
            e,
        )
    })
}
fn bridge_extract(
    r: &extract_step::sm::ProcessorExecuteRequest,
    out: &mut i32,
    e: &mut i32,
) -> bool {
    r.root_extract_outputs.map_or(false, |f| {
        f(
            &bridge_request(
                r.step_plan,
                r.output_out,
                r.lifecycle,
                r.tensor_machine,
                r.step_index,
                r.step_size,
                r.kv_tokens,
                r.memory_sm,
                r.memory_view,
                r.expected_outputs,
                r.compute_ctx,
                r.positions,
                r.positions_count,
                r.seq_masks,
                r.seq_mask_words,
                r.seq_masks_count,
                r.seq_primary_ids,
                r.seq_primary_ids_count,
            ),
            out,
            e,
        )
    })
}
sml! {
    GraphProcessor [temporary_context: &mut LifecycleDriver<'_, '_>] {
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

/// Synchronous graph processor actor.
pub struct GraphProcessor {
    machine: GraphProcessorStateMachine<GraphProcessorContext>,
    /// Reusable empty lifecycle storage for the callback-driven API.
    tensor: crate::tensor::sm::GraphTensor,
}

/// Actor-owned state and bounded phase results.
///
/// Dispatch state is intentionally private; callers observe only the immutable
/// [`ProcessorSnapshot`] returned by [`GraphProcessor::snapshot`].
pub struct GraphProcessorContext {
    output: ExecutionOutput,
    err: ProcessorError,
    validate_actor: ValidateActor,
    prepare_actor: PrepareActor,
    alloc_actor: AllocActor,
    bind_actor: BindActor,
    kernel_actor: KernelActor,
    extract_actor: ExtractActor,
}

impl Default for GraphProcessorContext {
    fn default() -> Self {
        Self {
            output: ExecutionOutput::default(),
            err: ProcessorError::None,
            validate_actor: ValidateActor::new(),
            prepare_actor: PrepareActor::new(),
            alloc_actor: AllocActor::new(),
            bind_actor: BindActor::new(),
            kernel_actor: KernelActor::new(),
            extract_actor: ExtractActor::new(),
        }
    }
}

impl fmt::Debug for GraphProcessorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphProcessorContext")
            .field("output", &self.output)
            .field("err", &self.err)
            .finish_non_exhaustive()
    }
}

impl GraphProcessorStateMachineContext for GraphProcessorContext {
    fn valid_execute(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(valid_request(&e.request))
    }
    fn invalid_execute_with_dispatchable_output(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(!valid_request(&e.request)
            && e.request.output_out != 0
            && e.request.dispatch_error.is_some())
    }
    fn invalid_execute_with_output_only(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(!valid_request(&e.request)
            && e.request.output_out != 0
            && e.request.dispatch_error.is_none())
    }
    fn invalid_execute_without_output(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(!valid_request(&e.request) && e.request.output_out == 0)
    }
    fn begin_execute(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        self.output = ExecutionOutput::default();
        self.err = ProcessorError::None;
        // Each nested actor reaches a terminal state after dispatch. Recreate
        // children at the valid-execute boundary while retaining actor ownership
        // in the persistent root context.
        self.validate_actor = ValidateActor::new();
        self.prepare_actor = PrepareActor::new();
        self.alloc_actor = AllocActor::new();
        self.bind_actor = BindActor::new();
        self.kernel_actor = KernelActor::new();
        self.extract_actor = ExtractActor::new();
        let _ = e;
        Ok(())
    }
    fn run_validate(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = validate_step::sm::ProcessorExecuteRequest {
            step_plan: handle_u64(r.step_plan),
            output_out: handle_u64(r.output_out),
            lifecycle: handle_u64(r.lifecycle),
            tensor_machine: handle_u64(r.tensor_machine),
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: handle_u64(r.memory_sm),
            memory_view: handle_u64(r.memory_view),
            expected_outputs: r.expected_outputs,
            compute_ctx: handle_u64(r.compute_ctx),
            positions: handle_u64(r.positions),
            positions_count: r.positions_count,
            seq_masks: handle_u64(r.seq_masks),
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: handle_u64(r.seq_primary_ids),
            seq_primary_ids_count: r.seq_primary_ids_count,
            validate: Some(bridge_validate),
            root_validate: r.validate,
        };
        let _ = self
            .validate_actor
            .process_event(validate_step::sm::ProcessorEventExecuteStep::new(child));
        self.err = match self.validate_actor.context().err {
            validate_step::sm::ProcessorError::None => ProcessorError::None,
            validate_step::sm::ProcessorError::InvalidRequest => ProcessorError::InvalidRequest,
            validate_step::sm::ProcessorError::KernelFailed => ProcessorError::KernelFailed,
            validate_step::sm::ProcessorError::InternalError => ProcessorError::InternalError,
            validate_step::sm::ProcessorError::Untracked => ProcessorError::Untracked,
            validate_step::sm::ProcessorError::Callback(_) => ProcessorError::Callback,
        };
        Ok(())
    }
    fn run_prepare(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = prepare_step::sm::ProcessorExecuteRequest {
            step_plan: handle_u64(r.step_plan),
            output_out: handle_u64(r.output_out),
            lifecycle: handle_u64(r.lifecycle),
            tensor_machine: handle_u64(r.tensor_machine),
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: handle_u64(r.memory_sm),
            memory_view: handle_u64(r.memory_view),
            expected_outputs: r.expected_outputs,
            compute_ctx: handle_u64(r.compute_ctx),
            positions: handle_u64(r.positions),
            positions_count: r.positions_count,
            seq_masks: handle_u64(r.seq_masks),
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: handle_u64(r.seq_primary_ids),
            seq_primary_ids_count: r.seq_primary_ids_count,
            prepare_graph: Some(bridge_prepare),
            root_prepare_graph: r.prepare_graph,
        };
        let _ = self
            .prepare_actor
            .process_event(prepare_step::sm::ProcessorEventExecuteStep::new(child));
        self.err = match self.prepare_actor.context().err {
            prepare_step::sm::ProcessorError::None => ProcessorError::None,
            prepare_step::sm::ProcessorError::InvalidRequest => ProcessorError::InvalidRequest,
            prepare_step::sm::ProcessorError::KernelFailed => ProcessorError::KernelFailed,
            prepare_step::sm::ProcessorError::InternalError => ProcessorError::InternalError,
            prepare_step::sm::ProcessorError::Untracked => ProcessorError::Untracked,
            prepare_step::sm::ProcessorError::Callback(_) => ProcessorError::Callback,
        };
        Ok(())
    }
    fn run_alloc(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = alloc_step::sm::ProcessorExecuteRequest {
            step_plan: handle_u64(r.step_plan),
            output_out: handle_u64(r.output_out),
            lifecycle: handle_u64(r.lifecycle),
            tensor_machine: handle_u64(r.tensor_machine),
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: handle_u64(r.memory_sm),
            memory_view: handle_u64(r.memory_view),
            expected_outputs: r.expected_outputs,
            compute_ctx: handle_u64(r.compute_ctx),
            positions: handle_u64(r.positions),
            positions_count: r.positions_count,
            seq_masks: handle_u64(r.seq_masks),
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: handle_u64(r.seq_primary_ids),
            seq_primary_ids_count: r.seq_primary_ids_count,
            alloc_graph: Some(bridge_alloc),
            root_alloc_graph: r.alloc_graph,
        };
        let _ = self
            .alloc_actor
            .process_event(alloc_step::sm::ProcessorEventExecuteStep::new(child));
        self.err = match self.alloc_actor.context().err {
            alloc_step::sm::ProcessorError::None => ProcessorError::None,
            alloc_step::sm::ProcessorError::InvalidRequest => ProcessorError::InvalidRequest,
            alloc_step::sm::ProcessorError::KernelFailed => ProcessorError::KernelFailed,
            alloc_step::sm::ProcessorError::InternalError => ProcessorError::InternalError,
            alloc_step::sm::ProcessorError::Untracked => ProcessorError::Untracked,
            alloc_step::sm::ProcessorError::Callback(_) => ProcessorError::Callback,
        };
        Ok(())
    }
    fn run_bind(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = bind_step::sm::ExecuteRequest {
            step_plan: r.step_plan,
            output_out: r.output_out,
            lifecycle: r.lifecycle,
            tensor_machine: r.tensor_machine,
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: r.memory_sm,
            memory_view: r.memory_view,
            expected_outputs: r.expected_outputs,
            compute_ctx: r.compute_ctx,
            positions: r.positions,
            positions_count: r.positions_count,
            seq_masks: r.seq_masks,
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: r.seq_primary_ids,
            seq_primary_ids_count: r.seq_primary_ids_count,
            bind_inputs: Some(bridge_bind),
            root_bind_inputs: r.bind_inputs,
        };
        let _ = self
            .bind_actor
            .process_event(bind_step::sm::ProcessorEventExecuteStep {
                request: child,
                initial_error: bind_step::sm::ProcessorError::NONE,
            });
        self.err = match self.bind_actor.context().err.0 {
            0 => ProcessorError::None,
            1 => ProcessorError::InvalidRequest,
            2 => ProcessorError::KernelFailed,
            4 => ProcessorError::InternalError,
            8 => ProcessorError::Untracked,
            _ => ProcessorError::Callback,
        };
        Ok(())
    }
    fn run_extract(
        &mut self,
        driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = extract_step::sm::ProcessorExecuteRequest {
            step_plan: handle_u64(r.step_plan),
            output_out: handle_u64(r.output_out),
            lifecycle: handle_u64(r.lifecycle),
            tensor_machine: handle_u64(r.tensor_machine),
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: handle_u64(r.memory_sm),
            memory_view: handle_u64(r.memory_view),
            expected_outputs: r.expected_outputs,
            compute_ctx: handle_u64(r.compute_ctx),
            positions: handle_u64(r.positions),
            positions_count: r.positions_count,
            seq_masks: handle_u64(r.seq_masks),
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: handle_u64(r.seq_primary_ids),
            seq_primary_ids_count: r.seq_primary_ids_count,
            extract_outputs: Some(bridge_extract),
            root_extract_outputs: r.extract_outputs,
        };
        let child_ok = self
            .extract_actor
            .process_event(extract_step::sm::ProcessorEventExecuteStep::new(child));
        if !child_ok
            || self.extract_actor.context().extract_outcome
                == extract_step::sm::PhaseOutcome::Failed
        {
            // Extract owns no tensor lifecycle handle: the processor root releases
            // the bound manifest IDs synchronously, exactly once on every failure.
            let _ = driver.release();
        }
        self.err = match self.extract_actor.context().err {
            extract_step::sm::ProcessorError::None => ProcessorError::None,
            extract_step::sm::ProcessorError::InvalidRequest => ProcessorError::InvalidRequest,
            extract_step::sm::ProcessorError::KernelFailed => ProcessorError::KernelFailed,
            extract_step::sm::ProcessorError::InternalError => ProcessorError::InternalError,
            extract_step::sm::ProcessorError::Untracked => ProcessorError::Untracked,
            extract_step::sm::ProcessorError::Callback(_) => ProcessorError::Callback,
        };
        Ok(())
    }
    fn run_kernel(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let r = e.request;
        let child = kernel_step::sm::ExecuteRequest {
            step_plan: r.step_plan,
            output_out: r.output_out,
            lifecycle: r.lifecycle,
            tensor_machine: r.tensor_machine,
            step_index: r.step_index,
            step_size: r.step_size,
            kv_tokens: r.kv_tokens,
            memory_sm: r.memory_sm,
            memory_view: r.memory_view,
            expected_outputs: r.expected_outputs,
            compute_ctx: r.compute_ctx,
            positions: r.positions,
            positions_count: r.positions_count,
            seq_masks: r.seq_masks,
            seq_mask_words: r.seq_mask_words,
            seq_masks_count: r.seq_masks_count,
            seq_primary_ids: r.seq_primary_ids,
            seq_primary_ids_count: r.seq_primary_ids_count,
            validate: None,
            prepare_graph: None,
            alloc_graph: None,
            bind_inputs: None,
            run_kernel: Some(bridge_kernel),
            root_run_kernel: r.run_kernel,
            extract_outputs: None,
            dispatch_done: None,
            dispatch_error: None,
        };
        let result = self
            .kernel_actor
            .process_event(kernel_step::sm::ProcessorEventExecuteStep {
                request: child,
                initial_error: kernel_step::sm::ProcessorError::NONE,
            });
        self.err = match result {
            Ok(_) => ProcessorError::None,
            Err(e) if e.0 == 1 => ProcessorError::InvalidRequest,
            Err(e) if e.0 == 2 => ProcessorError::KernelFailed,
            Err(e) if e.0 == 4 => ProcessorError::InternalError,
            Err(e) if e.0 == 8 => ProcessorError::Untracked,
            Err(_) => ProcessorError::Callback,
        };
        Ok(())
    }
    fn request_lifecycle_gate(
        &mut self,
        driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let manifest_ok = driver.gate();
        let callback_ok = e
            .request
            .lifecycle_gate
            .is_none_or(|callback| callback(&e.request));
        if !manifest_ok || !callback_ok {
            self.err = ProcessorError::InternalError;
        }
        Ok(())
    }
    fn request_lifecycle_publish(
        &mut self,
        driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let manifest_ok = driver.publish();
        let callback_ok = e
            .request
            .lifecycle_publish
            .is_none_or(|callback| callback(&e.request));
        if !manifest_ok || !callback_ok {
            self.err = ProcessorError::InternalError;
        }
        Ok(())
    }
    fn request_lifecycle_release(
        &mut self,
        driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        let manifest_ok = driver.release();
        let callback_ok = e
            .request
            .lifecycle_release
            .is_none_or(|callback| callback(&e.request));
        if !manifest_ok || !callback_ok {
            self.err = ProcessorError::InternalError;
        }
        Ok(())
    }
    fn commit_output(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        self.output = ExecutionOutput {
            outputs_produced: self.extract_actor.context().outputs_produced,
            graph_reused: u8::from(self.prepare_actor.context().graph_reused),
            lifecycle: e.request.lifecycle,
        };
        Ok(())
    }
    fn dispatch_done(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        if let Some(f) = e.request.dispatch_done {
            let _ = f(&self.output);
        }
        Ok(())
    }
    fn dispatch_error_from_execution_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        if let Some(f) = e.request.dispatch_error {
            let _ = f(&self.output, self.err as i32);
        }
        Ok(())
    }
    fn reject_invalid_execute_with_dispatch(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        e: &EventExecuteStep,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InvalidRequest;
        self.output = ExecutionOutput::default();
        if let Some(f) = e.request.dispatch_error {
            let _ = f(&self.output, self.err as i32);
        }
        Ok(())
    }
    fn reject_invalid_execute_with_output_only(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _e: &EventExecuteStep,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InvalidRequest;
        self.output = ExecutionOutput::default();
        Ok(())
    }
    fn reject_invalid_execute_without_output(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _e: &EventExecuteStep,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InvalidRequest;
        Ok(())
    }
    fn validate_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.validate_actor.context().validate_outcome
                == validate_step::sm::PhaseOutcome::Done)
    }
    fn validate_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.validate_actor.context().validate_outcome
                == validate_step::sm::PhaseOutcome::Failed)
    }
    fn prepare_done_reused(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.prepare_actor.context().prepare_outcome == prepare_step::sm::PhaseOutcome::Done
            && self.prepare_actor.context().graph_reused)
    }
    fn prepare_done_needs_allocation(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.prepare_actor.context().prepare_outcome == prepare_step::sm::PhaseOutcome::Done
            && !self.prepare_actor.context().graph_reused)
    }
    fn prepare_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.prepare_actor.context().prepare_outcome
                == prepare_step::sm::PhaseOutcome::Failed)
    }
    fn alloc_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.alloc_actor.context().alloc_outcome == alloc_step::sm::PhaseOutcome::Done)
    }
    fn alloc_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.alloc_actor.context().alloc_outcome == alloc_step::sm::PhaseOutcome::Failed)
    }
    fn bind_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.bind_actor.context().bind_outcome == bind_step::sm::PhaseOutcome::Done)
    }
    fn bind_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.bind_actor.context().bind_outcome == bind_step::sm::PhaseOutcome::Failed)
    }
    fn lifecycle_gate_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None)
    }
    fn lifecycle_gate_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }
    fn kernel_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.kernel_actor.context().kernel_outcome == kernel_step::sm::PhaseOutcome::Done)
    }
    fn kernel_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.kernel_actor.context().kernel_outcome == kernel_step::sm::PhaseOutcome::Failed)
    }
    fn publish_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None)
    }
    fn publish_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }
    fn extract_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None
            && self.extract_actor.context().extract_outcome == extract_step::sm::PhaseOutcome::Done)
    }
    fn extract_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None
            || self.extract_actor.context().extract_outcome
                == extract_step::sm::PhaseOutcome::Failed)
    }
    fn release_done(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None)
    }
    fn release_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err != ProcessorError::None)
    }
    fn execution_error_none(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::None)
    }
    fn execution_error_invalid_request(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::InvalidRequest)
    }
    fn execution_error_kernel_failed(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::KernelFailed)
    }
    fn execution_error_internal_error(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::InternalError)
    }
    fn execution_error_untracked(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(self.err == ProcessorError::Untracked)
    }
    fn execution_error_unknown(
        &self,
        _driver: &mut LifecycleDriver<'_, '_>,
        _: &EventExecuteStep,
    ) -> Result<bool, ()> {
        Ok(matches!(
            self.err,
            ProcessorError::Callback | ProcessorError::Unknown
        ))
    }
    fn on_unexpected_from_ready(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_validate_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_prepare_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_alloc_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_bind_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_lifecycle_gate_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_kernel_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_publish_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_extract_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_release_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
    fn on_unexpected_from_execution_decision(
        &mut self,
        _driver: &mut LifecycleDriver<'_, '_>,
    ) -> Result<(), ()> {
        self.err = ProcessorError::InternalError;
        Ok(())
    }
}

/// Synchronous graph processor actor.
impl fmt::Debug for GraphProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphProcessor")
            .field("output", &self.output())
            .field("error", &self.error())
            .finish_non_exhaustive()
    }
}
impl Default for GraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}
impl GraphProcessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphProcessorStateMachine::new(GraphProcessorContext::default()),
            tensor: crate::tensor::sm::GraphTensor::new(),
        }
    }
    pub fn process_event(&mut self, event: EventExecuteStep) -> bool {
        let mut driver = LifecycleDriver::new(&mut self.tensor, LifecycleManifest::default());
        if self
            .machine
            .process_event(&mut driver, GraphProcessorEvents::EventExecuteStep(event))
            .is_err()
        {
            self.machine.context_mut().err = ProcessorError::InternalError;
            return false;
        }
        self.machine.context().err == ProcessorError::None
    }
    /// Executes one request through the generated transition table.
    pub fn process_event_with_driver(
        &mut self,
        event: EventExecuteStep,
        tensor: &mut crate::tensor::sm::GraphTensor,
        manifest: LifecycleManifest<'_>,
    ) -> bool {
        let mut driver = LifecycleDriver::new(tensor, manifest);
        self.machine
            .process_event(&mut driver, GraphProcessorEvents::EventExecuteStep(event))
            .is_ok()
            && self.machine.context().err == ProcessorError::None
    }
    #[must_use]
    pub fn state(&self) -> &GraphProcessorStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &GraphProcessorStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the immutable domain result of the last dispatch.
    #[must_use]
    pub fn snapshot(&self) -> ProcessorSnapshot {
        ProcessorSnapshot {
            output: self.output(),
            error: self.error(),
        }
    }
    #[must_use]
    pub fn output(&self) -> ExecutionOutput {
        self.machine.context().output
    }
    #[must_use]
    pub fn error(&self) -> ProcessorError {
        self.machine.context().err
    }
}

/// Immutable public observation of a completed dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessorSnapshot {
    pub output: ExecutionOutput,
    pub error: ProcessorError,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifecycle_fails(_: &ExecuteRequest) -> bool {
        false
    }

    #[test]
    fn lifecycle_callback_false_sets_internal_error() {
        let mut context = GraphProcessorContext::default();
        let mut tensor = crate::tensor::sm::GraphTensor::new();
        let mut driver = LifecycleDriver::new(&mut tensor, LifecycleManifest::default());
        let request = ExecuteRequest {
            lifecycle_gate: Some(lifecycle_fails),
            ..ExecuteRequest::default()
        };
        context
            .request_lifecycle_gate(&mut driver, &EventExecuteStep::new(request))
            .unwrap();
        assert_eq!(context.err, ProcessorError::InternalError);
    }

    fn validate(_: &ExecuteRequest, _: &mut i32) -> bool {
        true
    }

    fn prepare(_: &ExecuteRequest, reused: &mut bool, _: &mut i32) -> bool {
        *reused = true;
        true
    }

    fn phase(_: &ExecuteRequest, error: &mut i32) -> bool {
        *error = 0;
        true
    }

    fn extract_fails(_: &ExecuteRequest, outputs: &mut i32, error: &mut i32) -> bool {
        *outputs = 0;
        *error = 37;
        false
    }

    fn extract_succeeds(_: &ExecuteRequest, outputs: &mut i32, error: &mut i32) -> bool {
        *outputs = 2;
        *error = 0;
        true
    }

    fn lifecycle(_: &ExecuteRequest) -> bool {
        true
    }

    fn dispatch_done(_: &ExecutionOutput) -> bool {
        true
    }

    fn dispatch_error(_: &ExecutionOutput, _: i32) -> bool {
        true
    }
    fn request(extract_outputs: ExtractOutputsFn) -> ExecuteRequest {
        ExecuteRequest {
            step_plan: 1,
            output_out: 1,
            lifecycle: 1,
            tensor_machine: 1,
            memory_sm: 1,
            memory_view: 1,
            step_size: 1,
            seq_mask_words: 1,
            validate: Some(validate),
            prepare_graph: Some(prepare),
            alloc_graph: Some(phase),
            bind_inputs: Some(phase),
            run_kernel: Some(phase),
            extract_outputs: Some(extract_outputs),
            lifecycle_gate: Some(lifecycle),
            lifecycle_publish: Some(lifecycle),
            lifecycle_release: Some(lifecycle),
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
            ..ExecuteRequest::default()
        }
    }

    fn reserve_and_publish(tensor: &mut crate::tensor::sm::GraphTensor, tensor_id: usize) {
        tensor
            .process_event(crate::tensor::sm::Event::Reserve(
                crate::tensor::sm::ReserveTensor {
                    tensor_id,
                    buffer: crate::tensor::sm::BufferHandle(1),
                    buffer_bytes: 64,
                    consumer_refs: 1,
                    is_leaf: false,
                },
            ))
            .unwrap();
        tensor
            .process_event(crate::tensor::sm::Event::PublishFilled(
                crate::tensor::sm::PublishFilledTensor { tensor_id },
            ))
            .unwrap();
    }

    fn capture(
        tensor: &mut crate::tensor::sm::GraphTensor,
        tensor_id: usize,
    ) -> crate::tensor::sm::TensorState {
        match tensor
            .process_event(crate::tensor::sm::Event::Capture(
                crate::tensor::sm::CaptureTensorState { tensor_id },
            ))
            .unwrap()
        {
            crate::tensor::sm::Outcome::State(state) => state,
            crate::tensor::sm::Outcome::Done => panic!("capture returned mutation outcome"),
        }
    }

    #[test]
    fn extract_failure_releases_bound_tensor_once_at_processor_root() {
        let tensor_id = 7;
        let mut tensor = crate::tensor::sm::GraphTensor::new();
        reserve_and_publish(&mut tensor, tensor_id);
        let manifest = LifecycleManifest {
            phase: LifecyclePhase {
                required_filled_ids: &[],
                publish_ids: &[],
                release_ids: &[tensor_id],
            },
        };
        let mut context = GraphProcessorContext::default();
        let mut driver = LifecycleDriver::new(&mut tensor, manifest);
        context
            .run_extract(&mut driver, &EventExecuteStep::new(request(extract_fails)))
            .unwrap();
        assert_eq!(context.err, ProcessorError::Callback);
        let state = capture(&mut tensor, tensor_id);
        assert_eq!(state.lifecycle, crate::tensor::sm::Lifecycle::Empty);
        assert_eq!(state.live_refs, 0);
    }

    #[test]
    fn successful_extract_defers_release_to_normal_completion() {
        let tensor_id = 9;
        let mut tensor = crate::tensor::sm::GraphTensor::new();
        reserve_and_publish(&mut tensor, tensor_id);
        let manifest = LifecycleManifest {
            phase: LifecyclePhase {
                required_filled_ids: &[],
                publish_ids: &[],
                release_ids: &[tensor_id],
            },
        };
        let mut context = GraphProcessorContext::default();
        {
            let mut driver = LifecycleDriver::new(&mut tensor, manifest);
            context
                .run_extract(
                    &mut driver,
                    &EventExecuteStep::new(request(extract_succeeds)),
                )
                .unwrap();
        }
        assert_eq!(context.err, ProcessorError::None);
        assert_eq!(context.extract_actor.context().outputs_produced, 2);
        assert_eq!(
            capture(&mut tensor, tensor_id).lifecycle,
            crate::tensor::sm::Lifecycle::Filled
        );
        let mut driver = LifecycleDriver::new(&mut tensor, manifest);
        assert!(driver.release());
        let state = capture(&mut tensor, tensor_id);
        assert_eq!(state.lifecycle, crate::tensor::sm::Lifecycle::Empty);
        assert_eq!(state.live_refs, 0);
    }
}
