//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

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

// --- machine ModelTensor from emel.cpp/src/emel/model/tensor/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailApplyEffectResultsRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailBindStorageRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailBindTensorRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailCaptureTensorStateRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailEvictTensorRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailPlanLoadRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReleaseMappedLoadRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailRequestMappedLoadRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailRequestReadLoadRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailRequestStagedLoadRuntime;

sml! {
    ModelTensor {
        "state_bind_storage_decision"_s <= *"ready"_s + event<DetailBindStorageRuntime> [guard_storage_bind_valid_without_mmap_resident] / effect_bind_storage,
        "state_bind_storage_error_decision"_s <= "ready"_s + event<DetailBindStorageRuntime> [storage_bind_invalid] / record_bind_storage_invalid_request_from_ready,
        "state_bind_storage_error_decision"_s <= "ready"_s + event<DetailBindStorageRuntime> [guard_storage_bind_valid_with_mmap_resident] / record_bind_storage_invalid_request_from_ready,
        "state_bind_storage_done_decision"_s <= "state_bind_storage_decision"_s + completion<DetailBindStorageRuntime>,
        "state_bind_storage_done_callback"_s <= "state_bind_storage_done_decision"_s + completion<DetailBindStorageRuntime> [bind_storage_done_callback_present] / publish_bind_storage_done,
        "ready"_s <= "state_bind_storage_done_decision"_s + completion<DetailBindStorageRuntime> [bind_storage_done_callback_absent] / record_bind_storage_done,
        "ready"_s <= "state_bind_storage_done_callback"_s + completion<DetailBindStorageRuntime>,
        "state_bind_storage_error_callback"_s <= "state_bind_storage_error_decision"_s + completion<DetailBindStorageRuntime> [bind_storage_error_callback_present] / publish_bind_storage_error_from_state_bind_storage_error_decision,
        "ready"_s <= "state_bind_storage_error_decision"_s + completion<DetailBindStorageRuntime> [bind_storage_error_callback_absent],
        "ready"_s <= "state_bind_storage_error_callback"_s + completion<DetailBindStorageRuntime>,
        "state_bind_storage_busy_error_callback"_s <= "state_awaiting_effects"_s + event<DetailBindStorageRuntime> [bind_storage_error_callback_present] / publish_bind_storage_error_from_state_awaiting_effects,
        "state_awaiting_effects"_s <= "state_awaiting_effects"_s + event<DetailBindStorageRuntime> [bind_storage_error_callback_absent] / record_bind_storage_invalid_request_from_state_awaiting_effects,
        "state_awaiting_effects"_s <= "state_bind_storage_busy_error_callback"_s + completion<DetailBindStorageRuntime>,
        "state_plan_load_done_decision"_s <= "ready"_s + event<DetailPlanLoadRuntime> [plan_load_valid_without_io_strategy] / effect_plan_load,
        "state_plan_load_done_decision"_s <= "ready"_s + event<DetailPlanLoadRuntime> [plan_load_valid_with_io_strategy] / effect_plan_io_load,
        "state_plan_load_error_decision"_s <= "ready"_s + event<DetailPlanLoadRuntime> [plan_load_invalid_request],
        "state_plan_load_capacity_error_decision"_s <= "ready"_s + event<DetailPlanLoadRuntime> [plan_load_invalid_capacity],
        "state_plan_load_done_callback"_s <= "state_plan_load_done_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_done_callback_present] / publish_plan_load_done,
        "state_awaiting_effects"_s <= "state_plan_load_done_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_done_callback_absent] / record_plan_load_done,
        "state_awaiting_effects"_s <= "state_plan_load_done_callback"_s + completion<DetailPlanLoadRuntime>,
        "state_plan_load_error_callback"_s <= "state_plan_load_error_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_error_callback_present] / publish_plan_load_invalid_request,
        "ready"_s <= "state_plan_load_error_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_error_callback_absent] / record_plan_load_invalid_request,
        "state_plan_load_error_callback"_s <= "state_plan_load_capacity_error_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_error_callback_present] / publish_plan_load_capacity_error,
        "ready"_s <= "state_plan_load_capacity_error_decision"_s + completion<DetailPlanLoadRuntime> [plan_load_error_callback_absent] / record_plan_load_capacity_error,
        "ready"_s <= "state_plan_load_error_callback"_s + completion<DetailPlanLoadRuntime>,
        "state_apply_effect_results_request_decision"_s <= "state_awaiting_effects"_s + event<DetailApplyEffectResultsRuntime>,
        "state_apply_effect_results_error_decision"_s <= "state_apply_effect_results_request_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_results_invalid],
        "state_apply_effect_results_backend_error_decision"_s <= "state_apply_effect_results_request_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_results_valid_with_effect_errors],
        "state_apply_effect_results_done_decision"_s <= "state_apply_effect_results_request_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_results_valid_without_effect_errors_without_record_output] / effect_apply_results,
        "state_apply_effect_results_done_decision"_s <= "state_apply_effect_results_request_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_results_valid_without_effect_errors_with_record_output] / effect_apply_results_with_record_output,
        "state_apply_effect_results_error_decision"_s <= "ready"_s + event<DetailApplyEffectResultsRuntime>,
        "state_apply_effect_results_done_callback"_s <= "state_apply_effect_results_done_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_done_callback_present] / publish_apply_effect_results_done,
        "ready"_s <= "state_apply_effect_results_done_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_done_callback_absent] / record_apply_effect_results_done,
        "ready"_s <= "state_apply_effect_results_done_callback"_s + completion<DetailApplyEffectResultsRuntime>,
        "state_apply_effect_results_error_callback"_s <= "state_apply_effect_results_error_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_error_callback_present] / publish_apply_effect_results_invalid_request,
        "ready"_s <= "state_apply_effect_results_error_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_error_callback_absent] / record_apply_effect_results_invalid_request,
        "state_apply_effect_results_error_callback"_s <= "state_apply_effect_results_backend_error_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_error_callback_present] / publish_apply_effect_results_backend_error,
        "ready"_s <= "state_apply_effect_results_backend_error_decision"_s + completion<DetailApplyEffectResultsRuntime> [apply_effect_results_error_callback_absent] / record_apply_effect_results_backend_error,
        "ready"_s <= "state_apply_effect_results_error_callback"_s + completion<DetailApplyEffectResultsRuntime>,
        "bind_tensor_request_decision"_s <= "ready"_s + event<DetailBindTensorRuntime> / begin_bind_tensor,
        "bind_tensor_exec"_s <= "bind_tensor_request_decision"_s + completion<DetailBindTensorRuntime> [bind_tensor_request_valid],
        "errored"_s <= "bind_tensor_request_decision"_s + completion<DetailBindTensorRuntime> [bind_tensor_request_invalid] / mark_invalid_request_detail_bind_tensor_runtime,
        "bind_tensor_result_decision"_s <= "bind_tensor_exec"_s + completion<DetailBindTensorRuntime> / exec_bind_tensor,
        "done"_s <= "bind_tensor_result_decision"_s + completion<DetailBindTensorRuntime> [operation_succeeded_detail_bind_tensor_runtime],
        "errored"_s <= "bind_tensor_result_decision"_s + completion<DetailBindTensorRuntime> [operation_not_dispatched_detail_bind_tensor_runtime] / mark_invalid_request_detail_bind_tensor_runtime,
        "evict_tensor_request_decision"_s <= "ready"_s + event<DetailEvictTensorRuntime> / begin_evict_tensor,
        "evict_tensor_exec"_s <= "evict_tensor_request_decision"_s + completion<DetailEvictTensorRuntime> [evict_tensor_request_valid],
        "errored"_s <= "evict_tensor_request_decision"_s + completion<DetailEvictTensorRuntime> [evict_tensor_request_invalid] / mark_invalid_request_detail_evict_tensor_runtime,
        "evict_tensor_result_decision"_s <= "evict_tensor_exec"_s + completion<DetailEvictTensorRuntime> / exec_evict_tensor,
        "done"_s <= "evict_tensor_result_decision"_s + completion<DetailEvictTensorRuntime> [operation_succeeded_detail_evict_tensor_runtime],
        "errored"_s <= "evict_tensor_result_decision"_s + completion<DetailEvictTensorRuntime> [operation_not_dispatched_detail_evict_tensor_runtime] / mark_invalid_request_detail_evict_tensor_runtime,
        "capture_tensor_state_request_decision"_s <= "ready"_s + event<DetailCaptureTensorStateRuntime> / begin_capture_tensor_state,
        "capture_tensor_state_exec"_s <= "capture_tensor_state_request_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_tensor_state_request_valid],
        "errored"_s <= "capture_tensor_state_request_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_tensor_state_request_invalid] / mark_invalid_request_detail_capture_tensor_state_runtime,
        "capture_tensor_state_result_decision"_s <= "capture_tensor_state_exec"_s + completion<DetailCaptureTensorStateRuntime> / exec_capture_tensor_state,
        "done"_s <= "capture_tensor_state_result_decision"_s + completion<DetailCaptureTensorStateRuntime> [operation_succeeded_detail_capture_tensor_state_runtime],
        "errored"_s <= "capture_tensor_state_result_decision"_s + completion<DetailCaptureTensorStateRuntime> [operation_not_dispatched_detail_capture_tensor_state_runtime] / mark_invalid_request_detail_capture_tensor_state_runtime,
        "ready"_s <= "done"_s + completion<DetailBindTensorRuntime> [error_code_output_present_detail_bind_tensor_runtime] / publish_done_with_error_code_detail_bind_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailBindTensorRuntime> [error_code_output_absent_detail_bind_tensor_runtime] / publish_done_detail_bind_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailBindTensorRuntime> [error_code_output_present_detail_bind_tensor_runtime] / publish_error_with_error_code_detail_bind_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailBindTensorRuntime> [error_code_output_absent_detail_bind_tensor_runtime] / publish_error_detail_bind_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailEvictTensorRuntime> [error_code_output_present_detail_evict_tensor_runtime] / publish_done_with_error_code_detail_evict_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailEvictTensorRuntime> [error_code_output_absent_detail_evict_tensor_runtime] / publish_done_detail_evict_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailEvictTensorRuntime> [error_code_output_present_detail_evict_tensor_runtime] / publish_error_with_error_code_detail_evict_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailEvictTensorRuntime> [error_code_output_absent_detail_evict_tensor_runtime] / publish_error_detail_evict_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailCaptureTensorStateRuntime> [error_code_output_present_detail_capture_tensor_state_runtime] / publish_done_with_error_code_detail_capture_tensor_state_runtime,
        "ready"_s <= "done"_s + completion<DetailCaptureTensorStateRuntime> [error_code_output_absent_detail_capture_tensor_state_runtime] / publish_done_detail_capture_tensor_state_runtime,
        "ready"_s <= "errored"_s + completion<DetailCaptureTensorStateRuntime> [error_code_output_present_detail_capture_tensor_state_runtime] / publish_error_with_error_code_detail_capture_tensor_state_runtime,
        "ready"_s <= "errored"_s + completion<DetailCaptureTensorStateRuntime> [error_code_output_absent_detail_capture_tensor_state_runtime] / publish_error_detail_capture_tensor_state_runtime,
        "state_request_mapped_load_decision"_s <= "ready"_s + event<DetailRequestMappedLoadRuntime> / effect_begin_request_mapped_load,
        "state_request_mapped_load_unsupported_io_mmap_error_decision"_s <= "state_request_mapped_load_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_absent] / effect_mark_request_mapped_load_unsupported_io_mmap,
        "state_request_mapped_load_invalid_request_error_decision"_s <= "state_request_mapped_load_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_present_request_invalid] / effect_mark_request_mapped_load_invalid_request,
        "state_request_mapped_load_already_resident_error_decision"_s <= "state_request_mapped_load_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_present_request_valid_already_resident] / effect_mark_request_mapped_load_tensor_already_resident,
        "state_request_mapped_load_dispatch_decision"_s <= "state_request_mapped_load_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_present_request_valid_tensor_unbound] / effect_attempt_request_mapped_load_dispatch,
        "state_request_mapped_load_done_callback"_s <= "state_request_mapped_load_dispatch_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_succeeded] / effect_commit_request_mapped_load,
        "state_request_mapped_load_io_mmap_error_decision"_s <= "state_request_mapped_load_dispatch_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_io_mmap_failed] / effect_mark_request_mapped_load_io_mmap_failed,
        "ready"_s <= "state_request_mapped_load_done_callback"_s + completion<DetailRequestMappedLoadRuntime> / effect_publish_request_mapped_load_done,
        "state_request_mapped_load_error_callback"_s <= "state_request_mapped_load_invalid_request_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_present] / effect_publish_request_mapped_load_error_from_state_request_mapped_load_invalid_request_error_decision,
        "ready"_s <= "state_request_mapped_load_invalid_request_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_absent] / effect_record_request_mapped_load_error_from_state_request_mapped_load_invalid_request_error_decision,
        "state_request_mapped_load_error_callback"_s <= "state_request_mapped_load_unsupported_io_mmap_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_present] / effect_publish_request_mapped_load_error_from_state_request_mapped_load_unsupported_io_mmap_error_decision,
        "ready"_s <= "state_request_mapped_load_unsupported_io_mmap_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_absent] / effect_record_request_mapped_load_error_from_state_request_mapped_load_unsupported_io_mmap_error_decision,
        "state_request_mapped_load_error_callback"_s <= "state_request_mapped_load_already_resident_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_present] / effect_publish_request_mapped_load_error_from_state_request_mapped_load_already_resident_error_decision,
        "ready"_s <= "state_request_mapped_load_already_resident_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_absent] / effect_record_request_mapped_load_error_from_state_request_mapped_load_already_resident_error_decision,
        "state_request_mapped_load_error_callback"_s <= "state_request_mapped_load_io_mmap_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_present] / effect_publish_request_mapped_load_error_from_state_request_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_request_mapped_load_io_mmap_error_decision"_s + completion<DetailRequestMappedLoadRuntime> [request_mapped_load_error_callback_absent] / effect_record_request_mapped_load_error_from_state_request_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_request_mapped_load_error_callback"_s + completion<DetailRequestMappedLoadRuntime> / effect_record_request_mapped_load_error_from_state_request_mapped_load_error_callback,
        "state_request_read_load_decision"_s <= "ready"_s + event<DetailRequestReadLoadRuntime> / effect_begin_request_read_load,
        "state_request_read_load_unsupported_io_read_error_decision"_s <= "state_request_read_load_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_absent] / effect_mark_request_read_load_unsupported_io_read_from_state_request_read_load_decision,
        "state_request_read_load_invalid_request_error_decision"_s <= "state_request_read_load_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_present_request_invalid] / effect_mark_request_read_load_invalid_request_from_state_request_read_load_decision,
        "state_request_read_load_already_resident_error_decision"_s <= "state_request_read_load_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_present_request_valid_already_resident] / effect_mark_request_read_load_tensor_already_resident,
        "state_request_read_load_dispatch_decision"_s <= "state_request_read_load_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_present_request_valid_tensor_unbound] / effect_attempt_request_read_load_dispatch,
        "state_request_read_load_done_callback"_s <= "state_request_read_load_dispatch_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_succeeded] / effect_commit_request_read_load,
        "state_request_read_load_invalid_request_error_decision"_s <= "state_request_read_load_dispatch_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_invalid_request] / effect_mark_request_read_load_invalid_request_from_state_request_read_load_dispatch_decision,
        "state_request_read_load_unsupported_io_read_error_decision"_s <= "state_request_read_load_dispatch_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_unsupported] / effect_mark_request_read_load_unsupported_io_read_from_state_request_read_load_dispatch_decision,
        "state_request_read_load_io_read_file_open_error_decision"_s <= "state_request_read_load_dispatch_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_file_open_failed] / effect_mark_request_read_load_io_read_failed_from_state_request_read_load_dispatch_decision,
        "state_request_read_load_io_read_file_read_error_decision"_s <= "state_request_read_load_dispatch_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_io_read_file_read_or_other_failed] / effect_mark_request_read_load_io_read_failed_from_state_request_read_load_dispatch_decision,
        "ready"_s <= "state_request_read_load_done_callback"_s + completion<DetailRequestReadLoadRuntime> / effect_publish_request_read_load_done,
        "state_request_read_load_error_callback"_s <= "state_request_read_load_invalid_request_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_present] / effect_publish_request_read_load_error_from_state_request_read_load_invalid_request_error_decision,
        "ready"_s <= "state_request_read_load_invalid_request_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_absent] / effect_record_request_read_load_error_from_state_request_read_load_invalid_request_error_decision,
        "state_request_read_load_error_callback"_s <= "state_request_read_load_unsupported_io_read_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_present] / effect_publish_request_read_load_error_from_state_request_read_load_unsupported_io_read_error_decision,
        "ready"_s <= "state_request_read_load_unsupported_io_read_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_absent] / effect_record_request_read_load_error_from_state_request_read_load_unsupported_io_read_error_decision,
        "state_request_read_load_error_callback"_s <= "state_request_read_load_already_resident_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_present] / effect_publish_request_read_load_error_from_state_request_read_load_already_resident_error_decision,
        "ready"_s <= "state_request_read_load_already_resident_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_absent] / effect_record_request_read_load_error_from_state_request_read_load_already_resident_error_decision,
        "state_request_read_load_error_callback"_s <= "state_request_read_load_io_read_file_open_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_present] / effect_publish_request_read_load_error_from_state_request_read_load_io_read_file_open_error_decision,
        "ready"_s <= "state_request_read_load_io_read_file_open_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_absent] / effect_record_request_read_load_error_from_state_request_read_load_io_read_file_open_error_decision,
        "state_request_read_load_error_callback"_s <= "state_request_read_load_io_read_file_read_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_present] / effect_publish_request_read_load_error_from_state_request_read_load_io_read_file_read_error_decision,
        "ready"_s <= "state_request_read_load_io_read_file_read_error_decision"_s + completion<DetailRequestReadLoadRuntime> [request_read_load_error_callback_absent] / effect_record_request_read_load_error_from_state_request_read_load_io_read_file_read_error_decision,
        "ready"_s <= "state_request_read_load_error_callback"_s + completion<DetailRequestReadLoadRuntime> / effect_record_request_read_load_error_from_state_request_read_load_error_callback,
        "state_request_staged_load_decision"_s <= "ready"_s + event<DetailRequestStagedLoadRuntime> / effect_begin_request_staged_load,
        "state_request_staged_load_unsupported_io_staged_read_error_decision"_s <= "state_request_staged_load_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_absent] / effect_mark_request_staged_load_unsupported_io_staged_read_from_state_request_staged_load_decision,
        "state_request_staged_load_invalid_request_error_decision"_s <= "state_request_staged_load_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_present_request_invalid] / effect_mark_request_staged_load_invalid_request_from_state_request_staged_load_decision,
        "state_request_staged_load_already_resident_error_decision"_s <= "state_request_staged_load_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_present_request_valid_already_resident] / effect_mark_request_staged_load_tensor_already_resident,
        "state_request_staged_load_dispatch_decision"_s <= "state_request_staged_load_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_present_request_valid_tensor_unbound] / effect_attempt_request_staged_load_dispatch,
        "state_request_staged_load_done_callback"_s <= "state_request_staged_load_dispatch_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_succeeded] / effect_commit_request_staged_load,
        "state_request_staged_load_invalid_request_error_decision"_s <= "state_request_staged_load_dispatch_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_invalid_request] / effect_mark_request_staged_load_invalid_request_from_state_request_staged_load_dispatch_decision,
        "state_request_staged_load_unsupported_io_staged_read_error_decision"_s <= "state_request_staged_load_dispatch_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_unsupported] / effect_mark_request_staged_load_unsupported_io_staged_read_from_state_request_staged_load_dispatch_decision,
        "state_request_staged_load_io_staged_read_error_decision"_s <= "state_request_staged_load_dispatch_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_io_staged_read_other_failed] / effect_mark_request_staged_load_io_staged_read_failed,
        "ready"_s <= "state_request_staged_load_done_callback"_s + completion<DetailRequestStagedLoadRuntime> / effect_publish_request_staged_load_done,
        "state_request_staged_load_error_callback"_s <= "state_request_staged_load_invalid_request_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_present] / effect_publish_request_staged_load_error_from_state_request_staged_load_invalid_request_error_decision,
        "ready"_s <= "state_request_staged_load_invalid_request_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_absent] / effect_record_request_staged_load_error_from_state_request_staged_load_invalid_request_error_decision,
        "state_request_staged_load_error_callback"_s <= "state_request_staged_load_unsupported_io_staged_read_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_present] / effect_publish_request_staged_load_error_from_state_request_staged_load_unsupported_io_staged_read_error_decision,
        "ready"_s <= "state_request_staged_load_unsupported_io_staged_read_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_absent] / effect_record_request_staged_load_error_from_state_request_staged_load_unsupported_io_staged_read_error_decision,
        "state_request_staged_load_error_callback"_s <= "state_request_staged_load_already_resident_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_present] / effect_publish_request_staged_load_error_from_state_request_staged_load_already_resident_error_decision,
        "ready"_s <= "state_request_staged_load_already_resident_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_absent] / effect_record_request_staged_load_error_from_state_request_staged_load_already_resident_error_decision,
        "state_request_staged_load_error_callback"_s <= "state_request_staged_load_io_staged_read_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_present] / effect_publish_request_staged_load_error_from_state_request_staged_load_io_staged_read_error_decision,
        "ready"_s <= "state_request_staged_load_io_staged_read_error_decision"_s + completion<DetailRequestStagedLoadRuntime> [request_staged_load_error_callback_absent] / effect_record_request_staged_load_error_from_state_request_staged_load_io_staged_read_error_decision,
        "ready"_s <= "state_request_staged_load_error_callback"_s + completion<DetailRequestStagedLoadRuntime> / effect_record_request_staged_load_error_from_state_request_staged_load_error_callback,
        "state_release_mapped_load_decision"_s <= "ready"_s + event<DetailReleaseMappedLoadRuntime> / effect_begin_release_mapped_load,
        "state_release_mapped_load_unsupported_io_mmap_error_decision"_s <= "state_release_mapped_load_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_absent] / effect_mark_release_mapped_load_unsupported_io_mmap,
        "state_release_mapped_load_invalid_request_error_decision"_s <= "state_release_mapped_load_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_present_request_invalid] / effect_mark_release_mapped_load_invalid_request,
        "state_release_mapped_load_handle_absent_error_decision"_s <= "state_release_mapped_load_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_present_request_valid_handle_absent] / effect_mark_release_mapped_load_handle_absent,
        "state_release_mapped_load_dispatch_decision"_s <= "state_release_mapped_load_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_present_request_valid_handle_present] / effect_attempt_release_mapped_load_dispatch,
        "state_release_mapped_load_publish_done_decision"_s <= "state_release_mapped_load_dispatch_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_succeeded] / effect_commit_release_mapped_load,
        "state_release_mapped_load_io_mmap_error_decision"_s <= "state_release_mapped_load_dispatch_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_io_mmap_failed] / effect_mark_release_mapped_load_io_mmap_failed,
        "state_release_mapped_load_done_callback"_s <= "state_release_mapped_load_publish_done_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_done_callback_present] / effect_publish_release_mapped_load_done,
        "ready"_s <= "state_release_mapped_load_publish_done_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_done_callback_absent] / effect_record_release_mapped_load_done_from_state_release_mapped_load_publish_done_decision,
        "ready"_s <= "state_release_mapped_load_done_callback"_s + completion<DetailReleaseMappedLoadRuntime> / effect_record_release_mapped_load_done_from_state_release_mapped_load_done_callback,
        "state_release_mapped_load_error_callback"_s <= "state_release_mapped_load_invalid_request_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_present] / effect_publish_release_mapped_load_error_from_state_release_mapped_load_invalid_request_error_decision,
        "ready"_s <= "state_release_mapped_load_invalid_request_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_absent] / effect_record_release_mapped_load_error_from_state_release_mapped_load_invalid_request_error_decision,
        "state_release_mapped_load_error_callback"_s <= "state_release_mapped_load_unsupported_io_mmap_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_present] / effect_publish_release_mapped_load_error_from_state_release_mapped_load_unsupported_io_mmap_error_decision,
        "ready"_s <= "state_release_mapped_load_unsupported_io_mmap_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_absent] / effect_record_release_mapped_load_error_from_state_release_mapped_load_unsupported_io_mmap_error_decision,
        "state_release_mapped_load_error_callback"_s <= "state_release_mapped_load_handle_absent_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_present] / effect_publish_release_mapped_load_error_from_state_release_mapped_load_handle_absent_error_decision,
        "ready"_s <= "state_release_mapped_load_handle_absent_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_absent] / effect_record_release_mapped_load_error_from_state_release_mapped_load_handle_absent_error_decision,
        "state_release_mapped_load_error_callback"_s <= "state_release_mapped_load_io_mmap_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_present] / effect_publish_release_mapped_load_error_from_state_release_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_release_mapped_load_io_mmap_error_decision"_s + completion<DetailReleaseMappedLoadRuntime> [release_mapped_load_error_callback_absent] / effect_record_release_mapped_load_error_from_state_release_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_release_mapped_load_error_callback"_s + completion<DetailReleaseMappedLoadRuntime> / effect_record_release_mapped_load_error_from_state_release_mapped_load_error_callback,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "state_bind_storage_decision"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_decision,
        "ready"_s <= "state_bind_storage_done_decision"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_done_decision,
        "ready"_s <= "state_bind_storage_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_done_callback,
        "ready"_s <= "state_bind_storage_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_error_decision,
        "ready"_s <= "state_bind_storage_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_error_callback,
        "state_awaiting_effects"_s <= "state_bind_storage_busy_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_bind_storage_busy_error_callback,
        "ready"_s <= "state_plan_load_done_decision"_s + unexpected_event<_> / on_unexpected_from_state_plan_load_done_decision,
        "ready"_s <= "state_plan_load_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_plan_load_done_callback,
        "ready"_s <= "state_plan_load_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_plan_load_error_decision,
        "ready"_s <= "state_plan_load_capacity_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_plan_load_capacity_error_decision,
        "ready"_s <= "state_plan_load_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_plan_load_error_callback,
        "ready"_s <= "state_awaiting_effects"_s + unexpected_event<_> / on_unexpected_from_state_awaiting_effects,
        "ready"_s <= "state_apply_effect_results_request_decision"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_request_decision,
        "ready"_s <= "state_apply_effect_results_done_decision"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_done_decision,
        "ready"_s <= "state_apply_effect_results_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_done_callback,
        "ready"_s <= "state_apply_effect_results_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_error_decision,
        "ready"_s <= "state_apply_effect_results_backend_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_backend_error_decision,
        "ready"_s <= "state_apply_effect_results_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_apply_effect_results_error_callback,
        "ready"_s <= "bind_tensor_request_decision"_s + unexpected_event<_> / on_unexpected_from_bind_tensor_request_decision,
        "ready"_s <= "bind_tensor_exec"_s + unexpected_event<_> / on_unexpected_from_bind_tensor_exec,
        "ready"_s <= "bind_tensor_result_decision"_s + unexpected_event<_> / on_unexpected_from_bind_tensor_result_decision,
        "ready"_s <= "evict_tensor_request_decision"_s + unexpected_event<_> / on_unexpected_from_evict_tensor_request_decision,
        "ready"_s <= "evict_tensor_exec"_s + unexpected_event<_> / on_unexpected_from_evict_tensor_exec,
        "ready"_s <= "evict_tensor_result_decision"_s + unexpected_event<_> / on_unexpected_from_evict_tensor_result_decision,
        "ready"_s <= "capture_tensor_state_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_request_decision,
        "ready"_s <= "capture_tensor_state_exec"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_exec,
        "ready"_s <= "capture_tensor_state_result_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_result_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "ready"_s <= "state_request_mapped_load_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_decision,
        "ready"_s <= "state_request_mapped_load_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_dispatch_decision,
        "ready"_s <= "state_request_mapped_load_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_done_callback,
        "ready"_s <= "state_request_mapped_load_invalid_request_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_invalid_request_error_decision,
        "ready"_s <= "state_request_mapped_load_unsupported_io_mmap_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_unsupported_io_mmap_error_decision,
        "ready"_s <= "state_request_mapped_load_already_resident_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_already_resident_error_decision,
        "ready"_s <= "state_request_mapped_load_io_mmap_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_request_mapped_load_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_mapped_load_error_callback,
        "ready"_s <= "state_request_read_load_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_decision,
        "ready"_s <= "state_request_read_load_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_dispatch_decision,
        "ready"_s <= "state_request_read_load_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_done_callback,
        "ready"_s <= "state_request_read_load_invalid_request_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_invalid_request_error_decision,
        "ready"_s <= "state_request_read_load_unsupported_io_read_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_unsupported_io_read_error_decision,
        "ready"_s <= "state_request_read_load_already_resident_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_already_resident_error_decision,
        "ready"_s <= "state_request_read_load_io_read_file_open_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_io_read_file_open_error_decision,
        "ready"_s <= "state_request_read_load_io_read_file_read_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_io_read_file_read_error_decision,
        "ready"_s <= "state_request_read_load_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_read_load_error_callback,
        "ready"_s <= "state_request_staged_load_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_decision,
        "ready"_s <= "state_request_staged_load_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_dispatch_decision,
        "ready"_s <= "state_request_staged_load_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_done_callback,
        "ready"_s <= "state_request_staged_load_invalid_request_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_invalid_request_error_decision,
        "ready"_s <= "state_request_staged_load_unsupported_io_staged_read_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_unsupported_io_staged_read_error_decision,
        "ready"_s <= "state_request_staged_load_already_resident_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_already_resident_error_decision,
        "ready"_s <= "state_request_staged_load_io_staged_read_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_io_staged_read_error_decision,
        "ready"_s <= "state_request_staged_load_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_request_staged_load_error_callback,
        "ready"_s <= "state_release_mapped_load_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_decision,
        "ready"_s <= "state_release_mapped_load_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_dispatch_decision,
        "ready"_s <= "state_release_mapped_load_publish_done_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_publish_done_decision,
        "ready"_s <= "state_release_mapped_load_done_callback"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_done_callback,
        "ready"_s <= "state_release_mapped_load_invalid_request_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_invalid_request_error_decision,
        "ready"_s <= "state_release_mapped_load_unsupported_io_mmap_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_unsupported_io_mmap_error_decision,
        "ready"_s <= "state_release_mapped_load_handle_absent_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_handle_absent_error_decision,
        "ready"_s <= "state_release_mapped_load_io_mmap_error_decision"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_io_mmap_error_decision,
        "ready"_s <= "state_release_mapped_load_error_callback"_s + unexpected_event<_> / on_unexpected_from_state_release_mapped_load_error_callback,
    }
}

/// Context for `ModelTensor` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct ModelTensorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl ModelTensorStateMachineContext for ModelTensorContext {
    fn apply_effect_results_done_callback_absent(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_effect_results_done_callback_absent
        todo!(
            "TODO: port guard `apply_effect_results_done_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_effect_results_done_callback_present(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_effect_results_done_callback_present
        todo!(
            "TODO: port guard `apply_effect_results_done_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_effect_results_error_callback_absent(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_effect_results_error_callback_absent
        todo!(
            "TODO: port guard `apply_effect_results_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_effect_results_error_callback_present(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_effect_results_error_callback_present
        todo!(
            "TODO: port guard `apply_effect_results_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_results_invalid(&self, _event: &DetailApplyEffectResultsRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_results_invalid
        todo!(
            "TODO: port guard `apply_results_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_results_valid_with_effect_errors(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_results_valid_with_effect_errors
        todo!(
            "TODO: port guard `apply_results_valid_with_effect_errors` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_results_valid_without_effect_errors_with_record_output(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_results_valid_without_effect_errors_with_record_output
        todo!(
            "TODO: port guard `apply_results_valid_without_effect_errors_with_record_output` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn apply_results_valid_without_effect_errors_without_record_output(
        &self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::apply_results_valid_without_effect_errors_without_record_output
        todo!(
            "TODO: port guard `apply_results_valid_without_effect_errors_without_record_output` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn begin_bind_tensor(&mut self, _event: &DetailBindTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::begin_bind_tensor
        todo!(
            "TODO: port action `begin_bind_tensor` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn begin_capture_tensor_state(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::begin_capture_tensor_state
        todo!(
            "TODO: port action `begin_capture_tensor_state` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn begin_evict_tensor(&mut self, _event: &DetailEvictTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::begin_evict_tensor
        todo!(
            "TODO: port action `begin_evict_tensor` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn bind_storage_done_callback_absent(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_storage_done_callback_absent
        todo!(
            "TODO: port guard `bind_storage_done_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn bind_storage_done_callback_present(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_storage_done_callback_present
        todo!(
            "TODO: port guard `bind_storage_done_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn bind_storage_error_callback_absent(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_storage_error_callback_absent
        todo!(
            "TODO: port guard `bind_storage_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn bind_storage_error_callback_present(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_storage_error_callback_present
        todo!(
            "TODO: port guard `bind_storage_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn bind_tensor_request_invalid(&self, _event: &DetailBindTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_tensor_request_invalid
        todo!(
            "TODO: port guard `bind_tensor_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn bind_tensor_request_valid(&self, _event: &DetailBindTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::bind_tensor_request_valid
        todo!(
            "TODO: port guard `bind_tensor_request_valid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn capture_tensor_state_request_invalid(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::capture_tensor_state_request_invalid
        todo!(
            "TODO: port guard `capture_tensor_state_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn capture_tensor_state_request_valid(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::capture_tensor_state_request_valid
        todo!(
            "TODO: port guard `capture_tensor_state_request_valid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn effect_apply_results(&mut self, _event: &DetailApplyEffectResultsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_apply_results
        todo!(
            "TODO: port action `effect_apply_results` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_apply_results_with_record_output(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_apply_results_with_record_output
        todo!(
            "TODO: port action `effect_apply_results_with_record_output` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_attempt_release_mapped_load_dispatch(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_attempt_release_mapped_load_dispatch
        todo!(
            "TODO: port action `effect_attempt_release_mapped_load_dispatch` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_attempt_request_mapped_load_dispatch(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_attempt_request_mapped_load_dispatch
        todo!(
            "TODO: port action `effect_attempt_request_mapped_load_dispatch` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_attempt_request_read_load_dispatch(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_attempt_request_read_load_dispatch
        todo!(
            "TODO: port action `effect_attempt_request_read_load_dispatch` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_attempt_request_staged_load_dispatch(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_attempt_request_staged_load_dispatch
        todo!(
            "TODO: port action `effect_attempt_request_staged_load_dispatch` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_begin_release_mapped_load(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_begin_release_mapped_load
        todo!(
            "TODO: port action `effect_begin_release_mapped_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_begin_request_mapped_load(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_begin_request_mapped_load
        todo!(
            "TODO: port action `effect_begin_request_mapped_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_begin_request_read_load(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_begin_request_read_load
        todo!(
            "TODO: port action `effect_begin_request_read_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_begin_request_staged_load(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_begin_request_staged_load
        todo!(
            "TODO: port action `effect_begin_request_staged_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_bind_storage(&mut self, _event: &DetailBindStorageRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_bind_storage
        todo!(
            "TODO: port action `effect_bind_storage` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_commit_release_mapped_load(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_commit_release_mapped_load
        todo!(
            "TODO: port action `effect_commit_release_mapped_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_commit_request_mapped_load(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_commit_request_mapped_load
        todo!(
            "TODO: port action `effect_commit_request_mapped_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_commit_request_read_load(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_commit_request_read_load
        todo!(
            "TODO: port action `effect_commit_request_read_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_commit_request_staged_load(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_commit_request_staged_load
        todo!(
            "TODO: port action `effect_commit_request_staged_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_release_mapped_load_handle_absent(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_release_mapped_load_handle_absent
        todo!(
            "TODO: port action `effect_mark_release_mapped_load_handle_absent` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_release_mapped_load_invalid_request(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_release_mapped_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_release_mapped_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_release_mapped_load_io_mmap_failed(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_release_mapped_load_io_mmap_failed
        todo!(
            "TODO: port action `effect_mark_release_mapped_load_io_mmap_failed` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_release_mapped_load_unsupported_io_mmap(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_release_mapped_load_unsupported_io_mmap
        todo!(
            "TODO: port action `effect_mark_release_mapped_load_unsupported_io_mmap` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_mapped_load_invalid_request(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_mapped_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_request_mapped_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_mapped_load_io_mmap_failed(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_mapped_load_io_mmap_failed
        todo!(
            "TODO: port action `effect_mark_request_mapped_load_io_mmap_failed` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_mapped_load_tensor_already_resident(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_mapped_load_tensor_already_resident
        todo!(
            "TODO: port action `effect_mark_request_mapped_load_tensor_already_resident` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_mapped_load_unsupported_io_mmap(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_mapped_load_unsupported_io_mmap
        todo!(
            "TODO: port action `effect_mark_request_mapped_load_unsupported_io_mmap` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_invalid_request_from_state_request_read_load_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_request_read_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_invalid_request_from_state_request_read_load_dispatch_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_request_read_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_io_read_failed_from_state_request_read_load_dispatch_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_io_read_failed
        todo!(
            "TODO: port action `effect_mark_request_read_load_io_read_failed` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_tensor_already_resident(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_tensor_already_resident
        todo!(
            "TODO: port action `effect_mark_request_read_load_tensor_already_resident` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_unsupported_io_read_from_state_request_read_load_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_unsupported_io_read
        todo!(
            "TODO: port action `effect_mark_request_read_load_unsupported_io_read` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_read_load_unsupported_io_read_from_state_request_read_load_dispatch_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_read_load_unsupported_io_read
        todo!(
            "TODO: port action `effect_mark_request_read_load_unsupported_io_read` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_invalid_request_from_state_request_staged_load_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_request_staged_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_invalid_request_from_state_request_staged_load_dispatch_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_invalid_request
        todo!(
            "TODO: port action `effect_mark_request_staged_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_io_staged_read_failed(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_io_staged_read_failed
        todo!(
            "TODO: port action `effect_mark_request_staged_load_io_staged_read_failed` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_tensor_already_resident(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_tensor_already_resident
        todo!(
            "TODO: port action `effect_mark_request_staged_load_tensor_already_resident` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_unsupported_io_staged_read_from_state_request_staged_load_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_unsupported_io_staged_read
        todo!(
            "TODO: port action `effect_mark_request_staged_load_unsupported_io_staged_read` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_mark_request_staged_load_unsupported_io_staged_read_from_state_request_staged_load_dispatch_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_mark_request_staged_load_unsupported_io_staged_read
        todo!(
            "TODO: port action `effect_mark_request_staged_load_unsupported_io_staged_read` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_plan_io_load(&mut self, _event: &DetailPlanLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_plan_io_load
        todo!(
            "TODO: port action `effect_plan_io_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_plan_load(&mut self, _event: &DetailPlanLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_plan_load
        todo!(
            "TODO: port action `effect_plan_load` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_release_mapped_load_done(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_release_mapped_load_done
        todo!(
            "TODO: port action `effect_publish_release_mapped_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_release_mapped_load_error_from_state_release_mapped_load_handle_absent_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_release_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_release_mapped_load_error_from_state_release_mapped_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_release_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_release_mapped_load_error_from_state_release_mapped_load_io_mmap_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_release_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_release_mapped_load_error_from_state_release_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_release_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_mapped_load_done(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_mapped_load_done
        todo!(
            "TODO: port action `effect_publish_request_mapped_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_mapped_load_error_from_state_request_mapped_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_mapped_load_error_from_state_request_mapped_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_mapped_load_error_from_state_request_mapped_load_io_mmap_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_mapped_load_error_from_state_request_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_mapped_load_error
        todo!(
            "TODO: port action `effect_publish_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_done(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_done
        todo!(
            "TODO: port action `effect_publish_request_read_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_error_from_state_request_read_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_error
        todo!(
            "TODO: port action `effect_publish_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_error_from_state_request_read_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_error
        todo!(
            "TODO: port action `effect_publish_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_error_from_state_request_read_load_io_read_file_open_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_error
        todo!(
            "TODO: port action `effect_publish_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_error_from_state_request_read_load_io_read_file_read_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_error
        todo!(
            "TODO: port action `effect_publish_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_read_load_error_from_state_request_read_load_unsupported_io_read_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_read_load_error
        todo!(
            "TODO: port action `effect_publish_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_staged_load_done(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_staged_load_done
        todo!(
            "TODO: port action `effect_publish_request_staged_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_staged_load_error_from_state_request_staged_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_staged_load_error
        todo!(
            "TODO: port action `effect_publish_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_staged_load_error_from_state_request_staged_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_staged_load_error
        todo!(
            "TODO: port action `effect_publish_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_staged_load_error_from_state_request_staged_load_io_staged_read_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_staged_load_error
        todo!(
            "TODO: port action `effect_publish_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_publish_request_staged_load_error_from_state_request_staged_load_unsupported_io_staged_read_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_publish_request_staged_load_error
        todo!(
            "TODO: port action `effect_publish_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_done_from_state_release_mapped_load_done_callback(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_done
        todo!(
            "TODO: port action `effect_record_release_mapped_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_done_from_state_release_mapped_load_publish_done_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_done
        todo!(
            "TODO: port action `effect_record_release_mapped_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_error_from_state_release_mapped_load_error_callback(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_error
        todo!(
            "TODO: port action `effect_record_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_error_from_state_release_mapped_load_handle_absent_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_error
        todo!(
            "TODO: port action `effect_record_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_error_from_state_release_mapped_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_error
        todo!(
            "TODO: port action `effect_record_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_error_from_state_release_mapped_load_io_mmap_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_error
        todo!(
            "TODO: port action `effect_record_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_release_mapped_load_error_from_state_release_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_release_mapped_load_error
        todo!(
            "TODO: port action `effect_record_release_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_mapped_load_error_from_state_request_mapped_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_mapped_load_error
        todo!(
            "TODO: port action `effect_record_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_mapped_load_error_from_state_request_mapped_load_error_callback(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_mapped_load_error
        todo!(
            "TODO: port action `effect_record_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_mapped_load_error_from_state_request_mapped_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_mapped_load_error
        todo!(
            "TODO: port action `effect_record_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_mapped_load_error_from_state_request_mapped_load_io_mmap_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_mapped_load_error
        todo!(
            "TODO: port action `effect_record_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_mapped_load_error_from_state_request_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_mapped_load_error
        todo!(
            "TODO: port action `effect_record_request_mapped_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_error_callback(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_io_read_file_open_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_io_read_file_read_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_read_load_error_from_state_request_read_load_unsupported_io_read_error_decision(
        &mut self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_read_load_error
        todo!(
            "TODO: port action `effect_record_request_read_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_staged_load_error_from_state_request_staged_load_already_resident_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_staged_load_error
        todo!(
            "TODO: port action `effect_record_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_staged_load_error_from_state_request_staged_load_error_callback(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_staged_load_error
        todo!(
            "TODO: port action `effect_record_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_staged_load_error_from_state_request_staged_load_invalid_request_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_staged_load_error
        todo!(
            "TODO: port action `effect_record_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_staged_load_error_from_state_request_staged_load_io_staged_read_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_staged_load_error
        todo!(
            "TODO: port action `effect_record_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn effect_record_request_staged_load_error_from_state_request_staged_load_unsupported_io_staged_read_error_decision(
        &mut self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::effect_record_request_staged_load_error
        todo!(
            "TODO: port action `effect_record_request_staged_load_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn error_code_output_absent_detail_bind_tensor_runtime(
        &self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_absent
        todo!(
            "TODO: port guard `error_code_output_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn error_code_output_absent_detail_capture_tensor_state_runtime(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_absent
        todo!(
            "TODO: port guard `error_code_output_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn error_code_output_absent_detail_evict_tensor_runtime(
        &self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_absent
        todo!(
            "TODO: port guard `error_code_output_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn error_code_output_present_detail_bind_tensor_runtime(
        &self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_present
        todo!(
            "TODO: port guard `error_code_output_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn error_code_output_present_detail_capture_tensor_state_runtime(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_present
        todo!(
            "TODO: port guard `error_code_output_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn error_code_output_present_detail_evict_tensor_runtime(
        &self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::error_code_output_present
        todo!(
            "TODO: port guard `error_code_output_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn evict_tensor_request_invalid(&self, _event: &DetailEvictTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::evict_tensor_request_invalid
        todo!(
            "TODO: port guard `evict_tensor_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn evict_tensor_request_valid(&self, _event: &DetailEvictTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::evict_tensor_request_valid
        todo!(
            "TODO: port guard `evict_tensor_request_valid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn exec_bind_tensor(&mut self, _event: &DetailBindTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::exec_bind_tensor
        todo!(
            "TODO: port action `exec_bind_tensor` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn exec_capture_tensor_state(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::exec_capture_tensor_state
        todo!(
            "TODO: port action `exec_capture_tensor_state` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn exec_evict_tensor(&mut self, _event: &DetailEvictTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::exec_evict_tensor
        todo!(
            "TODO: port action `exec_evict_tensor` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn guard_storage_bind_valid_with_mmap_resident(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::guard_storage_bind_valid_with_mmap_resident
        todo!(
            "TODO: port guard `guard_storage_bind_valid_with_mmap_resident` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn guard_storage_bind_valid_without_mmap_resident(
        &self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::guard_storage_bind_valid_without_mmap_resident
        todo!(
            "TODO: port guard `guard_storage_bind_valid_without_mmap_resident` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn mark_invalid_request_detail_bind_tensor_runtime(
        &mut self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_evict_tensor_runtime(
        &mut self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_tensor_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_bind_tensor_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_bind_tensor_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_state_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_state_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_state_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_evict_tensor_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_evict_tensor_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_evict_tensor_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_backend_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_apply_effect_results_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_awaiting_effects(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_busy_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_bind_storage_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_plan_load_capacity_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_plan_load_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_plan_load_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_plan_load_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_plan_load_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_handle_absent_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_invalid_request_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_io_mmap_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_publish_done_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_release_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_already_resident_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_invalid_request_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_io_mmap_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_mapped_load_unsupported_io_mmap_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_already_resident_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_invalid_request_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_io_read_file_open_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_io_read_file_read_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_read_load_unsupported_io_read_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_already_resident_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_invalid_request_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_io_staged_read_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn on_unexpected_from_state_request_staged_load_unsupported_io_staged_read_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn operation_not_dispatched_detail_bind_tensor_runtime(
        &self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_capture_tensor_state_runtime(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_evict_tensor_runtime(
        &self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_bind_tensor_runtime(
        &self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_capture_tensor_state_runtime(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_evict_tensor_runtime(
        &self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_done_callback_absent(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_done_callback_absent
        todo!(
            "TODO: port guard `plan_load_done_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_done_callback_present(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_done_callback_present
        todo!(
            "TODO: port guard `plan_load_done_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_error_callback_absent(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_error_callback_absent
        todo!(
            "TODO: port guard `plan_load_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_error_callback_present(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_error_callback_present
        todo!(
            "TODO: port guard `plan_load_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_invalid_capacity(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_invalid_capacity
        todo!(
            "TODO: port guard `plan_load_invalid_capacity` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_invalid_request(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_invalid_request
        todo!(
            "TODO: port guard `plan_load_invalid_request` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_valid_with_io_strategy(&self, _event: &DetailPlanLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_valid_with_io_strategy
        todo!(
            "TODO: port guard `plan_load_valid_with_io_strategy` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn plan_load_valid_without_io_strategy(
        &self,
        _event: &DetailPlanLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::plan_load_valid_without_io_strategy
        todo!(
            "TODO: port guard `plan_load_valid_without_io_strategy` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn publish_apply_effect_results_backend_error(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_apply_effect_results_backend_error
        todo!(
            "TODO: port action `publish_apply_effect_results_backend_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_apply_effect_results_done(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_apply_effect_results_done
        todo!(
            "TODO: port action `publish_apply_effect_results_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_apply_effect_results_invalid_request(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_apply_effect_results_invalid_request
        todo!(
            "TODO: port action `publish_apply_effect_results_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_bind_storage_done(&mut self, _event: &DetailBindStorageRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_bind_storage_done
        todo!(
            "TODO: port action `publish_bind_storage_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_bind_storage_error_from_state_awaiting_effects(
        &mut self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_bind_storage_error
        todo!(
            "TODO: port action `publish_bind_storage_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_bind_storage_error_from_state_bind_storage_error_decision(
        &mut self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_bind_storage_error
        todo!(
            "TODO: port action `publish_bind_storage_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_done_detail_bind_tensor_runtime(
        &mut self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_done_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_done_detail_evict_tensor_runtime(
        &mut self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_done_with_error_code_detail_bind_tensor_runtime(
        &mut self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done_with_error_code
        todo!(
            "TODO: port action `publish_done_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_done_with_error_code_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done_with_error_code
        todo!(
            "TODO: port action `publish_done_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_done_with_error_code_detail_evict_tensor_runtime(
        &mut self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_done_with_error_code
        todo!(
            "TODO: port action `publish_done_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_error_detail_bind_tensor_runtime(
        &mut self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_error_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_error_detail_evict_tensor_runtime(
        &mut self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/model/tensor/actions.hpp")
    }
    fn publish_error_with_error_code_detail_bind_tensor_runtime(
        &mut self,
        _event: &DetailBindTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error_with_error_code
        todo!(
            "TODO: port action `publish_error_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_error_with_error_code_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error_with_error_code
        todo!(
            "TODO: port action `publish_error_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_error_with_error_code_detail_evict_tensor_runtime(
        &mut self,
        _event: &DetailEvictTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_error_with_error_code
        todo!(
            "TODO: port action `publish_error_with_error_code` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_plan_load_capacity_error(
        &mut self,
        _event: &DetailPlanLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_plan_load_capacity_error
        todo!(
            "TODO: port action `publish_plan_load_capacity_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_plan_load_done(&mut self, _event: &DetailPlanLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_plan_load_done
        todo!(
            "TODO: port action `publish_plan_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn publish_plan_load_invalid_request(
        &mut self,
        _event: &DetailPlanLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::publish_plan_load_invalid_request
        todo!(
            "TODO: port action `publish_plan_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_apply_effect_results_backend_error(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_apply_effect_results_backend_error
        todo!(
            "TODO: port action `record_apply_effect_results_backend_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_apply_effect_results_done(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_apply_effect_results_done
        todo!(
            "TODO: port action `record_apply_effect_results_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_apply_effect_results_invalid_request(
        &mut self,
        _event: &DetailApplyEffectResultsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_apply_effect_results_invalid_request
        todo!(
            "TODO: port action `record_apply_effect_results_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_bind_storage_done(&mut self, _event: &DetailBindStorageRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_bind_storage_done
        todo!(
            "TODO: port action `record_bind_storage_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_bind_storage_invalid_request_from_ready(
        &mut self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_bind_storage_invalid_request
        todo!(
            "TODO: port action `record_bind_storage_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_bind_storage_invalid_request_from_state_awaiting_effects(
        &mut self,
        _event: &DetailBindStorageRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_bind_storage_invalid_request
        todo!(
            "TODO: port action `record_bind_storage_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_plan_load_capacity_error(
        &mut self,
        _event: &DetailPlanLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_plan_load_capacity_error
        todo!(
            "TODO: port action `record_plan_load_capacity_error` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_plan_load_done(&mut self, _event: &DetailPlanLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_plan_load_done
        todo!(
            "TODO: port action `record_plan_load_done` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn record_plan_load_invalid_request(
        &mut self,
        _event: &DetailPlanLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/actions.hpp::record_plan_load_invalid_request
        todo!(
            "TODO: port action `record_plan_load_invalid_request` from emel.cpp/src/emel/model/tensor/actions.hpp"
        )
    }
    fn release_mapped_load_done_callback_absent(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_done_callback_absent
        todo!(
            "TODO: port guard `release_mapped_load_done_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_done_callback_present(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_done_callback_present
        todo!(
            "TODO: port guard `release_mapped_load_done_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_error_callback_absent(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_error_callback_absent
        todo!(
            "TODO: port guard `release_mapped_load_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_error_callback_present(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_error_callback_present
        todo!(
            "TODO: port guard `release_mapped_load_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_absent(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_absent
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_failed(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_failed
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_failed` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_present_request_invalid(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_present_request_invalid
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_present_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_present_request_valid_handle_absent(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_present_request_valid_handle_absent
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_present_request_valid_handle_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_present_request_valid_handle_present(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_present_request_valid_handle_present
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_present_request_valid_handle_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn release_mapped_load_io_mmap_succeeded(
        &self,
        _event: &DetailReleaseMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::release_mapped_load_io_mmap_succeeded
        todo!(
            "TODO: port guard `release_mapped_load_io_mmap_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_error_callback_absent(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_error_callback_absent
        todo!(
            "TODO: port guard `request_mapped_load_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_error_callback_present(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_error_callback_present
        todo!(
            "TODO: port guard `request_mapped_load_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_absent(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_absent
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_failed(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_failed
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_failed` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_present_request_invalid(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_present_request_invalid
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_present_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_present_request_valid_already_resident(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_present_request_valid_already_resident
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_present_request_valid_already_resident` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_present_request_valid_tensor_unbound(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_present_request_valid_tensor_unbound
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_present_request_valid_tensor_unbound` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_mapped_load_io_mmap_succeeded(
        &self,
        _event: &DetailRequestMappedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_mapped_load_io_mmap_succeeded
        todo!(
            "TODO: port guard `request_mapped_load_io_mmap_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_error_callback_absent(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_error_callback_absent
        todo!(
            "TODO: port guard `request_read_load_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_error_callback_present(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_error_callback_present
        todo!(
            "TODO: port guard `request_read_load_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_absent(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_absent
        todo!(
            "TODO: port guard `request_read_load_io_read_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_file_open_failed(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_file_open_failed
        todo!(
            "TODO: port guard `request_read_load_io_read_file_open_failed` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_file_read_or_other_failed(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_file_read_or_other_failed
        todo!(
            "TODO: port guard `request_read_load_io_read_file_read_or_other_failed` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_invalid_request(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_invalid_request
        todo!(
            "TODO: port guard `request_read_load_io_read_invalid_request` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_present_request_invalid(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_present_request_invalid
        todo!(
            "TODO: port guard `request_read_load_io_read_present_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_present_request_valid_already_resident(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_present_request_valid_already_resident
        todo!(
            "TODO: port guard `request_read_load_io_read_present_request_valid_already_resident` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_present_request_valid_tensor_unbound(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_present_request_valid_tensor_unbound
        todo!(
            "TODO: port guard `request_read_load_io_read_present_request_valid_tensor_unbound` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_succeeded(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_succeeded
        todo!(
            "TODO: port guard `request_read_load_io_read_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_read_load_io_read_unsupported(
        &self,
        _event: &DetailRequestReadLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_read_load_io_read_unsupported
        todo!(
            "TODO: port guard `request_read_load_io_read_unsupported` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_error_callback_absent(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_error_callback_absent
        todo!(
            "TODO: port guard `request_staged_load_error_callback_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_error_callback_present(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_error_callback_present
        todo!(
            "TODO: port guard `request_staged_load_error_callback_present` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_absent(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_absent
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_absent` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_invalid_request(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_invalid_request
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_invalid_request` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_other_failed(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_other_failed
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_other_failed` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_present_request_invalid(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_present_request_invalid
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_present_request_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_present_request_valid_already_resident(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_present_request_valid_already_resident
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_present_request_valid_already_resident` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_present_request_valid_tensor_unbound(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_present_request_valid_tensor_unbound
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_present_request_valid_tensor_unbound` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_succeeded(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_succeeded
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_succeeded` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn request_staged_load_io_staged_read_unsupported(
        &self,
        _event: &DetailRequestStagedLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::request_staged_load_io_staged_read_unsupported
        todo!(
            "TODO: port guard `request_staged_load_io_staged_read_unsupported` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
    fn storage_bind_invalid(&self, _event: &DetailBindStorageRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/guards.hpp::storage_bind_invalid
        todo!(
            "TODO: port guard `storage_bind_invalid` from emel.cpp/src/emel/model/tensor/guards.hpp"
        )
    }
}
