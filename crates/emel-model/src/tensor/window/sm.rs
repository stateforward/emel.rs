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

// --- machine ModelTensorWindow from emel.cpp/src/emel/model/tensor/window/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailAcquirePublishRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailAcquireResolveRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailAcquireRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailBindWindowRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailUnbindFinishRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailUnbindRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EmelEventCompletion;

sml! {
    ModelTensorWindow {
        "state_bind_request_decision"_s <= *"state_unbound"_s + event<DetailBindWindowRuntime> / effect_begin_bind,
        "state_bind_source_decision"_s <= "state_bind_request_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_request_valid] / effect_map_source,
        "state_bind_error_ready"_s <= "state_bind_request_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_request_invalid] / effect_mark_bind_invalid,
        "state_bind_budget_decision"_s <= "state_bind_source_decision"_s + completion<DetailBindWindowRuntime> [guard_source_map_succeeded] / effect_scan_layer_plan,
        "state_bind_error_ready"_s <= "state_bind_source_decision"_s + completion<DetailBindWindowRuntime> [guard_source_map_failed] / effect_mark_source_map_failed,
        "state_passthrough_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_fits_budget_callback_present] / effect_activate_passthrough_and_publish,
        "state_passthrough_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_fits_budget_callback_absent] / effect_activate_passthrough_and_record,
        "state_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_requires_streaming_callback_present] / effect_activate_streaming_and_publish,
        "state_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_requires_streaming_callback_absent] / effect_activate_streaming_and_record,
        "state_bind_error_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_budget_too_small] / effect_release_and_mark_budget_too_small,
        "state_bind_error_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_slot_storage_too_small] / effect_release_and_mark_slot_storage_too_small,
        "state_bind_error_ready"_s <= "state_bind_budget_decision"_s + completion<DetailBindWindowRuntime> [guard_bind_streaming_config_invalid] / effect_release_and_mark_streaming_config_invalid,
        "state_bind_error_callback"_s <= "state_bind_error_ready"_s + completion<DetailBindWindowRuntime> [guard_bind_error_callback_present] / effect_publish_bind_error,
        "state_unbound"_s <= "state_bind_error_ready"_s + completion<DetailBindWindowRuntime> [guard_bind_error_callback_absent_unbound] / effect_reset_rejected_window_from_state_bind_error_ready,
        "state_passthrough_ready"_s <= "state_bind_error_ready"_s + completion<DetailBindWindowRuntime> [guard_bind_error_callback_absent_retained] / effect_retain_source_for_release_retry_from_state_bind_error_ready,
        "state_unbound"_s <= "state_bind_error_callback"_s + completion<DetailBindWindowRuntime> [guard_bind_error_exit_unbound] / effect_reset_rejected_window_from_state_bind_error_callback,
        "state_passthrough_ready"_s <= "state_bind_error_callback"_s + completion<DetailBindWindowRuntime> [guard_bind_error_exit_retained] / effect_retain_source_for_release_retry_from_state_bind_error_callback,
        "state_ready"_s <= "state_ready"_s + event<DetailBindWindowRuntime> [guard_bind_error_callback_present] / effect_mark_already_bound_and_publish_from_state_ready,
        "state_ready"_s <= "state_ready"_s + event<DetailBindWindowRuntime> [guard_bind_error_callback_absent] / effect_mark_bind_already_bound_from_state_ready,
        "state_passthrough_ready"_s <= "state_passthrough_ready"_s + event<DetailBindWindowRuntime> [guard_bind_error_callback_present] / effect_mark_already_bound_and_publish_from_state_passthrough_ready,
        "state_passthrough_ready"_s <= "state_passthrough_ready"_s + event<DetailBindWindowRuntime> [guard_bind_error_callback_absent] / effect_mark_bind_already_bound_from_state_passthrough_ready,
        "state_acquire_resolve_decision"_s <= "state_ready"_s + event<DetailAcquireResolveRuntime> / effect_begin_acquire_resolve,
        "state_acquire_resolved"_s <= "state_acquire_resolve_decision"_s + completion<DetailAcquireResolveRuntime> [guard_resolve_layer_out_of_range] / effect_mark_resolve_out_of_range,
        "state_acquire_resolved"_s <= "state_acquire_resolve_decision"_s + completion<DetailAcquireResolveRuntime> [guard_resolve_slot_busy_other_layer] / effect_require_busy_slot,
        "state_acquire_resolved"_s <= "state_acquire_resolve_decision"_s + completion<DetailAcquireResolveRuntime> [guard_resolve_slot_ready_for_target],
        "state_acquire_resolved"_s <= "state_acquire_resolved"_s + event<EmelEventCompletion> [guard_completion_slot_armed] / effect_commit_slot_load_from_state_acquire_resolved,
        "state_acquire_resolved"_s <= "state_acquire_resolved"_s + event<EmelEventCompletion> [guard_completion_slot_stray] / effect_record_stray_completion_from_state_acquire_resolved,
        "state_acquire_decision"_s <= "state_acquire_resolved"_s + event<DetailAcquireRuntime>,
        "state_acquire_pending"_s <= "state_acquire_decision"_s + completion<DetailAcquireRuntime> [guard_acquire_layer_out_of_range] / effect_mark_acquire_out_of_range,
        "state_acquire_pending"_s <= "state_acquire_decision"_s + completion<DetailAcquireRuntime> [guard_acquire_layer_resident],
        "state_acquire_pending"_s <= "state_acquire_decision"_s + completion<DetailAcquireRuntime> [guard_acquire_layer_loading] / effect_require_layer_completion,
        "state_acquire_pending"_s <= "state_acquire_decision"_s + completion<DetailAcquireRuntime> [guard_acquire_layer_unscheduled] / effect_submit_and_require_layer_from_state_acquire_decision,
        "state_acquire_pending"_s <= "state_acquire_decision"_s + completion<DetailAcquireRuntime> [guard_acquire_layer_failed] / effect_submit_and_require_layer_from_state_acquire_decision,
        "state_acquire_publish_decision"_s <= "state_acquire_pending"_s + event<DetailAcquirePublishRuntime>,
        "state_acquire_advance_decision"_s <= "state_acquire_publish_decision"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_result_ready] / effect_stage_acquire_result,
        "state_acquire_error_ready"_s <= "state_acquire_publish_decision"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_error_pending],
        "state_acquire_error_ready"_s <= "state_acquire_publish_decision"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_copy_failed] / effect_mark_slot_copy_failed,
        "state_acquire_publish_ready"_s <= "state_acquire_advance_decision"_s + completion<DetailAcquirePublishRuntime> [guard_prefetch_ahead_needed] / effect_advance_window,
        "state_acquire_publish_ready"_s <= "state_acquire_advance_decision"_s + completion<DetailAcquirePublishRuntime> [guard_prefetch_ahead_not_needed],
        "state_acquire_done_callback"_s <= "state_acquire_publish_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_done_callback_present] / effect_publish_acquire_done,
        "state_ready"_s <= "state_acquire_publish_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_done_callback_absent] / effect_record_acquire_done_from_state_acquire_publish_ready,
        "state_ready"_s <= "state_acquire_done_callback"_s + completion<DetailAcquirePublishRuntime> / effect_record_acquire_done_from_state_acquire_done_callback,
        "state_acquire_error_callback"_s <= "state_acquire_error_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_error_callback_present] / effect_publish_acquire_error_from_state_acquire_error_ready,
        "state_ready"_s <= "state_acquire_error_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_error_callback_absent] / effect_record_acquire_error_from_state_acquire_error_ready,
        "state_ready"_s <= "state_acquire_error_callback"_s + completion<DetailAcquirePublishRuntime> / effect_record_acquire_error_from_state_acquire_error_callback,
        "state_passthrough_acquire_resolved"_s <= "state_passthrough_ready"_s + event<DetailAcquireResolveRuntime> / effect_begin_resolve_not_streaming,
        "state_passthrough_acquire_pending"_s <= "state_passthrough_acquire_resolved"_s + event<DetailAcquireRuntime>,
        "state_passthrough_acquire_error_ready"_s <= "state_passthrough_acquire_pending"_s + event<DetailAcquirePublishRuntime>,
        "state_passthrough_acquire_error_callback"_s <= "state_passthrough_acquire_error_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_error_callback_present] / effect_publish_acquire_error_from_state_passthrough_acquire_error_ready,
        "state_passthrough_ready"_s <= "state_passthrough_acquire_error_ready"_s + completion<DetailAcquirePublishRuntime> [guard_acquire_error_callback_absent] / effect_record_acquire_error_from_state_passthrough_acquire_error_ready,
        "state_passthrough_ready"_s <= "state_passthrough_acquire_error_callback"_s + completion<DetailAcquirePublishRuntime> / effect_record_acquire_error_from_state_passthrough_acquire_error_callback,
        "state_unbound"_s <= "state_unbound"_s + event<DetailAcquireResolveRuntime> / effect_begin_resolve_not_bound,
        "state_unbound"_s <= "state_unbound"_s + event<DetailAcquireRuntime>,
        "state_unbound"_s <= "state_unbound"_s + event<DetailAcquirePublishRuntime> [guard_acquire_error_callback_present] / effect_publish_acquire_error_from_state_unbound,
        "state_unbound"_s <= "state_unbound"_s + event<DetailAcquirePublishRuntime> [guard_acquire_error_callback_absent] / effect_record_acquire_error_from_state_unbound,
        "state_ready"_s <= "state_ready"_s + event<EmelEventCompletion> [guard_completion_slot_armed] / effect_commit_slot_load_from_state_ready,
        "state_ready"_s <= "state_ready"_s + event<EmelEventCompletion> [guard_completion_slot_stray] / effect_record_stray_completion_from_state_ready,
        "state_acquire_pending"_s <= "state_acquire_pending"_s + event<EmelEventCompletion> [guard_completion_slot_armed] / effect_commit_slot_load_from_state_acquire_pending,
        "state_acquire_pending"_s <= "state_acquire_pending"_s + event<EmelEventCompletion> [guard_completion_slot_stray] / effect_record_stray_completion_from_state_acquire_pending,
        "state_unbind_pending"_s <= "state_unbind_pending"_s + event<EmelEventCompletion> [guard_completion_slot_armed] / effect_commit_slot_load_from_state_unbind_pending,
        "state_unbind_pending"_s <= "state_unbind_pending"_s + event<EmelEventCompletion> [guard_completion_slot_stray] / effect_record_stray_completion_from_state_unbind_pending,
        "state_passthrough_ready"_s <= "state_passthrough_ready"_s + event<EmelEventCompletion> / effect_record_stray_completion_from_state_passthrough_ready,
        "state_unbound"_s <= "state_unbound"_s + event<EmelEventCompletion> / effect_record_stray_completion_from_state_unbound,
        "state_unbind_pending"_s <= "state_ready"_s + event<DetailUnbindRuntime> / effect_begin_unbind_from_state_ready,
        "state_unbind_pending"_s <= "state_passthrough_ready"_s + event<DetailUnbindRuntime> / effect_begin_unbind_from_state_passthrough_ready,
        "state_unbind_finish_decision"_s <= "state_unbind_pending"_s + event<DetailUnbindFinishRuntime> / effect_attempt_release_source,
        "state_unbind_publish_ready"_s <= "state_unbind_finish_decision"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_release_succeeded] / effect_reset_window_after_release,
        "state_unbind_error_ready"_s <= "state_unbind_finish_decision"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_release_failed] / effect_mark_unbind_release_failed,
        "state_unbind_done_callback"_s <= "state_unbind_publish_ready"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_done_callback_present] / effect_publish_unbind_done,
        "state_unbound"_s <= "state_unbind_publish_ready"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_done_callback_absent] / effect_record_unbind_done_from_state_unbind_publish_ready,
        "state_unbound"_s <= "state_unbind_done_callback"_s + completion<DetailUnbindFinishRuntime> / effect_record_unbind_done_from_state_unbind_done_callback,
        "state_unbind_error_callback"_s <= "state_unbind_error_ready"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_error_callback_present] / effect_publish_unbind_error_from_state_unbind_error_ready,
        "state_ready"_s <= "state_unbind_error_ready"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_error_callback_absent_streaming] / effect_record_unbind_error_from_state_unbind_error_ready,
        "state_passthrough_ready"_s <= "state_unbind_error_ready"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_error_callback_absent_passthrough] / effect_record_unbind_error_from_state_unbind_error_ready,
        "state_ready"_s <= "state_unbind_error_callback"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_window_streaming] / effect_record_unbind_error_from_state_unbind_error_callback,
        "state_passthrough_ready"_s <= "state_unbind_error_callback"_s + completion<DetailUnbindFinishRuntime> [guard_unbind_window_passthrough] / effect_record_unbind_error_from_state_unbind_error_callback,
        "state_unbound"_s <= "state_unbound"_s + event<DetailUnbindRuntime> / effect_mark_unbind_not_bound,
        "state_unbound"_s <= "state_unbound"_s + event<DetailUnbindFinishRuntime> [guard_unbind_error_callback_present] / effect_publish_unbind_error_from_state_unbound,
        "state_unbound"_s <= "state_unbound"_s + event<DetailUnbindFinishRuntime> [guard_unbind_error_callback_absent] / effect_record_unbind_error_from_state_unbound,
        "state_unbound"_s <= "state_unbound"_s + unexpected_event<_> / effect_on_unexpected_from_state_unbound,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_passthrough_ready"_s <= "state_passthrough_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_passthrough_ready,
        "state_acquire_resolved"_s <= "state_acquire_resolved"_s + unexpected_event<_> / effect_on_unexpected_from_state_acquire_resolved,
        "state_passthrough_acquire_resolved"_s <= "state_passthrough_acquire_resolved"_s + unexpected_event<_> / effect_on_unexpected_from_state_passthrough_acquire_resolved,
        "state_acquire_pending"_s <= "state_acquire_pending"_s + unexpected_event<_> / effect_on_unexpected_from_state_acquire_pending,
        "state_passthrough_acquire_pending"_s <= "state_passthrough_acquire_pending"_s + unexpected_event<_> / effect_on_unexpected_from_state_passthrough_acquire_pending,
        "state_unbind_pending"_s <= "state_unbind_pending"_s + unexpected_event<_> / effect_on_unexpected_from_state_unbind_pending,
    }
}

/// Context for `ModelTensorWindow` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct ModelTensorWindowContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl ModelTensorWindowStateMachineContext for ModelTensorWindowContext {
    fn effect_activate_passthrough_and_publish(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_activate_passthrough_and_publish
        todo!(
            "TODO: port action `effect_activate_passthrough_and_publish` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_activate_passthrough_and_record(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_activate_passthrough_and_record
        todo!(
            "TODO: port action `effect_activate_passthrough_and_record` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_activate_streaming_and_publish(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_activate_streaming_and_publish
        todo!(
            "TODO: port action `effect_activate_streaming_and_publish` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_activate_streaming_and_record(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_activate_streaming_and_record
        todo!(
            "TODO: port action `effect_activate_streaming_and_record` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_advance_window(&mut self, _event: &DetailAcquirePublishRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_advance_window
        todo!(
            "TODO: port action `effect_advance_window` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_attempt_release_source(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_attempt_release_source
        todo!(
            "TODO: port action `effect_attempt_release_source` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_acquire_resolve(
        &mut self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_acquire_resolve
        todo!(
            "TODO: port action `effect_begin_acquire_resolve` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_bind(&mut self, _event: &DetailBindWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_bind
        todo!(
            "TODO: port action `effect_begin_bind` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_resolve_not_bound(
        &mut self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_resolve_not_bound
        todo!(
            "TODO: port action `effect_begin_resolve_not_bound` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_resolve_not_streaming(
        &mut self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_resolve_not_streaming
        todo!(
            "TODO: port action `effect_begin_resolve_not_streaming` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_unbind_from_state_passthrough_ready(
        &mut self,
        _event: &DetailUnbindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_unbind
        todo!(
            "TODO: port action `effect_begin_unbind` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_begin_unbind_from_state_ready(
        &mut self,
        _event: &DetailUnbindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_begin_unbind
        todo!(
            "TODO: port action `effect_begin_unbind` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_commit_slot_load_from_state_acquire_pending(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_commit_slot_load
        todo!(
            "TODO: port action `effect_commit_slot_load` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_commit_slot_load_from_state_acquire_resolved(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_commit_slot_load
        todo!(
            "TODO: port action `effect_commit_slot_load` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_commit_slot_load_from_state_ready(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_commit_slot_load
        todo!(
            "TODO: port action `effect_commit_slot_load` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_commit_slot_load_from_state_unbind_pending(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_commit_slot_load
        todo!(
            "TODO: port action `effect_commit_slot_load` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_map_source(&mut self, _event: &DetailBindWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_map_source
        todo!(
            "TODO: port action `effect_map_source` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_acquire_out_of_range(
        &mut self,
        _event: &DetailAcquireRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_acquire_out_of_range
        todo!(
            "TODO: port action `effect_mark_acquire_out_of_range` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_already_bound_and_publish_from_state_passthrough_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_already_bound_and_publish
        todo!(
            "TODO: port action `effect_mark_already_bound_and_publish` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_already_bound_and_publish_from_state_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_already_bound_and_publish
        todo!(
            "TODO: port action `effect_mark_already_bound_and_publish` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_bind_already_bound_from_state_passthrough_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_bind_already_bound
        todo!(
            "TODO: port action `effect_mark_bind_already_bound` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_bind_already_bound_from_state_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_bind_already_bound
        todo!(
            "TODO: port action `effect_mark_bind_already_bound` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_bind_invalid(&mut self, _event: &DetailBindWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_bind_invalid
        todo!(
            "TODO: port action `effect_mark_bind_invalid` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_resolve_out_of_range(
        &mut self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_resolve_out_of_range
        todo!(
            "TODO: port action `effect_mark_resolve_out_of_range` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_slot_copy_failed(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_slot_copy_failed
        todo!(
            "TODO: port action `effect_mark_slot_copy_failed` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_source_map_failed(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_source_map_failed
        todo!(
            "TODO: port action `effect_mark_source_map_failed` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_unbind_not_bound(&mut self, _event: &DetailUnbindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_unbind_not_bound
        todo!(
            "TODO: port action `effect_mark_unbind_not_bound` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_mark_unbind_release_failed(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_mark_unbind_release_failed
        todo!(
            "TODO: port action `effect_mark_unbind_release_failed` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_acquire_pending(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_acquire_resolved(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_passthrough_acquire_pending(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_passthrough_acquire_resolved(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_passthrough_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_unbind_pending(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_unbound(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_acquire_done(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_acquire_done
        todo!(
            "TODO: port action `effect_publish_acquire_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_acquire_error_from_state_acquire_error_ready(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_acquire_error
        todo!(
            "TODO: port action `effect_publish_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_acquire_error_from_state_passthrough_acquire_error_ready(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_acquire_error
        todo!(
            "TODO: port action `effect_publish_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_acquire_error_from_state_unbound(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_acquire_error
        todo!(
            "TODO: port action `effect_publish_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_bind_error(&mut self, _event: &DetailBindWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_bind_error
        todo!(
            "TODO: port action `effect_publish_bind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_unbind_done(&mut self, _event: &DetailUnbindFinishRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_unbind_done
        todo!(
            "TODO: port action `effect_publish_unbind_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_unbind_error_from_state_unbind_error_ready(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_unbind_error
        todo!(
            "TODO: port action `effect_publish_unbind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_publish_unbind_error_from_state_unbound(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_publish_unbind_error
        todo!(
            "TODO: port action `effect_publish_unbind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_done_from_state_acquire_done_callback(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_done
        todo!(
            "TODO: port action `effect_record_acquire_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_done_from_state_acquire_publish_ready(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_done
        todo!(
            "TODO: port action `effect_record_acquire_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_error_from_state_acquire_error_callback(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_error
        todo!(
            "TODO: port action `effect_record_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_error_from_state_acquire_error_ready(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_error
        todo!(
            "TODO: port action `effect_record_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_error_from_state_passthrough_acquire_error_callback(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_error
        todo!(
            "TODO: port action `effect_record_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_error_from_state_passthrough_acquire_error_ready(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_error
        todo!(
            "TODO: port action `effect_record_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_acquire_error_from_state_unbound(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_acquire_error
        todo!(
            "TODO: port action `effect_record_acquire_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_acquire_pending(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_acquire_resolved(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_passthrough_ready(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_ready(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_unbind_pending(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_stray_completion_from_state_unbound(
        &mut self,
        _event: &EmelEventCompletion,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_stray_completion
        todo!(
            "TODO: port action `effect_record_stray_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_unbind_done_from_state_unbind_done_callback(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_unbind_done
        todo!(
            "TODO: port action `effect_record_unbind_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_unbind_done_from_state_unbind_publish_ready(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_unbind_done
        todo!(
            "TODO: port action `effect_record_unbind_done` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_unbind_error_from_state_unbind_error_callback(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_unbind_error
        todo!(
            "TODO: port action `effect_record_unbind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_unbind_error_from_state_unbind_error_ready(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_unbind_error
        todo!(
            "TODO: port action `effect_record_unbind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_record_unbind_error_from_state_unbound(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_record_unbind_error
        todo!(
            "TODO: port action `effect_record_unbind_error` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_release_and_mark_budget_too_small(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_release_and_mark_budget_too_small
        todo!(
            "TODO: port action `effect_release_and_mark_budget_too_small` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_release_and_mark_slot_storage_too_small(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_release_and_mark_slot_storage_too_small
        todo!(
            "TODO: port action `effect_release_and_mark_slot_storage_too_small` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_release_and_mark_streaming_config_invalid(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_release_and_mark_streaming_config_invalid
        todo!(
            "TODO: port action `effect_release_and_mark_streaming_config_invalid` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_require_busy_slot(&mut self, _event: &DetailAcquireResolveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_require_busy_slot
        todo!(
            "TODO: port action `effect_require_busy_slot` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_require_layer_completion(&mut self, _event: &DetailAcquireRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_require_layer_completion
        todo!(
            "TODO: port action `effect_require_layer_completion` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_reset_rejected_window_from_state_bind_error_callback(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_reset_rejected_window
        todo!(
            "TODO: port action `effect_reset_rejected_window` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_reset_rejected_window_from_state_bind_error_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_reset_rejected_window
        todo!(
            "TODO: port action `effect_reset_rejected_window` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_reset_window_after_release(
        &mut self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_reset_window_after_release
        todo!(
            "TODO: port action `effect_reset_window_after_release` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_retain_source_for_release_retry_from_state_bind_error_callback(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_retain_source_for_release_retry
        todo!(
            "TODO: port action `effect_retain_source_for_release_retry` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_retain_source_for_release_retry_from_state_bind_error_ready(
        &mut self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_retain_source_for_release_retry
        todo!(
            "TODO: port action `effect_retain_source_for_release_retry` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_scan_layer_plan(&mut self, _event: &DetailBindWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_scan_layer_plan
        todo!(
            "TODO: port action `effect_scan_layer_plan` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_stage_acquire_result(
        &mut self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_stage_acquire_result
        todo!(
            "TODO: port action `effect_stage_acquire_result` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn effect_submit_and_require_layer_from_state_acquire_decision(
        &mut self,
        _event: &DetailAcquireRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/actions.hpp::effect_submit_and_require_layer
        todo!(
            "TODO: port action `effect_submit_and_require_layer` from emel.cpp/src/emel/model/tensor/window/actions.hpp"
        )
    }
    fn guard_acquire_copy_failed(&self, _event: &DetailAcquirePublishRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_copy_failed
        todo!(
            "TODO: port guard `guard_acquire_copy_failed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_done_callback_absent(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_done_callback_absent
        todo!(
            "TODO: port guard `guard_acquire_done_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_done_callback_present(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_done_callback_present
        todo!(
            "TODO: port guard `guard_acquire_done_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_error_callback_absent(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_error_callback_absent
        todo!(
            "TODO: port guard `guard_acquire_error_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_error_callback_present(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_error_callback_present
        todo!(
            "TODO: port guard `guard_acquire_error_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_error_pending(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_error_pending
        todo!(
            "TODO: port guard `guard_acquire_error_pending` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_layer_failed(&self, _event: &DetailAcquireRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_layer_failed
        todo!(
            "TODO: port guard `guard_acquire_layer_failed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_layer_loading(&self, _event: &DetailAcquireRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_layer_loading
        todo!(
            "TODO: port guard `guard_acquire_layer_loading` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_layer_out_of_range(&self, _event: &DetailAcquireRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_layer_out_of_range
        todo!(
            "TODO: port guard `guard_acquire_layer_out_of_range` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_layer_resident(&self, _event: &DetailAcquireRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_layer_resident
        todo!(
            "TODO: port guard `guard_acquire_layer_resident` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_layer_unscheduled(&self, _event: &DetailAcquireRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_layer_unscheduled
        todo!(
            "TODO: port guard `guard_acquire_layer_unscheduled` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_acquire_result_ready(&self, _event: &DetailAcquirePublishRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_acquire_result_ready
        todo!(
            "TODO: port guard `guard_acquire_result_ready` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_budget_too_small(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_budget_too_small
        todo!(
            "TODO: port guard `guard_bind_budget_too_small` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_callback_absent(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_callback_absent
        todo!(
            "TODO: port guard `guard_bind_error_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_callback_absent_retained(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_callback_absent_retained
        todo!(
            "TODO: port guard `guard_bind_error_callback_absent_retained` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_callback_absent_unbound(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_callback_absent_unbound
        todo!(
            "TODO: port guard `guard_bind_error_callback_absent_unbound` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_callback_present(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_callback_present
        todo!(
            "TODO: port guard `guard_bind_error_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_exit_retained(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_exit_retained
        todo!(
            "TODO: port guard `guard_bind_error_exit_retained` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_error_exit_unbound(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_error_exit_unbound
        todo!(
            "TODO: port guard `guard_bind_error_exit_unbound` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_fits_budget_callback_absent(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_fits_budget_callback_absent
        todo!(
            "TODO: port guard `guard_bind_fits_budget_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_fits_budget_callback_present(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_fits_budget_callback_present
        todo!(
            "TODO: port guard `guard_bind_fits_budget_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_request_invalid(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_request_invalid
        todo!(
            "TODO: port guard `guard_bind_request_invalid` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_request_valid(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_request_valid
        todo!(
            "TODO: port guard `guard_bind_request_valid` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_requires_streaming_callback_absent(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_requires_streaming_callback_absent
        todo!(
            "TODO: port guard `guard_bind_requires_streaming_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_requires_streaming_callback_present(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_requires_streaming_callback_present
        todo!(
            "TODO: port guard `guard_bind_requires_streaming_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_slot_storage_too_small(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_slot_storage_too_small
        todo!(
            "TODO: port guard `guard_bind_slot_storage_too_small` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_bind_streaming_config_invalid(
        &self,
        _event: &DetailBindWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_bind_streaming_config_invalid
        todo!(
            "TODO: port guard `guard_bind_streaming_config_invalid` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_completion_slot_armed(&self, _event: &EmelEventCompletion) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_completion_slot_armed
        todo!(
            "TODO: port guard `guard_completion_slot_armed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_completion_slot_stray(&self, _event: &EmelEventCompletion) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_completion_slot_stray
        todo!(
            "TODO: port guard `guard_completion_slot_stray` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_prefetch_ahead_needed(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_prefetch_ahead_needed
        todo!(
            "TODO: port guard `guard_prefetch_ahead_needed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_prefetch_ahead_not_needed(
        &self,
        _event: &DetailAcquirePublishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_prefetch_ahead_not_needed
        todo!(
            "TODO: port guard `guard_prefetch_ahead_not_needed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_resolve_layer_out_of_range(
        &self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_resolve_layer_out_of_range
        todo!(
            "TODO: port guard `guard_resolve_layer_out_of_range` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_resolve_slot_busy_other_layer(
        &self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_resolve_slot_busy_other_layer
        todo!(
            "TODO: port guard `guard_resolve_slot_busy_other_layer` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_resolve_slot_ready_for_target(
        &self,
        _event: &DetailAcquireResolveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_resolve_slot_ready_for_target
        todo!(
            "TODO: port guard `guard_resolve_slot_ready_for_target` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_source_map_failed(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_source_map_failed
        todo!(
            "TODO: port guard `guard_source_map_failed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_source_map_succeeded(&self, _event: &DetailBindWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_source_map_succeeded
        todo!(
            "TODO: port guard `guard_source_map_succeeded` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_done_callback_absent(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_done_callback_absent
        todo!(
            "TODO: port guard `guard_unbind_done_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_done_callback_present(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_done_callback_present
        todo!(
            "TODO: port guard `guard_unbind_done_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_error_callback_absent(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_error_callback_absent
        todo!(
            "TODO: port guard `guard_unbind_error_callback_absent` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_error_callback_absent_passthrough(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_error_callback_absent_passthrough
        todo!(
            "TODO: port guard `guard_unbind_error_callback_absent_passthrough` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_error_callback_absent_streaming(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_error_callback_absent_streaming
        todo!(
            "TODO: port guard `guard_unbind_error_callback_absent_streaming` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_error_callback_present(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_error_callback_present
        todo!(
            "TODO: port guard `guard_unbind_error_callback_present` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_release_failed(&self, _event: &DetailUnbindFinishRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_release_failed
        todo!(
            "TODO: port guard `guard_unbind_release_failed` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_release_succeeded(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_release_succeeded
        todo!(
            "TODO: port guard `guard_unbind_release_succeeded` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_window_passthrough(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_window_passthrough
        todo!(
            "TODO: port guard `guard_unbind_window_passthrough` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
    fn guard_unbind_window_streaming(
        &self,
        _event: &DetailUnbindFinishRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/tensor/window/guards.hpp::guard_unbind_window_streaming
        todo!(
            "TODO: port guard `guard_unbind_window_streaming` from emel.cpp/src/emel/model/tensor/window/guards.hpp"
        )
    }
}
