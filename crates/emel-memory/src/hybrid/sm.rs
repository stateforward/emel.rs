//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::enum_variant_names,
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

// --- machine MemoryHybrid from emel.cpp/src/emel/memory/hybrid/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAllocateSequenceRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAllocateSlotsRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBranchSequenceRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventCaptureViewRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventFreeSequenceRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReserveRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRollbackSlotsRuntime;

sml! {
    MemoryHybrid {
        "reserve_kv"_s <= *"ready"_s + event<EventReserveRuntime> / begin_reserve,
        "reserve_kv_decision"_s <= "reserve_kv"_s + completion<EventReserveRuntime> / exec_reserve_kv,
        "reserve_recurrent"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime> [kv_accepted_event_reserve_runtime],
        "errored"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime> [kv_rejected_with_error_event_reserve_runtime] / mark_error_from_kv_event_reserve_runtime,
        "errored"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime> [kv_rejected_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "reserve_recurrent_decision"_s <= "reserve_recurrent"_s + completion<EventReserveRuntime> / exec_reserve_recurrent,
        "done"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime> [recurrent_accepted_event_reserve_runtime],
        "errored"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime> [recurrent_rejected_with_error_event_reserve_runtime] / mark_error_from_recurrent_event_reserve_runtime,
        "errored"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime> [recurrent_rejected_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "allocate_sequence_kv"_s <= "ready"_s + event<EventAllocateSequenceRuntime> / begin_allocate_sequence,
        "allocate_sequence_kv_decision"_s <= "allocate_sequence_kv"_s + completion<EventAllocateSequenceRuntime> / exec_allocate_sequence_kv,
        "allocate_sequence_recurrent"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime> [kv_accepted_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime> [kv_rejected_with_error_event_allocate_sequence_runtime] / mark_error_from_kv_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime> [kv_rejected_without_error_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "allocate_sequence_recurrent_decision"_s <= "allocate_sequence_recurrent"_s + completion<EventAllocateSequenceRuntime> / exec_allocate_sequence_recurrent,
        "done"_s <= "allocate_sequence_recurrent_decision"_s + completion<EventAllocateSequenceRuntime> [recurrent_accepted_event_allocate_sequence_runtime],
        "allocate_sequence_rollback_kv"_s <= "allocate_sequence_recurrent_decision"_s + completion<EventAllocateSequenceRuntime> [recurrent_rejected_any_event_allocate_sequence_runtime] / exec_allocate_sequence_rollback_kv,
        "allocate_sequence_rollback_result_decision"_s <= "allocate_sequence_rollback_kv"_s + completion<EventAllocateSequenceRuntime>,
        "allocate_sequence_recurrent_error_decision"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime> [rollback_accepted_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime> [rollback_rejected_with_error_event_allocate_sequence_runtime] / mark_error_from_rollback_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime> [rollback_rejected_without_error_event_allocate_sequence_runtime] / mark_internal_error_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime> / mark_internal_error_event_allocate_sequence_runtime,
        "out_of_memory"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime> [recurrent_rejected_out_of_memory_event_allocate_sequence_runtime] / mark_out_of_memory_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime> [recurrent_rejected_backend_or_none_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime> [recurrent_rejected_non_backend_error_event_allocate_sequence_runtime] / mark_error_from_recurrent_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime> / mark_internal_error_event_allocate_sequence_runtime,
        "allocate_slots_kv"_s <= "ready"_s + event<EventAllocateSlotsRuntime> / begin_allocate_slots,
        "allocate_slots_kv_decision"_s <= "allocate_slots_kv"_s + completion<EventAllocateSlotsRuntime> [guard_owned_kv_cache_event_allocate_slots_runtime] / effect_allocate_slots_owned_kv,
        "allocate_slots_kv_decision"_s <= "allocate_slots_kv"_s + completion<EventAllocateSlotsRuntime> [guard_bound_kv_cache_event_allocate_slots_runtime] / effect_allocate_slots_bound_kv,
        "allocate_slots_recurrent"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime> [kv_accepted_event_allocate_slots_runtime],
        "out_of_memory"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime> [kv_rejected_out_of_memory_event_allocate_slots_runtime] / mark_out_of_memory_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime> [kv_rejected_backend_or_none_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime> [kv_rejected_non_backend_error_event_allocate_slots_runtime] / mark_error_from_kv_event_allocate_slots_runtime,
        "allocate_slots_recurrent_decision"_s <= "allocate_slots_recurrent"_s + completion<EventAllocateSlotsRuntime> / exec_allocate_slots_recurrent,
        "done"_s <= "allocate_slots_recurrent_decision"_s + completion<EventAllocateSlotsRuntime> [recurrent_accepted_event_allocate_slots_runtime],
        "allocate_slots_rollback_kv"_s <= "allocate_slots_recurrent_decision"_s + completion<EventAllocateSlotsRuntime> [recurrent_rejected_any_event_allocate_slots_runtime] / exec_allocate_slots_rollback_kv,
        "allocate_slots_rollback_result_decision"_s <= "allocate_slots_rollback_kv"_s + completion<EventAllocateSlotsRuntime>,
        "allocate_slots_recurrent_error_decision"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime> [rollback_accepted_event_allocate_slots_runtime],
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime> [rollback_rejected_with_error_event_allocate_slots_runtime] / mark_error_from_rollback_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime> [rollback_rejected_without_error_event_allocate_slots_runtime] / mark_internal_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime> / mark_internal_error_event_allocate_slots_runtime,
        "out_of_memory"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime> [recurrent_rejected_out_of_memory_event_allocate_slots_runtime] / mark_out_of_memory_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime> [recurrent_rejected_backend_or_none_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime> [recurrent_rejected_non_backend_error_event_allocate_slots_runtime] / mark_error_from_recurrent_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime> / mark_internal_error_event_allocate_slots_runtime,
        "branch_sequence_kv"_s <= "ready"_s + event<EventBranchSequenceRuntime> / begin_branch_sequence,
        "branch_sequence_kv_decision"_s <= "branch_sequence_kv"_s + completion<EventBranchSequenceRuntime> / exec_branch_sequence_kv,
        "branch_sequence_recurrent"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime> [kv_accepted_event_branch_sequence_runtime],
        "out_of_memory"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime> [kv_rejected_out_of_memory_event_branch_sequence_runtime] / mark_out_of_memory_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime> [kv_rejected_backend_or_none_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime> [kv_rejected_non_backend_error_event_branch_sequence_runtime] / mark_error_from_kv_event_branch_sequence_runtime,
        "branch_sequence_recurrent_decision"_s <= "branch_sequence_recurrent"_s + completion<EventBranchSequenceRuntime> / exec_branch_sequence_recurrent,
        "done"_s <= "branch_sequence_recurrent_decision"_s + completion<EventBranchSequenceRuntime> [recurrent_accepted_event_branch_sequence_runtime],
        "branch_sequence_rollback_kv"_s <= "branch_sequence_recurrent_decision"_s + completion<EventBranchSequenceRuntime> [recurrent_rejected_any_event_branch_sequence_runtime] / exec_branch_sequence_rollback_kv,
        "branch_sequence_rollback_result_decision"_s <= "branch_sequence_rollback_kv"_s + completion<EventBranchSequenceRuntime>,
        "branch_sequence_recurrent_error_decision"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime> [rollback_accepted_event_branch_sequence_runtime],
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime> [rollback_rejected_with_error_event_branch_sequence_runtime] / mark_error_from_rollback_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime> [rollback_rejected_without_error_event_branch_sequence_runtime] / mark_internal_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime> / mark_internal_error_event_branch_sequence_runtime,
        "out_of_memory"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime> [recurrent_rejected_out_of_memory_event_branch_sequence_runtime] / mark_out_of_memory_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime> [recurrent_rejected_backend_or_none_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime> [recurrent_rejected_non_backend_error_event_branch_sequence_runtime] / mark_error_from_recurrent_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime> / mark_internal_error_event_branch_sequence_runtime,
        "free_sequence_kv"_s <= "ready"_s + event<EventFreeSequenceRuntime> / begin_free_sequence,
        "free_sequence_kv_decision"_s <= "free_sequence_kv"_s + completion<EventFreeSequenceRuntime> / exec_free_sequence_kv,
        "free_sequence_recurrent"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime> [kv_accepted_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime> [kv_rejected_with_error_event_free_sequence_runtime] / mark_error_from_kv_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime> [kv_rejected_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "free_sequence_recurrent_decision"_s <= "free_sequence_recurrent"_s + completion<EventFreeSequenceRuntime> / exec_free_sequence_recurrent,
        "done"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime> [recurrent_accepted_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime> [recurrent_rejected_with_error_event_free_sequence_runtime] / mark_error_from_recurrent_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime> [recurrent_rejected_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "rollback_slots_kv"_s <= "ready"_s + event<EventRollbackSlotsRuntime> / begin_rollback_slots,
        "rollback_slots_kv_decision"_s <= "rollback_slots_kv"_s + completion<EventRollbackSlotsRuntime> / exec_rollback_slots_kv,
        "rollback_slots_recurrent"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime> [kv_accepted_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime> [kv_rejected_with_error_event_rollback_slots_runtime] / mark_error_from_kv_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime> [kv_rejected_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "rollback_slots_recurrent_decision"_s <= "rollback_slots_recurrent"_s + completion<EventRollbackSlotsRuntime> / exec_rollback_slots_recurrent,
        "done"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime> [recurrent_accepted_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime> [recurrent_rejected_with_error_event_rollback_slots_runtime] / mark_error_from_recurrent_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime> [recurrent_rejected_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "capture_request_decision"_s <= "ready"_s + event<EventCaptureViewRuntime> / begin_capture_view,
        "capture_kv"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime> [capture_request_valid],
        "errored"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime> [capture_request_invalid] / mark_invalid_request,
        "capture_kv_decision"_s <= "capture_kv"_s + completion<EventCaptureViewRuntime> [guard_owned_kv_cache_event_capture_view_runtime] / effect_capture_owned_kv,
        "capture_kv_decision"_s <= "capture_kv"_s + completion<EventCaptureViewRuntime> [guard_bound_kv_cache_event_capture_view_runtime] / effect_capture_bound_kv,
        "capture_recurrent"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime> [kv_accepted_event_capture_view_runtime],
        "errored"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime> [kv_rejected_with_error_event_capture_view_runtime] / mark_error_from_kv_event_capture_view_runtime,
        "errored"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime> [kv_rejected_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "capture_recurrent_decision"_s <= "capture_recurrent"_s + completion<EventCaptureViewRuntime> / exec_capture_recurrent,
        "capture_merge"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime> [recurrent_accepted_event_capture_view_runtime],
        "errored"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime> [recurrent_rejected_with_error_event_capture_view_runtime] / mark_error_from_recurrent_event_capture_view_runtime,
        "errored"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime> [recurrent_rejected_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "done"_s <= "capture_merge"_s + completion<EventCaptureViewRuntime> / merge_capture_snapshots,
        "ready"_s <= "done"_s + completion<EventReserveRuntime> / publish_done_event_reserve_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventReserveRuntime> / publish_error_event_reserve_runtime,
        "ready"_s <= "errored"_s + completion<EventReserveRuntime> / publish_error_event_reserve_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSequenceRuntime> / publish_done_event_allocate_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventAllocateSequenceRuntime> / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSequenceRuntime> / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSlotsRuntime> / publish_done_event_allocate_slots_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventAllocateSlotsRuntime> / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSlotsRuntime> / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "done"_s + completion<EventBranchSequenceRuntime> / publish_done_event_branch_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventBranchSequenceRuntime> / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventBranchSequenceRuntime> / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventFreeSequenceRuntime> / publish_done_event_free_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventFreeSequenceRuntime> / publish_error_event_free_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventFreeSequenceRuntime> / publish_error_event_free_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventRollbackSlotsRuntime> / publish_done_event_rollback_slots_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventRollbackSlotsRuntime> / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventRollbackSlotsRuntime> / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "done"_s + completion<EventCaptureViewRuntime> / publish_done_event_capture_view_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventCaptureViewRuntime> / publish_error_event_capture_view_runtime,
        "ready"_s <= "errored"_s + completion<EventCaptureViewRuntime> / publish_error_event_capture_view_runtime,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "reserve_kv"_s + unexpected_event<_> / on_unexpected_from_reserve_kv,
        "ready"_s <= "reserve_kv_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_kv_decision,
        "ready"_s <= "reserve_recurrent"_s + unexpected_event<_> / on_unexpected_from_reserve_recurrent,
        "ready"_s <= "reserve_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_recurrent_decision,
        "ready"_s <= "allocate_sequence_kv"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_kv,
        "ready"_s <= "allocate_sequence_kv_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_kv_decision,
        "ready"_s <= "allocate_sequence_recurrent"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_recurrent,
        "ready"_s <= "allocate_sequence_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_recurrent_decision,
        "ready"_s <= "allocate_sequence_rollback_kv"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_rollback_kv,
        "ready"_s <= "allocate_sequence_rollback_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_rollback_result_decision,
        "ready"_s <= "allocate_sequence_recurrent_error_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_recurrent_error_decision,
        "ready"_s <= "allocate_slots_kv"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_kv,
        "ready"_s <= "allocate_slots_kv_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_kv_decision,
        "ready"_s <= "allocate_slots_recurrent"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_recurrent,
        "ready"_s <= "allocate_slots_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_recurrent_decision,
        "ready"_s <= "allocate_slots_rollback_kv"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_rollback_kv,
        "ready"_s <= "allocate_slots_rollback_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_rollback_result_decision,
        "ready"_s <= "allocate_slots_recurrent_error_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_recurrent_error_decision,
        "ready"_s <= "branch_sequence_kv"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_kv,
        "ready"_s <= "branch_sequence_kv_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_kv_decision,
        "ready"_s <= "branch_sequence_recurrent"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_recurrent,
        "ready"_s <= "branch_sequence_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_recurrent_decision,
        "ready"_s <= "branch_sequence_rollback_kv"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_rollback_kv,
        "ready"_s <= "branch_sequence_rollback_result_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_rollback_result_decision,
        "ready"_s <= "branch_sequence_recurrent_error_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_recurrent_error_decision,
        "ready"_s <= "free_sequence_kv"_s + unexpected_event<_> / on_unexpected_from_free_sequence_kv,
        "ready"_s <= "free_sequence_kv_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_kv_decision,
        "ready"_s <= "free_sequence_recurrent"_s + unexpected_event<_> / on_unexpected_from_free_sequence_recurrent,
        "ready"_s <= "free_sequence_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_recurrent_decision,
        "ready"_s <= "rollback_slots_kv"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_kv,
        "ready"_s <= "rollback_slots_kv_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_kv_decision,
        "ready"_s <= "rollback_slots_recurrent"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_recurrent,
        "ready"_s <= "rollback_slots_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_recurrent_decision,
        "ready"_s <= "capture_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_request_decision,
        "ready"_s <= "capture_kv"_s + unexpected_event<_> / on_unexpected_from_capture_kv,
        "ready"_s <= "capture_kv_decision"_s + unexpected_event<_> / on_unexpected_from_capture_kv_decision,
        "ready"_s <= "capture_recurrent"_s + unexpected_event<_> / on_unexpected_from_capture_recurrent,
        "ready"_s <= "capture_recurrent_decision"_s + unexpected_event<_> / on_unexpected_from_capture_recurrent_decision,
        "ready"_s <= "capture_merge"_s + unexpected_event<_> / on_unexpected_from_capture_merge,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "out_of_memory"_s + unexpected_event<_> / on_unexpected_from_out_of_memory,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `MemoryHybrid` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct MemoryHybridContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl MemoryHybridStateMachineContext for MemoryHybridContext {
    fn begin_allocate_sequence(&mut self, _event: &EventAllocateSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_allocate_sequence
        todo!(
            "TODO: port action `begin_allocate_sequence` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn begin_allocate_slots(&mut self, _event: &EventAllocateSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_allocate_slots
        todo!(
            "TODO: port action `begin_allocate_slots` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn begin_branch_sequence(&mut self, _event: &EventBranchSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_branch_sequence
        todo!(
            "TODO: port action `begin_branch_sequence` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn begin_capture_view(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_capture_view
        todo!(
            "TODO: port action `begin_capture_view` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn begin_free_sequence(&mut self, _event: &EventFreeSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_free_sequence
        todo!(
            "TODO: port action `begin_free_sequence` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn begin_reserve(&mut self, _event: &EventReserveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_reserve
        todo!("TODO: port action `begin_reserve` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn begin_rollback_slots(&mut self, _event: &EventRollbackSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::begin_rollback_slots
        todo!(
            "TODO: port action `begin_rollback_slots` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn capture_request_invalid(&self, _event: &EventCaptureViewRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::capture_request_invalid
        todo!(
            "TODO: port guard `capture_request_invalid` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn capture_request_valid(&self, _event: &EventCaptureViewRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::capture_request_valid
        todo!(
            "TODO: port guard `capture_request_valid` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn effect_allocate_slots_bound_kv(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::effect_allocate_slots_bound_kv
        todo!(
            "TODO: port action `effect_allocate_slots_bound_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn effect_allocate_slots_owned_kv(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::effect_allocate_slots_owned_kv
        todo!(
            "TODO: port action `effect_allocate_slots_owned_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn effect_capture_bound_kv(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::effect_capture_bound_kv
        todo!(
            "TODO: port action `effect_capture_bound_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn effect_capture_owned_kv(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::effect_capture_owned_kv
        todo!(
            "TODO: port action `effect_capture_owned_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_allocate_sequence_kv(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_allocate_sequence_kv
        todo!(
            "TODO: port action `exec_allocate_sequence_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_allocate_sequence_recurrent(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_allocate_sequence_recurrent
        todo!(
            "TODO: port action `exec_allocate_sequence_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_allocate_sequence_rollback_kv(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_allocate_sequence_rollback_kv
        todo!(
            "TODO: port action `exec_allocate_sequence_rollback_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_allocate_slots_recurrent(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_allocate_slots_recurrent
        todo!(
            "TODO: port action `exec_allocate_slots_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_allocate_slots_rollback_kv(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_allocate_slots_rollback_kv
        todo!(
            "TODO: port action `exec_allocate_slots_rollback_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_branch_sequence_kv(&mut self, _event: &EventBranchSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_branch_sequence_kv
        todo!(
            "TODO: port action `exec_branch_sequence_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_branch_sequence_recurrent(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_branch_sequence_recurrent
        todo!(
            "TODO: port action `exec_branch_sequence_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_branch_sequence_rollback_kv(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_branch_sequence_rollback_kv
        todo!(
            "TODO: port action `exec_branch_sequence_rollback_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_capture_recurrent(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_capture_recurrent
        todo!(
            "TODO: port action `exec_capture_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_free_sequence_kv(&mut self, _event: &EventFreeSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_free_sequence_kv
        todo!(
            "TODO: port action `exec_free_sequence_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_free_sequence_recurrent(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_free_sequence_recurrent
        todo!(
            "TODO: port action `exec_free_sequence_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_reserve_kv(&mut self, _event: &EventReserveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_reserve_kv
        todo!(
            "TODO: port action `exec_reserve_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_reserve_recurrent(&mut self, _event: &EventReserveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_reserve_recurrent
        todo!(
            "TODO: port action `exec_reserve_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_rollback_slots_kv(&mut self, _event: &EventRollbackSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_rollback_slots_kv
        todo!(
            "TODO: port action `exec_rollback_slots_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn exec_rollback_slots_recurrent(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::exec_rollback_slots_recurrent
        todo!(
            "TODO: port action `exec_rollback_slots_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn guard_bound_kv_cache_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::guard_bound_kv_cache
        todo!(
            "TODO: port guard `guard_bound_kv_cache` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn guard_bound_kv_cache_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::guard_bound_kv_cache
        todo!(
            "TODO: port guard `guard_bound_kv_cache` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn guard_owned_kv_cache_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::guard_owned_kv_cache
        todo!(
            "TODO: port guard `guard_owned_kv_cache` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn guard_owned_kv_cache_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::guard_owned_kv_cache
        todo!(
            "TODO: port guard `guard_owned_kv_cache` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_accepted_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_reserve_runtime(&self, _event: &EventReserveRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_accepted_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_accepted
        todo!("TODO: port guard `kv_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp")
    }
    fn kv_rejected_backend_or_none_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_backend_or_none
        todo!(
            "TODO: port guard `kv_rejected_backend_or_none` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_backend_or_none_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_backend_or_none
        todo!(
            "TODO: port guard `kv_rejected_backend_or_none` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_non_backend_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_non_backend_error
        todo!(
            "TODO: port guard `kv_rejected_non_backend_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_non_backend_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_non_backend_error
        todo!(
            "TODO: port guard `kv_rejected_non_backend_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_out_of_memory_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_out_of_memory
        todo!(
            "TODO: port guard `kv_rejected_out_of_memory` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_out_of_memory_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_out_of_memory
        todo!(
            "TODO: port guard `kv_rejected_out_of_memory` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_with_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_with_error
        todo!(
            "TODO: port guard `kv_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_with_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_with_error
        todo!(
            "TODO: port guard `kv_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_with_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_with_error
        todo!(
            "TODO: port guard `kv_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_with_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_with_error
        todo!(
            "TODO: port guard `kv_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_with_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_with_error
        todo!(
            "TODO: port guard `kv_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_without_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_without_error
        todo!(
            "TODO: port guard `kv_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_without_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_without_error
        todo!(
            "TODO: port guard `kv_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_without_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_without_error
        todo!(
            "TODO: port guard `kv_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_without_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_without_error
        todo!(
            "TODO: port guard `kv_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn kv_rejected_without_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::kv_rejected_without_error
        todo!(
            "TODO: port guard `kv_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn mark_backend_error_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_backend_error_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_kv_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_kv
        todo!(
            "TODO: port action `mark_error_from_kv` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_recurrent_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_recurrent
        todo!(
            "TODO: port action `mark_error_from_recurrent` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_rollback_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_rollback
        todo!(
            "TODO: port action `mark_error_from_rollback` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_rollback_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_rollback
        todo!(
            "TODO: port action `mark_error_from_rollback` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_error_from_rollback_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_error_from_rollback
        todo!(
            "TODO: port action `mark_error_from_rollback` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_internal_error_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_internal_error_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_internal_error_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_invalid_request(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_out_of_memory_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_out_of_memory
        todo!(
            "TODO: port action `mark_out_of_memory` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_out_of_memory_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_out_of_memory
        todo!(
            "TODO: port action `mark_out_of_memory` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn mark_out_of_memory_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::mark_out_of_memory
        todo!(
            "TODO: port action `mark_out_of_memory` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn merge_capture_snapshots(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::merge_capture_snapshots
        todo!(
            "TODO: port action `merge_capture_snapshots` from emel.cpp/src/emel/memory/hybrid/actions.hpp"
        )
    }
    fn on_unexpected_from_allocate_sequence_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_recurrent_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_rollback_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_rollback_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_recurrent_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_rollback_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_rollback_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_recurrent_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_rollback_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_rollback_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_merge(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_capture_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_out_of_memory(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_reserve_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_reserve_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_reserve_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_reserve_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_kv(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_kv_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_recurrent(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_recurrent_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_done_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn publish_error_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/hybrid/actions.hpp")
    }
    fn recurrent_accepted_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_accepted_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_accepted
        todo!(
            "TODO: port guard `recurrent_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_any_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_any
        todo!(
            "TODO: port guard `recurrent_rejected_any` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_any_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_any
        todo!(
            "TODO: port guard `recurrent_rejected_any` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_any_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_any
        todo!(
            "TODO: port guard `recurrent_rejected_any` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_backend_or_none_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_backend_or_none
        todo!(
            "TODO: port guard `recurrent_rejected_backend_or_none` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_backend_or_none_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_backend_or_none
        todo!(
            "TODO: port guard `recurrent_rejected_backend_or_none` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_backend_or_none_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_backend_or_none
        todo!(
            "TODO: port guard `recurrent_rejected_backend_or_none` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_non_backend_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_non_backend_error
        todo!(
            "TODO: port guard `recurrent_rejected_non_backend_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_non_backend_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_non_backend_error
        todo!(
            "TODO: port guard `recurrent_rejected_non_backend_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_non_backend_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_non_backend_error
        todo!(
            "TODO: port guard `recurrent_rejected_non_backend_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_out_of_memory_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_out_of_memory
        todo!(
            "TODO: port guard `recurrent_rejected_out_of_memory` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_out_of_memory_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_out_of_memory
        todo!(
            "TODO: port guard `recurrent_rejected_out_of_memory` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_out_of_memory_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_out_of_memory
        todo!(
            "TODO: port guard `recurrent_rejected_out_of_memory` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_with_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_with_error
        todo!(
            "TODO: port guard `recurrent_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_with_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_with_error
        todo!(
            "TODO: port guard `recurrent_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_with_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_with_error
        todo!(
            "TODO: port guard `recurrent_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_with_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_with_error
        todo!(
            "TODO: port guard `recurrent_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_without_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_without_error
        todo!(
            "TODO: port guard `recurrent_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_without_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_without_error
        todo!(
            "TODO: port guard `recurrent_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_without_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_without_error
        todo!(
            "TODO: port guard `recurrent_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn recurrent_rejected_without_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::recurrent_rejected_without_error
        todo!(
            "TODO: port guard `recurrent_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_accepted_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_accepted
        todo!(
            "TODO: port guard `rollback_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_accepted_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_accepted
        todo!(
            "TODO: port guard `rollback_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_accepted_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_accepted
        todo!(
            "TODO: port guard `rollback_accepted` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_with_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_with_error
        todo!(
            "TODO: port guard `rollback_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_with_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_with_error
        todo!(
            "TODO: port guard `rollback_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_with_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_with_error
        todo!(
            "TODO: port guard `rollback_rejected_with_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_without_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_without_error
        todo!(
            "TODO: port guard `rollback_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_without_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_without_error
        todo!(
            "TODO: port guard `rollback_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
    fn rollback_rejected_without_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/hybrid/guards.hpp::rollback_rejected_without_error
        todo!(
            "TODO: port guard `rollback_rejected_without_error` from emel.cpp/src/emel/memory/hybrid/guards.hpp"
        )
    }
}
