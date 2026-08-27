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

// --- machine MemoryKv from emel.cpp/src/emel/memory/kv/sm.hpp ---
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
    MemoryKv {
        "reserve_request_decision"_s <= *"ready"_s + event<EventReserveRuntime> / begin_reserve,
        "reserve_exec"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime> [reserve_request_valid],
        "errored"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime> [reserve_request_invalid] / mark_invalid_request_event_reserve_runtime,
        "reserve_result_decision"_s <= "reserve_exec"_s + completion<EventReserveRuntime> / exec_reserve,
        "done"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime> [operation_succeeded_event_reserve_runtime],
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime> [operation_failed_with_error_event_reserve_runtime] / mark_error_from_operation_event_reserve_runtime,
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime> [operation_failed_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "allocate_sequence_request_decision"_s <= "ready"_s + event<EventAllocateSequenceRuntime> / begin_allocate_sequence,
        "allocate_sequence_exec"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime> [allocate_sequence_request_valid],
        "errored"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime> [allocate_sequence_request_invalid] / mark_invalid_request_event_allocate_sequence_runtime,
        "allocate_sequence_result_decision"_s <= "allocate_sequence_exec"_s + completion<EventAllocateSequenceRuntime> / exec_allocate_sequence,
        "done"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime> [operation_succeeded_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime> [operation_failed_with_error_event_allocate_sequence_runtime] / mark_error_from_operation_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime> [operation_failed_without_error_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "allocate_slots_request_decision"_s <= "ready"_s + event<EventAllocateSlotsRuntime> / begin_allocate_slots,
        "allocate_slots_request_shape_decision"_s <= "allocate_slots_request_decision"_s + completion<EventAllocateSlotsRuntime>,
        "allocate_slots_request_length_decision"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_shape_valid],
        "errored"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_shape_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_request_block_layout_decision"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_length_valid],
        "errored"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_length_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_request_capacity_decision"_s <= "allocate_slots_request_block_layout_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_block_layout_valid],
        "errored"_s <= "allocate_slots_request_block_layout_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_block_layout_invalid] / mark_backend_error_event_allocate_slots_runtime,
        "state_allocate_slots_tail_decision"_s <= "allocate_slots_request_capacity_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_capacity_valid],
        "errored"_s <= "allocate_slots_request_capacity_decision"_s + completion<EventAllocateSlotsRuntime> [allocate_slots_request_capacity_invalid] / mark_out_of_memory,
        "state_allocate_slots_tail_copy_exec"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime> [guard_allocate_slots_shared_tail_copy_ready],
        "errored"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime> [guard_allocate_slots_shared_tail_copy_missing] / mark_invalid_request_event_allocate_slots_runtime,
        "state_allocate_slots_direct_exec"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime> [guard_allocate_slots_shared_tail_split_not_required],
        "state_allocate_slots_tail_copy_result_decision"_s <= "state_allocate_slots_tail_copy_exec"_s + completion<EventAllocateSlotsRuntime> / effect_copy_shared_tail_block,
        "state_allocate_slots_tail_split_exec"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_succeeded_event_allocate_slots_runtime],
        "errored"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_failed_with_error_event_allocate_slots_runtime] / mark_error_from_operation_event_allocate_slots_runtime,
        "errored"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_failed_without_error_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "allocate_slots_result_decision"_s <= "state_allocate_slots_tail_split_exec"_s + completion<EventAllocateSlotsRuntime> / effect_allocate_slots_with_tail_split,
        "allocate_slots_result_decision"_s <= "state_allocate_slots_direct_exec"_s + completion<EventAllocateSlotsRuntime> / effect_allocate_slots_without_tail_split,
        "done"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_succeeded_event_allocate_slots_runtime],
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_failed_with_error_event_allocate_slots_runtime] / mark_error_from_operation_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime> [operation_failed_without_error_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "branch_sequence_request_decision"_s <= "ready"_s + event<EventBranchSequenceRuntime> / begin_branch_sequence,
        "branch_sequence_exec"_s <= "branch_sequence_request_decision"_s + completion<EventBranchSequenceRuntime> [branch_sequence_request_valid],
        "errored"_s <= "branch_sequence_request_decision"_s + completion<EventBranchSequenceRuntime> [branch_sequence_request_invalid] / mark_invalid_request_event_branch_sequence_runtime,
        "branch_sequence_result_decision"_s <= "branch_sequence_exec"_s + completion<EventBranchSequenceRuntime> / exec_branch_sequence,
        "done"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime> [operation_succeeded_event_branch_sequence_runtime],
        "errored"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime> [operation_failed_with_error_event_branch_sequence_runtime] / mark_error_from_operation_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime> [operation_failed_without_error_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "free_sequence_request_decision"_s <= "ready"_s + event<EventFreeSequenceRuntime> / begin_free_sequence,
        "free_sequence_exec"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime> [free_sequence_request_valid],
        "errored"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime> [free_sequence_request_invalid] / mark_invalid_request_event_free_sequence_runtime,
        "free_sequence_result_decision"_s <= "free_sequence_exec"_s + completion<EventFreeSequenceRuntime> / exec_free_sequence,
        "done"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime> [operation_succeeded_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime> [operation_failed_with_error_event_free_sequence_runtime] / mark_error_from_operation_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime> [operation_failed_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "rollback_slots_request_decision"_s <= "ready"_s + event<EventRollbackSlotsRuntime> / begin_rollback_slots,
        "rollback_slots_exec"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime> [rollback_slots_request_valid],
        "errored"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime> [rollback_slots_request_invalid] / mark_invalid_request_event_rollback_slots_runtime,
        "rollback_slots_result_decision"_s <= "rollback_slots_exec"_s + completion<EventRollbackSlotsRuntime> / exec_rollback_slots,
        "done"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime> [operation_succeeded_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime> [operation_failed_with_error_event_rollback_slots_runtime] / mark_error_from_operation_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime> [operation_failed_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "capture_request_decision"_s <= "ready"_s + event<EventCaptureViewRuntime> / begin_capture_view,
        "capture_exec"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime> [capture_request_valid],
        "errored"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime> [capture_request_invalid] / mark_invalid_request_event_capture_view_runtime,
        "capture_result_decision"_s <= "capture_exec"_s + completion<EventCaptureViewRuntime> / exec_capture_view,
        "done"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime> [operation_succeeded_event_capture_view_runtime],
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime> [operation_failed_with_error_event_capture_view_runtime] / mark_error_from_operation_event_capture_view_runtime,
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime> [operation_failed_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "ready"_s <= "done"_s + completion<EventReserveRuntime> / publish_done_event_reserve_runtime,
        "ready"_s <= "errored"_s + completion<EventReserveRuntime> / publish_error_event_reserve_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSequenceRuntime> / publish_done_event_allocate_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSequenceRuntime> / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSlotsRuntime> / publish_done_event_allocate_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSlotsRuntime> / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "done"_s + completion<EventBranchSequenceRuntime> / publish_done_event_branch_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventBranchSequenceRuntime> / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventFreeSequenceRuntime> / publish_done_event_free_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventFreeSequenceRuntime> / publish_error_event_free_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventRollbackSlotsRuntime> / publish_done_event_rollback_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventRollbackSlotsRuntime> / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "done"_s + completion<EventCaptureViewRuntime> / publish_done_event_capture_view_runtime,
        "ready"_s <= "errored"_s + completion<EventCaptureViewRuntime> / publish_error_event_capture_view_runtime,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "reserve_request_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_request_decision,
        "ready"_s <= "reserve_exec"_s + unexpected_event<_> / on_unexpected_from_reserve_exec,
        "ready"_s <= "reserve_result_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_result_decision,
        "ready"_s <= "allocate_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_request_decision,
        "ready"_s <= "allocate_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_exec,
        "ready"_s <= "allocate_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_result_decision,
        "ready"_s <= "allocate_slots_request_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_decision,
        "ready"_s <= "allocate_slots_request_shape_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_shape_decision,
        "ready"_s <= "allocate_slots_request_length_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_length_decision,
        "ready"_s <= "allocate_slots_request_block_layout_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_block_layout_decision,
        "ready"_s <= "allocate_slots_request_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_capacity_decision,
        "ready"_s <= "state_allocate_slots_tail_decision"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_decision,
        "ready"_s <= "state_allocate_slots_tail_copy_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_copy_exec,
        "ready"_s <= "state_allocate_slots_tail_copy_result_decision"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_copy_result_decision,
        "ready"_s <= "state_allocate_slots_direct_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_direct_exec,
        "ready"_s <= "state_allocate_slots_tail_split_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_split_exec,
        "ready"_s <= "allocate_slots_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_result_decision,
        "ready"_s <= "branch_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_request_decision,
        "ready"_s <= "branch_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_exec,
        "ready"_s <= "branch_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_result_decision,
        "ready"_s <= "free_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_request_decision,
        "ready"_s <= "free_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_free_sequence_exec,
        "ready"_s <= "free_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_result_decision,
        "ready"_s <= "rollback_slots_request_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_request_decision,
        "ready"_s <= "rollback_slots_exec"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_exec,
        "ready"_s <= "rollback_slots_result_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_result_decision,
        "ready"_s <= "capture_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_request_decision,
        "ready"_s <= "capture_exec"_s + unexpected_event<_> / on_unexpected_from_capture_exec,
        "ready"_s <= "capture_result_decision"_s + unexpected_event<_> / on_unexpected_from_capture_result_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `MemoryKv` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct MemoryKvContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl MemoryKvStateMachineContext for MemoryKvContext {
    fn allocate_sequence_request_invalid(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_sequence_request_invalid
        todo!(
            "TODO: port guard `allocate_sequence_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_sequence_request_valid(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_sequence_request_valid
        todo!(
            "TODO: port guard `allocate_sequence_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_block_layout_invalid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_block_layout_invalid
        todo!(
            "TODO: port guard `allocate_slots_request_block_layout_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_block_layout_valid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_block_layout_valid
        todo!(
            "TODO: port guard `allocate_slots_request_block_layout_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_capacity_invalid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_capacity_invalid
        todo!(
            "TODO: port guard `allocate_slots_request_capacity_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_capacity_valid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_capacity_valid
        todo!(
            "TODO: port guard `allocate_slots_request_capacity_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_length_invalid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_length_invalid
        todo!(
            "TODO: port guard `allocate_slots_request_length_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_length_valid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_length_valid
        todo!(
            "TODO: port guard `allocate_slots_request_length_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_shape_invalid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_shape_invalid
        todo!(
            "TODO: port guard `allocate_slots_request_shape_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn allocate_slots_request_shape_valid(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::allocate_slots_request_shape_valid
        todo!(
            "TODO: port guard `allocate_slots_request_shape_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn begin_allocate_sequence(&mut self, _event: &EventAllocateSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_allocate_sequence
        todo!(
            "TODO: port action `begin_allocate_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn begin_allocate_slots(&mut self, _event: &EventAllocateSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_allocate_slots
        todo!(
            "TODO: port action `begin_allocate_slots` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn begin_branch_sequence(&mut self, _event: &EventBranchSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_branch_sequence
        todo!(
            "TODO: port action `begin_branch_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn begin_capture_view(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_capture_view
        todo!("TODO: port action `begin_capture_view` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn begin_free_sequence(&mut self, _event: &EventFreeSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_free_sequence
        todo!(
            "TODO: port action `begin_free_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn begin_reserve(&mut self, _event: &EventReserveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_reserve
        todo!("TODO: port action `begin_reserve` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn begin_rollback_slots(&mut self, _event: &EventRollbackSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::begin_rollback_slots
        todo!(
            "TODO: port action `begin_rollback_slots` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn branch_sequence_request_invalid(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::branch_sequence_request_invalid
        todo!(
            "TODO: port guard `branch_sequence_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn branch_sequence_request_valid(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::branch_sequence_request_valid
        todo!(
            "TODO: port guard `branch_sequence_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn capture_request_invalid(&self, _event: &EventCaptureViewRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::capture_request_invalid
        todo!(
            "TODO: port guard `capture_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn capture_request_valid(&self, _event: &EventCaptureViewRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::capture_request_valid
        todo!(
            "TODO: port guard `capture_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn effect_allocate_slots_with_tail_split(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::effect_allocate_slots_with_tail_split
        todo!(
            "TODO: port action `effect_allocate_slots_with_tail_split` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn effect_allocate_slots_without_tail_split(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::effect_allocate_slots_without_tail_split
        todo!(
            "TODO: port action `effect_allocate_slots_without_tail_split` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn effect_copy_shared_tail_block(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::effect_copy_shared_tail_block
        todo!(
            "TODO: port action `effect_copy_shared_tail_block` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn exec_allocate_sequence(&mut self, _event: &EventAllocateSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_allocate_sequence
        todo!(
            "TODO: port action `exec_allocate_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn exec_branch_sequence(&mut self, _event: &EventBranchSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_branch_sequence
        todo!(
            "TODO: port action `exec_branch_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn exec_capture_view(&mut self, _event: &EventCaptureViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_capture_view
        todo!("TODO: port action `exec_capture_view` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn exec_free_sequence(&mut self, _event: &EventFreeSequenceRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_free_sequence
        todo!("TODO: port action `exec_free_sequence` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn exec_reserve(&mut self, _event: &EventReserveRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_reserve
        todo!("TODO: port action `exec_reserve` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn exec_rollback_slots(&mut self, _event: &EventRollbackSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::exec_rollback_slots
        todo!(
            "TODO: port action `exec_rollback_slots` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn free_sequence_request_invalid(&self, _event: &EventFreeSequenceRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::free_sequence_request_invalid
        todo!(
            "TODO: port guard `free_sequence_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn free_sequence_request_valid(&self, _event: &EventFreeSequenceRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::free_sequence_request_valid
        todo!(
            "TODO: port guard `free_sequence_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn guard_allocate_slots_shared_tail_copy_missing(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::guard_allocate_slots_shared_tail_copy_missing
        todo!(
            "TODO: port guard `guard_allocate_slots_shared_tail_copy_missing` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn guard_allocate_slots_shared_tail_copy_ready(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::guard_allocate_slots_shared_tail_copy_ready
        todo!(
            "TODO: port guard `guard_allocate_slots_shared_tail_copy_ready` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn guard_allocate_slots_shared_tail_split_not_required(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::guard_allocate_slots_shared_tail_split_not_required
        todo!(
            "TODO: port guard `guard_allocate_slots_shared_tail_split_not_required` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn mark_backend_error_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_backend_error_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_backend_error
        todo!("TODO: port action `mark_backend_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn mark_error_from_operation_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_error_from_operation_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_invalid_request_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/memory/kv/actions.hpp"
        )
    }
    fn mark_out_of_memory(&mut self, _event: &EventAllocateSlotsRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::mark_out_of_memory
        todo!("TODO: port action `mark_out_of_memory` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_sequence_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_request_block_layout_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_request_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_request_length_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_request_shape_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_allocate_slots_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_branch_sequence_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_capture_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_capture_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_capture_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_free_sequence_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_reserve_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_reserve_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_reserve_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_rollback_slots_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_state_allocate_slots_direct_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_state_allocate_slots_tail_copy_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_state_allocate_slots_tail_copy_result_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_state_allocate_slots_tail_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn on_unexpected_from_state_allocate_slots_tail_split_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn operation_failed_with_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_with_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_failed_without_error_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn operation_succeeded_event_allocate_sequence_runtime(
        &self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_allocate_slots_runtime(
        &self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_branch_sequence_runtime(
        &self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_capture_view_runtime(
        &self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_free_sequence_runtime(
        &self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_reserve_runtime(
        &self,
        _event: &EventReserveRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn operation_succeeded_event_rollback_slots_runtime(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::operation_succeeded
        todo!("TODO: port guard `operation_succeeded` from emel.cpp/src/emel/memory/kv/guards.hpp")
    }
    fn publish_done_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_done_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_allocate_sequence_runtime(
        &mut self,
        _event: &EventAllocateSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_allocate_slots_runtime(
        &mut self,
        _event: &EventAllocateSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_branch_sequence_runtime(
        &mut self,
        _event: &EventBranchSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_capture_view_runtime(
        &mut self,
        _event: &EventCaptureViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_free_sequence_runtime(
        &mut self,
        _event: &EventFreeSequenceRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_reserve_runtime(
        &mut self,
        _event: &EventReserveRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn publish_error_event_rollback_slots_runtime(
        &mut self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/memory/kv/actions.hpp")
    }
    fn reserve_request_invalid(&self, _event: &EventReserveRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::reserve_request_invalid
        todo!(
            "TODO: port guard `reserve_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn reserve_request_valid(&self, _event: &EventReserveRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::reserve_request_valid
        todo!(
            "TODO: port guard `reserve_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn rollback_slots_request_invalid(
        &self,
        _event: &EventRollbackSlotsRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::rollback_slots_request_invalid
        todo!(
            "TODO: port guard `rollback_slots_request_invalid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
    fn rollback_slots_request_valid(&self, _event: &EventRollbackSlotsRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/kv/guards.hpp::rollback_slots_request_valid
        todo!(
            "TODO: port guard `rollback_slots_request_valid` from emel.cpp/src/emel/memory/kv/guards.hpp"
        )
    }
}
