//! Source-aligned bounded hybrid memory state machine.

#![allow(
    clippy::enum_variant_names, clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions, clippy::missing_errors_doc,
    clippy::must_use_candidate, clippy::return_self_not_must_use,
    clippy::missing_const_for_fn, dead_code, missing_docs
)]

use core::cell::RefCell;
use sml::sml;

use crate::kv::sm as kv;
use crate::recurrent::sm as recurrent;

pub const MAX_SEQUENCES: usize = kv::MAX_SEQUENCES;
pub type Snapshot = kv::Snapshot;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HybridError {
    #[default] None = 0, InvalidRequest = 1, BackendError = 2,
    InternalError = 4, OutOfMemory = 8, Untracked = 16,
}
impl HybridError { const fn code(self) -> i32 { self as i32 } }

#[derive(Clone, Copy, Debug, Default)]
pub struct ReserveContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSequenceContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) rollback_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32, pub(crate) rollback_error: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSlotsContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) rollback_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32, pub(crate) rollback_error: i32, pub(crate) kv_block_count: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct BranchSequenceContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) rollback_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32, pub(crate) rollback_error: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeSequenceContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct RollbackSlotsContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32, pub(crate) kv_block_count: i32 }
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureViewContext { pub(crate) err: HybridError, pub(crate) kv_accepted: bool, pub(crate) recurrent_accepted: bool, pub(crate) kv_error: i32, pub(crate) recurrent_error: i32 }

#[derive(Clone)]
pub struct EventAllocateSequenceRuntime<'event> { pub seq_id: i32, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<AllocateSequenceContext> }
#[derive(Clone)]
pub struct EventAllocateSlotsRuntime<'event> { pub seq_id: i32, pub token_count: i32, pub block_count_out: Option<&'event RefCell<i32>>, pub error_out: Option<&'event RefCell<i32>>, pub copy_block: Option<&'event dyn kv::BlockCopier>, pub context: &'event RefCell<AllocateSlotsContext> }
#[derive(Clone)]
pub struct EventBranchSequenceRuntime<'event> { pub parent_seq_id: i32, pub child_seq_id: i32, pub copy_state: Option<&'event dyn recurrent::StateCopier>, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<BranchSequenceContext> }
#[derive(Clone)]
pub struct EventCaptureViewRuntime<'event> { pub snapshot_out: Option<&'event RefCell<Snapshot>>, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<CaptureViewContext> }
#[derive(Clone)]
pub struct EventFreeSequenceRuntime<'event> { pub seq_id: i32, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<FreeSequenceContext> }
#[derive(Clone)]
pub struct EventReserveRuntime<'event> { pub max_sequences: i32, pub max_blocks: i32, pub block_tokens: i32, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<ReserveContext> }
#[derive(Clone)]
pub struct EventRollbackSlotsRuntime<'event> { pub seq_id: i32, pub token_count: i32, pub block_count_out: Option<&'event RefCell<i32>>, pub error_out: Option<&'event RefCell<i32>>, pub context: &'event RefCell<RollbackSlotsContext> }
macro_rules! event_debug { ($($name:ident),+ $(,)?) => { $(impl core::fmt::Debug for $name<'_> { fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct(stringify!($name)).finish() } })+ }; }
event_debug!(EventAllocateSequenceRuntime, EventAllocateSlotsRuntime, EventBranchSequenceRuntime, EventCaptureViewRuntime, EventFreeSequenceRuntime, EventReserveRuntime, EventRollbackSlotsRuntime);

sml! {
    MemoryHybrid<'event> {
        "reserve_kv"_s <= *"ready"_s + event<EventReserveRuntime<'event>> / begin_reserve,
        "reserve_kv_decision"_s <= "reserve_kv"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / exec_reserve_kv,
        "reserve_recurrent"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [kv_accepted_event_reserve_runtime],
        "errored"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [kv_rejected_with_error_event_reserve_runtime] / mark_error_from_kv_event_reserve_runtime,
        "errored"_s <= "reserve_kv_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [kv_rejected_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "reserve_recurrent_decision"_s <= "reserve_recurrent"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / exec_reserve_recurrent,
        "done"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [recurrent_accepted_event_reserve_runtime],
        "errored"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [recurrent_rejected_with_error_event_reserve_runtime] / mark_error_from_recurrent_event_reserve_runtime,
        "errored"_s <= "reserve_recurrent_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [recurrent_rejected_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "allocate_sequence_kv"_s <= "ready"_s + event<EventAllocateSequenceRuntime<'event>> / begin_allocate_sequence,
        "allocate_sequence_kv_decision"_s <= "allocate_sequence_kv"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / exec_allocate_sequence_kv,
        "allocate_sequence_recurrent"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [kv_accepted_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [kv_rejected_with_error_event_allocate_sequence_runtime] / mark_error_from_kv_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_kv_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [kv_rejected_without_error_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "allocate_sequence_recurrent_decision"_s <= "allocate_sequence_recurrent"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / exec_allocate_sequence_recurrent,
        "done"_s <= "allocate_sequence_recurrent_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [recurrent_accepted_event_allocate_sequence_runtime],
        "allocate_sequence_rollback_kv"_s <= "allocate_sequence_recurrent_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [recurrent_rejected_any_event_allocate_sequence_runtime] / exec_allocate_sequence_rollback_kv,
        "allocate_sequence_rollback_result_decision"_s <= "allocate_sequence_rollback_kv"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>),
        "allocate_sequence_recurrent_error_decision"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [rollback_accepted_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [rollback_rejected_with_error_event_allocate_sequence_runtime] / mark_error_from_rollback_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [rollback_rejected_without_error_event_allocate_sequence_runtime] / mark_internal_error_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_rollback_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / mark_internal_error_event_allocate_sequence_runtime,
        "out_of_memory"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [recurrent_rejected_out_of_memory_event_allocate_sequence_runtime] / mark_out_of_memory_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [recurrent_rejected_backend_or_none_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [recurrent_rejected_non_backend_error_event_allocate_sequence_runtime] / mark_error_from_recurrent_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_recurrent_error_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / mark_internal_error_event_allocate_sequence_runtime,
        "allocate_slots_kv"_s <= "ready"_s + event<EventAllocateSlotsRuntime<'event>> / begin_allocate_slots,
        "allocate_slots_kv_decision"_s <= "allocate_slots_kv"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [guard_owned_kv_cache_event_allocate_slots_runtime] / effect_allocate_slots_owned_kv,
        "allocate_slots_kv_decision"_s <= "allocate_slots_kv"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [guard_bound_kv_cache_event_allocate_slots_runtime] / effect_allocate_slots_bound_kv,
        "allocate_slots_recurrent"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [kv_accepted_event_allocate_slots_runtime],
        "out_of_memory"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [kv_rejected_out_of_memory_event_allocate_slots_runtime] / mark_out_of_memory_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [kv_rejected_backend_or_none_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_kv_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [kv_rejected_non_backend_error_event_allocate_slots_runtime] / mark_error_from_kv_event_allocate_slots_runtime,
        "allocate_slots_recurrent_decision"_s <= "allocate_slots_recurrent"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / exec_allocate_slots_recurrent,
        "done"_s <= "allocate_slots_recurrent_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [recurrent_accepted_event_allocate_slots_runtime],
        "allocate_slots_rollback_kv"_s <= "allocate_slots_recurrent_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [recurrent_rejected_any_event_allocate_slots_runtime] / exec_allocate_slots_rollback_kv,
        "allocate_slots_rollback_result_decision"_s <= "allocate_slots_rollback_kv"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>),
        "allocate_slots_recurrent_error_decision"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [rollback_accepted_event_allocate_slots_runtime],
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [rollback_rejected_with_error_event_allocate_slots_runtime] / mark_error_from_rollback_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [rollback_rejected_without_error_event_allocate_slots_runtime] / mark_internal_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_rollback_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / mark_internal_error_event_allocate_slots_runtime,
        "out_of_memory"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [recurrent_rejected_out_of_memory_event_allocate_slots_runtime] / mark_out_of_memory_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [recurrent_rejected_backend_or_none_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [recurrent_rejected_non_backend_error_event_allocate_slots_runtime] / mark_error_from_recurrent_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_recurrent_error_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / mark_internal_error_event_allocate_slots_runtime,
        "branch_sequence_kv"_s <= "ready"_s + event<EventBranchSequenceRuntime<'event>> / begin_branch_sequence,
        "branch_sequence_kv_decision"_s <= "branch_sequence_kv"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence_kv,
        "branch_sequence_recurrent"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [kv_accepted_event_branch_sequence_runtime],
        "out_of_memory"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [kv_rejected_out_of_memory_event_branch_sequence_runtime] / mark_out_of_memory_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [kv_rejected_backend_or_none_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_kv_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [kv_rejected_non_backend_error_event_branch_sequence_runtime] / mark_error_from_kv_event_branch_sequence_runtime,
        "branch_sequence_recurrent_decision"_s <= "branch_sequence_recurrent"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence_recurrent,
        "done"_s <= "branch_sequence_recurrent_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [recurrent_accepted_event_branch_sequence_runtime],
        "branch_sequence_rollback_kv"_s <= "branch_sequence_recurrent_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [recurrent_rejected_any_event_branch_sequence_runtime] / exec_branch_sequence_rollback_kv,
        "branch_sequence_rollback_result_decision"_s <= "branch_sequence_rollback_kv"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>),
        "branch_sequence_recurrent_error_decision"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [rollback_accepted_event_branch_sequence_runtime],
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [rollback_rejected_with_error_event_branch_sequence_runtime] / mark_error_from_rollback_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [rollback_rejected_without_error_event_branch_sequence_runtime] / mark_internal_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_rollback_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / mark_internal_error_event_branch_sequence_runtime,
        "out_of_memory"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [recurrent_rejected_out_of_memory_event_branch_sequence_runtime] / mark_out_of_memory_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [recurrent_rejected_backend_or_none_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [recurrent_rejected_non_backend_error_event_branch_sequence_runtime] / mark_error_from_recurrent_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_recurrent_error_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / mark_internal_error_event_branch_sequence_runtime,
        "free_sequence_kv"_s <= "ready"_s + event<EventFreeSequenceRuntime<'event>> / begin_free_sequence,
        "free_sequence_kv_decision"_s <= "free_sequence_kv"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / exec_free_sequence_kv,
        "free_sequence_recurrent"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [kv_accepted_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [kv_rejected_with_error_event_free_sequence_runtime] / mark_error_from_kv_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_kv_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [kv_rejected_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "free_sequence_recurrent_decision"_s <= "free_sequence_recurrent"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / exec_free_sequence_recurrent,
        "done"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [recurrent_accepted_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [recurrent_rejected_with_error_event_free_sequence_runtime] / mark_error_from_recurrent_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_recurrent_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [recurrent_rejected_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "rollback_slots_kv"_s <= "ready"_s + event<EventRollbackSlotsRuntime<'event>> / begin_rollback_slots,
        "rollback_slots_kv_decision"_s <= "rollback_slots_kv"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / exec_rollback_slots_kv,
        "rollback_slots_recurrent"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [kv_accepted_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [kv_rejected_with_error_event_rollback_slots_runtime] / mark_error_from_kv_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_kv_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [kv_rejected_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "rollback_slots_recurrent_decision"_s <= "rollback_slots_recurrent"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / exec_rollback_slots_recurrent,
        "done"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [recurrent_accepted_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [recurrent_rejected_with_error_event_rollback_slots_runtime] / mark_error_from_recurrent_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_recurrent_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [recurrent_rejected_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "capture_request_decision"_s <= "ready"_s + event<EventCaptureViewRuntime<'event>> / begin_capture_view,
        "capture_kv"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_valid],
        "errored"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_invalid] / mark_invalid_request,
        "capture_kv_decision"_s <= "capture_kv"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [guard_owned_kv_cache_event_capture_view_runtime] / effect_capture_owned_kv,
        "capture_kv_decision"_s <= "capture_kv"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [guard_bound_kv_cache_event_capture_view_runtime] / effect_capture_bound_kv,
        "capture_recurrent"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [kv_accepted_event_capture_view_runtime],
        "errored"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [kv_rejected_with_error_event_capture_view_runtime] / mark_error_from_kv_event_capture_view_runtime,
        "errored"_s <= "capture_kv_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [kv_rejected_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "capture_recurrent_decision"_s <= "capture_recurrent"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / exec_capture_recurrent,
        "capture_merge"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [recurrent_accepted_event_capture_view_runtime],
        "errored"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [recurrent_rejected_with_error_event_capture_view_runtime] / mark_error_from_recurrent_event_capture_view_runtime,
        "errored"_s <= "capture_recurrent_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [recurrent_rejected_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "done"_s <= "capture_merge"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / merge_capture_snapshots,
        "ready"_s <= "done"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_done_event_reserve_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_error_event_reserve_runtime,
        "ready"_s <= "errored"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_error_event_reserve_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_done_event_allocate_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_done_event_allocate_slots_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "done"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_done_event_branch_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_done_event_free_sequence_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_error_event_free_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_error_event_free_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_done_event_rollback_slots_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "done"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_done_event_capture_view_runtime,
        "ready"_s <= "out_of_memory"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_error_event_capture_view_runtime,
        "ready"_s <= "errored"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_error_event_capture_view_runtime,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheRoute { Owned, Bound }

/// Public, safe boundary implemented by injected KV actors.
pub trait HybridKvActor {
    fn reserve(&mut self, max_sequences: i32, max_blocks: i32, block_tokens: i32, error_out: &RefCell<i32>) -> bool;
    fn allocate_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool;
    fn allocate_slots(&mut self, seq_id: i32, token_count: i32, block_count_out: &RefCell<i32>, error_out: &RefCell<i32>, copy_block: Option<&dyn kv::BlockCopier>) -> bool;
    fn free_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool;
    fn rollback_slots(&mut self, seq_id: i32, token_count: i32, block_count_out: &RefCell<i32>, error_out: &RefCell<i32>) -> bool;
    fn capture_view(&mut self, snapshot_out: &RefCell<Snapshot>, error_out: &RefCell<i32>) -> bool;
    fn branch_sequence(&mut self, parent_seq_id: i32, child_seq_id: i32, error_out: &RefCell<i32>) -> bool;
}

impl HybridKvActor for kv::MemoryKvStateMachine<kv::MemoryKvContext> {
    fn reserve(&mut self, max_sequences: i32, max_blocks: i32, block_tokens: i32, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::ReserveContext::default()); self.process_event(kv::EventReserveRuntime { max_sequences, max_blocks, block_tokens, error_out: Some(error_out), context: &c }).is_ok() }
    fn allocate_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::AllocateSequenceContext::default()); self.process_event(kv::EventAllocateSequenceRuntime { seq_id, error_out: Some(error_out), context: &c }).is_ok() }
    fn allocate_slots(&mut self, seq_id: i32, token_count: i32, block_count_out: &RefCell<i32>, error_out: &RefCell<i32>, copy_block: Option<&dyn kv::BlockCopier>) -> bool { let c = RefCell::new(kv::AllocateSlotsContext::default()); self.process_event(kv::EventAllocateSlotsRuntime { seq_id, token_count, block_count_out: Some(block_count_out), error_out: Some(error_out), copy_block, context: &c }).is_ok() }
    fn free_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::FreeSequenceContext::default()); self.process_event(kv::EventFreeSequenceRuntime { seq_id, error_out: Some(error_out), context: &c }).is_ok() }
    fn rollback_slots(&mut self, seq_id: i32, token_count: i32, block_count_out: &RefCell<i32>, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::RollbackSlotsContext::default()); self.process_event(kv::EventRollbackSlotsRuntime { seq_id, token_count, block_count_out: Some(block_count_out), error_out: Some(error_out), context: &c }).is_ok() }
    fn capture_view(&mut self, snapshot_out: &RefCell<Snapshot>, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::CaptureViewContext::default()); self.process_event(kv::EventCaptureViewRuntime { snapshot_out: Some(snapshot_out), error_out: Some(error_out), context: &c }).is_ok() }
    fn branch_sequence(&mut self, parent_seq_id: i32, child_seq_id: i32, error_out: &RefCell<i32>) -> bool { let c = RefCell::new(kv::BranchSequenceContext::default()); self.process_event(kv::EventBranchSequenceRuntime { parent_seq_id, child_seq_id, copy_state: None, error_out: Some(error_out), context: &c }).is_ok() }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvBindingKind { Empty, Complete, Invalid }

#[derive(Clone)]
pub struct KvBinding { actor: Option<std::rc::Rc<RefCell<dyn HybridKvActor>>>, kind: KvBindingKind }
impl Default for KvBinding { fn default() -> Self { Self { actor: None, kind: KvBindingKind::Empty } } }
impl KvBinding {
    pub fn empty() -> Self { Self::default() }
    pub fn invalid() -> Self { Self { actor: None, kind: KvBindingKind::Invalid } }
    pub fn from_shared(actor: std::rc::Rc<RefCell<dyn HybridKvActor>>) -> Self { Self { actor: Some(actor), kind: KvBindingKind::Complete } }
    pub fn from_actor<A: HybridKvActor + 'static>(actor: A) -> Self { Self::from_shared(std::rc::Rc::new(RefCell::new(actor))) }
    fn route(&self) -> KvCacheRoute { if matches!(self.kind, KvBindingKind::Empty) { KvCacheRoute::Owned } else { KvCacheRoute::Bound } }
    fn is_invalid(&self) -> bool { matches!(self.kind, KvBindingKind::Invalid) }
}

pub struct MemoryHybridContext {
    pub kv: kv::MemoryKvStateMachine<kv::MemoryKvContext>,
    pub recurrent: recurrent::MemoryRecurrentStateMachine<recurrent::MemoryRecurrentContext>,
    pub kv_route: KvCacheRoute,
    pub kv_binding: KvBinding,
    pub kv_snapshot: Snapshot,
    pub recurrent_snapshot: recurrent::Snapshot,
}
impl Default for MemoryHybridContext {
    fn default() -> Self {
        Self { kv: kv::MemoryKvStateMachine::new(kv::MemoryKvContext::default()), recurrent: recurrent::MemoryRecurrentStateMachine::new(recurrent::MemoryRecurrentContext::default()), kv_route: KvCacheRoute::Owned, kv_binding: KvBinding::default(), kv_snapshot: Snapshot::default(), recurrent_snapshot: recurrent::Snapshot::default() }
    }
}
impl MemoryHybridContext {
    pub fn with_kv_binding(binding: KvBinding) -> Self { let mut context = Self::default(); context.kv_route = binding.route(); context.kv_binding = binding; context }
    fn bound_actor(&self) -> Option<std::rc::Rc<RefCell<dyn HybridKvActor>>> { self.kv_binding.actor.clone() }
}

fn set_error(out: Option<&RefCell<i32>>, error: HybridError) { if let Some(out) = out { *out.borrow_mut() = error.code(); } }
fn api_error(code: i32) -> HybridError { match code { 1 => HybridError::InvalidRequest, 2 => HybridError::BackendError, 4 => HybridError::InternalError, 8 => HybridError::OutOfMemory, 16 => HybridError::Untracked, _ => HybridError::InternalError } }
fn backend_family(code: i32) -> bool { matches!(code, 0 | 2 | 8) }
fn backend_or_none(code: i32) -> bool { matches!(code, 0 | 2) }
fn merge(kv: &Snapshot, rec: &recurrent::Snapshot, out: &mut Snapshot) {
    *out = Snapshot::default(); out.max_sequences = kv.max_sequences.min(rec.max_sequences); out.block_tokens = kv.block_tokens;
    let count = usize::try_from(out.max_sequences.max(0)).unwrap_or(0).min(MAX_SEQUENCES);
    for i in 0..count { let active = kv.sequence_active[i] != 0 && rec.sequence_active[i] != 0; out.sequence_active[i] = u8::from(active); out.sequence_length_values[i] = if active { kv.sequence_length_values[i].min(rec.sequence_length_values[i]) } else { 0 }; out.sequence_kv_block_count[i] = if active { kv.sequence_kv_block_count[i] } else { 0 }; out.sequence_kv_blocks[i] = kv.sequence_kv_blocks[i]; out.sequence_recurrent_slot[i] = if active { rec.sequence_recurrent_slot[i] } else { -1 }; }
}

fn effect_allocate_slots_impl(ctx: &mut MemoryHybridContext, e: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> {
    let c = RefCell::new(kv::AllocateSlotsContext::default());
    let err = RefCell::new(0);
    let ok = ctx.kv.process_event(kv::EventAllocateSlotsRuntime { seq_id: e.seq_id, token_count: e.token_count, block_count_out: None, error_out: Some(&err), copy_block: e.copy_block, context: &c }).is_ok() && *err.borrow() == 0;
    let mut h = e.context.borrow_mut(); h.kv_accepted = ok; h.kv_error = if ok { 0 } else { *err.borrow() }; h.kv_block_count = 0; Ok(())
}
fn effect_capture_impl(ctx: &mut MemoryHybridContext, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
    let c = RefCell::new(kv::CaptureViewContext::default());
    let err = RefCell::new(0);
    let snapshot = RefCell::new(Snapshot::default());
    let ok = ctx.kv.process_event(kv::EventCaptureViewRuntime { snapshot_out: Some(&snapshot), error_out: Some(&err), context: &c }).is_ok() && *err.borrow() == 0;
    let mut h = e.context.borrow_mut(); h.kv_accepted = ok; h.kv_error = if ok { 0 } else { *err.borrow() }; Ok(())
}

impl MemoryHybridStateMachineContext for MemoryHybridContext {
    fn begin_allocate_sequence(&mut self, e: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = AllocateSequenceContext::default(); set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_reserve(&mut self, e: &EventReserveRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = ReserveContext::default(); set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_allocate_slots(&mut self, e: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = AllocateSlotsContext::default(); if let Some(o)=e.block_count_out {*o.borrow_mut()=0;} set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_branch_sequence(&mut self, e: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = BranchSequenceContext::default(); set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_free_sequence(&mut self, e: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = FreeSequenceContext::default(); set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_rollback_slots(&mut self, e: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = RollbackSlotsContext::default(); if let Some(o)=e.block_count_out {*o.borrow_mut()=0;} set_error(e.error_out, HybridError::None); Ok(()) }
    fn begin_capture_view(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> { *e.context.borrow_mut() = CaptureViewContext::default(); set_error(e.error_out, HybridError::None); Ok(()) }
    fn capture_request_valid(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { Ok(e.snapshot_out.is_some()) } fn capture_request_invalid(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { Ok(e.snapshot_out.is_none()) }
    fn exec_reserve_kv(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> { let mut c=RefCell::new(kv::ReserveContext::default()); let mut err=0; let ok=self.kv.process_event(kv::EventReserveRuntime{max_sequences:e.max_sequences,max_blocks:e.max_blocks,block_tokens:e.block_tokens,error_out:Some(&RefCell::new(err)),context:&c}).is_ok(); let x=c.borrow(); let mut h=e.context.borrow_mut(); h.kv_accepted=ok; h.kv_error=if ok {0} else {err}; drop(x); Ok(()) }
    fn exec_reserve_recurrent(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::ReserveContext::default()); let err=RefCell::new(0); let ok=self.recurrent.process_event(recurrent::EventReserveRuntime{max_sequences:e.max_sequences,max_blocks:e.max_blocks,block_tokens:e.block_tokens,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_allocate_sequence_kv(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::AllocateSequenceContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventAllocateSequenceRuntime{seq_id:e.seq_id,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.kv_accepted=ok; h.kv_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_allocate_sequence_recurrent(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::AllocateSequenceContext::default()); let err=RefCell::new(0); let ok=self.recurrent.process_event(recurrent::EventAllocateSequenceRuntime{seq_id:e.seq_id,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_allocate_sequence_rollback_kv(&mut self, e: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { let c = RefCell::new(kv::FreeSequenceContext::default()); let err = RefCell::new(0); let ok = self.kv.process_event(kv::EventFreeSequenceRuntime { seq_id: e.seq_id, error_out: Some(&err), context: &c }).is_ok() && *err.borrow() == 0; let mut h = e.context.borrow_mut(); h.rollback_accepted = ok; h.rollback_error = if ok { 0 } else { *err.borrow() }; Ok(()) }
    fn exec_rollback_slots_recurrent(&mut self, e: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> { let c = RefCell::new(recurrent::RollbackSlotsContext::default()); let err = RefCell::new(0); let ok = self.recurrent.process_event(recurrent::EventRollbackSlotsRuntime { seq_id: e.seq_id, token_count: e.token_count, block_count_out: None, error_out: Some(&err), context: &c }).is_ok() && *err.borrow() == 0; let mut h = e.context.borrow_mut(); h.recurrent_accepted = ok; h.recurrent_error = if ok { 0 } else { *err.borrow() }; Ok(()) }
    fn effect_allocate_slots_owned_kv(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> { effect_allocate_slots_impl(self, e) } fn effect_allocate_slots_bound_kv(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> { effect_allocate_slots_impl(self, e) }
    fn exec_allocate_slots_recurrent(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::AllocateSlotsContext::default()); let err=RefCell::new(0); let ok=self.recurrent.process_event(recurrent::EventAllocateSlotsRuntime{seq_id:e.seq_id,token_count:e.token_count,block_count_out:None,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_allocate_slots_rollback_kv(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::RollbackSlotsContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventRollbackSlotsRuntime{seq_id:e.seq_id,token_count:e.token_count,block_count_out:None,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.rollback_accepted=ok; h.rollback_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_branch_sequence_kv(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::BranchSequenceContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventBranchSequenceRuntime{parent_seq_id:e.parent_seq_id,child_seq_id:e.child_seq_id,copy_state:None,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.kv_accepted=ok; h.kv_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_branch_sequence_recurrent(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::BranchSequenceContext::default()); let err=RefCell::new(0); let ok=self.recurrent.process_event(recurrent::EventBranchSequenceRuntime{parent_seq_id:e.parent_seq_id,child_seq_id:e.child_seq_id,copy_state:e.copy_state,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_branch_sequence_rollback_kv(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::FreeSequenceContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventFreeSequenceRuntime{seq_id:e.child_seq_id,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.rollback_accepted=ok; h.rollback_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_free_sequence_kv(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::FreeSequenceContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventFreeSequenceRuntime{seq_id:e.seq_id,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.kv_accepted=ok; h.kv_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_free_sequence_recurrent(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::FreeSequenceContext::default()); let err=RefCell::new(0); let ok=self.recurrent.process_event(recurrent::EventFreeSequenceRuntime{seq_id:e.seq_id,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn exec_rollback_slots_kv(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> { let c=RefCell::new(kv::RollbackSlotsContext::default()); let err=RefCell::new(0); let ok=self.kv.process_event(kv::EventRollbackSlotsRuntime{seq_id:e.seq_id,token_count:e.token_count,block_count_out:None,error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.kv_accepted=ok; h.kv_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn effect_capture_owned_kv(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> { effect_capture_impl(self, e) } fn effect_capture_bound_kv(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> { effect_capture_impl(self, e) }
    fn exec_capture_recurrent(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> { let c=RefCell::new(recurrent::CaptureViewContext::default()); let err=RefCell::new(0); let snap=RefCell::new(recurrent::Snapshot::default()); let ok=self.recurrent.process_event(recurrent::EventCaptureViewRuntime{snapshot_out:Some(&snap),error_out:Some(&err),context:&c}).is_ok() && *err.borrow() == 0; let mut h=e.context.borrow_mut(); h.recurrent_accepted=ok; h.recurrent_error=if ok {0} else {*err.borrow()}; Ok(()) }
    fn merge_capture_snapshots(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> { let kv=Snapshot::default(); let rec=recurrent::Snapshot::default(); if let Some(out)=e.snapshot_out { merge(&kv,&rec,&mut out.borrow_mut()); } Ok(()) }
    fn kv_accepted_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn kv_rejected_with_error_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error != 0) }
    fn kv_rejected_without_error_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error == 0) }
    fn recurrent_accepted_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn kv_accepted_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn recurrent_accepted_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn kv_accepted_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn recurrent_accepted_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn kv_accepted_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn kv_rejected_with_error_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error != 0) }
    fn kv_rejected_without_error_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error == 0) }
    fn recurrent_accepted_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn recurrent_rejected_with_error_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error != 0) }
    fn recurrent_rejected_without_error_event_capture_view_runtime(&self,e:&EventCaptureViewRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error == 0) }
    fn kv_accepted_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn kv_rejected_with_error_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error != 0) }
    fn kv_rejected_without_error_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error == 0) }
    fn recurrent_accepted_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn recurrent_rejected_with_error_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error != 0) }
    fn recurrent_rejected_without_error_event_free_sequence_runtime(&self,e:&EventFreeSequenceRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error == 0) }
    fn kv_accepted_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn kv_rejected_with_error_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error != 0) }
    fn kv_rejected_without_error_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error == 0) }
    fn recurrent_accepted_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn recurrent_rejected_with_error_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error != 0) }
    fn recurrent_rejected_without_error_event_reserve_runtime(&self,e:&EventReserveRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error == 0) }
    fn kv_accepted_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().kv_accepted) }
    fn kv_rejected_with_error_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error != 0) }
    fn kv_rejected_without_error_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error == 0) }
    fn recurrent_accepted_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { Ok(e.context.borrow().recurrent_accepted) }
    fn recurrent_rejected_with_error_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error != 0) }
    fn recurrent_rejected_without_error_event_rollback_slots_runtime(&self,e:&EventRollbackSlotsRuntime<'_>)->Result<bool,()> { let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error == 0) }
    fn recurrent_rejected_any_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()>{Ok(!e.context.borrow().recurrent_accepted)} fn recurrent_rejected_any_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()>{Ok(!e.context.borrow().recurrent_accepted)} fn recurrent_rejected_any_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()>{Ok(!e.context.borrow().recurrent_accepted)}
    fn recurrent_rejected_out_of_memory_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error==8)}
    fn recurrent_rejected_backend_or_none_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))}
    fn recurrent_rejected_non_backend_error_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error!=0 && !backend_family(c.recurrent_error))}
    fn kv_rejected_out_of_memory_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error==8)}
    fn recurrent_rejected_out_of_memory_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error==8)}
    fn kv_rejected_backend_or_none_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && backend_or_none(c.kv_error))}
    fn recurrent_rejected_backend_or_none_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))}
    fn kv_rejected_non_backend_error_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error!=0 && !backend_family(c.kv_error))}
    fn recurrent_rejected_non_backend_error_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error!=0 && !backend_family(c.recurrent_error))}
    fn kv_rejected_out_of_memory_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error==8)}
    fn recurrent_rejected_out_of_memory_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error==8)}
    fn kv_rejected_backend_or_none_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && backend_or_none(c.kv_error))}
    fn recurrent_rejected_backend_or_none_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))}
    fn kv_rejected_non_backend_error_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.kv_accepted && c.kv_error!=0 && !backend_family(c.kv_error))}
    fn recurrent_rejected_non_backend_error_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow(); Ok(!c.recurrent_accepted && c.recurrent_error!=0 && !backend_family(c.recurrent_error))}
    fn rollback_accepted_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {Ok(e.context.borrow().rollback_accepted)}
    fn rollback_rejected_with_error_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error!=0)}
    fn rollback_rejected_without_error_event_allocate_sequence_runtime(&self,e:&EventAllocateSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error==0)}
    fn rollback_accepted_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {Ok(e.context.borrow().rollback_accepted)}
    fn rollback_rejected_with_error_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error!=0)}
    fn rollback_rejected_without_error_event_allocate_slots_runtime(&self,e:&EventAllocateSlotsRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error==0)}
    fn rollback_accepted_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {Ok(e.context.borrow().rollback_accepted)}
    fn rollback_rejected_with_error_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error!=0)}
    fn rollback_rejected_without_error_event_branch_sequence_runtime(&self,e:&EventBranchSequenceRuntime<'_>)->Result<bool,()> {let c=e.context.borrow();Ok(!c.rollback_accepted&&c.rollback_error==0)}
    fn mark_backend_error_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_capture_view_runtime(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_free_sequence_runtime(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_reserve_runtime(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_backend_error_event_rollback_slots_runtime(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::BackendError;set_error(e.error_out,HybridError::BackendError);Ok(())}
    fn mark_out_of_memory_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::OutOfMemory;set_error(e.error_out,HybridError::OutOfMemory);Ok(())}
    fn mark_out_of_memory_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::OutOfMemory;set_error(e.error_out,HybridError::OutOfMemory);Ok(())}
    fn mark_out_of_memory_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::OutOfMemory;set_error(e.error_out,HybridError::OutOfMemory);Ok(())}
    fn mark_internal_error_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::InternalError;set_error(e.error_out,HybridError::InternalError);Ok(())}
    fn mark_internal_error_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::InternalError;set_error(e.error_out,HybridError::InternalError);Ok(())}
    fn mark_internal_error_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::InternalError;set_error(e.error_out,HybridError::InternalError);Ok(())}
    fn mark_error_from_kv_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_capture_view_runtime(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_free_sequence_runtime(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_reserve_runtime(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_kv_event_rollback_slots_runtime(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> {let code=e.context.borrow().kv_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_capture_view_runtime(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_free_sequence_runtime(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_reserve_runtime(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_recurrent_event_rollback_slots_runtime(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> {let code=e.context.borrow().recurrent_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_rollback_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().rollback_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_rollback_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {let code=e.context.borrow().rollback_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn mark_error_from_rollback_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {let code=e.context.borrow().rollback_error;let err=api_error(code);e.context.borrow_mut().err=err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_allocate_sequence_runtime(&mut self,e:&EventAllocateSequenceRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_allocate_slots_runtime(&mut self,e:&EventAllocateSlotsRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);if let Some(o)=e.block_count_out{*o.borrow_mut()=0;}Ok(())}
    fn publish_done_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_branch_sequence_runtime(&mut self,e:&EventBranchSequenceRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_capture_view_runtime(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_capture_view_runtime(&mut self,e:&EventCaptureViewRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_free_sequence_runtime(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_free_sequence_runtime(&mut self,e:&EventFreeSequenceRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_reserve_runtime(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_reserve_runtime(&mut self,e:&EventReserveRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);Ok(())}
    fn publish_done_event_rollback_slots_runtime(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> {e.context.borrow_mut().err=HybridError::None;set_error(e.error_out,HybridError::None);Ok(())}
    fn publish_error_event_rollback_slots_runtime(&mut self,e:&EventRollbackSlotsRuntime<'_>)->Result<(),()> {let err=e.context.borrow().err;set_error(e.error_out,err);if let Some(o)=e.block_count_out{*o.borrow_mut()=0;}Ok(())}
    fn on_unexpected_from_allocate_sequence_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_recurrent_error_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_rollback_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_sequence_rollback_result_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_recurrent_error_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_rollback_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_allocate_slots_rollback_result_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_recurrent_error_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_rollback_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_branch_sequence_rollback_result_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_merge(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_capture_request_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_done(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_errored(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_free_sequence_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_free_sequence_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_free_sequence_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_free_sequence_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_out_of_memory(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_ready(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_reserve_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_reserve_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_reserve_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_reserve_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_rollback_slots_kv(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_rollback_slots_kv_decision(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_rollback_slots_recurrent(&mut self)->Result<(),()>{Ok(())}
    fn on_unexpected_from_rollback_slots_recurrent_decision(&mut self)->Result<(),()>{Ok(())}
    fn guard_bound_kv_cache_event_allocate_slots_runtime(&self, _: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { Ok(matches!(self.kv_route, KvCacheRoute::Bound)) }
    fn guard_bound_kv_cache_event_capture_view_runtime(&self, _: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> { Ok(matches!(self.kv_route, KvCacheRoute::Bound)) }
    fn guard_owned_kv_cache_event_allocate_slots_runtime(&self, _: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { Ok(matches!(self.kv_route, KvCacheRoute::Owned)) }
    fn guard_owned_kv_cache_event_capture_view_runtime(&self, _: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> { Ok(matches!(self.kv_route, KvCacheRoute::Owned)) }
    fn mark_invalid_request(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> { e.context.borrow_mut().err = HybridError::InvalidRequest; set_error(e.error_out, HybridError::InvalidRequest); Ok(()) }
}

pub type Hybrid = MemoryHybridStateMachine<MemoryHybridContext>;
