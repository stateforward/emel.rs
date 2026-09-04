//! Source-aligned bounded hybrid memory state machine.

// The SML transition DSL intentionally uses assignment-like `<=` expressions;
// this generated state-machine syntax triggers Clippy's formatting heuristic.
#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    clippy::suspicious_assignment_formatting,
    dead_code,
    missing_docs
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
    #[default]
    None = 0,
    InvalidRequest = 1,
    BackendError = 2,
    InternalError = 4,
    OutOfMemory = 8,
    Untracked = 16,
}
impl HybridError {
    const fn code(self) -> i32 {
        self as i32
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReserveContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSequenceContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) rollback_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
    pub(crate) rollback_error: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSlotsContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) rollback_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
    pub(crate) rollback_error: i32,
    pub(crate) kv_block_count: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BranchSequenceContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) rollback_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
    pub(crate) rollback_error: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeSequenceContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RollbackSlotsContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
    pub(crate) kv_block_count: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureViewContext {
    pub(crate) err: HybridError,
    pub(crate) kv_accepted: bool,
    pub(crate) recurrent_accepted: bool,
    pub(crate) kv_error: i32,
    pub(crate) recurrent_error: i32,
}

#[derive(Clone)]
pub struct EventAllocateSequenceRuntime<'event> {
    pub seq_id: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<AllocateSequenceContext>,
}
#[derive(Clone)]
pub struct EventAllocateSlotsRuntime<'event> {
    pub seq_id: i32,
    pub token_count: i32,
    pub block_count_out: Option<&'event RefCell<i32>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub copy_block: Option<&'event dyn kv::BlockCopier>,
    pub context: &'event RefCell<AllocateSlotsContext>,
}
#[derive(Clone)]
pub struct EventBranchSequenceRuntime<'event> {
    pub parent_seq_id: i32,
    pub child_seq_id: i32,
    pub copy_state: Option<&'event dyn recurrent::StateCopier>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<BranchSequenceContext>,
}
#[derive(Clone)]
pub struct EventCaptureViewRuntime<'event> {
    pub snapshot_out: Option<&'event RefCell<Snapshot>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<CaptureViewContext>,
}
#[derive(Clone)]
pub struct EventFreeSequenceRuntime<'event> {
    pub seq_id: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<FreeSequenceContext>,
}
#[derive(Clone)]
pub struct EventReserveRuntime<'event> {
    pub max_sequences: i32,
    pub max_blocks: i32,
    pub block_tokens: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<ReserveContext>,
}
#[derive(Clone)]
pub struct EventRollbackSlotsRuntime<'event> {
    pub seq_id: i32,
    pub token_count: i32,
    pub block_count_out: Option<&'event RefCell<i32>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<RollbackSlotsContext>,
}
macro_rules! event_debug { ($($name:ident),+ $(,)?) => { $(impl core::fmt::Debug for $name<'_> { fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct(stringify!($name)).finish() } })+ }; }
event_debug!(
    EventAllocateSequenceRuntime,
    EventAllocateSlotsRuntime,
    EventBranchSequenceRuntime,
    EventCaptureViewRuntime,
    EventFreeSequenceRuntime,
    EventReserveRuntime,
    EventRollbackSlotsRuntime
);

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
pub enum KvCacheRoute {
    Owned,
    Bound,
}

/// Public, safe boundary implemented by injected KV actors.
pub trait HybridKvActor {
    fn reserve(
        &mut self,
        max_sequences: i32,
        max_blocks: i32,
        block_tokens: i32,
        error_out: &RefCell<i32>,
    ) -> bool;
    fn allocate_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool;
    fn allocate_slots(
        &mut self,
        seq_id: i32,
        token_count: i32,
        block_count_out: &RefCell<i32>,
        error_out: &RefCell<i32>,
        copy_block: Option<&dyn kv::BlockCopier>,
    ) -> bool;
    fn free_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool;
    fn rollback_slots(
        &mut self,
        seq_id: i32,
        token_count: i32,
        block_count_out: &RefCell<i32>,
        error_out: &RefCell<i32>,
    ) -> bool;
    fn capture_view(&mut self, snapshot_out: &RefCell<Snapshot>, error_out: &RefCell<i32>) -> bool;
    fn branch_sequence(
        &mut self,
        parent_seq_id: i32,
        child_seq_id: i32,
        error_out: &RefCell<i32>,
    ) -> bool;
}

impl HybridKvActor for kv::MemoryKvStateMachine<kv::MemoryKvContext> {
    fn reserve(
        &mut self,
        max_sequences: i32,
        max_blocks: i32,
        block_tokens: i32,
        error_out: &RefCell<i32>,
    ) -> bool {
        let c = RefCell::new(kv::ReserveContext::default());
        self.process_event(kv::EventReserveRuntime {
            max_sequences,
            max_blocks,
            block_tokens,
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
    fn allocate_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool {
        let c = RefCell::new(kv::AllocateSequenceContext::default());
        self.process_event(kv::EventAllocateSequenceRuntime {
            seq_id,
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
    fn allocate_slots(
        &mut self,
        seq_id: i32,
        token_count: i32,
        block_count_out: &RefCell<i32>,
        error_out: &RefCell<i32>,
        copy_block: Option<&dyn kv::BlockCopier>,
    ) -> bool {
        let c = RefCell::new(kv::AllocateSlotsContext::default());
        self.process_event(kv::EventAllocateSlotsRuntime {
            seq_id,
            token_count,
            block_count_out: Some(block_count_out),
            error_out: Some(error_out),
            copy_block,
            context: &c,
        })
        .is_ok()
    }
    fn free_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool {
        let c = RefCell::new(kv::FreeSequenceContext::default());
        self.process_event(kv::EventFreeSequenceRuntime {
            seq_id,
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
    fn rollback_slots(
        &mut self,
        seq_id: i32,
        token_count: i32,
        block_count_out: &RefCell<i32>,
        error_out: &RefCell<i32>,
    ) -> bool {
        let c = RefCell::new(kv::RollbackSlotsContext::default());
        self.process_event(kv::EventRollbackSlotsRuntime {
            seq_id,
            token_count,
            block_count_out: Some(block_count_out),
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
    fn capture_view(&mut self, snapshot_out: &RefCell<Snapshot>, error_out: &RefCell<i32>) -> bool {
        let c = RefCell::new(kv::CaptureViewContext::default());
        self.process_event(kv::EventCaptureViewRuntime {
            snapshot_out: Some(snapshot_out),
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
    fn branch_sequence(
        &mut self,
        parent_seq_id: i32,
        child_seq_id: i32,
        error_out: &RefCell<i32>,
    ) -> bool {
        let c = RefCell::new(kv::BranchSequenceContext::default());
        self.process_event(kv::EventBranchSequenceRuntime {
            parent_seq_id,
            child_seq_id,
            copy_state: None,
            error_out: Some(error_out),
            context: &c,
        })
        .is_ok()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvBindingKind {
    Empty,
    Complete,
    Invalid,
}

#[derive(Clone)]
pub struct KvBinding {
    actor: Option<std::rc::Rc<RefCell<dyn HybridKvActor>>>,
    kind: KvBindingKind,
}
impl core::fmt::Debug for KvBinding {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("KvBinding")
            .field("kind", &self.kind)
            .field("actor_bound", &self.actor.is_some())
            .finish()
    }
}

impl Default for KvBinding {
    fn default() -> Self {
        Self::empty()
    }
}
impl KvBinding {
    pub fn empty() -> Self {
        Self {
            actor: Some(std::rc::Rc::new(RefCell::new(
                kv::MemoryKvStateMachine::new(kv::MemoryKvContext::default()),
            ))),
            kind: KvBindingKind::Empty,
        }
    }
    pub fn invalid() -> Self {
        Self {
            actor: None,
            kind: KvBindingKind::Invalid,
        }
    }
    pub fn from_shared(actor: std::rc::Rc<RefCell<dyn HybridKvActor>>) -> Self {
        Self {
            actor: Some(actor),
            kind: KvBindingKind::Complete,
        }
    }
    pub fn from_actor<A: HybridKvActor + 'static>(actor: A) -> Self {
        Self::from_shared(std::rc::Rc::new(RefCell::new(actor)))
    }
    fn route(&self) -> KvCacheRoute {
        if matches!(self.kind, KvBindingKind::Empty) {
            KvCacheRoute::Owned
        } else {
            KvCacheRoute::Bound
        }
    }
    fn is_invalid(&self) -> bool {
        matches!(self.kind, KvBindingKind::Invalid)
    }
}

pub struct MemoryHybridContext {
    pub recurrent: recurrent::MemoryRecurrentStateMachine<recurrent::MemoryRecurrentContext>,
    pub kv_route: KvCacheRoute,
    pub kv_binding: KvBinding,
    pub kv_snapshot: RefCell<Snapshot>,
    pub recurrent_snapshot: RefCell<recurrent::Snapshot>,
}
impl core::fmt::Debug for MemoryHybridContext {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MemoryHybridContext")
            .field("kv_route", &self.kv_route)
            .field("kv_binding", &self.kv_binding)
            .field("kv_snapshot", &self.kv_snapshot)
            .field("recurrent_snapshot", &self.recurrent_snapshot)
            .finish_non_exhaustive()
    }
}

impl Default for MemoryHybridContext {
    fn default() -> Self {
        let binding = KvBinding::empty();
        Self {
            recurrent: recurrent::MemoryRecurrentStateMachine::new(
                recurrent::MemoryRecurrentContext::default(),
            ),
            kv_route: binding.route(),
            kv_binding: binding,
            kv_snapshot: RefCell::new(Snapshot::default()),
            recurrent_snapshot: RefCell::new(recurrent::Snapshot::default()),
        }
    }
}
impl MemoryHybridContext {
    pub fn with_kv_binding(binding: KvBinding) -> Self {
        let kv_route = binding.route();
        Self {
            recurrent: recurrent::MemoryRecurrentStateMachine::new(
                recurrent::MemoryRecurrentContext::default(),
            ),
            kv_route,
            kv_binding: binding,
            kv_snapshot: RefCell::new(Snapshot::default()),
            recurrent_snapshot: RefCell::new(recurrent::Snapshot::default()),
        }
    }
    fn bound_actor(&self) -> Option<std::rc::Rc<RefCell<dyn HybridKvActor>>> {
        self.kv_binding.actor.clone()
    }
    fn kv_free_sequence(&self, seq_id: i32, err: &RefCell<i32>) -> bool {
        if self.kv_binding.is_invalid() {
            *err.borrow_mut() = HybridError::BackendError.code();
            return false;
        }
        self.bound_actor()
            .is_some_and(|actor| actor.borrow_mut().free_sequence(seq_id, err))
    }
    fn kv_rollback_slots(
        &self,
        seq_id: i32,
        token_count: i32,
        count: &RefCell<i32>,
        err: &RefCell<i32>,
    ) -> bool {
        if self.kv_binding.is_invalid() {
            *err.borrow_mut() = HybridError::BackendError.code();
            return false;
        }
        self.bound_actor().is_some_and(|actor| {
            actor
                .borrow_mut()
                .rollback_slots(seq_id, token_count, count, err)
        })
    }
    fn kv_branch(&self, parent_seq_id: i32, child_seq_id: i32, err: &RefCell<i32>) -> bool {
        if self.kv_binding.is_invalid() {
            *err.borrow_mut() = HybridError::BackendError.code();
            return false;
        }
        self.bound_actor().is_some_and(|actor| {
            actor
                .borrow_mut()
                .branch_sequence(parent_seq_id, child_seq_id, err)
        })
    }
}

fn set_error(out: Option<&RefCell<i32>>, error: HybridError) {
    if let Some(out) = out {
        *out.borrow_mut() = error.code();
    }
}
fn api_error(code: i32) -> HybridError {
    match code {
        1 => HybridError::InvalidRequest,
        2 => HybridError::BackendError,
        8 => HybridError::OutOfMemory,
        16 => HybridError::Untracked,
        _ => HybridError::InternalError,
    }
}
fn backend_family(code: i32) -> bool {
    matches!(code, 0 | 2 | 8)
}
fn backend_or_none(code: i32) -> bool {
    matches!(code, 0 | 2)
}
fn merge(kv: &Snapshot, rec: &recurrent::Snapshot, out: &mut Snapshot) {
    out.max_sequences = 0;
    out.block_tokens = kv::DEFAULT_BLOCK_TOKENS;
    out.sequence_active.fill(0);
    out.sequence_length_values.fill(0);
    out.sequence_kv_block_count.fill(0);
    for row in &mut out.sequence_kv_blocks {
        row.fill(0);
    }
    out.sequence_recurrent_slot.fill(0);
    out.max_sequences = kv.max_sequences.min(rec.max_sequences);
    out.block_tokens = kv.block_tokens;
    let count = usize::try_from(out.max_sequences.max(0))
        .unwrap_or(0)
        .min(MAX_SEQUENCES);
    for i in 0..count {
        let active = kv.sequence_active[i] != 0 && rec.sequence_active[i] != 0;
        out.sequence_active[i] = u8::from(active);
        out.sequence_length_values[i] = if active {
            kv.sequence_length_values[i].min(rec.sequence_length_values[i])
        } else {
            0
        };
        out.sequence_kv_block_count[i] = if active {
            kv.sequence_kv_block_count[i]
        } else {
            0
        };
        out.sequence_kv_blocks[i] = kv.sequence_kv_blocks[i];
        out.sequence_recurrent_slot[i] = if active {
            rec.sequence_recurrent_slot[i]
        } else {
            -1
        };
    }
}
fn effect_allocate_slots_impl(ctx: &MemoryHybridContext, e: &EventAllocateSlotsRuntime<'_>) {
    let err = RefCell::new(0);
    let count = RefCell::new(0);
    let ok = if ctx.kv_binding.is_invalid() {
        *err.borrow_mut() = HybridError::BackendError.code();
        false
    } else if let Some(actor) = ctx.bound_actor() {
        actor
            .borrow_mut()
            .allocate_slots(e.seq_id, e.token_count, &count, &err, e.copy_block)
    } else {
        false
    };
    let mut h = e.context.borrow_mut();
    h.kv_accepted = ok && *err.borrow() == 0;
    h.kv_error = *err.borrow();
    h.kv_block_count = *count.borrow();
    if let Some(out) = e.block_count_out {
        *out.borrow_mut() = h.kv_block_count;
    }
}
fn effect_capture_impl(ctx: &MemoryHybridContext, e: &EventCaptureViewRuntime<'_>) {
    let err = RefCell::new(0);
    let ok = if ctx.kv_binding.is_invalid() {
        *err.borrow_mut() = HybridError::BackendError.code();
        false
    } else if let Some(actor) = ctx.bound_actor() {
        actor.borrow_mut().capture_view(&ctx.kv_snapshot, &err)
    } else {
        false
    };
    let mut h = e.context.borrow_mut();
    h.kv_accepted = ok && *err.borrow() == 0;
    h.kv_error = *err.borrow();
}

impl MemoryHybridStateMachineContext for MemoryHybridContext {
    fn begin_allocate_sequence(&mut self, e: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = AllocateSequenceContext::default();
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_reserve(&mut self, e: &EventReserveRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = ReserveContext::default();
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_allocate_slots(&mut self, e: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = AllocateSlotsContext::default();
        if let Some(o) = e.block_count_out {
            *o.borrow_mut() = 0;
        }
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_branch_sequence(&mut self, e: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = BranchSequenceContext::default();
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_free_sequence(&mut self, e: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = FreeSequenceContext::default();
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_rollback_slots(&mut self, e: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = RollbackSlotsContext::default();
        if let Some(o) = e.block_count_out {
            *o.borrow_mut() = 0;
        }
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn begin_capture_view(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        *e.context.borrow_mut() = CaptureViewContext::default();
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn capture_request_valid(&self, e: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> {
        Ok(e.snapshot_out.is_some())
    }
    fn capture_request_invalid(&self, e: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> {
        Ok(e.snapshot_out.is_none())
    }
    fn exec_reserve_kv(&mut self, e: &EventReserveRuntime<'_>) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = if self.kv_binding.is_invalid() {
            *err.borrow_mut() = HybridError::BackendError.code();
            false
        } else {
            self.bound_actor().is_some_and(|actor| {
                actor
                    .borrow_mut()
                    .reserve(e.max_sequences, e.max_blocks, e.block_tokens, &err)
            })
        };
        let mut h = e.context.borrow_mut();
        h.kv_accepted = ok && *err.borrow() == 0;
        h.kv_error = *err.borrow();
        Ok(())
    }
    fn exec_reserve_recurrent(&mut self, e: &EventReserveRuntime<'_>) -> Result<(), ()> {
        let c = RefCell::new(recurrent::ReserveContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventReserveRuntime {
                max_sequences: e.max_sequences,
                max_blocks: e.max_blocks,
                block_tokens: e.block_tokens,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn exec_allocate_sequence_kv(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = if self.kv_binding.is_invalid() {
            *err.borrow_mut() = HybridError::BackendError.code();
            false
        } else {
            self.bound_actor()
                .is_some_and(|actor| actor.borrow_mut().allocate_sequence(e.seq_id, &err))
        };
        let mut h = e.context.borrow_mut();
        h.kv_accepted = ok && *err.borrow() == 0;
        h.kv_error = *err.borrow();
        Ok(())
    }
    fn exec_allocate_sequence_recurrent(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let c = RefCell::new(recurrent::AllocateSequenceContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventAllocateSequenceRuntime {
                seq_id: e.seq_id,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn exec_allocate_sequence_rollback_kv(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = self.kv_free_sequence(e.seq_id, &err);
        let mut h = e.context.borrow_mut();
        h.rollback_accepted = ok && *err.borrow() == 0;
        h.rollback_error = *err.borrow();
        Ok(())
    }
    fn exec_rollback_slots_recurrent(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let c = RefCell::new(recurrent::RollbackSlotsContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventRollbackSlotsRuntime {
                seq_id: e.seq_id,
                token_count: e.token_count,
                block_count_out: None,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn effect_allocate_slots_owned_kv(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        effect_allocate_slots_impl(self, e);
        Ok(())
    }
    fn effect_allocate_slots_bound_kv(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        effect_allocate_slots_impl(self, e);
        Ok(())
    }
    fn exec_allocate_slots_recurrent(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let c = RefCell::new(recurrent::AllocateSlotsContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventAllocateSlotsRuntime {
                seq_id: e.seq_id,
                token_count: e.token_count,
                block_count_out: None,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn exec_allocate_slots_rollback_kv(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = self.kv_rollback_slots(e.seq_id, e.token_count, &RefCell::new(0), &err);
        let mut h = e.context.borrow_mut();
        h.rollback_accepted = ok && *err.borrow() == 0;
        h.rollback_error = *err.borrow();
        Ok(())
    }
    fn exec_branch_sequence_kv(&mut self, e: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = self.kv_branch(e.parent_seq_id, e.child_seq_id, &err);
        let mut h = e.context.borrow_mut();
        h.kv_accepted = ok && *err.borrow() == 0;
        h.kv_error = *err.borrow();
        Ok(())
    }
    fn exec_branch_sequence_recurrent(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let c = RefCell::new(recurrent::BranchSequenceContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventBranchSequenceRuntime {
                parent_seq_id: e.parent_seq_id,
                child_seq_id: e.child_seq_id,
                copy_state: e.copy_state,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn exec_branch_sequence_rollback_kv(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = self.kv_free_sequence(e.child_seq_id, &err);
        let mut h = e.context.borrow_mut();
        h.rollback_accepted = ok && *err.borrow() == 0;
        h.rollback_error = *err.borrow();
        Ok(())
    }
    fn exec_free_sequence_kv(&mut self, e: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> {
        let err = RefCell::new(0);
        let ok = self.kv_free_sequence(e.seq_id, &err);
        let mut h = e.context.borrow_mut();
        h.kv_accepted = ok && *err.borrow() == 0;
        h.kv_error = *err.borrow();
        Ok(())
    }
    fn exec_free_sequence_recurrent(&mut self, e: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> {
        let c = RefCell::new(recurrent::FreeSequenceContext::default());
        let err = RefCell::new(0);
        let ok = self
            .recurrent
            .process_event(recurrent::EventFreeSequenceRuntime {
                seq_id: e.seq_id,
                error_out: Some(&err),
                context: &c,
            })
            .is_ok()
            && *err.borrow() == 0;
        let mut h = e.context.borrow_mut();
        h.recurrent_accepted = ok;
        h.recurrent_error = *err.borrow();
        Ok(())
    }
    fn exec_rollback_slots_kv(&mut self, e: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> {
        let count = RefCell::new(0);
        let err = RefCell::new(0);
        let ok = self.kv_rollback_slots(e.seq_id, e.token_count, &count, &err);
        let mut h = e.context.borrow_mut();
        h.kv_accepted = ok && *err.borrow() == 0;
        h.kv_error = *err.borrow();
        h.kv_block_count = *count.borrow();
        if let Some(o) = e.block_count_out {
            *o.borrow_mut() = *count.borrow();
        }
        Ok(())
    }
    fn effect_capture_owned_kv(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        effect_capture_impl(self, e);
        Ok(())
    }
    fn effect_capture_bound_kv(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        effect_capture_impl(self, e);
        Ok(())
    }
    fn exec_capture_recurrent(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        let context = RefCell::new(recurrent::CaptureViewContext::default());
        let error = RefCell::new(0);
        let accepted = self
            .recurrent
            .process_event(recurrent::EventCaptureViewRuntime {
                snapshot_out: Some(&self.recurrent_snapshot),
                error_out: Some(&error),
                context: &context,
            })
            .is_ok()
            && *error.borrow() == 0;
        let mut hybrid = e.context.borrow_mut();
        hybrid.recurrent_accepted = accepted;
        hybrid.recurrent_error = *error.borrow();
        Ok(())
    }
    fn merge_capture_snapshots(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        if let Some(out) = e.snapshot_out {
            merge(
                &self.kv_snapshot.borrow(),
                &self.recurrent_snapshot.borrow(),
                &mut out.borrow_mut(),
            );
        }
        Ok(())
    }
    fn kv_accepted_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn kv_rejected_with_error_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0)
    }
    fn kv_rejected_without_error_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 0)
    }
    fn recurrent_accepted_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn kv_accepted_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn recurrent_accepted_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn kv_accepted_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn recurrent_accepted_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn kv_accepted_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn kv_rejected_with_error_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0)
    }
    fn kv_rejected_without_error_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 0)
    }
    fn recurrent_accepted_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_with_error_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0)
    }
    fn recurrent_rejected_without_error_event_capture_view_runtime(
        &self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 0)
    }
    fn kv_accepted_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn kv_rejected_with_error_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0)
    }
    fn kv_rejected_without_error_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 0)
    }
    fn recurrent_accepted_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_with_error_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0)
    }
    fn recurrent_rejected_without_error_event_free_sequence_runtime(
        &self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 0)
    }
    fn kv_accepted_event_reserve_runtime(&self, e: &EventReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn kv_rejected_with_error_event_reserve_runtime(
        &self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0)
    }
    fn kv_rejected_without_error_event_reserve_runtime(
        &self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 0)
    }
    fn recurrent_accepted_event_reserve_runtime(
        &self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_with_error_event_reserve_runtime(
        &self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0)
    }
    fn recurrent_rejected_without_error_event_reserve_runtime(
        &self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 0)
    }
    fn kv_accepted_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().kv_accepted)
    }
    fn kv_rejected_with_error_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0)
    }
    fn kv_rejected_without_error_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 0)
    }
    fn recurrent_accepted_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_with_error_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0)
    }
    fn recurrent_rejected_without_error_event_rollback_slots_runtime(
        &self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 0)
    }
    fn recurrent_rejected_any_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_any_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_any_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!e.context.borrow().recurrent_accepted)
    }
    fn recurrent_rejected_out_of_memory_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 8)
    }
    fn recurrent_rejected_backend_or_none_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))
    }
    fn recurrent_rejected_non_backend_error_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0 && !backend_family(c.recurrent_error))
    }
    fn kv_rejected_out_of_memory_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 8)
    }
    fn recurrent_rejected_out_of_memory_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 8)
    }
    fn kv_rejected_backend_or_none_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && backend_or_none(c.kv_error))
    }
    fn recurrent_rejected_backend_or_none_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))
    }
    fn kv_rejected_non_backend_error_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0 && !backend_family(c.kv_error))
    }
    fn recurrent_rejected_non_backend_error_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0 && !backend_family(c.recurrent_error))
    }
    fn kv_rejected_out_of_memory_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error == 8)
    }
    fn recurrent_rejected_out_of_memory_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error == 8)
    }
    fn kv_rejected_backend_or_none_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && backend_or_none(c.kv_error))
    }
    fn recurrent_rejected_backend_or_none_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && backend_or_none(c.recurrent_error))
    }
    fn kv_rejected_non_backend_error_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.kv_accepted && c.kv_error != 0 && !backend_family(c.kv_error))
    }
    fn recurrent_rejected_non_backend_error_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.recurrent_accepted && c.recurrent_error != 0 && !backend_family(c.recurrent_error))
    }
    fn rollback_accepted_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().rollback_accepted)
    }
    fn rollback_rejected_with_error_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error != 0)
    }
    fn rollback_rejected_without_error_event_allocate_sequence_runtime(
        &self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error == 0)
    }
    fn rollback_accepted_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().rollback_accepted)
    }
    fn rollback_rejected_with_error_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error != 0)
    }
    fn rollback_rejected_without_error_event_allocate_slots_runtime(
        &self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error == 0)
    }
    fn rollback_accepted_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(e.context.borrow().rollback_accepted)
    }
    fn rollback_rejected_with_error_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error != 0)
    }
    fn rollback_rejected_without_error_event_branch_sequence_runtime(
        &self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!c.rollback_accepted && c.rollback_error == 0)
    }
    fn mark_backend_error_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_capture_view_runtime(
        &mut self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_free_sequence_runtime(
        &mut self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_reserve_runtime(
        &mut self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_backend_error_event_rollback_slots_runtime(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::BackendError;
        set_error(e.error_out, HybridError::BackendError);
        Ok(())
    }
    fn mark_out_of_memory_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::OutOfMemory;
        set_error(e.error_out, HybridError::OutOfMemory);
        Ok(())
    }
    fn mark_out_of_memory_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::OutOfMemory;
        set_error(e.error_out, HybridError::OutOfMemory);
        Ok(())
    }
    fn mark_out_of_memory_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::OutOfMemory;
        set_error(e.error_out, HybridError::OutOfMemory);
        Ok(())
    }
    fn mark_internal_error_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::InternalError;
        set_error(e.error_out, HybridError::InternalError);
        Ok(())
    }
    fn mark_internal_error_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::InternalError;
        set_error(e.error_out, HybridError::InternalError);
        Ok(())
    }
    fn mark_internal_error_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::InternalError;
        set_error(e.error_out, HybridError::InternalError);
        Ok(())
    }
    fn mark_error_from_kv_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_capture_view_runtime(
        &mut self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_free_sequence_runtime(
        &mut self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_reserve_runtime(
        &mut self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_kv_event_rollback_slots_runtime(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().kv_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_capture_view_runtime(
        &mut self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_free_sequence_runtime(
        &mut self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_reserve_runtime(
        &mut self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_recurrent_event_rollback_slots_runtime(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().recurrent_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_rollback_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().rollback_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_rollback_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().rollback_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn mark_error_from_rollback_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let code = e.context.borrow().rollback_error;
        let err = api_error(code);
        e.context.borrow_mut().err = err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_allocate_sequence_runtime(
        &mut self,
        e: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_allocate_slots_runtime(
        &mut self,
        e: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        if let Some(o) = e.block_count_out {
            *o.borrow_mut() = 0;
        }
        Ok(())
    }
    fn publish_done_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_branch_sequence_runtime(
        &mut self,
        e: &EventBranchSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_capture_view_runtime(
        &mut self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_capture_view_runtime(
        &mut self,
        e: &EventCaptureViewRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_free_sequence_runtime(
        &mut self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_free_sequence_runtime(
        &mut self,
        e: &EventFreeSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_reserve_runtime(
        &mut self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_reserve_runtime(
        &mut self,
        e: &EventReserveRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        Ok(())
    }
    fn publish_done_event_rollback_slots_runtime(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::None;
        set_error(e.error_out, HybridError::None);
        Ok(())
    }
    fn publish_error_event_rollback_slots_runtime(
        &mut self,
        e: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let err = e.context.borrow().err;
        set_error(e.error_out, err);
        if let Some(o) = e.block_count_out {
            *o.borrow_mut() = 0;
        }
        Ok(())
    }
    fn on_unexpected_from_allocate_sequence_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_recurrent_error_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_rollback_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_sequence_rollback_result_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_recurrent_error_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_rollback_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_allocate_slots_rollback_result_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_recurrent_error_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_rollback_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_branch_sequence_rollback_result_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_merge(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_capture_request_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_free_sequence_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_free_sequence_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_free_sequence_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_free_sequence_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_out_of_memory(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_reserve_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_reserve_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_reserve_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_reserve_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_rollback_slots_kv(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_rollback_slots_kv_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_rollback_slots_recurrent(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn on_unexpected_from_rollback_slots_recurrent_decision(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn guard_bound_kv_cache_event_allocate_slots_runtime(
        &self,
        _: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(self.kv_route, KvCacheRoute::Bound))
    }
    fn guard_bound_kv_cache_event_capture_view_runtime(
        &self,
        _: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(self.kv_route, KvCacheRoute::Bound))
    }
    fn guard_owned_kv_cache_event_allocate_slots_runtime(
        &self,
        _: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(self.kv_route, KvCacheRoute::Owned))
    }
    fn guard_owned_kv_cache_event_capture_view_runtime(
        &self,
        _: &EventCaptureViewRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(self.kv_route, KvCacheRoute::Owned))
    }
    fn mark_invalid_request(&mut self, e: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        e.context.borrow_mut().err = HybridError::InvalidRequest;
        set_error(e.error_out, HybridError::InvalidRequest);
        Ok(())
    }
}

pub type Hybrid = MemoryHybridStateMachine<MemoryHybridContext>;
#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[derive(Default)]
    struct SpyKv {
        calls: [u32; 7],
        active: [bool; 3],
        capture_count: u32,
    }

    impl HybridKvActor for SpyKv {
        fn reserve(&mut self, _: i32, _: i32, _: i32, error_out: &RefCell<i32>) -> bool {
            self.calls[0] += 1;
            *error_out.borrow_mut() = 0;
            true
        }
        fn allocate_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool {
            self.calls[1] += 1;
            if let Ok(index) = usize::try_from(seq_id)
                && let Some(active) = self.active.get_mut(index)
            {
                *active = true;
            }
            *error_out.borrow_mut() = 0;
            true
        }
        fn allocate_slots(
            &mut self,
            _: i32,
            _: i32,
            count: &RefCell<i32>,
            error_out: &RefCell<i32>,
            _: Option<&dyn kv::BlockCopier>,
        ) -> bool {
            self.calls[2] += 1;
            *count.borrow_mut() = 7;
            *error_out.borrow_mut() = 0;
            true
        }
        fn free_sequence(&mut self, seq_id: i32, error_out: &RefCell<i32>) -> bool {
            self.calls[3] += 1;
            if let Ok(index) = usize::try_from(seq_id)
                && let Some(active) = self.active.get_mut(index)
            {
                *active = false;
            }
            *error_out.borrow_mut() = 0;
            true
        }
        fn rollback_slots(
            &mut self,
            _: i32,
            _: i32,
            count: &RefCell<i32>,
            error_out: &RefCell<i32>,
        ) -> bool {
            self.calls[4] += 1;
            *count.borrow_mut() = 3;
            *error_out.borrow_mut() = 0;
            true
        }
        fn capture_view(
            &mut self,
            snapshot_out: &RefCell<Snapshot>,
            error_out: &RefCell<i32>,
        ) -> bool {
            self.calls[5] += 1;
            self.capture_count += 1;
            let mut snapshot = snapshot_out.borrow_mut();
            let max_sequences = if self.capture_count == 1 { 3 } else { 1 };
            snapshot.max_sequences = max_sequences;
            snapshot.block_tokens = 16;
            for index in 0..usize::try_from(max_sequences).unwrap_or(0) {
                let active = self.active[index];
                snapshot.sequence_active[index] = u8::from(active);
                snapshot.sequence_length_values[index] = if active {
                    4 + i32::try_from(index).unwrap_or(0)
                } else {
                    0
                };
                snapshot.sequence_kv_block_count[index] = if active { 7 } else { 0 };
                if active {
                    snapshot.sequence_kv_blocks[index][0] = u16::try_from(100 + index).unwrap_or(0);
                    snapshot.sequence_kv_blocks[index][5] = u16::try_from(200 + index).unwrap_or(0);
                }
            }
            *error_out.borrow_mut() = 0;
            true
        }
        fn branch_sequence(&mut self, _: i32, child_seq_id: i32, error_out: &RefCell<i32>) -> bool {
            self.calls[6] += 1;
            if let Ok(index) = usize::try_from(child_seq_id)
                && let Some(active) = self.active.get_mut(index)
            {
                *active = true;
            }
            *error_out.borrow_mut() = 0;
            true
        }
    }
    #[test]
    fn injected_binding_receives_all_kv_routes_and_capture_survives_merge() {
        let actor = Rc::new(RefCell::new(SpyKv::default()));
        let mut machine = Hybrid::new(MemoryHybridContext::with_kv_binding(
            KvBinding::from_shared(actor.clone()),
        ));
        let error = RefCell::new(-1);
        let reserve_context = RefCell::new(ReserveContext::default());
        machine
            .process_event(EventReserveRuntime {
                max_sequences: 2,
                max_blocks: 2,
                block_tokens: 16,
                error_out: Some(&error),
                context: &reserve_context,
            })
            .unwrap();
        let sequence_context = RefCell::new(AllocateSequenceContext::default());
        machine
            .process_event(EventAllocateSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &sequence_context,
            })
            .unwrap();
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        let block_count = RefCell::new(-1);
        machine
            .process_event(EventAllocateSlotsRuntime {
                seq_id: 0,
                token_count: 4,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                copy_block: None,
                context: &slots_context,
            })
            .unwrap();
        assert_eq!(*block_count.borrow(), 7);
        let rollback_context = RefCell::new(RollbackSlotsContext::default());
        machine
            .process_event(EventRollbackSlotsRuntime {
                seq_id: 0,
                token_count: 1,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                context: &rollback_context,
            })
            .unwrap();
        assert_eq!(*block_count.borrow(), 3);
        let branch_context = RefCell::new(BranchSequenceContext::default());
        machine
            .process_event(EventBranchSequenceRuntime {
                parent_seq_id: 0,
                child_seq_id: 1,
                copy_state: None,
                error_out: Some(&error),
                context: &branch_context,
            })
            .unwrap();
        assert_eq!(*error.borrow(), HybridError::InvalidRequest.code());
        assert!(branch_context.borrow().kv_accepted);
        assert!(!branch_context.borrow().recurrent_accepted);
        assert!(branch_context.borrow().rollback_accepted);
        assert_eq!(branch_context.borrow().err, HybridError::InvalidRequest);
        assert!(!actor.borrow().active[1]);
        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        machine
            .process_event(EventCaptureViewRuntime {
                snapshot_out: Some(&snapshot),
                error_out: Some(&error),
                context: &capture_context,
            })
            .unwrap();
        assert!(capture_context.borrow().kv_accepted);
        assert!(capture_context.borrow().recurrent_accepted);
        assert_eq!(capture_context.borrow().err, HybridError::None);
        assert_eq!(snapshot.borrow().sequence_active[1], 0);
        assert_eq!(snapshot.borrow().sequence_kv_block_count[0], 7);
        assert_eq!(snapshot.borrow().sequence_length_values[0], 3);
        let free_context = RefCell::new(FreeSequenceContext::default());
        machine
            .process_event(EventFreeSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &free_context,
            })
            .unwrap();
        assert_eq!(actor.borrow().calls, [1, 1, 1, 2, 1, 1, 1]);
    }
    #[test]
    fn bound_kv_merge_uses_recurrent_length_bound() {
        let actor = Rc::new(RefCell::new(SpyKv::default()));
        let mut machine = Hybrid::new(MemoryHybridContext::with_kv_binding(
            KvBinding::from_shared(actor),
        ));
        let error = RefCell::new(-1);
        let reserve_context = RefCell::new(ReserveContext::default());
        machine
            .process_event(EventReserveRuntime {
                max_sequences: 1,
                max_blocks: 2,
                block_tokens: 16,
                error_out: Some(&error),
                context: &reserve_context,
            })
            .unwrap();
        let sequence_context = RefCell::new(AllocateSequenceContext::default());
        machine
            .process_event(EventAllocateSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &sequence_context,
            })
            .unwrap();
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        let block_count = RefCell::new(-1);
        machine
            .process_event(EventAllocateSlotsRuntime {
                seq_id: 0,
                token_count: 3,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                copy_block: None,
                context: &slots_context,
            })
            .unwrap();
        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        machine
            .process_event(EventCaptureViewRuntime {
                snapshot_out: Some(&snapshot),
                error_out: Some(&error),
                context: &capture_context,
            })
            .unwrap();

        assert_eq!(snapshot.borrow().sequence_length(0), 3);
    }

    fn assert_snapshot_tail_is_cleared(snapshot: &Snapshot) {
        assert!(
            snapshot.sequence_active[1..]
                .iter()
                .all(|&value| value == 0)
        );
        assert!(
            snapshot.sequence_length_values[1..]
                .iter()
                .all(|&value| value == 0)
        );
        assert!(
            snapshot.sequence_kv_block_count[1..]
                .iter()
                .all(|&value| value == 0)
        );
        assert!(
            snapshot.sequence_recurrent_slot[1..]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(
            snapshot.sequence_kv_blocks[1],
            [0; kv::MAX_BLOCKS_PER_SEQUENCE]
        );
        assert_eq!(
            snapshot.sequence_kv_blocks[2],
            [0; kv::MAX_BLOCKS_PER_SEQUENCE]
        );
    }

    #[test]
    fn repeated_capture_clears_tail_data_without_replacing_snapshot_storage() {
        let actor = Rc::new(RefCell::new(SpyKv::default()));
        let mut machine = Hybrid::new(MemoryHybridContext::with_kv_binding(
            KvBinding::from_shared(actor),
        ));
        let error = RefCell::new(-1);
        let reserve_context = RefCell::new(ReserveContext::default());
        machine
            .process_event(EventReserveRuntime {
                max_sequences: 3,
                max_blocks: 8,
                block_tokens: 16,
                error_out: Some(&error),
                context: &reserve_context,
            })
            .unwrap();
        for seq_id in 0..3 {
            let context = RefCell::new(AllocateSequenceContext::default());
            machine
                .process_event(EventAllocateSequenceRuntime {
                    seq_id,
                    error_out: Some(&error),
                    context: &context,
                })
                .unwrap();
        }

        for seq_id in 0..3 {
            let context = RefCell::new(AllocateSlotsContext::default());
            let block_count = RefCell::new(-1);
            machine
                .process_event(EventAllocateSlotsRuntime {
                    seq_id,
                    token_count: 8,
                    block_count_out: Some(&block_count),
                    error_out: Some(&error),
                    copy_block: None,
                    context: &context,
                })
                .unwrap();
        }
        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        machine
            .process_event(EventCaptureViewRuntime {
                snapshot_out: Some(&snapshot),
                error_out: Some(&error),
                context: &capture_context,
            })
            .unwrap();
        {
            let snapshot = snapshot.borrow();
            assert_eq!(snapshot.max_sequences, 3);
            assert_eq!(snapshot.sequence_active[2], 1);
            assert_eq!(snapshot.sequence_length_values[2], 6);
            assert_eq!(snapshot.sequence_kv_block_count[2], 7);
            assert_eq!(snapshot.sequence_kv_blocks[2][5], 202);
            assert_eq!(snapshot.sequence_recurrent_slot[2], 2);
        }
        let active_ptr = snapshot.borrow().sequence_active.as_ptr();
        let lengths_ptr = snapshot.borrow().sequence_length_values.as_ptr();
        let counts_ptr = snapshot.borrow().sequence_kv_block_count.as_ptr();
        let rows_ptr = snapshot.borrow().sequence_kv_blocks[2].as_ptr();
        let recurrent_ptr = snapshot.borrow().sequence_recurrent_slot.as_ptr();

        machine
            .process_event(EventCaptureViewRuntime {
                snapshot_out: Some(&snapshot),
                error_out: Some(&error),
                context: &capture_context,
            })
            .unwrap();
        let snapshot = snapshot.borrow();
        assert_eq!(snapshot.max_sequences, 1);
        assert_snapshot_tail_is_cleared(&snapshot);
        assert_eq!(snapshot.sequence_active.as_ptr(), active_ptr);
        assert_eq!(snapshot.sequence_length_values.as_ptr(), lengths_ptr);
        assert_eq!(snapshot.sequence_kv_block_count.as_ptr(), counts_ptr);
        assert_eq!(snapshot.sequence_kv_blocks[2].as_ptr(), rows_ptr);
        assert_eq!(snapshot.sequence_recurrent_slot.as_ptr(), recurrent_ptr);
    }

    fn assert_reserve(
        machine: &mut Hybrid,
        error: &RefCell<i32>,
        context: &RefCell<ReserveContext>,
    ) {
        assert!(
            machine
                .process_event(EventReserveRuntime {
                    max_sequences: 3,
                    max_blocks: 4,
                    block_tokens: 2,
                    error_out: Some(error),
                    context,
                })
                .is_ok()
        );
    }

    fn assert_allocate_sequence(
        machine: &mut Hybrid,
        seq_id: i32,
        error: &RefCell<i32>,
        context: &RefCell<AllocateSequenceContext>,
    ) {
        assert!(
            machine
                .process_event(EventAllocateSequenceRuntime {
                    seq_id,
                    error_out: Some(error),
                    context,
                })
                .is_ok()
        );
    }

    fn assert_allocate_slots(
        machine: &mut Hybrid,
        seq_id: i32,
        token_count: i32,
        error: &RefCell<i32>,
        context: &RefCell<AllocateSlotsContext>,
        blocks: &RefCell<i32>,
    ) {
        assert!(
            machine
                .process_event(EventAllocateSlotsRuntime {
                    seq_id,
                    token_count,
                    block_count_out: Some(blocks),
                    error_out: Some(error),
                    copy_block: None,
                    context,
                })
                .is_ok()
        );
        assert_eq!(*error.borrow(), HybridError::None.code());
        assert!(*blocks.borrow() >= 0);
    }

    fn assert_capture(
        machine: &mut Hybrid,
        error: &RefCell<i32>,
        snapshot: &RefCell<Snapshot>,
        context: &RefCell<CaptureViewContext>,
    ) {
        assert!(
            machine
                .process_event(EventCaptureViewRuntime {
                    snapshot_out: Some(snapshot),
                    error_out: Some(error),
                    context,
                })
                .is_ok()
        );
    }

    fn assert_free_sequence(
        machine: &mut Hybrid,
        seq_id: i32,
        error: &RefCell<i32>,
        context: &RefCell<FreeSequenceContext>,
    ) {
        assert!(
            machine
                .process_event(EventFreeSequenceRuntime {
                    seq_id,
                    error_out: Some(error),
                    context,
                })
                .is_ok()
        );
    }

    #[test]
    fn owned_lifecycle_tracks_isolated_blocks_and_recycles_after_free() {
        let mut machine = Hybrid::new(MemoryHybridContext::default());
        let error = RefCell::new(-1);
        let reserve_context = RefCell::new(ReserveContext::default());
        assert_reserve(&mut machine, &error, &reserve_context);
        for seq_id in [0, 1] {
            let context = RefCell::new(AllocateSequenceContext::default());
            assert_allocate_sequence(&mut machine, seq_id, &error, &context);
        }
        for (seq_id, token_count) in [(0, 3), (1, 2), (0, 1)] {
            let context = RefCell::new(AllocateSlotsContext::default());
            let blocks = RefCell::new(-1);
            assert_allocate_slots(&mut machine, seq_id, token_count, &error, &context, &blocks);
        }

        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        assert_capture(&mut machine, &error, &snapshot, &capture_context);
        let snapshot_ref = snapshot.borrow();
        assert!(snapshot_ref.is_sequence_active(0));
        assert!(snapshot_ref.is_sequence_active(1));
        assert_eq!(snapshot_ref.sequence_length(0), 4);
        assert_eq!(snapshot_ref.sequence_length(1), 2);
        let seq0_block = snapshot_ref.lookup_kv_block(0, 0);
        let seq1_block = snapshot_ref.lookup_kv_block(1, 0);
        assert!(seq0_block >= 0);
        assert!(seq1_block >= 0);
        assert_ne!(seq0_block, seq1_block);
        assert_ne!(
            snapshot_ref.lookup_recurrent_slot(0),
            snapshot_ref.lookup_recurrent_slot(1)
        );
        drop(snapshot_ref);

        let free_context = RefCell::new(FreeSequenceContext::default());
        assert_free_sequence(&mut machine, 0, &error, &free_context);
        assert_eq!(*error.borrow(), HybridError::None.code());
        let sequence_context = RefCell::new(AllocateSequenceContext::default());
        assert_allocate_sequence(&mut machine, 2, &error, &sequence_context);
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        let blocks = RefCell::new(-1);
        assert_allocate_slots(&mut machine, 2, 4, &error, &slots_context, &blocks);
        assert_eq!(*blocks.borrow(), 2);
        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        assert_capture(&mut machine, &error, &snapshot, &capture_context);
        let snapshot = snapshot.borrow();
        assert!(!snapshot.is_sequence_active(0));
        assert!(snapshot.is_sequence_active(1));
        assert!(snapshot.is_sequence_active(2));
        assert_eq!(snapshot.lookup_kv_block(2, 0), 0);
        assert_eq!(snapshot.lookup_kv_block(2, 2), 1);
    }

    #[test]
    fn invalid_requests_reset_caller_outputs_and_recover_to_ready() {
        let mut machine = Hybrid::new(MemoryHybridContext::default());
        let error = RefCell::new(-1);
        let blocks = RefCell::new(91);
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        assert!(
            machine
                .process_event(EventAllocateSlotsRuntime {
                    seq_id: -1,
                    token_count: 1,
                    block_count_out: Some(&blocks),
                    error_out: Some(&error),
                    copy_block: None,
                    context: &slots_context,
                })
                .is_ok()
        );
        assert_eq!(*blocks.borrow(), 0);
        assert_eq!(*error.borrow(), HybridError::InvalidRequest.code());

        let capture_context = RefCell::new(CaptureViewContext::default());
        assert!(
            machine
                .process_event(EventCaptureViewRuntime {
                    snapshot_out: None,
                    error_out: Some(&error),
                    context: &capture_context,
                })
                .is_ok()
        );
        assert_eq!(*error.borrow(), HybridError::InvalidRequest.code());

        let reserve_context = RefCell::new(ReserveContext::default());
        assert!(
            machine
                .process_event(EventReserveRuntime {
                    max_sequences: 1,
                    max_blocks: 1,
                    block_tokens: 1,
                    error_out: Some(&error),
                    context: &reserve_context,
                })
                .is_ok()
        );
        assert_eq!(*error.borrow(), HybridError::None.code());
    }
}
