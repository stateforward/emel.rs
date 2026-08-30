//! Source-aligned bounded recurrent memory state machine.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

pub const MAX_SEQUENCES: usize = 256;
const MAX_SEQUENCES_I32: i32 = MAX_SEQUENCES as i32;
pub const INVALID_SLOT: i32 = -1;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecurrentError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    BackendError = 2,
    InternalError = 4,
    OutOfMemory = 8,
    Untracked = 16,
}

impl RecurrentError {
    const fn code(self) -> i32 { self as i32 }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub max_sequences: i32,
    pub sequence_active: Box<[u8]>,
    pub sequence_length_values: Box<[i32]>,
    pub sequence_recurrent_slot: Box<[i32]>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            max_sequences: 0,
            sequence_active: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_length_values: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_recurrent_slot: vec![INVALID_SLOT; MAX_SEQUENCES].into_boxed_slice(),
        }
    }
}

impl Snapshot {
    fn index(&self, seq_id: i32) -> Option<usize> {
        (seq_id >= 0 && seq_id < self.max_sequences)
            .then(|| usize::try_from(seq_id).ok())
            .flatten()
            .filter(|&index| index < MAX_SEQUENCES)
    }
    pub fn valid_seq_id(&self, seq_id: i32) -> bool { self.index(seq_id).is_some() }
    pub fn is_sequence_active(&self, seq_id: i32) -> bool {
        self.index(seq_id).is_some_and(|index| self.sequence_active[index] != 0)
    }
    pub fn sequence_length(&self, seq_id: i32) -> i32 {
        self.index(seq_id).filter(|&index| self.sequence_active[index] != 0)
            .map_or(0, |index| self.sequence_length_values[index])
    }
    pub fn lookup_recurrent_slot(&self, seq_id: i32) -> i32 {
        self.index(seq_id).filter(|&index| self.sequence_active[index] != 0)
            .map_or(INVALID_SLOT, |index| self.sequence_recurrent_slot[index])
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReserveContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    resolved_max_sequences: i32,
    resolved_slots: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSequenceContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    slot_id: i32,
    slot_activated: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSlotsContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    block_count: i32,
    old_length: i32,
    new_length: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BranchSequenceContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    child_slot: i32,
    slot_activated: bool,
    copy_accepted: bool,
    copy_error: RecurrentError,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeSequenceContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    slot_id: i32,
    slot_deactivated: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RollbackSlotsContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
    block_count: i32,
    current_length: i32,
    new_length: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureViewContext {
    err: RecurrentError,
    accepted: bool,
    operation_error: RecurrentError,
}

pub trait StateCopier {
    fn copy_state(&self, source_slot: i32, destination_slot: i32) -> Result<(), RecurrentError>;
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
    pub context: &'event RefCell<AllocateSlotsContext>,
}
#[derive(Clone)]
pub struct EventBranchSequenceRuntime<'event> {
    pub parent_seq_id: i32,
    pub child_seq_id: i32,
    pub copy_state: Option<&'event dyn StateCopier>,
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

macro_rules! event_debug {
    ($($name:ident),+ $(,)?) => { $(impl core::fmt::Debug for $name<'_> { fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct(stringify!($name)).finish() } })+ };
}
event_debug!(EventAllocateSequenceRuntime, EventAllocateSlotsRuntime, EventBranchSequenceRuntime,
    EventCaptureViewRuntime, EventFreeSequenceRuntime, EventReserveRuntime, EventRollbackSlotsRuntime);

sml! {
    MemoryRecurrent<'event> {
        "reserve_request_decision"_s <= *"ready"_s + event<EventReserveRuntime<'event>> / begin_reserve,
        "reserve_exec"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [reserve_request_valid],
        "errored"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [reserve_request_invalid] / mark_invalid_request_event_reserve_runtime,
        "reserve_result_decision"_s <= "reserve_exec"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / exec_reserve,
        "done"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_succeeded_event_reserve_runtime],
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_failed_with_error_event_reserve_runtime] / mark_error_from_operation_event_reserve_runtime,
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_failed_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "allocate_sequence_request_decision"_s <= "ready"_s + event<EventAllocateSequenceRuntime<'event>> / begin_allocate_sequence,
        "allocate_sequence_exec"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_inactive_with_slot],
        "done"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_active] / mark_operation_success_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_inactive_without_slot] / mark_backend_error_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_invalid] / mark_invalid_request_event_allocate_sequence_runtime,
        "allocate_sequence_result_decision"_s <= "allocate_sequence_exec"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / exec_allocate_sequence_inactive,
        "done"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_succeeded_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_failed_with_error_event_allocate_sequence_runtime] / mark_error_from_operation_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_failed_without_error_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "allocate_slots_request_decision"_s <= "ready"_s + event<EventAllocateSlotsRuntime<'event>> / begin_allocate_slots,
        "allocate_slots_request_shape_decision"_s <= "allocate_slots_request_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>),
        "allocate_slots_request_length_decision"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_shape_valid],
        "errored"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_shape_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_exec"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_length_valid],
        "errored"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_length_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_result_decision"_s <= "allocate_slots_exec"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / exec_allocate_slots,
        "done"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_succeeded_event_allocate_slots_runtime],
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_with_error_event_allocate_slots_runtime] / mark_error_from_operation_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_without_error_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "branch_sequence_request_decision"_s <= "ready"_s + event<EventBranchSequenceRuntime<'event>> / begin_branch_sequence,
        "branch_sequence_request_shape_decision"_s <= "branch_sequence_request_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>),
        "branch_sequence_request_capacity_decision"_s <= "branch_sequence_request_shape_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_shape_valid],
        "errored"_s <= "branch_sequence_request_shape_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_shape_invalid] / mark_invalid_request_event_branch_sequence_runtime,
        "branch_sequence_exec"_s <= "branch_sequence_request_capacity_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_capacity_available],
        "errored"_s <= "branch_sequence_request_capacity_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_capacity_exhausted] / mark_backend_error_event_branch_sequence_runtime,
        "branch_sequence_result_decision"_s <= "branch_sequence_exec"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence_prepare_child_slot,
        "branch_sequence_copy_exec"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_slot_activation_succeeded],
        "errored"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_slot_activation_failed] / mark_backend_error_event_branch_sequence_runtime,
        "branch_sequence_copy_result_decision"_s <= "branch_sequence_copy_exec"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence_copy_callback,
        "done"_s <= "branch_sequence_copy_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_copy_succeeded] / finalize_branch_sequence_success,
        "branch_sequence_rollback_exec"_s <= "branch_sequence_copy_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_copy_failed_with_error] / mark_error_from_operation_event_branch_sequence_runtime,
        "branch_sequence_rollback_exec"_s <= "branch_sequence_copy_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_copy_failed_without_error] / mark_backend_error_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_rollback_exec"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence_rollback_child_slot,
        "free_sequence_request_decision"_s <= "ready"_s + event<EventFreeSequenceRuntime<'event>> / begin_free_sequence,
        "free_sequence_exec"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [free_sequence_request_active],
        "done"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [free_sequence_request_inactive] / mark_operation_success_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [free_sequence_request_invalid] / mark_invalid_request_event_free_sequence_runtime,
        "free_sequence_result_decision"_s <= "free_sequence_exec"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / exec_free_sequence_active,
        "done"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_succeeded_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_failed_with_error_event_free_sequence_runtime] / mark_error_from_operation_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_failed_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "rollback_slots_request_decision"_s <= "ready"_s + event<EventRollbackSlotsRuntime<'event>> / begin_rollback_slots,
        "rollback_slots_exec"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [rollback_slots_request_valid],
        "errored"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [rollback_slots_request_invalid] / mark_invalid_request_event_rollback_slots_runtime,
        "rollback_slots_result_decision"_s <= "rollback_slots_exec"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / exec_rollback_slots,
        "done"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_succeeded_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_failed_with_error_event_rollback_slots_runtime] / mark_error_from_operation_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_failed_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "capture_request_decision"_s <= "ready"_s + event<EventCaptureViewRuntime<'event>> / begin_capture_view,
        "capture_exec"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_valid],
        "errored"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_invalid] / mark_invalid_request_event_capture_view_runtime,
        "capture_result_decision"_s <= "capture_exec"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / exec_capture_view,
        "done"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_succeeded_event_capture_view_runtime],
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_failed_with_error_event_capture_view_runtime] / mark_error_from_operation_event_capture_view_runtime,
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_failed_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "ready"_s <= "done"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_done_event_reserve_runtime,
        "ready"_s <= "errored"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_error_event_reserve_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_done_event_allocate_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_done_event_allocate_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "done"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_done_event_branch_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_done_event_free_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_error_event_free_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_done_event_rollback_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "done"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_done_event_capture_view_runtime,
        "ready"_s <= "errored"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_error_event_capture_view_runtime,
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
        "ready"_s <= "allocate_slots_exec"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_exec,
        "ready"_s <= "allocate_slots_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_result_decision,
        "ready"_s <= "branch_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_request_decision,
        "ready"_s <= "branch_sequence_request_shape_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_request_shape_decision,
        "ready"_s <= "branch_sequence_request_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_request_capacity_decision,
        "ready"_s <= "branch_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_exec,
        "ready"_s <= "branch_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_result_decision,
        "ready"_s <= "branch_sequence_copy_exec"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_copy_exec,
        "ready"_s <= "branch_sequence_copy_result_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_copy_result_decision,
        "ready"_s <= "branch_sequence_rollback_exec"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_rollback_exec,
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

#[derive(Debug)]
pub struct MemoryRecurrentContext {
    pub max_sequences: i32,
    pub max_slots: i32,
    pub slot_active: [bool; MAX_SEQUENCES],
    pub free_stack: [i32; MAX_SEQUENCES],
    pub free_count: i32,
    pub seq_to_slot: [i32; MAX_SEQUENCES],
    pub slot_owner_seq: [i32; MAX_SEQUENCES],
    pub sequence_length: [i32; MAX_SEQUENCES],
}

impl Default for MemoryRecurrentContext {
    fn default() -> Self {
        let mut context = Self {
            max_sequences: MAX_SEQUENCES_I32,
            max_slots: MAX_SEQUENCES_I32,
            slot_active: [false; MAX_SEQUENCES],
            free_stack: [0; MAX_SEQUENCES],
            free_count: 0,
            seq_to_slot: [INVALID_SLOT; MAX_SEQUENCES],
            slot_owner_seq: [INVALID_SLOT; MAX_SEQUENCES],
            sequence_length: [0; MAX_SEQUENCES],
        };
        context.reset_runtime();
        context
    }
}

impl MemoryRecurrentContext {
    fn reset_runtime(&mut self) {
        self.slot_active = [false; MAX_SEQUENCES];
        self.seq_to_slot = [INVALID_SLOT; MAX_SEQUENCES];
        self.slot_owner_seq = [INVALID_SLOT; MAX_SEQUENCES];
        self.sequence_length = [0; MAX_SEQUENCES];
        self.free_count = self.max_slots.clamp(0, MAX_SEQUENCES_I32);
        for index in 0..usize::try_from(self.free_count).unwrap_or(0) {
            self.free_stack[index] = self.free_count - 1 - index as i32;
        }
    }
    fn valid_seq(&self, id: i32) -> bool { id >= 0 && id < self.max_sequences && id < MAX_SEQUENCES_I32 }
    fn active_seq(&self, id: i32) -> bool { self.valid_seq(id) && self.seq_to_slot[id as usize] != INVALID_SLOT }
    fn set_error(out: Option<&RefCell<i32>>, error: RecurrentError) { if let Some(out) = out { *out.borrow_mut() = error.code(); } }
    fn fill_snapshot(&self, snapshot: &mut Snapshot) {
        snapshot.max_sequences = self.max_sequences;
        snapshot.sequence_active.fill(0);
        snapshot.sequence_length_values.fill(0);
        snapshot.sequence_recurrent_slot.fill(INVALID_SLOT);
        let count = usize::try_from(self.max_sequences.clamp(0, MAX_SEQUENCES_I32)).unwrap_or(0);
        for index in 0..count {
            let active = u8::from(self.seq_to_slot[index] != INVALID_SLOT);
            snapshot.sequence_active[index] = active;
            snapshot.sequence_length_values[index] = self.sequence_length[index] * i32::from(active);
            snapshot.sequence_recurrent_slot[index] = if active != 0 { self.seq_to_slot[index] } else { INVALID_SLOT };
        }
    }
}

macro_rules! begin_actions { ($(($name:ident, $ctx:ty, $event:ident)),+ $(,)?) => { $(fn $name(&mut self, event: &$event<'_>) -> Result<(), ()> { *event.context.borrow_mut() = <$ctx>::default(); MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) })+ }; }
macro_rules! outcome_guards { ($(($with:ident, $without:ident, $success:ident, $event:ident)),+ $(,)?) => { $(fn $with(&self, event: &$event<'_>) -> Result<bool, ()> { let op = event.context.borrow(); Ok(!op.accepted && op.operation_error != RecurrentError::None) } fn $without(&self, event: &$event<'_>) -> Result<bool, ()> { let op = event.context.borrow(); Ok(!op.accepted && op.operation_error == RecurrentError::None) } fn $success(&self, event: &$event<'_>) -> Result<bool, ()> { Ok(event.context.borrow().accepted) })+ }; }
macro_rules! mark_actions { ($(($backend:ident, $invalid:ident, $from:ident, $event:ident)),+ $(,)?) => { $(fn $backend(&mut self, event: &$event<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::BackendError; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::BackendError); Ok(()) } fn $invalid(&mut self, event: &$event<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::InvalidRequest; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::InvalidRequest); Ok(()) } fn $from(&mut self, event: &$event<'_>) -> Result<(), ()> { let error = event.context.borrow().operation_error; event.context.borrow_mut().err = error; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) })+ }; }
macro_rules! unexpected_actions { ($($name:ident),+ $(,)?) => { $(fn $name(&mut self) -> Result<(), ()> { Ok(()) })+ }; }

impl MemoryRecurrentStateMachineContext for MemoryRecurrentContext {
    fn reserve_request_invalid(&self, event: &EventReserveRuntime<'_>) -> Result<bool, ()> { let max = if event.max_sequences > 0 { event.max_sequences } else { MAX_SEQUENCES_I32 }; let slots = if event.max_blocks > 0 { event.max_blocks } else { max }; Ok(!(max > 0 && max <= MAX_SEQUENCES_I32 && slots > 0)) }
    fn reserve_request_valid(&self, event: &EventReserveRuntime<'_>) -> Result<bool, ()> { Ok(!self.reserve_request_invalid(event)?) }
    fn allocate_sequence_request_invalid(&self, event: &EventAllocateSequenceRuntime<'_>) -> Result<bool, ()> { Ok(!self.valid_seq(event.seq_id)) }
    fn allocate_sequence_request_active(&self, event: &EventAllocateSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.valid_seq(event.seq_id) && self.active_seq(event.seq_id)) }
    fn allocate_sequence_request_inactive_with_slot(&self, event: &EventAllocateSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.valid_seq(event.seq_id) && !self.active_seq(event.seq_id) && self.free_count > 0) }
    fn allocate_sequence_request_inactive_without_slot(&self, event: &EventAllocateSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.valid_seq(event.seq_id) && !self.active_seq(event.seq_id) && self.free_count <= 0) }
    fn allocate_slots_request_shape_invalid(&self, event: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { Ok(!self.valid_seq(event.seq_id) || event.token_count <= 0 || !self.active_seq(event.seq_id)) }
    fn allocate_slots_request_shape_valid(&self, event: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { Ok(!self.allocate_slots_request_shape_invalid(event)?) }
    fn allocate_slots_request_length_invalid(&self, event: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { if !self.allocate_slots_request_shape_valid(event)? { return Ok(true); } let old = self.sequence_length[event.seq_id as usize]; Ok(i64::from(old) + i64::from(event.token_count) > i64::from(i32::MAX)) }
    fn allocate_slots_request_length_valid(&self, event: &EventAllocateSlotsRuntime<'_>) -> Result<bool, ()> { Ok(!self.allocate_slots_request_length_invalid(event)?) }
    fn branch_sequence_request_shape_invalid(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(event.copy_state.is_none() || !self.valid_seq(event.parent_seq_id) || !self.valid_seq(event.child_seq_id) || event.parent_seq_id == event.child_seq_id || !self.active_seq(event.parent_seq_id) || self.active_seq(event.child_seq_id)) }
    fn branch_sequence_request_shape_valid(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(!self.branch_sequence_request_shape_invalid(event)?) }
    fn branch_sequence_request_capacity_available(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.branch_sequence_request_shape_valid(event)? && self.free_count > 0) }
    fn branch_sequence_request_capacity_exhausted(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.branch_sequence_request_shape_valid(event)? && self.free_count <= 0) }
    fn branch_slot_activation_succeeded(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().slot_activated) }
    fn branch_slot_activation_failed(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(!event.context.borrow().slot_activated) }
    fn branch_copy_succeeded(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { let op = event.context.borrow(); Ok(op.copy_accepted && op.copy_error == RecurrentError::None) }
    fn branch_copy_failed_with_error(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().copy_error != RecurrentError::None) }
    fn branch_copy_failed_without_error(&self, event: &EventBranchSequenceRuntime<'_>) -> Result<bool, ()> { let op = event.context.borrow(); Ok(!op.copy_accepted && op.copy_error == RecurrentError::None) }
    fn free_sequence_request_invalid(&self, event: &EventFreeSequenceRuntime<'_>) -> Result<bool, ()> { Ok(!self.valid_seq(event.seq_id)) }
    fn free_sequence_request_active(&self, event: &EventFreeSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.valid_seq(event.seq_id) && self.active_seq(event.seq_id)) }
    fn free_sequence_request_inactive(&self, event: &EventFreeSequenceRuntime<'_>) -> Result<bool, ()> { Ok(self.valid_seq(event.seq_id) && !self.active_seq(event.seq_id)) }
    fn rollback_slots_request_invalid(&self, event: &EventRollbackSlotsRuntime<'_>) -> Result<bool, ()> { Ok(!self.valid_seq(event.seq_id) || event.token_count <= 0 || !self.active_seq(event.seq_id)) }
    fn rollback_slots_request_valid(&self, event: &EventRollbackSlotsRuntime<'_>) -> Result<bool, ()> { Ok(!self.rollback_slots_request_invalid(event)?) }
    fn capture_request_invalid(&self, event: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> { Ok(event.snapshot_out.is_none()) }
    fn capture_request_valid(&self, event: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> { Ok(event.snapshot_out.is_some()) }

    begin_actions!((begin_allocate_sequence, AllocateSequenceContext, EventAllocateSequenceRuntime), (begin_allocate_slots, AllocateSlotsContext, EventAllocateSlotsRuntime), (begin_branch_sequence, BranchSequenceContext, EventBranchSequenceRuntime), (begin_capture_view, CaptureViewContext, EventCaptureViewRuntime), (begin_free_sequence, FreeSequenceContext, EventFreeSequenceRuntime), (begin_reserve, ReserveContext, EventReserveRuntime), (begin_rollback_slots, RollbackSlotsContext, EventRollbackSlotsRuntime));

    fn exec_reserve(&mut self, event: &EventReserveRuntime<'_>) -> Result<(), ()> { let max = if event.max_sequences > 0 { event.max_sequences } else { MAX_SEQUENCES_I32 }; let requested = if event.max_blocks > 0 { event.max_blocks } else { max }; let slots = max.min(requested); self.max_sequences = max; self.max_slots = slots; self.reset_runtime(); let mut op = event.context.borrow_mut(); op.resolved_max_sequences = max; op.resolved_slots = slots; op.accepted = true; op.operation_error = RecurrentError::None; Ok(()) }
    fn exec_allocate_sequence_inactive(&mut self, event: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { let index = event.seq_id as usize; let slot = self.free_stack[(self.free_count - 1) as usize]; self.free_count -= 1; let activated = !self.slot_active[slot as usize]; if activated { self.slot_active[slot as usize] = true; self.seq_to_slot[index] = slot; self.slot_owner_seq[slot as usize] = event.seq_id; self.sequence_length[index] = 0; } else { self.free_count += 1; } let mut op = event.context.borrow_mut(); op.slot_id = slot; op.slot_activated = activated; op.accepted = activated; op.operation_error = RecurrentError::None; Ok(()) }
    fn exec_allocate_slots(&mut self, event: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> { let index = event.seq_id as usize; let mut op = event.context.borrow_mut(); op.old_length = self.sequence_length[index]; op.new_length = op.old_length.saturating_add(event.token_count); self.sequence_length[index] = op.new_length; op.block_count = 0; op.accepted = true; op.operation_error = RecurrentError::None; Ok(()) }
    fn exec_branch_sequence_prepare_child_slot(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { let slot = self.free_stack[(self.free_count - 1) as usize]; self.free_count -= 1; let activated = !self.slot_active[slot as usize]; if activated { self.slot_active[slot as usize] = true; } else { self.free_count += 1; } let mut op = event.context.borrow_mut(); op.child_slot = slot; op.slot_activated = activated; Ok(()) }
    fn exec_branch_sequence_copy_callback(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { let parent_slot = self.seq_to_slot[event.parent_seq_id as usize]; let mut op = event.context.borrow_mut(); op.copy_error = RecurrentError::None; let Some(copier) = event.copy_state else { op.copy_accepted = false; op.copy_error = RecurrentError::InternalError; op.accepted = false; op.operation_error = RecurrentError::InternalError; return Ok(()); }; match copier.copy_state(parent_slot, op.child_slot) { Ok(()) => { op.copy_accepted = true; op.accepted = true; op.operation_error = RecurrentError::None; }, Err(error) => { op.copy_accepted = false; op.copy_error = error; op.accepted = false; op.operation_error = error; } } Ok(()) }
    fn exec_branch_sequence_rollback_child_slot(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { let slot = event.context.borrow().child_slot; if slot >= 0 && (slot as usize) < MAX_SEQUENCES { self.slot_active[slot as usize] = false; self.slot_owner_seq[slot as usize] = INVALID_SLOT; if self.free_count < self.max_slots { self.free_stack[self.free_count as usize] = slot; self.free_count += 1; } } Ok(()) }
    fn finalize_branch_sequence_success(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { let op = event.context.borrow(); self.seq_to_slot[event.child_seq_id as usize] = op.child_slot; self.slot_owner_seq[op.child_slot as usize] = event.child_seq_id; self.sequence_length[event.child_seq_id as usize] = self.sequence_length[event.parent_seq_id as usize]; Ok(()) }
    fn exec_free_sequence_active(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> { let index = event.seq_id as usize; let slot = self.seq_to_slot[index]; let deactivated = slot >= 0 && !self.slot_active[slot as usize].then_some(true).unwrap_or(false); let was_active = slot >= 0 && self.slot_active[slot as usize]; if was_active { self.slot_active[slot as usize] = false; self.slot_owner_seq[slot as usize] = INVALID_SLOT; self.seq_to_slot[index] = INVALID_SLOT; self.sequence_length[index] = 0; self.free_stack[self.free_count as usize] = slot; self.free_count += 1; } let mut op = event.context.borrow_mut(); op.slot_id = slot; op.slot_deactivated = was_active || deactivated; op.accepted = was_active; op.operation_error = RecurrentError::None; Ok(()) }
    fn exec_rollback_slots(&mut self, event: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> { let index = event.seq_id as usize; let mut op = event.context.borrow_mut(); op.current_length = self.sequence_length[index]; op.new_length = op.current_length.saturating_sub(event.token_count).max(0); self.sequence_length[index] = op.new_length; op.block_count = 0; op.accepted = true; op.operation_error = RecurrentError::None; Ok(()) }
    fn exec_capture_view(&mut self, event: &EventCaptureViewRuntime<'_>) -> Result<(), ()> { if let Some(snapshot) = event.snapshot_out { self.fill_snapshot(&mut snapshot.borrow_mut()); let mut op = event.context.borrow_mut(); op.accepted = true; op.operation_error = RecurrentError::None; } Ok(()) }
    fn mark_operation_success_event_allocate_sequence_runtime(&mut self, event: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { let mut op = event.context.borrow_mut(); op.accepted = true; op.operation_error = RecurrentError::None; Ok(()) }
    fn mark_operation_success_event_free_sequence_runtime(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> { let mut op = event.context.borrow_mut(); op.accepted = true; op.operation_error = RecurrentError::None; Ok(()) }

    outcome_guards!((operation_failed_with_error_event_allocate_sequence_runtime, operation_failed_without_error_event_allocate_sequence_runtime, operation_succeeded_event_allocate_sequence_runtime, EventAllocateSequenceRuntime), (operation_failed_with_error_event_allocate_slots_runtime, operation_failed_without_error_event_allocate_slots_runtime, operation_succeeded_event_allocate_slots_runtime, EventAllocateSlotsRuntime), (operation_failed_with_error_event_capture_view_runtime, operation_failed_without_error_event_capture_view_runtime, operation_succeeded_event_capture_view_runtime, EventCaptureViewRuntime), (operation_failed_with_error_event_free_sequence_runtime, operation_failed_without_error_event_free_sequence_runtime, operation_succeeded_event_free_sequence_runtime, EventFreeSequenceRuntime), (operation_failed_with_error_event_reserve_runtime, operation_failed_without_error_event_reserve_runtime, operation_succeeded_event_reserve_runtime, EventReserveRuntime), (operation_failed_with_error_event_rollback_slots_runtime, operation_failed_without_error_event_rollback_slots_runtime, operation_succeeded_event_rollback_slots_runtime, EventRollbackSlotsRuntime));
    mark_actions!((mark_backend_error_event_allocate_sequence_runtime, mark_invalid_request_event_allocate_sequence_runtime, mark_error_from_operation_event_allocate_sequence_runtime, EventAllocateSequenceRuntime), (mark_backend_error_event_allocate_slots_runtime, mark_invalid_request_event_allocate_slots_runtime, mark_error_from_operation_event_allocate_slots_runtime, EventAllocateSlotsRuntime), (mark_backend_error_event_branch_sequence_runtime, mark_invalid_request_event_branch_sequence_runtime, mark_error_from_operation_event_branch_sequence_runtime, EventBranchSequenceRuntime), (mark_backend_error_event_capture_view_runtime, mark_invalid_request_event_capture_view_runtime, mark_error_from_operation_event_capture_view_runtime, EventCaptureViewRuntime), (mark_backend_error_event_free_sequence_runtime, mark_invalid_request_event_free_sequence_runtime, mark_error_from_operation_event_free_sequence_runtime, EventFreeSequenceRuntime), (mark_backend_error_event_reserve_runtime, mark_invalid_request_event_reserve_runtime, mark_error_from_operation_event_reserve_runtime, EventReserveRuntime), (mark_backend_error_event_rollback_slots_runtime, mark_invalid_request_event_rollback_slots_runtime, mark_error_from_operation_event_rollback_slots_runtime, EventRollbackSlotsRuntime));
    fn publish_done_event_allocate_sequence_runtime(&mut self, event: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::None; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_allocate_slots_runtime(&mut self, event: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> { let count = event.context.borrow().block_count; event.context.borrow_mut().err = RecurrentError::None; if let Some(out) = event.block_count_out { *out.borrow_mut() = count; } MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_branch_sequence_runtime(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::None; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_capture_view_runtime(&mut self, event: &EventCaptureViewRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::None; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_free_sequence_runtime(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::None; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_reserve_runtime(&mut self, event: &EventReserveRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().err = RecurrentError::None; MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_done_event_rollback_slots_runtime(&mut self, event: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> { let count = event.context.borrow().block_count; event.context.borrow_mut().err = RecurrentError::None; if let Some(out) = event.block_count_out { *out.borrow_mut() = count; } MemoryRecurrentContext::set_error(event.error_out, RecurrentError::None); Ok(()) }
    fn publish_error_event_allocate_sequence_runtime(&mut self, event: &EventAllocateSequenceRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_allocate_slots_runtime(&mut self, event: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; if let Some(out) = event.block_count_out { *out.borrow_mut() = 0; } MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_branch_sequence_runtime(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_capture_view_runtime(&mut self, event: &EventCaptureViewRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_free_sequence_runtime(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_reserve_runtime(&mut self, event: &EventReserveRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }
    fn publish_error_event_rollback_slots_runtime(&mut self, event: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> { let error = event.context.borrow().err; if let Some(out) = event.block_count_out { *out.borrow_mut() = 0; } MemoryRecurrentContext::set_error(event.error_out, error); Ok(()) }


    unexpected_actions!(on_unexpected_from_ready, on_unexpected_from_reserve_request_decision, on_unexpected_from_reserve_exec, on_unexpected_from_reserve_result_decision, on_unexpected_from_allocate_sequence_request_decision, on_unexpected_from_allocate_sequence_exec, on_unexpected_from_allocate_sequence_result_decision, on_unexpected_from_allocate_slots_request_decision, on_unexpected_from_allocate_slots_request_shape_decision, on_unexpected_from_allocate_slots_request_length_decision, on_unexpected_from_allocate_slots_exec, on_unexpected_from_allocate_slots_result_decision, on_unexpected_from_branch_sequence_request_decision, on_unexpected_from_branch_sequence_request_shape_decision, on_unexpected_from_branch_sequence_request_capacity_decision, on_unexpected_from_branch_sequence_exec, on_unexpected_from_branch_sequence_result_decision, on_unexpected_from_branch_sequence_copy_exec, on_unexpected_from_branch_sequence_copy_result_decision, on_unexpected_from_branch_sequence_rollback_exec, on_unexpected_from_free_sequence_request_decision, on_unexpected_from_free_sequence_exec, on_unexpected_from_free_sequence_result_decision, on_unexpected_from_rollback_slots_request_decision, on_unexpected_from_rollback_slots_exec, on_unexpected_from_rollback_slots_result_decision, on_unexpected_from_capture_request_decision, on_unexpected_from_capture_exec, on_unexpected_from_capture_result_decision, on_unexpected_from_done, on_unexpected_from_errored);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CopyState;
    impl StateCopier for CopyState {
        fn copy_state(&self, _source_slot: i32, _destination_slot: i32) -> Result<(), RecurrentError> {
            Ok(())
        }
    }

    #[test]
    fn bounded_lifecycle_reserves_allocates_branches_captures_rolls_back_and_frees() {
        let mut machine = MemoryRecurrentStateMachine::new(MemoryRecurrentContext::default());
        let error = RefCell::new(-1);
        let reserve_context = RefCell::new(ReserveContext::default());
        machine.process_event(EventReserveRuntime {
            max_sequences: 2,
            max_blocks: 2,
            block_tokens: 16,
            error_out: Some(&error),
            context: &reserve_context,
        }).unwrap();
        assert_eq!(*error.borrow(), RecurrentError::None.code());
        assert_eq!(machine.context().free_count, 2);

        let sequence_context = RefCell::new(AllocateSequenceContext::default());
        machine.process_event(EventAllocateSequenceRuntime { seq_id: 0, error_out: Some(&error), context: &sequence_context }).unwrap();
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        let blocks = RefCell::new(-1);
        machine.process_event(EventAllocateSlotsRuntime { seq_id: 0, token_count: 3, block_count_out: Some(&blocks), error_out: Some(&error), context: &slots_context }).unwrap();
        assert_eq!(machine.context().sequence_length[0], 3);

        let copier = CopyState;
        let branch_context = RefCell::new(BranchSequenceContext::default());
        machine.process_event(EventBranchSequenceRuntime { parent_seq_id: 0, child_seq_id: 1, copy_state: Some(&copier), error_out: Some(&error), context: &branch_context }).unwrap();
        assert_eq!(machine.context().sequence_length[1], 3);

        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        machine.process_event(EventCaptureViewRuntime { snapshot_out: Some(&snapshot), error_out: Some(&error), context: &capture_context }).unwrap();
        assert!(snapshot.borrow().is_sequence_active(0));
        assert_eq!(snapshot.borrow().lookup_recurrent_slot(1), machine.context().seq_to_slot[1]);

        let rollback_context = RefCell::new(RollbackSlotsContext::default());
        machine.process_event(EventRollbackSlotsRuntime { seq_id: 0, token_count: 2, block_count_out: Some(&blocks), error_out: Some(&error), context: &rollback_context }).unwrap();
        assert_eq!(machine.context().sequence_length[0], 1);

        let free_context = RefCell::new(FreeSequenceContext::default());
        machine.process_event(EventFreeSequenceRuntime { seq_id: 1, error_out: Some(&error), context: &free_context }).unwrap();
        machine.process_event(EventFreeSequenceRuntime { seq_id: 0, error_out: Some(&error), context: &free_context }).unwrap();
        assert_eq!(machine.context().free_count, 2);
    }
}
