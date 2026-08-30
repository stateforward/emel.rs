//! Bounded graph-allocation placement phase.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/placement_pass/{sm,events,guards,actions}.hpp`.

#![allow(clippy::derive_partial_eq_without_eq, clippy::module_name_repetitions, dead_code, unused_imports, missing_docs)]

use sml::sml;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome { #[default] Unknown = 0, Done = 1, Failed = 2 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AllocationError { #[default] None = 0, InvalidRequest = 1, Capacity = 2, Internal = 4, Untracked = 8 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocatorEventAllocateGraphPlan;

sml! {
    GraphAllocatorPlacementPass {
        "allocate_failed"_s <= *"deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prefailed] / mark_failed_prefailed,
        "allocated"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_done] / mark_done,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prereq_failed] / mark_failed_prereq,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_capacity_exceeded] / mark_failed_capacity,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_invalid_request] / mark_failed_invalid_request,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> / mark_failed_internal,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "allocated"_s + unexpected_event<_> / on_unexpected_from_allocated,
        "unexpected_event"_s <= "allocate_failed"_s + unexpected_event<_> / on_unexpected_from_allocate_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "allocated"_s = X,
        "allocate_failed"_s = X,
    }
}

#[derive(Debug, Default)]
pub struct GraphAllocatorPlacementPassContext {
    pub outcome: PhaseOutcome,
    pub error: AllocationError,
    pub ordering_outcome: PhaseOutcome,
    pub sorted_tensor_count: u32,
    pub required_buffer_bytes: u64,
    pub workspace_capacity_bytes: u64,
    pub has_plan_output: bool,
}
impl GraphAllocatorPlacementPassContext {
    pub fn set_request(&mut self, ordering_outcome: PhaseOutcome, sorted_tensor_count: u32, required_buffer_bytes: u64, workspace_capacity_bytes: u64, has_plan_output: bool) { self.outcome=PhaseOutcome::Unknown; self.error=AllocationError::None; self.ordering_outcome=ordering_outcome; self.sorted_tensor_count=sorted_tensor_count; self.required_buffer_bytes=required_buffer_bytes; self.workspace_capacity_bytes=workspace_capacity_bytes; self.has_plan_output=has_plan_output; }
}
impl GraphAllocatorPlacementPassStateMachineContext for GraphAllocatorPlacementPassContext {
    fn mark_done(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Done; self.error=AllocationError::None; Ok(()) }
    fn mark_failed_capacity(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Capacity; Ok(()) }
    fn mark_failed_internal(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::InvalidRequest; Ok(()) }
    fn mark_failed_prefailed(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; Ok(()) }
    fn mark_failed_prereq(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_allocate_failed(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_allocated(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; Ok(()) }
    fn phase_capacity_exceeded(&self) -> Result<bool, ()> { Ok(self.error==AllocationError::None && self.ordering_outcome==PhaseOutcome::Done && self.required_buffer_bytes>self.workspace_capacity_bytes) }
    fn phase_done(&self) -> Result<bool, ()> { Ok(self.error==AllocationError::None && self.ordering_outcome==PhaseOutcome::Done && self.has_plan_output && self.sorted_tensor_count!=0 && self.required_buffer_bytes<=self.workspace_capacity_bytes) }
    fn phase_invalid_request(&self) -> Result<bool, ()> { Ok(self.error==AllocationError::None && self.ordering_outcome==PhaseOutcome::Done && (!self.has_plan_output || self.sorted_tensor_count==0)) }
    fn phase_prefailed(&self) -> Result<bool, ()> { Ok(self.error!=AllocationError::None) }
    fn phase_prereq_failed(&self) -> Result<bool, ()> { Ok(self.error==AllocationError::None && self.ordering_outcome!=PhaseOutcome::Done) }
}
