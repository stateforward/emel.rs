//! Bounded graph-allocation liveness phase.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/liveness_pass/{sm,events,guards,actions}.hpp`.

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
    GraphAllocatorLivenessPass {
        "allocate_failed"_s <= *"deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prefailed] / mark_failed_prefailed,
        "allocated"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_done] / mark_done,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_invalid_request] / mark_failed_invalid_request,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_capacity_exceeded] / mark_failed_capacity,
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
pub struct GraphAllocatorLivenessPassContext {
    pub outcome: PhaseOutcome,
    pub error: AllocationError,
    pub has_graph_topology: bool,
    pub node_count: u32,
    pub tensor_count: u32,
    pub tensor_capacity: u32,
    pub required_intervals: u32,
}

impl GraphAllocatorLivenessPassContext {
    pub fn set_request(&mut self, has_graph_topology: bool, node_count: u32, tensor_count: u32, tensor_capacity: u32) {
        self.has_graph_topology = has_graph_topology;
        self.node_count = node_count;
        self.tensor_count = tensor_count;
        self.tensor_capacity = tensor_capacity;
        self.outcome = PhaseOutcome::Unknown;
        self.error = AllocationError::None;
        self.required_intervals = 0;
    }
}

impl GraphAllocatorLivenessPassStateMachineContext for GraphAllocatorLivenessPassContext {
    fn mark_done(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Done; self.required_intervals = self.tensor_count; self.error = AllocationError::None; Ok(()) }
    fn mark_failed_capacity(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Capacity; Ok(()) }
    fn mark_failed_internal(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Internal; Ok(()) }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::InvalidRequest; Ok(()) }
    fn mark_failed_prefailed(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; Ok(()) }
    fn on_unexpected_from_allocate_failed(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_allocated(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.outcome = PhaseOutcome::Failed; self.error = AllocationError::Internal; Ok(()) }
    fn phase_capacity_exceeded(&self) -> Result<bool, ()> { Ok(self.error == AllocationError::None && self.has_graph_topology && self.node_count != 0 && self.tensor_count != 0 && self.tensor_count > self.tensor_capacity) }
    fn phase_done(&self) -> Result<bool, ()> { Ok(self.error == AllocationError::None && self.has_graph_topology && self.node_count != 0 && self.tensor_count != 0 && self.tensor_count <= self.tensor_capacity) }
    fn phase_invalid_request(&self) -> Result<bool, ()> { Ok(self.error == AllocationError::None && (!self.has_graph_topology || self.node_count == 0 || self.tensor_count == 0)) }
    fn phase_prefailed(&self) -> Result<bool, ()> { Ok(self.error != AllocationError::None) }
}
