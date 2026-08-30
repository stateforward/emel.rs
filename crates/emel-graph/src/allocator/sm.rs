//! Bounded graph allocator state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/{sm,context,events,guards,actions,errors}.hpp`.
//! The public Rust surface owns request/result values and never carries raw pointers.

#![allow(clippy::derive_partial_eq_without_eq, clippy::module_name_repetitions, dead_code, unused_imports, missing_docs)]

use sml::sml;

/// Allocation errors matching the pinned C++ error values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AllocationError { #[default] None = 0, InvalidRequest = 1, Capacity = 2, Internal = 4, Untracked = 8 }

/// Bounded allocation plan produced by the allocator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlan { pub tensor_count: u32, pub interval_count: u32, pub required_buffer_bytes: u64 }

/// Successful callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationDone { pub plan: AllocationPlan }
/// Failed callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationErrorEvent { pub plan: AllocationPlan, pub error: AllocationError }
/// Synchronous callback types; callbacks must not retain or re-enter the actor.
pub type DoneCallback = fn(AllocationDone) -> bool;
pub type ErrorCallback = fn(AllocationErrorEvent) -> bool;

/// Caller request corresponding to C++ `event::allocate_graph`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateGraph {
    pub graph_topology: usize,
    pub plan_out: bool,
    pub node_count: u32,
    pub tensor_count: u32,
    pub tensor_capacity: u32,
    pub interval_capacity: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}

/// Internal event corresponding to C++ `event::allocate_graph_plan`.
#[derive(Clone, Copy, Debug, Default)]
pub struct EventAllocateGraphPlan { pub request: AllocateGraph }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome { #[default] Unknown = 0, Done = 1, Failed = 2 }

sml! {
    GraphAllocator {
        "liveness_pass_model"_s <= *"ready"_s + event<EventAllocateGraphPlan> [valid_allocate] / begin_allocate,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_with_dispatchable_output] / reject_invalid_allocate_with_dispatch,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_with_output_only] / reject_invalid_allocate_with_output_only,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_without_output] / reject_invalid_allocate_without_output,
        "liveness_decision"_s <= "liveness_pass_model"_s + completion<EventAllocateGraphPlan> / run_liveness,
        "ordering_pass_model"_s <= "liveness_decision"_s + completion<EventAllocateGraphPlan> [liveness_done],
        "allocation_decision"_s <= "liveness_decision"_s + completion<EventAllocateGraphPlan> [liveness_failed],
        "ordering_decision"_s <= "ordering_pass_model"_s + completion<EventAllocateGraphPlan> / run_ordering,
        "placement_pass_model"_s <= "ordering_decision"_s + completion<EventAllocateGraphPlan> [ordering_done],
        "allocation_decision"_s <= "ordering_decision"_s + completion<EventAllocateGraphPlan> [ordering_failed],
        "placement_decision"_s <= "placement_pass_model"_s + completion<EventAllocateGraphPlan> / run_placement,
        "allocation_decision"_s <= "placement_decision"_s + completion<EventAllocateGraphPlan> [placement_done] / commit_plan,
        "allocation_decision"_s <= "placement_decision"_s + completion<EventAllocateGraphPlan> [placement_failed],
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_none] / dispatch_done,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_invalid_request] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_capacity] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_internal_error] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_untracked] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_unknown] / dispatch_error_from_allocation_decision,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "allocation_decision"_s <= "liveness_decision"_s + unexpected_event<_> / on_unexpected_from_liveness_decision,
        "allocation_decision"_s <= "ordering_decision"_s + unexpected_event<_> / on_unexpected_from_ordering_decision,
        "allocation_decision"_s <= "placement_decision"_s + unexpected_event<_> / on_unexpected_from_placement_decision,
        "ready"_s <= "allocation_decision"_s + unexpected_event<_> / on_unexpected_from_allocation_decision,
    }
}

/// Persistent actor context containing only bounded scalar state.
#[derive(Debug, Default)]
pub struct GraphAllocatorContext {
    pub error: AllocationError,
    pub liveness_outcome: PhaseOutcome,
    pub ordering_outcome: PhaseOutcome,
    pub placement_outcome: PhaseOutcome,
    pub required_intervals: u32,
    pub sorted_tensor_count: u32,
    pub required_buffer_bytes: u64,
    pub plan: AllocationPlan,
    pub dispatch_generation: u32,
}

impl GraphAllocatorStateMachineContext for GraphAllocatorContext {
    fn allocation_error_capacity(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.error == AllocationError::Capacity) }
    fn allocation_error_internal_error(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.error == AllocationError::Internal) }
    fn allocation_error_invalid_request(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.error == AllocationError::InvalidRequest) }
    fn allocation_error_none(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.error == AllocationError::None) }
    fn allocation_error_unknown(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(false) }
    fn allocation_error_untracked(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.error == AllocationError::Untracked) }
    fn begin_allocate(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> { self.error=AllocationError::None; self.liveness_outcome=PhaseOutcome::Unknown; self.ordering_outcome=PhaseOutcome::Unknown; self.placement_outcome=PhaseOutcome::Unknown; self.required_intervals=0; self.sorted_tensor_count=0; self.required_buffer_bytes=0; self.plan=AllocationPlan::default(); self.dispatch_generation=self.dispatch_generation.wrapping_add(1); Ok(()) }
    fn commit_plan(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> { self.plan=AllocationPlan { tensor_count:self.sorted_tensor_count, interval_count:self.required_intervals, required_buffer_bytes:self.required_buffer_bytes }; Ok(()) }
    fn dispatch_done(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { if let Some(callback)=event.request.dispatch_done { callback(AllocationDone { plan:self.plan }); } Ok(()) }
    fn dispatch_error_from_allocation_decision(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { if let Some(callback)=event.request.dispatch_error { callback(AllocationErrorEvent { plan:self.plan, error:self.error }); } Ok(()) }
    fn invalid_allocate_with_dispatchable_output(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(!self.valid_allocate(event)? && event.request.plan_out && event.request.dispatch_error.is_some()) }
    fn invalid_allocate_with_output_only(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(!self.valid_allocate(event)? && event.request.plan_out && event.request.dispatch_error.is_none()) }
    fn invalid_allocate_without_output(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(!self.valid_allocate(event)? && !event.request.plan_out) }
    fn liveness_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.liveness_outcome==PhaseOutcome::Done && self.error==AllocationError::None) }
    fn liveness_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.liveness_outcome==PhaseOutcome::Failed || self.error!=AllocationError::None) }
    fn on_unexpected_from_allocation_decision(&mut self) -> Result<(), ()> { self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_liveness_decision(&mut self) -> Result<(), ()> { self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_ordering_decision(&mut self) -> Result<(), ()> { self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_placement_decision(&mut self) -> Result<(), ()> { self.error=AllocationError::Internal; Ok(()) }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> { self.error=AllocationError::Internal; Ok(()) }
    fn ordering_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.ordering_outcome==PhaseOutcome::Done && self.error==AllocationError::None) }
    fn ordering_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.ordering_outcome==PhaseOutcome::Failed || self.error!=AllocationError::None) }
    fn placement_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.placement_outcome==PhaseOutcome::Done && self.error==AllocationError::None) }
    fn placement_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(self.placement_outcome==PhaseOutcome::Failed || self.error!=AllocationError::None) }
    fn reject_invalid_allocate_with_dispatch(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { self.error=AllocationError::InvalidRequest; self.plan=AllocationPlan::default(); if let Some(callback)=event.request.dispatch_error { callback(AllocationErrorEvent { plan:self.plan, error:self.error }); } Ok(()) }
    fn reject_invalid_allocate_with_output_only(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> { self.error=AllocationError::InvalidRequest; self.plan=AllocationPlan::default(); Ok(()) }
    fn reject_invalid_allocate_without_output(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> { self.error=AllocationError::InvalidRequest; Ok(()) }
    fn run_liveness(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { if event.request.graph_topology==0 || event.request.node_count==0 || event.request.tensor_count==0 { self.liveness_outcome=PhaseOutcome::Failed; self.error=AllocationError::InvalidRequest; } else if event.request.tensor_count>event.request.tensor_capacity { self.liveness_outcome=PhaseOutcome::Failed; self.error=AllocationError::Capacity; } else { self.liveness_outcome=PhaseOutcome::Done; self.required_intervals=event.request.tensor_count; } Ok(()) }
    fn run_ordering(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { if self.liveness_outcome!=PhaseOutcome::Done { self.ordering_outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; } else if self.required_intervals==0 || event.request.bytes_per_tensor==0 { self.ordering_outcome=PhaseOutcome::Failed; self.error=AllocationError::InvalidRequest; } else if self.required_intervals>event.request.interval_capacity { self.ordering_outcome=PhaseOutcome::Failed; self.error=AllocationError::Capacity; } else if u64::from(self.required_intervals)>u64::MAX/event.request.bytes_per_tensor { self.ordering_outcome=PhaseOutcome::Failed; self.error=AllocationError::Capacity; } else { self.ordering_outcome=PhaseOutcome::Done; self.sorted_tensor_count=self.required_intervals; self.required_buffer_bytes=u64::from(self.required_intervals)*event.request.bytes_per_tensor; } Ok(()) }
    fn run_placement(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> { if self.ordering_outcome!=PhaseOutcome::Done { self.placement_outcome=PhaseOutcome::Failed; self.error=AllocationError::Internal; } else if !event.request.plan_out || self.sorted_tensor_count==0 { self.placement_outcome=PhaseOutcome::Failed; self.error=AllocationError::InvalidRequest; } else if self.required_buffer_bytes>event.request.workspace_capacity_bytes { self.placement_outcome=PhaseOutcome::Failed; self.error=AllocationError::Capacity; } else { self.placement_outcome=PhaseOutcome::Done; } Ok(()) }
    fn valid_allocate(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> { Ok(event.request.graph_topology!=0 && event.request.plan_out && event.request.node_count!=0 && event.request.tensor_count!=0 && event.request.tensor_capacity!=0 && event.request.interval_capacity!=0 && event.request.bytes_per_tensor!=0 && event.request.workspace_capacity_bytes!=0 && event.request.dispatch_done.is_some() && event.request.dispatch_error.is_some()) }
}

/// Single-writer allocator actor.
pub struct Allocator { machine: GraphAllocatorStateMachine<GraphAllocatorContext> }
impl Default for Allocator { fn default() -> Self { Self::new() } }
impl Allocator {
    #[must_use] pub fn new() -> Self { Self { machine: GraphAllocatorStateMachine::new(GraphAllocatorContext::default()) } }
    pub fn process_event(&mut self, event: EventAllocateGraphPlan) -> bool { self.machine.process_event(event).is_ok() }
    #[must_use] pub fn is_ready(&self) -> bool { self.machine.is(&GraphAllocatorStates::Ready) }
    #[must_use] pub fn plan(&self) -> AllocationPlan { self.machine.context().plan }
    #[must_use] pub fn error(&self) -> AllocationError { self.machine.context().error }
}
