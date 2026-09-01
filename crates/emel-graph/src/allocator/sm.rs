//! Bounded graph allocator state machine.
//!
#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::if_same_then_else,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]
use super::{
    liveness_pass::sm as liveness,
    ordering_pass::sm as ordering,
    placement_pass::sm as placement,
};
use sml::sml;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AllocationError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Capacity = 2,
    Internal = 4,
    Untracked = 8,
}

/// Bounded allocation plan produced by the allocator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlan {
    pub tensor_count: u32,
    pub interval_count: u32,
    pub required_buffer_bytes: u64,
}

/// Successful callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationDone {
    pub plan: AllocationPlan,
}
/// Failed callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationErrorEvent {
    pub plan: AllocationPlan,
    pub error: AllocationError,
}
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
pub struct EventAllocateGraphPlan {
    pub request: AllocateGraph,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

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

/// Persistent actor context containing bounded scalar state and owned child actors.
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
    liveness: liveness::Actor,
    ordering: ordering::Actor,
    placement: placement::Actor,
}

impl Default for GraphAllocatorContext {
    fn default() -> Self {
        Self {
            error: AllocationError::None,
            liveness_outcome: PhaseOutcome::Unknown,
            ordering_outcome: PhaseOutcome::Unknown,
            placement_outcome: PhaseOutcome::Unknown,
            required_intervals: 0,
            sorted_tensor_count: 0,
            required_buffer_bytes: 0,
            plan: AllocationPlan::default(),
            dispatch_generation: 0,
            liveness: liveness::Actor::new(),
            ordering: ordering::Actor::new(),
            placement: placement::Actor::new(),
        }
    }
}

fn map_liveness_outcome(outcome: liveness::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        liveness::PhaseOutcome::Done => PhaseOutcome::Done,
        liveness::PhaseOutcome::Failed => PhaseOutcome::Failed,
        liveness::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_liveness_error(error: liveness::AllocationError) -> AllocationError {
    match error {
        liveness::AllocationError::None => AllocationError::None,
        liveness::AllocationError::InvalidRequest => AllocationError::InvalidRequest,
        liveness::AllocationError::Capacity => AllocationError::Capacity,
        liveness::AllocationError::Internal => AllocationError::Internal,
        liveness::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_liveness(error: AllocationError) -> liveness::AllocationError {
    match error {
        AllocationError::None => liveness::AllocationError::None,
        AllocationError::InvalidRequest => liveness::AllocationError::InvalidRequest,
        AllocationError::Capacity => liveness::AllocationError::Capacity,
        AllocationError::Internal => liveness::AllocationError::Internal,
        AllocationError::Untracked => liveness::AllocationError::Untracked,
    }
}
fn map_phase_to_ordering(outcome: PhaseOutcome) -> ordering::PhaseOutcome {
    match outcome {
        PhaseOutcome::Done => ordering::PhaseOutcome::Done,
        PhaseOutcome::Failed => ordering::PhaseOutcome::Failed,
        PhaseOutcome::Unknown => ordering::PhaseOutcome::Unknown,
    }
}
fn map_ordering_outcome(outcome: ordering::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        ordering::PhaseOutcome::Done => PhaseOutcome::Done,
        ordering::PhaseOutcome::Failed => PhaseOutcome::Failed,
        ordering::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_ordering_error(error: ordering::AllocationError) -> AllocationError {
    match error {
        ordering::AllocationError::None => AllocationError::None,
        ordering::AllocationError::InvalidRequest => AllocationError::InvalidRequest,
        ordering::AllocationError::Capacity => AllocationError::Capacity,
        ordering::AllocationError::Internal => AllocationError::Internal,
        ordering::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_ordering(error: AllocationError) -> ordering::AllocationError {
    match error {
        AllocationError::None => ordering::AllocationError::None,
        AllocationError::InvalidRequest => ordering::AllocationError::InvalidRequest,
        AllocationError::Capacity => ordering::AllocationError::Capacity,
        AllocationError::Internal => ordering::AllocationError::Internal,
        AllocationError::Untracked => ordering::AllocationError::Untracked,
    }
}
fn map_phase_to_placement(outcome: PhaseOutcome) -> placement::PhaseOutcome {
    match outcome {
        PhaseOutcome::Done => placement::PhaseOutcome::Done,
        PhaseOutcome::Failed => placement::PhaseOutcome::Failed,
        PhaseOutcome::Unknown => placement::PhaseOutcome::Unknown,
    }
}
fn map_placement_outcome(outcome: placement::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        placement::PhaseOutcome::Done => PhaseOutcome::Done,
        placement::PhaseOutcome::Failed => PhaseOutcome::Failed,
        placement::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_placement_error(error: placement::AllocationError) -> AllocationError {
    match error {
        placement::AllocationError::None => AllocationError::None,
        placement::AllocationError::InvalidRequest => AllocationError::InvalidRequest,
        placement::AllocationError::Capacity => AllocationError::Capacity,
        placement::AllocationError::Internal => AllocationError::Internal,
        placement::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_placement(error: AllocationError) -> placement::AllocationError {
    match error {
        AllocationError::None => placement::AllocationError::None,
        AllocationError::InvalidRequest => placement::AllocationError::InvalidRequest,
        AllocationError::Capacity => placement::AllocationError::Capacity,
        AllocationError::Internal => placement::AllocationError::Internal,
        AllocationError::Untracked => placement::AllocationError::Untracked,
    }
}
impl GraphAllocatorStateMachineContext for GraphAllocatorContext {
    fn allocation_error_capacity(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::Capacity)
    }
    fn allocation_error_internal_error(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::Internal)
    }
    fn allocation_error_invalid_request(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::InvalidRequest)
    }
    fn allocation_error_none(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None)
    }
    fn allocation_error_unknown(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(!matches!(
            self.error,
            AllocationError::None
                | AllocationError::InvalidRequest
                | AllocationError::Capacity
                | AllocationError::Internal
                | AllocationError::Untracked
        ))
    }
    fn allocation_error_untracked(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::Untracked)
    }
    fn begin_allocate(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> {
        self.error = AllocationError::None;
        self.liveness_outcome = PhaseOutcome::Unknown;
        self.ordering_outcome = PhaseOutcome::Unknown;
        self.placement_outcome = PhaseOutcome::Unknown;
        self.required_intervals = 0;
        self.sorted_tensor_count = 0;
        self.required_buffer_bytes = 0;
        self.plan = AllocationPlan::default();
        self.dispatch_generation = self.dispatch_generation.wrapping_add(1);
        Ok(())
    }
    fn commit_plan(&mut self, _: &EventAllocateGraphPlan) -> Result<(), ()> {
        self.plan = AllocationPlan {
            tensor_count: self.sorted_tensor_count,
            interval_count: self.required_intervals,
            required_buffer_bytes: self.required_buffer_bytes,
        };
        Ok(())
    }
    fn dispatch_done(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> {
        if let Some(callback) = event.request.dispatch_done {
            callback(AllocationDone { plan: self.plan });
        }
        Ok(())
    }
    fn dispatch_error_from_allocation_decision(
        &mut self,
        event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        if let Some(callback) = event.request.dispatch_error {
            callback(AllocationErrorEvent {
                plan: self.plan,
                error: self.error,
            });
        }
        Ok(())
    }
    fn invalid_allocate_with_dispatchable_output(
        &self,
        event: &EventAllocateGraphPlan,
    ) -> Result<bool, ()> {
        Ok(!self.valid_allocate(event)?
            && event.request.plan_out
            && event.request.dispatch_error.is_some())
    }
    fn invalid_allocate_with_output_only(
        &self,
        event: &EventAllocateGraphPlan,
    ) -> Result<bool, ()> {
        Ok(!self.valid_allocate(event)?
            && event.request.plan_out
            && event.request.dispatch_error.is_none())
    }
    fn reject_invalid_allocate_without_output(
        &mut self,
        _: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        self.error = AllocationError::InvalidRequest;
        Ok(())
    }
    fn invalid_allocate_without_output(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(!self.valid_allocate(event)? && !event.request.plan_out)
    }
    fn liveness_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.liveness_outcome == PhaseOutcome::Done && self.error == AllocationError::None)
    }
    fn liveness_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.liveness_outcome == PhaseOutcome::Failed || self.error != AllocationError::None)
    }
    fn on_unexpected_from_allocation_decision(&mut self) -> Result<(), ()> {
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_liveness_decision(&mut self) -> Result<(), ()> {
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_ordering_decision(&mut self) -> Result<(), ()> {
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_placement_decision(&mut self) -> Result<(), ()> {
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn ordering_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.ordering_outcome == PhaseOutcome::Done && self.error == AllocationError::None)
    }
    fn ordering_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.ordering_outcome == PhaseOutcome::Failed || self.error != AllocationError::None)
    }
    fn placement_done(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.placement_outcome == PhaseOutcome::Done && self.error == AllocationError::None)
    }
    fn placement_failed(&self, _: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.placement_outcome == PhaseOutcome::Failed || self.error != AllocationError::None)
    }
    fn reject_invalid_allocate_with_dispatch(
        &mut self,
        event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        self.error = AllocationError::InvalidRequest;
        self.plan = AllocationPlan::default();
        if let Some(callback) = event.request.dispatch_error {
            callback(AllocationErrorEvent {
                plan: self.plan,
                error: self.error,
            });
        }
        Ok(())
    }
    fn reject_invalid_allocate_with_output_only(
        &mut self,
        _: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        self.error = AllocationError::InvalidRequest;
        self.plan = AllocationPlan::default();
        Ok(())
    }
    fn run_liveness(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> {
        let request = liveness::LivenessGraphRequest::new(
            event.request.graph_topology != 0,
            event.request.node_count,
            event.request.tensor_count,
            event.request.tensor_capacity,
        );
        let ok = self.liveness.process_event(liveness::LivenessEventAllocateGraphPlan::with_error(
            request,
            map_error_to_liveness(self.error),
        ));
        self.liveness_outcome = map_liveness_outcome(self.liveness.context().outcome());
        self.required_intervals = self.liveness.context().required_intervals;
        self.error = map_liveness_error(self.liveness.context().error());
        if !ok && self.error == AllocationError::None {
            self.error = AllocationError::Internal;
        }
        Ok(())
    }
    fn run_ordering(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> {
        let pass_event = ordering::AllocatorEventAllocateGraphPlan::with_error(
            map_phase_to_ordering(self.liveness_outcome),
            self.required_intervals,
            event.request.interval_capacity,
            event.request.bytes_per_tensor,
            map_error_to_ordering(self.error),
        );
        let ok = self.ordering.process_event(pass_event);
        self.ordering_outcome = map_ordering_outcome(self.ordering.context().outcome());
        self.sorted_tensor_count = self.ordering.context().sorted_tensor_count;
        self.required_buffer_bytes = self.ordering.context().required_buffer_bytes;
        self.error = map_ordering_error(self.ordering.context().error());
        if !ok && self.error == AllocationError::None {
            self.error = AllocationError::Internal;
        }
        Ok(())
    }
    fn run_placement(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> {
        let pass_event = placement::AllocatorEventAllocateGraphPlan::with_error(
            map_phase_to_placement(self.ordering_outcome),
            self.sorted_tensor_count,
            self.required_buffer_bytes,
            event.request.workspace_capacity_bytes,
            event.request.plan_out,
            map_error_to_placement(self.error),
        );
        let ok = self.placement.process_event(pass_event);
        self.placement_outcome = map_placement_outcome(self.placement.context().outcome());
        self.error = map_placement_error(self.placement.context().error());
        if !ok && self.error == AllocationError::None {
            self.error = AllocationError::Internal;
        }
        Ok(())
    }
    fn valid_allocate(&self, event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(event.request.graph_topology != 0
            && event.request.plan_out
            && event.request.node_count != 0
            && event.request.tensor_count != 0
            && event.request.tensor_capacity != 0
            && event.request.interval_capacity != 0
            && event.request.bytes_per_tensor != 0
            && event.request.workspace_capacity_bytes != 0
            && event.request.dispatch_done.is_some()
            && event.request.dispatch_error.is_some())
    }
}

/// Single-writer allocator actor.
pub struct Allocator {
    machine: GraphAllocatorStateMachine<GraphAllocatorContext>,
}
impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}
impl Allocator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAllocatorStateMachine::new(GraphAllocatorContext::default()),
        }
    }
    pub fn process_event(&mut self, event: EventAllocateGraphPlan) -> bool {
        self.machine.process_event(event).is_ok()
    }
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&GraphAllocatorStates::Ready)
    }
    #[must_use]
    pub fn plan(&self) -> AllocationPlan {
        self.machine.context().plan
    }
    #[must_use]
    pub fn error(&self) -> AllocationError {
        self.machine.context().error
    }
}
