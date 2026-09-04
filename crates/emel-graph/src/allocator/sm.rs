//! Bounded graph allocator state machine.
//!
//! The allocator owns three terminal, synchronously joined phase actors. The
//! pinned `stateforward-sml` composite DSL can compose adjacent tables, but it
//! cannot embed already-generated actor values, so this parent keeps explicit
//! phase ownership and ordering in a flat table. Phase results stay in their
//! owning child contexts and are consumed immediately by the next typed phase
//! event; they are not mirrored in the parent context.
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
    liveness_pass::sm::GraphAllocatorLivenessPassActor,
    ordering_pass::sm::GraphAllocatorOrderingPassActor,
    placement_pass::sm::GraphAllocatorPlacementPassActor,
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
    liveness: GraphAllocatorLivenessPassActor,
    ordering: GraphAllocatorOrderingPassActor,
    placement: GraphAllocatorPlacementPassActor,
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
            liveness: GraphAllocatorLivenessPassActor::new(),
            ordering: GraphAllocatorOrderingPassActor::new(),
            placement: GraphAllocatorPlacementPassActor::new(),
        }
    }
}

fn map_liveness_outcome(outcome: super::liveness_pass::sm::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        super::liveness_pass::sm::PhaseOutcome::Done => PhaseOutcome::Done,
        super::liveness_pass::sm::PhaseOutcome::Failed => PhaseOutcome::Failed,
        super::liveness_pass::sm::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_liveness_error(error: super::liveness_pass::sm::AllocationError) -> AllocationError {
    match error {
        super::liveness_pass::sm::AllocationError::None => AllocationError::None,
        super::liveness_pass::sm::AllocationError::InvalidRequest => {
            AllocationError::InvalidRequest
        }
        super::liveness_pass::sm::AllocationError::Capacity => AllocationError::Capacity,
        super::liveness_pass::sm::AllocationError::Internal => AllocationError::Internal,
        super::liveness_pass::sm::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_liveness(error: AllocationError) -> super::liveness_pass::sm::AllocationError {
    match error {
        AllocationError::None => super::liveness_pass::sm::AllocationError::None,
        AllocationError::InvalidRequest => {
            super::liveness_pass::sm::AllocationError::InvalidRequest
        }
        AllocationError::Capacity => super::liveness_pass::sm::AllocationError::Capacity,
        AllocationError::Internal => super::liveness_pass::sm::AllocationError::Internal,
        AllocationError::Untracked => super::liveness_pass::sm::AllocationError::Untracked,
    }
}
fn map_phase_to_ordering(outcome: PhaseOutcome) -> super::ordering_pass::sm::PhaseOutcome {
    match outcome {
        PhaseOutcome::Done => super::ordering_pass::sm::PhaseOutcome::Done,
        PhaseOutcome::Failed => super::ordering_pass::sm::PhaseOutcome::Failed,
        PhaseOutcome::Unknown => super::ordering_pass::sm::PhaseOutcome::Unknown,
    }
}
fn map_ordering_outcome(outcome: super::ordering_pass::sm::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        super::ordering_pass::sm::PhaseOutcome::Done => PhaseOutcome::Done,
        super::ordering_pass::sm::PhaseOutcome::Failed => PhaseOutcome::Failed,
        super::ordering_pass::sm::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_ordering_error(error: super::ordering_pass::sm::AllocationError) -> AllocationError {
    match error {
        super::ordering_pass::sm::AllocationError::None => AllocationError::None,
        super::ordering_pass::sm::AllocationError::InvalidRequest => {
            AllocationError::InvalidRequest
        }
        super::ordering_pass::sm::AllocationError::Capacity => AllocationError::Capacity,
        super::ordering_pass::sm::AllocationError::Internal => AllocationError::Internal,
        super::ordering_pass::sm::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_ordering(error: AllocationError) -> super::ordering_pass::sm::AllocationError {
    match error {
        AllocationError::None => super::ordering_pass::sm::AllocationError::None,
        AllocationError::InvalidRequest => {
            super::ordering_pass::sm::AllocationError::InvalidRequest
        }
        AllocationError::Capacity => super::ordering_pass::sm::AllocationError::Capacity,
        AllocationError::Internal => super::ordering_pass::sm::AllocationError::Internal,
        AllocationError::Untracked => super::ordering_pass::sm::AllocationError::Untracked,
    }
}
fn map_phase_to_placement(outcome: PhaseOutcome) -> super::placement_pass::sm::PhaseOutcome {
    match outcome {
        PhaseOutcome::Done => super::placement_pass::sm::PhaseOutcome::Done,
        PhaseOutcome::Failed => super::placement_pass::sm::PhaseOutcome::Failed,
        PhaseOutcome::Unknown => super::placement_pass::sm::PhaseOutcome::Unknown,
    }
}
fn map_placement_outcome(outcome: super::placement_pass::sm::PhaseOutcome) -> PhaseOutcome {
    match outcome {
        super::placement_pass::sm::PhaseOutcome::Done => PhaseOutcome::Done,
        super::placement_pass::sm::PhaseOutcome::Failed => PhaseOutcome::Failed,
        super::placement_pass::sm::PhaseOutcome::Unknown => PhaseOutcome::Unknown,
    }
}
fn map_placement_error(error: super::placement_pass::sm::AllocationError) -> AllocationError {
    match error {
        super::placement_pass::sm::AllocationError::None => AllocationError::None,
        super::placement_pass::sm::AllocationError::InvalidRequest => {
            AllocationError::InvalidRequest
        }
        super::placement_pass::sm::AllocationError::Capacity => AllocationError::Capacity,
        super::placement_pass::sm::AllocationError::Internal => AllocationError::Internal,
        super::placement_pass::sm::AllocationError::Untracked => AllocationError::Untracked,
    }
}
fn map_error_to_placement(error: AllocationError) -> super::placement_pass::sm::AllocationError {
    match error {
        AllocationError::None => super::placement_pass::sm::AllocationError::None,
        AllocationError::InvalidRequest => {
            super::placement_pass::sm::AllocationError::InvalidRequest
        }
        AllocationError::Capacity => super::placement_pass::sm::AllocationError::Capacity,
        AllocationError::Internal => super::placement_pass::sm::AllocationError::Internal,
        AllocationError::Untracked => super::placement_pass::sm::AllocationError::Untracked,
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
        // The C++ allocator re-enters each nested phase model for every valid
        // request. Recreate the terminal child actors at that boundary so a
        // ready allocator remains reusable after a prior allocation.
        self.liveness = GraphAllocatorLivenessPassActor::new();
        self.ordering = GraphAllocatorOrderingPassActor::new();
        self.placement = GraphAllocatorPlacementPassActor::new();
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
        let request = super::liveness_pass::sm::LivenessGraphRequest::new(
            event.request.graph_topology != 0,
            event.request.node_count,
            event.request.tensor_count,
            event.request.tensor_capacity,
        );
        let ok = self.liveness.process_event(
            super::liveness_pass::sm::LivenessEventAllocateGraphPlan::with_error(
                request,
                map_error_to_liveness(self.error),
            ),
        );
        self.liveness_outcome = map_liveness_outcome(self.liveness.context().outcome());
        self.required_intervals = self.liveness.context().required_intervals;
        self.error = map_liveness_error(self.liveness.context().error());
        if !ok && self.error == AllocationError::None {
            self.error = AllocationError::Internal;
        }
        Ok(())
    }
    fn run_ordering(&mut self, event: &EventAllocateGraphPlan) -> Result<(), ()> {
        let pass_event = super::ordering_pass::sm::AllocatorEventAllocateGraphPlan::with_error(
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
        let pass_event = super::placement_pass::sm::AllocatorEventAllocateGraphPlan::with_error(
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

impl core::fmt::Debug for Allocator {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Allocator")
            .field("ready", &self.is_ready())
            .field("error", &self.error())
            .finish()
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
        let accepted = self.machine.process_event(event).is_ok();
        accepted && self.machine.context().error == AllocationError::None
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        AllocateGraph, AllocationDone, AllocationErrorEvent, AllocationPlan, Allocator,
        EventAllocateGraphPlan,
    };

    static DONE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static ERROR_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

    fn dispatch_done(_: AllocationDone) -> bool {
        DONE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn dispatch_error(_: AllocationErrorEvent) -> bool {
        ERROR_CALLBACKS.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn valid_request() -> EventAllocateGraphPlan {
        EventAllocateGraphPlan {
            request: AllocateGraph {
                graph_topology: 1,
                plan_out: true,
                node_count: 2,
                tensor_count: 3,
                tensor_capacity: 4,
                interval_capacity: 4,
                bytes_per_tensor: 8,
                workspace_capacity_bytes: 32,
                dispatch_done: Some(dispatch_done),
                dispatch_error: Some(dispatch_error),
            },
        }
    }

    #[test]
    fn ready_allocator_reuses_child_phases_for_repeated_requests() {
        DONE_CALLBACKS.store(0, Ordering::Relaxed);
        ERROR_CALLBACKS.store(0, Ordering::Relaxed);
        let mut allocator = Allocator::new();

        assert!(allocator.process_event(valid_request()));
        assert!(allocator.is_ready());
        assert_eq!(allocator.error(), super::AllocationError::None);
        assert_eq!(
            allocator.plan(),
            AllocationPlan {
                tensor_count: 3,
                interval_count: 3,
                required_buffer_bytes: 24,
            }
        );

        assert!(allocator.process_event(valid_request()));
        assert!(allocator.is_ready());
        assert_eq!(allocator.error(), super::AllocationError::None);
        assert_eq!(DONE_CALLBACKS.load(Ordering::Relaxed), 2);
        assert_eq!(ERROR_CALLBACKS.load(Ordering::Relaxed), 0);
    }
}
