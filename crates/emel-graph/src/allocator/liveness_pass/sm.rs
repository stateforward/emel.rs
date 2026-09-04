//! Bounded graph-allocation liveness phase.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/liveness_pass/{sm,events,guards,actions}.hpp`.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AllocationError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Capacity = 2,
    Internal = 4,
    Untracked = 8,
}

/// Completion event carrying copied request data and the preceding allocator
/// error, matching the pinned C++ `allocate_graph_plan` context boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LivenessEventAllocateGraphPlan {
    pub request: LivenessGraphRequest,
    pub error: AllocationError,
}

impl LivenessEventAllocateGraphPlan {
    #[must_use]
    pub const fn new(request: LivenessGraphRequest) -> Self {
        Self {
            request,
            error: AllocationError::None,
        }
    }

    #[must_use]
    pub const fn with_error(request: LivenessGraphRequest, error: AllocationError) -> Self {
        Self { request, error }
    }
}

/// Copied scalar request consumed by the liveness guards.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LivenessGraphRequest {
    pub has_graph_topology: bool,
    pub node_count: u32,
    pub tensor_count: u32,
    pub tensor_capacity: u32,
}

impl LivenessGraphRequest {
    #[must_use]
    pub const fn new(
        has_graph_topology: bool,
        node_count: u32,
        tensor_count: u32,
        tensor_capacity: u32,
    ) -> Self {
        Self {
            has_graph_topology,
            node_count,
            tensor_count,
            tensor_capacity,
        }
    }
}

sml! {
    GraphAllocatorLivenessPass {
        "allocate_failed"_s <= *"deciding"_s + event<LivenessEventAllocateGraphPlan> [phase_prefailed] / mark_failed_prefailed,
        "allocated"_s <= "deciding"_s + event<LivenessEventAllocateGraphPlan> [phase_done] / mark_done,
        "allocate_failed"_s <= "deciding"_s + event<LivenessEventAllocateGraphPlan> [phase_invalid_request] / mark_failed_invalid_request,
        "allocate_failed"_s <= "deciding"_s + event<LivenessEventAllocateGraphPlan> [phase_capacity_exceeded] / mark_failed_capacity,
        "allocate_failed"_s <= "deciding"_s + event<LivenessEventAllocateGraphPlan> / mark_failed_internal,
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
    pub fn set_request(
        &mut self,
        has_graph_topology: bool,
        node_count: u32,
        tensor_count: u32,
        tensor_capacity: u32,
    ) {
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
    fn mark_done(&mut self, _: &LivenessEventAllocateGraphPlan) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Done;
        self.required_intervals = self.tensor_count;
        self.error = AllocationError::None;
        Ok(())
    }
    fn mark_failed_capacity(&mut self, _: &LivenessEventAllocateGraphPlan) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Capacity;
        Ok(())
    }
    fn mark_failed_internal(&mut self, _: &LivenessEventAllocateGraphPlan) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn mark_failed_invalid_request(
        &mut self,
        _: &LivenessEventAllocateGraphPlan,
    ) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::InvalidRequest;
        Ok(())
    }
    fn mark_failed_prefailed(&mut self, _: &LivenessEventAllocateGraphPlan) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        Ok(())
    }
    fn on_unexpected_from_allocate_failed(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_allocated(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn phase_capacity_exceeded(&self, _: &LivenessEventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.has_graph_topology
            && self.node_count != 0
            && self.tensor_count != 0
            && self.tensor_count > self.tensor_capacity)
    }
    fn phase_done(&self, _: &LivenessEventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.has_graph_topology
            && self.node_count != 0
            && self.tensor_count != 0
            && self.tensor_count <= self.tensor_capacity)
    }
    fn phase_invalid_request(&self, _: &LivenessEventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && (!self.has_graph_topology || self.node_count == 0 || self.tensor_count == 0))
    }
    fn phase_prefailed(&self, _: &LivenessEventAllocateGraphPlan) -> Result<bool, ()> {
        Ok(self.error != AllocationError::None)
    }
}

impl GraphAllocatorLivenessPassContext {
    /// Copies one bounded completion event into the pass context.
    pub fn set_event(&mut self, event: LivenessEventAllocateGraphPlan) {
        self.has_graph_topology = event.request.has_graph_topology;
        self.node_count = event.request.node_count;
        self.tensor_count = event.request.tensor_count;
        self.tensor_capacity = event.request.tensor_capacity;
        self.outcome = PhaseOutcome::Unknown;
        self.error = event.error;
        self.required_intervals = 0;
    }

    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome {
        self.outcome
    }

    #[must_use]
    pub const fn error(&self) -> AllocationError {
        self.error
    }
}

/// Synchronous single-writer actor around the generated liveness pass machine.
pub struct GraphAllocatorLivenessPassActor {
    machine: GraphAllocatorLivenessPassStateMachine<GraphAllocatorLivenessPassContext>,
}

impl Default for GraphAllocatorLivenessPassActor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAllocatorLivenessPassActor {
    /// Creates an actor in the generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAllocatorLivenessPassStateMachine::new(
                GraphAllocatorLivenessPassContext::default(),
            ),
        }
    }

    /// Copies one completion event and dispatches it synchronously.
    pub fn process_event(&mut self, event: LivenessEventAllocateGraphPlan) -> bool {
        if !self.machine.is(&GraphAllocatorLivenessPassStates::Deciding) {
            self.machine.context_mut().outcome = PhaseOutcome::Failed;
            self.machine.context_mut().error = AllocationError::Internal;
            return false;
        }
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAllocatorLivenessPassEvents::LivenessEventAllocateGraphPlan(event))
            .is_ok()
    }

    /// Records an explicit unexpected event and enters its generated state.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.context_mut().outcome = PhaseOutcome::Failed;
        self.machine.context_mut().error = AllocationError::Internal;
        self.machine
            .set_state(GraphAllocatorLivenessPassStates::UnexpectedEvent);
        false
    }

    #[must_use]
    pub fn state(&self) -> &GraphAllocatorLivenessPassStates {
        self.machine.state()
    }

    #[must_use]
    pub fn is(&self, state: &GraphAllocatorLivenessPassStates) -> bool {
        self.machine.is(state)
    }

    #[must_use]
    pub fn context(&self) -> &GraphAllocatorLivenessPassContext {
        self.machine.context()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AllocationError, GraphAllocatorLivenessPassActor, LivenessEventAllocateGraphPlan,
        LivenessGraphRequest, PhaseOutcome,
    };

    fn request(
        topology: bool,
        nodes: u32,
        tensors: u32,
        capacity: u32,
    ) -> LivenessEventAllocateGraphPlan {
        LivenessEventAllocateGraphPlan::new(LivenessGraphRequest::new(
            topology, nodes, tensors, capacity,
        ))
    }

    #[test]
    fn accepts_valid_topology_and_records_required_intervals() {
        let mut actor = GraphAllocatorLivenessPassActor::new();
        assert!(actor.process_event(request(true, 2, 3, 4)));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Done);
        assert_eq!(actor.context().error(), AllocationError::None);
        assert_eq!(actor.context().required_intervals, 3);
    }

    #[test]
    fn distinguishes_invalid_request_and_capacity() {
        let mut invalid = GraphAllocatorLivenessPassActor::new();
        assert!(invalid.process_event(request(false, 2, 3, 4)));
        assert_eq!(invalid.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(invalid.context().error(), AllocationError::InvalidRequest);

        let mut capacity = GraphAllocatorLivenessPassActor::new();
        assert!(capacity.process_event(request(true, 2, 5, 4)));
        assert_eq!(capacity.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(capacity.context().error(), AllocationError::Capacity);
    }

    #[test]
    fn propagates_prefailed_context_and_rejects_reentrant_completion() {
        let mut prefailed = GraphAllocatorLivenessPassActor::new();
        assert!(
            prefailed.process_event(LivenessEventAllocateGraphPlan::with_error(
                LivenessGraphRequest::new(true, 2, 3, 4),
                AllocationError::Capacity,
            ))
        );
        assert_eq!(prefailed.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(prefailed.context().error(), AllocationError::Capacity);

        let mut actor = GraphAllocatorLivenessPassActor::new();
        assert!(actor.process_event(request(true, 1, 1, 1)));
        assert!(!actor.process_event(request(true, 1, 1, 1)));
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), AllocationError::Internal);
    }

    #[test]
    fn explicit_unexpected_event_is_typed_and_terminal() {
        let mut actor = GraphAllocatorLivenessPassActor::new();
        assert!(!actor.process_unexpected_event());
        assert_eq!(actor.context().outcome(), PhaseOutcome::Failed);
        assert_eq!(actor.context().error(), AllocationError::Internal);
    }
}
