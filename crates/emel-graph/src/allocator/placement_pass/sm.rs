//! Bounded graph-allocation placement phase.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/placement_pass/{sm,events,guards,actions}.hpp`.

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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocatorEventAllocateGraphPlan {
    pub ordering_outcome: PhaseOutcome,
    pub sorted_tensor_count: u32,
    pub required_buffer_bytes: u64,
    pub workspace_capacity_bytes: u64,
    pub has_plan_output: bool,
    pub error: AllocationError,
}

impl AllocatorEventAllocateGraphPlan {
    #[must_use]
    pub const fn new(
        ordering_outcome: PhaseOutcome,
        sorted_tensor_count: u32,
        required_buffer_bytes: u64,
        workspace_capacity_bytes: u64,
        has_plan_output: bool,
    ) -> Self {
        Self {
            ordering_outcome,
            sorted_tensor_count,
            required_buffer_bytes,
            workspace_capacity_bytes,
            has_plan_output,
            error: AllocationError::None,
        }
    }

    #[must_use]
    pub const fn with_error(
        ordering_outcome: PhaseOutcome,
        sorted_tensor_count: u32,
        required_buffer_bytes: u64,
        workspace_capacity_bytes: u64,
        has_plan_output: bool,
        error: AllocationError,
    ) -> Self {
        Self {
            ordering_outcome,
            sorted_tensor_count,
            required_buffer_bytes,
            workspace_capacity_bytes,
            has_plan_output,
            error,
        }
    }
}

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
    pub fn set_event(&mut self, event: AllocatorEventAllocateGraphPlan) {
        self.outcome = PhaseOutcome::Unknown;
        self.error = event.error;
        self.ordering_outcome = event.ordering_outcome;
        self.sorted_tensor_count = event.sorted_tensor_count;
        self.required_buffer_bytes = event.required_buffer_bytes;
        self.workspace_capacity_bytes = event.workspace_capacity_bytes;
        self.has_plan_output = event.has_plan_output;
    }

    pub fn set_request(
        &mut self,
        ordering_outcome: PhaseOutcome,
        sorted_tensor_count: u32,
        required_buffer_bytes: u64,
        workspace_capacity_bytes: u64,
        has_plan_output: bool,
    ) {
        self.set_event(AllocatorEventAllocateGraphPlan::new(
            ordering_outcome,
            sorted_tensor_count,
            required_buffer_bytes,
            workspace_capacity_bytes,
            has_plan_output,
        ));
    }

    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.outcome }

    #[must_use]
    pub const fn error(&self) -> AllocationError { self.error }
}

impl GraphAllocatorPlacementPassStateMachineContext for GraphAllocatorPlacementPassContext {
    fn mark_done(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Done;
        self.error = AllocationError::None;
        Ok(())
    }
    fn mark_failed_capacity(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Capacity;
        Ok(())
    }
    fn mark_failed_internal(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
        Ok(())
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::InvalidRequest;
        Ok(())
    }
    fn mark_failed_prefailed(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        Ok(())
    }
    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Internal;
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
    fn phase_capacity_exceeded(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.ordering_outcome == PhaseOutcome::Done
            && self.required_buffer_bytes > self.workspace_capacity_bytes)
    }
    fn phase_done(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.ordering_outcome == PhaseOutcome::Done
            && self.has_plan_output
            && self.sorted_tensor_count != 0
            && self.required_buffer_bytes <= self.workspace_capacity_bytes)
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.ordering_outcome == PhaseOutcome::Done
            && (!self.has_plan_output || self.sorted_tensor_count == 0))
    }
    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.error != AllocationError::None)
    }
    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None && self.ordering_outcome != PhaseOutcome::Done)
    }
}

pub struct GraphAllocatorPlacementPassActor {
    machine: GraphAllocatorPlacementPassStateMachine<GraphAllocatorPlacementPassContext>,
}

impl Default for GraphAllocatorPlacementPassActor {
    fn default() -> Self { Self::new() }
}

impl GraphAllocatorPlacementPassActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAllocatorPlacementPassStateMachine::new(
                GraphAllocatorPlacementPassContext::default(),
            ),
        }
    }

    pub fn process_event(&mut self, event: AllocatorEventAllocateGraphPlan) -> bool {
        if !self.machine.is(&GraphAllocatorPlacementPassStates::Deciding) {
            self.machine.context_mut().outcome = PhaseOutcome::Failed;
            self.machine.context_mut().error = AllocationError::Internal;
            return false;
        }
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAllocatorPlacementPassEvents::AllocatorEventAllocateGraphPlan)
            .is_ok()
    }

    #[must_use]
    pub fn context(&self) -> &GraphAllocatorPlacementPassContext { self.machine.context() }
}

pub type Actor = GraphAllocatorPlacementPassActor;
