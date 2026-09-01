//! Bounded graph-allocation ordering phase.
//!
//! Source mapping: `emel.cpp/src/emel/graph/allocator/ordering_pass/{sm,events,guards,actions}.hpp`.

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
    pub liveness_outcome: PhaseOutcome,
    pub required_intervals: u32,
    pub interval_capacity: u32,
    pub bytes_per_tensor: u64,
    pub error: AllocationError,
}

impl AllocatorEventAllocateGraphPlan {
    #[must_use]
    pub const fn new(
        liveness_outcome: PhaseOutcome,
        required_intervals: u32,
        interval_capacity: u32,
        bytes_per_tensor: u64,
    ) -> Self {
        Self {
            liveness_outcome,
            required_intervals,
            interval_capacity,
            bytes_per_tensor,
            error: AllocationError::None,
        }
    }

    #[must_use]
    pub const fn with_error(
        liveness_outcome: PhaseOutcome,
        required_intervals: u32,
        interval_capacity: u32,
        bytes_per_tensor: u64,
        error: AllocationError,
    ) -> Self {
        Self {
            liveness_outcome,
            required_intervals,
            interval_capacity,
            bytes_per_tensor,
            error,
        }
    }
}

sml! {
    GraphAllocatorOrderingPass {
        "allocate_failed"_s <= *"deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prefailed] / mark_failed_prefailed,
        "allocated"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_done] / mark_done,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prereq_failed] / mark_failed_prereq,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_capacity_exceeded] / mark_failed_capacity,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_overflow] / mark_failed_overflow,
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
pub struct GraphAllocatorOrderingPassContext {
    pub outcome: PhaseOutcome,
    pub error: AllocationError,
    pub liveness_outcome: PhaseOutcome,
    pub required_intervals: u32,
    pub interval_capacity: u32,
    pub bytes_per_tensor: u64,
    pub sorted_tensor_count: u32,
    pub required_buffer_bytes: u64,
}
impl GraphAllocatorOrderingPassContext {
    pub fn set_request(
        &mut self,
        liveness_outcome: PhaseOutcome,
        required_intervals: u32,
        interval_capacity: u32,
        bytes_per_tensor: u64,
    ) {
        self.set_event(AllocatorEventAllocateGraphPlan::new(
            liveness_outcome,
            required_intervals,
            interval_capacity,
            bytes_per_tensor,
        ));
    }

    pub fn set_event(&mut self, event: AllocatorEventAllocateGraphPlan) {
        self.outcome = PhaseOutcome::Unknown;
        self.error = event.error;
        self.liveness_outcome = event.liveness_outcome;
        self.required_intervals = event.required_intervals;
        self.interval_capacity = event.interval_capacity;
        self.bytes_per_tensor = event.bytes_per_tensor;
        self.sorted_tensor_count = 0;
        self.required_buffer_bytes = 0;
    }
}

impl GraphAllocatorOrderingPassContext {
    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.outcome }

    #[must_use]
    pub const fn error(&self) -> AllocationError { self.error }
}

fn overflow(lhs: u32, rhs: u64) -> bool {
    lhs != 0 && rhs > u64::MAX / u64::from(lhs)
}
impl GraphAllocatorOrderingPassStateMachineContext for GraphAllocatorOrderingPassContext {
    fn mark_done(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Done;
        self.sorted_tensor_count = self.required_intervals;
        self.required_buffer_bytes = u64::from(self.required_intervals) * self.bytes_per_tensor;
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
    fn mark_failed_overflow(&mut self) -> Result<(), ()> {
        self.outcome = PhaseOutcome::Failed;
        self.error = AllocationError::Capacity;
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
            && self.liveness_outcome == PhaseOutcome::Done
            && self.required_intervals > self.interval_capacity)
    }
    fn phase_done(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.liveness_outcome == PhaseOutcome::Done
            && self.required_intervals != 0
            && self.required_intervals <= self.interval_capacity
            && self.bytes_per_tensor != 0
            && !overflow(self.required_intervals, self.bytes_per_tensor))
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.liveness_outcome == PhaseOutcome::Done
            && (self.required_intervals == 0 || self.bytes_per_tensor == 0))
    }
    fn phase_overflow(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None
            && self.liveness_outcome == PhaseOutcome::Done
            && self.required_intervals != 0
            && self.required_intervals <= self.interval_capacity
            && self.bytes_per_tensor != 0
            && overflow(self.required_intervals, self.bytes_per_tensor))
    }
    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.error != AllocationError::None)
    }
    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.error == AllocationError::None && self.liveness_outcome != PhaseOutcome::Done)
    }
}

pub struct GraphAllocatorOrderingPassActor {
    machine: GraphAllocatorOrderingPassStateMachine<GraphAllocatorOrderingPassContext>,
}

impl Default for GraphAllocatorOrderingPassActor {
    fn default() -> Self { Self::new() }
}

impl GraphAllocatorOrderingPassActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAllocatorOrderingPassStateMachine::new(
                GraphAllocatorOrderingPassContext::default(),
            ),
        }
    }

    pub fn process_event(&mut self, event: AllocatorEventAllocateGraphPlan) -> bool {
        if !self.machine.is(&GraphAllocatorOrderingPassStates::Deciding) {
            self.machine.context_mut().outcome = PhaseOutcome::Failed;
            self.machine.context_mut().error = AllocationError::Internal;
            return false;
        }
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAllocatorOrderingPassEvents::AllocatorEventAllocateGraphPlan)
            .is_ok()
    }

    #[must_use]
    pub fn context(&self) -> &GraphAllocatorOrderingPassContext { self.machine.context() }
}

pub type Actor = GraphAllocatorOrderingPassActor;
