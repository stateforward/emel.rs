//! Bounded graph-assembler reserve-allocation phase state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/assembler/reserve_alloc_pass/{sm,context,events,actions,guards,errors}.hpp`.
//! Requests and phase results are copied into the single-writer context before synchronous
//! dispatch. Actions are bounded, allocation-free, and non-routing.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]
use crate::allocator::sm as allocator;
use crate::assembler::sm::AllocationPlan;
use sml::sml;

/// Outcome retained by the reserve-allocation phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Error values matching the assembler error contract.
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

/// Short alias for callers using the phase's error type.
pub type Error = AllocationError;

/// Copied reserve request fields used by source guards and actions.
///
/// C++ pointer-bearing values are represented by bounded presence bits; this child never
/// dereferences or retains caller-owned objects.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReserveGraphRequest {
    pub has_model_topology: bool,
    pub has_output_out: bool,
    pub max_node_count: u32,
    pub max_tensor_count: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
}

impl ReserveGraphRequest {
    #[must_use]
    pub const fn new(
        has_model_topology: bool,
        has_output_out: bool,
        max_node_count: u32,
        max_tensor_count: u32,
        bytes_per_tensor: u64,
        workspace_capacity_bytes: u64,
    ) -> Self {
        Self {
            has_model_topology,
            has_output_out,
            max_node_count,
            max_tensor_count,
            bytes_per_tensor,
            workspace_capacity_bytes,
        }
    }
}

/// Runtime event shell corresponding to C++ `assembler::event::reserve_graph`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventReserveGraph {
    pub request: ReserveGraphRequest,
    pub build_outcome: PhaseOutcome,
    pub err: AllocationError,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
}

impl AssemblerEventReserveGraph {
    #[must_use]
    pub const fn new(request: ReserveGraphRequest) -> Self {
        Self {
            request,
            build_outcome: PhaseOutcome::Unknown,
            err: AllocationError::None,
            assembled_node_count: 0,
            assembled_tensor_count: 0,
        }
    }

    #[must_use]
    pub const fn with_prerequisites(
        request: ReserveGraphRequest,
        build_outcome: PhaseOutcome,
        err: AllocationError,
        assembled_node_count: u32,
        assembled_tensor_count: u32,
    ) -> Self {
        Self {
            request,
            build_outcome,
            err,
            assembled_node_count,
            assembled_tensor_count,
        }
    }
}

sml! {
    GraphAssemblerReserveAllocPass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventReserveGraph> [phase_request_allocator] / request_allocator_plan,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Persistent bounded context for `GraphAssemblerReserveAllocPass`.
#[derive(Default)]
pub struct GraphAssemblerReserveAllocPassContext {
    pub request: ReserveGraphRequest,
    pub alloc_outcome: PhaseOutcome,
    pub err: AllocationError,
    pub build_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub alloc_plan: AllocationPlan,
    allocator: allocator::Allocator,
}

impl GraphAssemblerReserveAllocPassContext {
    /// Copies one request and prerequisite state into the actor context.
    pub fn set_request(&mut self, event: AssemblerEventReserveGraph) {
        self.request = event.request;
        self.alloc_outcome = PhaseOutcome::Unknown;
        self.err = event.err;
        self.build_outcome = event.build_outcome;
        self.assembled_node_count = event.assembled_node_count;
        self.assembled_tensor_count = event.assembled_tensor_count;
        self.alloc_plan = AllocationPlan::default();
    }

    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome {
        self.alloc_outcome
    }

    #[must_use]
    pub const fn error(&self) -> AllocationError {
        self.err
    }

    #[must_use]
    pub const fn plan(&self) -> AllocationPlan {
        self.alloc_plan
    }
}

const fn map_allocator_error(error: allocator::AllocationError) -> AllocationError {
    match error {
        allocator::AllocationError::None => AllocationError::None,
        allocator::AllocationError::InvalidRequest => AllocationError::InvalidRequest,
        allocator::AllocationError::Capacity => AllocationError::Capacity,
        allocator::AllocationError::Internal => AllocationError::Internal,
        allocator::AllocationError::Untracked => AllocationError::Untracked,
    }
}

const fn allocation_done(_: allocator::AllocationDone) -> bool {
    true
}

const fn allocation_error(_: allocator::AllocationErrorEvent) -> bool {
    true
}

impl GraphAssemblerReserveAllocPassStateMachineContext for GraphAssemblerReserveAllocPassContext {
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::InvalidRequest;
        Ok(())
    }

    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::Internal;
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::Internal;
        Ok(())
    }

    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::Internal;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::Internal;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AllocationError::Internal;
        Ok(())
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.err == AllocationError::None
            && self.build_outcome == PhaseOutcome::Done
            && (!self.request.has_model_topology
                || !self.request.has_output_out
                || self.assembled_node_count == 0
                || self.assembled_tensor_count == 0))
    }

    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.err == AllocationError::None && self.build_outcome != PhaseOutcome::Done)
    }

    fn phase_request_allocator(&self) -> Result<bool, ()> {
        Ok(self.err == AllocationError::None
            && self.build_outcome == PhaseOutcome::Done
            && self.request.has_model_topology
            && self.request.has_output_out
            && self.assembled_node_count != 0
            && self.assembled_tensor_count != 0)
    }

    fn request_allocator_plan(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.alloc_plan = AllocationPlan::default();
        self.err = AllocationError::Internal;

        let request = allocator::EventAllocateGraphPlan {
            request: allocator::AllocateGraph {
                graph_topology: 1,
                plan_out: self.request.has_output_out,
                node_count: self.assembled_node_count,
                tensor_count: self.assembled_tensor_count,
                tensor_capacity: self.assembled_tensor_count,
                interval_capacity: self.assembled_tensor_count,
                bytes_per_tensor: self.request.bytes_per_tensor,
                workspace_capacity_bytes: self.request.workspace_capacity_bytes,
                dispatch_done: Some(allocation_done),
                dispatch_error: Some(allocation_error),
            },
        };
        let accepted = self.allocator.process_event(request);
        let error = map_allocator_error(self.allocator.error());
        self.err = error;
        if accepted && error == AllocationError::None {
            let plan = self.allocator.plan();
            self.alloc_plan = AllocationPlan {
                tensor_count: plan.tensor_count,
                interval_count: plan.interval_count,
                required_buffer_bytes: plan.required_buffer_bytes,
            };
            self.alloc_outcome = PhaseOutcome::Done;
        }
        Ok(())
    }
}

/// Synchronous, single-writer reserve-allocation actor.
pub struct GraphAssemblerReserveAllocPass {
    machine: GraphAssemblerReserveAllocPassStateMachine<GraphAssemblerReserveAllocPassContext>,
}

impl Default for GraphAssemblerReserveAllocPass {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAssemblerReserveAllocPass {
    /// Constructs the actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerReserveAllocPassStateMachine::new(
                GraphAssemblerReserveAllocPassContext::default(),
            ),
        }
    }

    /// Copies and dispatches one reserve-allocation completion synchronously.
    pub fn process_event(&mut self, event: AssemblerEventReserveGraph) -> bool {
        if !self
            .machine
            .is(&GraphAssemblerReserveAllocPassStates::Deciding)
        {
            self.machine.context_mut().alloc_outcome = PhaseOutcome::Failed;
            self.machine.context_mut().err = AllocationError::Internal;
            return false;
        }
        self.machine.context_mut().set_request(event);
        self.machine
            .process_event(GraphAssemblerReserveAllocPassEvents::AssemblerEventReserveGraph)
            .is_ok()
    }

    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.context_mut().alloc_outcome = PhaseOutcome::Failed;
        self.machine.context_mut().err = AllocationError::Internal;
        self.machine
            .set_state(GraphAssemblerReserveAllocPassStates::UnexpectedEvent);
        false
    }

    #[must_use]
    pub fn state(&self) -> &GraphAssemblerReserveAllocPassStates {
        self.machine.state()
    }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphAssemblerReserveAllocPassStates) -> bool {
        self.machine.is(&state)
    }

    /// Returns the retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerReserveAllocPassContext {
        self.machine.context()
    }
}
