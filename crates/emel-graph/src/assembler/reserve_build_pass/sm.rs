//! Bounded reserve-build phase of the graph assembler.
//!
//! Source mapping: `emel.cpp/src/emel/graph/assembler/reserve_build_pass/{sm,context,events,actions,guards}.hpp`.
//! The event and context contain copied scalar state only; actions are synchronous,
//! allocation-free, and never perform routing.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

/// Outcome of an assembler phase, matching the C++ phase outcome values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Error values used by the assembler phases.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AssemblerError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Capacity = 2,
    Internal = 4,
    Untracked = 8,
}
/// Bounded copied reserve request consumed by the reserve-build actor.
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

/// Runtime event corresponding to C++ `assembler::event::reserve_graph`.
/// Inputs and prerequisite state are copied before dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventReserveGraph {
    pub request: ReserveGraphRequest,
    pub validate_outcome: PhaseOutcome,
    pub build_outcome: PhaseOutcome,
    pub alloc_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub alloc_plan: AllocationPlan,
    pub err: AssemblerError,
}

impl AssemblerEventReserveGraph {
    #[must_use]
    pub const fn new(request: ReserveGraphRequest) -> Self {
        Self {
            request,
            validate_outcome: PhaseOutcome::Unknown,
            build_outcome: PhaseOutcome::Unknown,
            alloc_outcome: PhaseOutcome::Unknown,
            assembled_node_count: 0,
            assembled_tensor_count: 0,
            alloc_plan: AllocationPlan {
                tensor_count: 0,
                interval_count: 0,
                required_buffer_bytes: 0,
            },
            err: AssemblerError::None,
        }
    }

    #[must_use]
    pub const fn with_prerequisites(
        request: ReserveGraphRequest,
        validate_outcome: PhaseOutcome,
        err: AssemblerError,
    ) -> Self {
        let mut event = Self::new(request);
        event.validate_outcome = validate_outcome;
        event.err = err;
        event
    }
}

sml! {
    GraphAssemblerReserveBuildPass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventReserveGraph> [phase_done] / mark_done,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_capacity_exceeded] / mark_failed_capacity,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Context for `GraphAssemblerReserveBuildPass` with bounded copied state.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphAssemblerReserveBuildPassContext {
    pub request: ReserveGraphRequest,
    pub validate_outcome: PhaseOutcome,
    pub build_outcome: PhaseOutcome,
    pub alloc_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub alloc_plan: AllocationPlan,
    pub err: AssemblerError,
}

impl GraphAssemblerReserveBuildPassContext {
    /// Copies an input event into the single-writer actor context.
    pub fn set_event(&mut self, event: AssemblerEventReserveGraph) {
        self.request = event.request;
        self.validate_outcome = event.validate_outcome;
        self.build_outcome = event.build_outcome;
        self.alloc_outcome = event.alloc_outcome;
        self.assembled_node_count = event.assembled_node_count;
        self.assembled_tensor_count = event.assembled_tensor_count;
        self.alloc_plan = event.alloc_plan;
        self.err = event.err;
    }
}

impl GraphAssemblerReserveBuildPassStateMachineContext for GraphAssemblerReserveBuildPassContext {
    fn mark_done(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Done;
        self.assembled_node_count = self.request.max_node_count;
        self.assembled_tensor_count = self.request.max_tensor_count;
        self.err = AssemblerError::None;
        Ok(())
    }
    fn mark_failed_capacity(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Capacity;
        Ok(())
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InvalidRequest;
        Ok(())
    }
    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }
    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }
    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }
    fn phase_capacity_exceeded(&self) -> Result<bool, ()> {
        let request = self.request;
        let overflow = product_overflows_u64(request.max_tensor_count as u64, request.bytes_per_tensor);
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && request.max_tensor_count != 0
            && request.bytes_per_tensor != 0
            && (overflow
                || (request.max_tensor_count as u64).saturating_mul(request.bytes_per_tensor)
                    > request.workspace_capacity_bytes))
    }
    fn phase_done(&self) -> Result<bool, ()> {
        let request = self.request;
        let overflow = product_overflows_u64(request.max_tensor_count as u64, request.bytes_per_tensor);
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && request.max_node_count != 0
            && request.max_tensor_count != 0
            && request.bytes_per_tensor != 0
            && !overflow
            && (request.max_tensor_count as u64) * request.bytes_per_tensor
                <= request.workspace_capacity_bytes)
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        let request = self.request;
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && (request.max_node_count == 0
                || request.max_tensor_count == 0
                || request.bytes_per_tensor == 0))
    }
}

fn product_overflows_u64(lhs: u64, rhs: u64) -> bool {
    lhs != 0 && rhs > u64::MAX / lhs
}

impl GraphAssemblerReserveBuildPassContext {
    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.build_outcome }

    #[must_use]
    pub const fn error(&self) -> AssemblerError { self.err }
}

/// Single-writer, synchronous reserve-build actor.
pub struct GraphAssemblerReserveBuildPass {
    machine: GraphAssemblerReserveBuildPassStateMachine<GraphAssemblerReserveBuildPassContext>,
}

impl Default for GraphAssemblerReserveBuildPass {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAssemblerReserveBuildPass {
    /// Creates an actor in its generated deciding state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerReserveBuildPassStateMachine::new(
                GraphAssemblerReserveBuildPassContext::default(),
            ),
        }
    }

    /// Processes one copied reserve-graph completion synchronously.
    pub fn process_event(&mut self, event: AssemblerEventReserveGraph) -> bool {
        if !self.machine.is(&GraphAssemblerReserveBuildPassStates::Deciding) {
            self.machine.context_mut().build_outcome = PhaseOutcome::Failed;
            self.machine.context_mut().err = AssemblerError::Internal;
            return false;
        }
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAssemblerReserveBuildPassEvents::AssemblerEventReserveGraph)
            .is_ok()
    }

    /// Dispatches an explicit unexpected event synchronously.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine
            .process_event(GraphAssemblerReserveBuildPassEvents::UnexpectedEvent)
            .is_ok()
    }

    /// Returns the generated machine state.
    #[must_use]
    pub fn state(&self) -> GraphAssemblerReserveBuildPassStates {
        *self.machine.state()
    }

    /// Returns whether the actor is waiting for the completion event.
    #[must_use]
    pub fn is_deciding(&self) -> bool {
        self.machine.is(&GraphAssemblerReserveBuildPassStates::Deciding)
    }

    /// Returns whether the build completed successfully.
    #[must_use]
    pub fn is_assembled(&self) -> bool {
        self.machine.is(&GraphAssemblerReserveBuildPassStates::Assembled)
    }

    /// Returns whether the build failed.
    #[must_use]
    pub fn is_assemble_failed(&self) -> bool {
        self.machine
            .is(&GraphAssemblerReserveBuildPassStates::AssembleFailed)
    }

    /// Returns the copied phase context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerReserveBuildPassContext {
        self.machine.context()
    }
}
/// Short actor alias matching the phase name.
pub type Actor = GraphAssemblerReserveBuildPass;
