//! Source-aligned bounded reuse-decision phase for graph assembly.
//!
//! The transition table mirrors the pinned C++
//! `reuse_decision_pass/{sm,actions,guards}.hpp` contract. Runtime requests and
//! completion diagnostics are copied into a bounded single-writer context;
//! opaque C++ handles are represented by integer identities.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

/// Outcome values used by assembler completion phases.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Outcome produced by the reuse-decision phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ReuseOutcome {
    #[default]
    Unknown = 0,
    Reused = 1,
    Rebuild = 2,
    Failed = 3,
}

/// Error values matching the pinned assembler error contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AssemblerError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Capacity = 2,
    InternalError = 4,
    Untracked = 8,
}

/// Bounded allocation plan copied to and from the completion context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlan {
    /// Number of tensors represented by the plan.
    pub tensor_count: u32,
    /// Number of allocation intervals represented by the plan.
    pub interval_count: u32,
    /// Required workspace bytes.
    pub required_buffer_bytes: u64,
}

/// Copied assemble request and completion context consumed by this phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventAssembleGraph {
    /// Opaque step-plan identity/presence bit.
    pub step_plan: u64,
    /// Opaque output destination identity/presence bit.
    pub output_out: u64,
    /// Opaque lifecycle-manifest identity/presence bit.
    pub lifecycle: u64,
    /// Expected node count used for reuse selection.
    pub node_count_hint: u32,
    /// Expected tensor count used for reuse selection.
    pub tensor_count_hint: u32,
    /// Bytes reserved per tensor.
    pub bytes_per_tensor: u64,
    /// Workspace capacity in bytes.
    pub workspace_capacity_bytes: u64,
    /// Validation outcome carried by the preceding phase.
    pub validate_outcome: PhaseOutcome,
    /// Reuse outcome carried by this phase.
    pub reuse_outcome: ReuseOutcome,
    /// Build outcome carried by a later phase.
    pub build_outcome: PhaseOutcome,
    /// Allocation outcome carried by a later phase.
    pub alloc_outcome: PhaseOutcome,
    /// Number of assembled nodes carried by the completion context.
    pub assembled_node_count: u32,
    /// Number of assembled tensors carried by the completion context.
    pub assembled_tensor_count: u32,
    /// Number of tensors in the allocation plan.
    pub alloc_tensor_count: u32,
    /// Number of intervals in the allocation plan.
    pub alloc_interval_count: u32,
    /// Required bytes in the allocation plan.
    pub alloc_required_buffer_bytes: u64,
    /// Whether the selected topology is reused.
    pub reused_topology: bool,
    /// Error carried by an earlier phase.
    pub err: AssemblerError,
}

impl AssemblerEventAssembleGraph {
    /// Creates a copied event from the request and validation completion.
    #[must_use]
    pub const fn new(
        step_plan: u64,
        output_out: u64,
        lifecycle: u64,
        node_count_hint: u32,
        tensor_count_hint: u32,
        bytes_per_tensor: u64,
        workspace_capacity_bytes: u64,
        validate_outcome: PhaseOutcome,
        err: AssemblerError,
    ) -> Self {
        Self {
            step_plan,
            output_out,
            lifecycle,
            node_count_hint,
            tensor_count_hint,
            bytes_per_tensor,
            workspace_capacity_bytes,
            validate_outcome,
            reuse_outcome: ReuseOutcome::Unknown,
            build_outcome: PhaseOutcome::Unknown,
            alloc_outcome: PhaseOutcome::Unknown,
            assembled_node_count: 0,
            assembled_tensor_count: 0,
            alloc_tensor_count: 0,
            alloc_interval_count: 0,
            alloc_required_buffer_bytes: 0,
            reused_topology: false,
            err,
        }
    }
}

sml! {
    GraphAssemblerReuseDecisionPass {
        "assemble_failed"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prefailed] / mark_failed_prefailed,
        "reuse_selected"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_reuse] / mark_reuse,
        "rebuild_selected"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_rebuild] / mark_rebuild,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "reuse_selected"_s + unexpected_event<_> / on_unexpected_from_reuse_selected,
        "unexpected_event"_s <= "rebuild_selected"_s + unexpected_event<_> / on_unexpected_from_rebuild_selected,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "reuse_selected"_s = X,
        "rebuild_selected"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Persistent bounded context for `GraphAssemblerReuseDecisionPass`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphAssemblerReuseDecisionPassContext {
    /// Copied request fields.
    pub step_plan: u64,
    pub output_out: u64,
    pub lifecycle: u64,
    pub node_count_hint: u32,
    pub tensor_count_hint: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
    /// Copied completion fields.
    pub validate_outcome: PhaseOutcome,
    pub reuse_outcome: ReuseOutcome,
    pub build_outcome: PhaseOutcome,
    pub alloc_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub alloc_tensor_count: u32,
    pub alloc_interval_count: u32,
    pub alloc_required_buffer_bytes: u64,
    pub reused_topology: bool,
    pub err: AssemblerError,
    /// Reservation retained by the owning assembler context.
    pub reserved_topology: u64,
    pub reserved_node_count: u32,
    pub reserved_tensor_count: u32,
    pub reserved_required_buffer_bytes: u64,
    pub has_reserved_topology: bool,
    /// Bounded plan produced by the reuse action.
    pub alloc_plan: AllocationPlan,
}

impl GraphAssemblerReuseDecisionPassContext {
    /// Copies one runtime event into the actor context and resets this phase.
    pub fn set_event(&mut self, event: AssemblerEventAssembleGraph) {
        self.step_plan = event.step_plan;
        self.output_out = event.output_out;
        self.lifecycle = event.lifecycle;
        self.node_count_hint = event.node_count_hint;
        self.tensor_count_hint = event.tensor_count_hint;
        self.bytes_per_tensor = event.bytes_per_tensor;
        self.workspace_capacity_bytes = event.workspace_capacity_bytes;
        self.validate_outcome = event.validate_outcome;
        self.reuse_outcome = ReuseOutcome::Unknown;
        self.build_outcome = event.build_outcome;
        self.alloc_outcome = event.alloc_outcome;
        self.assembled_node_count = event.assembled_node_count;
        self.assembled_tensor_count = event.assembled_tensor_count;
        self.alloc_tensor_count = event.alloc_tensor_count;
        self.alloc_interval_count = event.alloc_interval_count;
        self.alloc_required_buffer_bytes = event.alloc_required_buffer_bytes;
        self.reused_topology = event.reused_topology;
        self.err = event.err;
        self.alloc_plan = AllocationPlan::default();
    }

    #[must_use]
    pub const fn outcome(&self) -> ReuseOutcome {
        self.reuse_outcome
    }

    #[must_use]
    pub const fn error(&self) -> AssemblerError {
        self.err
    }
}

impl GraphAssemblerReuseDecisionPassStateMachineContext for GraphAssemblerReuseDecisionPassContext {
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InvalidRequest;
        Ok(())
    }

    fn mark_failed_prefailed(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        Ok(())
    }

    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn mark_rebuild(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Rebuild;
        self.reused_topology = false;
        self.assembled_node_count = self.node_count_hint;
        self.assembled_tensor_count = self.tensor_count_hint;
        self.alloc_plan = AllocationPlan::default();
        self.err = AssemblerError::None;
        Ok(())
    }

    fn mark_reuse(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Reused;
        self.reused_topology = true;
        self.assembled_node_count = self.reserved_node_count;
        self.assembled_tensor_count = self.reserved_tensor_count;
        self.alloc_plan = AllocationPlan {
            tensor_count: self.reserved_tensor_count,
            interval_count: self.reserved_tensor_count,
            required_buffer_bytes: self.reserved_required_buffer_bytes,
        };
        self.err = AssemblerError::None;
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_rebuild_selected(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_reuse_selected(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.reuse_outcome = ReuseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        let reuse_candidate = self.has_reserved_topology
            && self.reserved_topology != 0
            && self.node_count_hint == self.reserved_node_count
            && self.tensor_count_hint == self.reserved_tensor_count;
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && !reuse_candidate
            && (self.node_count_hint == 0 || self.tensor_count_hint == 0))
    }

    fn phase_prefailed(&self) -> Result<bool, ()> {
        Ok(self.err != AssemblerError::None)
    }

    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None && self.validate_outcome != PhaseOutcome::Done)
    }

    fn phase_rebuild(&self) -> Result<bool, ()> {
        let reuse_candidate = self.has_reserved_topology
            && self.reserved_topology != 0
            && self.node_count_hint == self.reserved_node_count
            && self.tensor_count_hint == self.reserved_tensor_count;
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && !reuse_candidate
            && self.node_count_hint != 0
            && self.tensor_count_hint != 0)
    }

    fn phase_reuse(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.validate_outcome == PhaseOutcome::Done
            && self.has_reserved_topology
            && self.reserved_topology != 0
            && self.node_count_hint == self.reserved_node_count
            && self.tensor_count_hint == self.reserved_tensor_count)
    }
}

/// Synchronous single-writer actor around the generated reuse-decision machine.
pub struct ReuseDecisionPass {
    machine: GraphAssemblerReuseDecisionPassStateMachine<GraphAssemblerReuseDecisionPassContext>,
}

impl Default for ReuseDecisionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ReuseDecisionPass {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerReuseDecisionPassStateMachine::new(
                GraphAssemblerReuseDecisionPassContext::default(),
            ),
        }
    }

    /// Copies and dispatches one completion event synchronously.
    pub fn process_event(&mut self, event: AssemblerEventAssembleGraph) -> ReuseOutcome {
        if !self
            .machine
            .is(&GraphAssemblerReuseDecisionPassStates::Deciding)
        {
            return self.process_unexpected_event();
        }
        self.machine.context_mut().set_event(event);
        let _ = self
            .machine
            .process_event(GraphAssemblerReuseDecisionPassEvents::AssemblerEventAssembleGraph);
        self.machine.context().reuse_outcome
    }

    /// Records an unexpected event and enters the generated unexpected state.
    pub fn process_unexpected_event(&mut self) -> ReuseOutcome {
        let context = self.machine.context_mut();
        context.reuse_outcome = ReuseOutcome::Failed;
        context.err = AssemblerError::InternalError;
        self.machine
            .set_state(GraphAssemblerReuseDecisionPassStates::UnexpectedEvent);
        self.machine.context().reuse_outcome
    }

    /// Returns the generated state.
    #[must_use]
    pub fn state(&self) -> &GraphAssemblerReuseDecisionPassStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GraphAssemblerReuseDecisionPassStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerReuseDecisionPassContext {
        self.machine.context()
    }

    /// Returns the selected outcome.
    #[must_use]
    pub fn outcome(&self) -> ReuseOutcome {
        self.machine.context().reuse_outcome
    }

    /// Returns the effective phase error.
    #[must_use]
    pub fn error(&self) -> AssemblerError {
        self.machine.context().err
    }
}

/// Short actor alias matching the phase name.
pub type Actor = ReuseDecisionPass;
