//! Source-aligned bounded graph-assembler build phase.
//!
//! The transition table mirrors the pinned C++ `assemble_build_pass` machine.
//! Runtime input is copied into the single-writer context before synchronous
//! completion dispatch; opaque C++ handles are represented by integer handles.

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
    missing_docs,
    private_interfaces
)]

use sml::sml;

/// Outcome retained by an assembler phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PhaseOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
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

/// Runtime event carrying the bounded copied fields used by this phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventAssembleGraph {
    /// Opaque step-plan handle copied from the request.
    pub step_plan: u64,
    /// Opaque output handle copied from the request.
    pub output_out: u64,
    /// Opaque lifecycle-manifest handle copied from the request.
    pub lifecycle: u64,
    /// Expected node count from the reuse decision.
    pub node_count_hint: u32,
    /// Expected tensor count from the reuse decision.
    pub tensor_count_hint: u32,
    /// Bytes reserved per tensor.
    pub bytes_per_tensor: u64,
    /// Available workspace capacity.
    pub workspace_capacity_bytes: u64,
    /// Validation phase outcome.
    pub validate_outcome: PhaseOutcome,
    /// Reuse decision outcome; `Rebuild` selects this phase.
    pub reuse_outcome: PhaseOutcome,
    /// Build phase outcome carried by the internal context.
    pub build_outcome: PhaseOutcome,
    /// Allocation phase outcome.
    pub alloc_outcome: PhaseOutcome,
    /// Number of assembled nodes.
    pub assembled_node_count: u32,
    /// Number of assembled tensors.
    pub assembled_tensor_count: u32,
    /// Number of tensors in the allocation plan.
    pub alloc_tensor_count: u32,
    /// Number of intervals in the allocation plan.
    pub alloc_interval_count: u32,
    /// Required bytes in the allocation plan.
    pub alloc_required_buffer_bytes: u64,
    /// Whether the topology was reused.
    pub reused_topology: bool,
    /// Error already reported by an earlier phase.
    pub err: AssemblerError,
}

sml! {
    GraphAssemblerAssembleBuildPass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_done] / mark_done,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_capacity_exceeded] / mark_failed_capacity,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Persistent bounded context for `GraphAssemblerAssembleBuildPass`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphAssemblerAssembleBuildPassContext {
    pub step_plan: u64,
    pub output_out: u64,
    pub lifecycle: u64,
    pub node_count_hint: u32,
    pub tensor_count_hint: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
    pub validate_outcome: PhaseOutcome,
    pub reuse_outcome: PhaseOutcome,
    pub build_outcome: PhaseOutcome,
    pub alloc_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub alloc_tensor_count: u32,
    pub alloc_interval_count: u32,
    pub alloc_required_buffer_bytes: u64,
    pub reused_topology: bool,
    pub err: AssemblerError,
}

impl GraphAssemblerAssembleBuildPassContext {
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
        self.reuse_outcome = event.reuse_outcome;
        self.build_outcome = PhaseOutcome::Unknown;
        self.alloc_outcome = event.alloc_outcome;
        self.assembled_node_count = event.assembled_node_count;
        self.assembled_tensor_count = event.assembled_tensor_count;
        self.alloc_tensor_count = event.alloc_tensor_count;
        self.alloc_interval_count = event.alloc_interval_count;
        self.alloc_required_buffer_bytes = event.alloc_required_buffer_bytes;
        self.reused_topology = event.reused_topology;
        self.err = event.err;
    }

    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.build_outcome }

    #[must_use]
    pub const fn error(&self) -> AssemblerError { self.err }
}

fn product_overflows_u64(lhs: u64, rhs: u64) -> bool {
    lhs != 0 && rhs > u64::MAX / lhs
}

impl GraphAssemblerAssembleBuildPassStateMachineContext
    for GraphAssemblerAssembleBuildPassContext
{
    fn mark_done(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Done;
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
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.build_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn phase_capacity_exceeded(&self) -> Result<bool, ()> {
        let tensor_count = u64::from(self.assembled_tensor_count);
        let overflow = product_overflows_u64(tensor_count, self.bytes_per_tensor);
        Ok(self.err == AssemblerError::None
            && self.reuse_outcome == PhaseOutcome::Rebuild
            && self.assembled_tensor_count != 0
            && self.bytes_per_tensor != 0
            && (overflow
                || tensor_count * self.bytes_per_tensor > self.workspace_capacity_bytes))
    }

    fn phase_done(&self) -> Result<bool, ()> {
        let tensor_count = u64::from(self.assembled_tensor_count);
        Ok(self.err == AssemblerError::None
            && self.reuse_outcome == PhaseOutcome::Rebuild
            && self.assembled_node_count != 0
            && self.assembled_tensor_count != 0
            && self.bytes_per_tensor != 0
            && !product_overflows_u64(tensor_count, self.bytes_per_tensor)
            && tensor_count * self.bytes_per_tensor <= self.workspace_capacity_bytes)
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.reuse_outcome == PhaseOutcome::Rebuild
            && (self.assembled_node_count == 0
                || self.assembled_tensor_count == 0
                || self.bytes_per_tensor == 0))
    }

    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None && self.reuse_outcome != PhaseOutcome::Rebuild)
    }
}

/// Synchronous single-writer actor around the generated build-pass machine.
pub struct AssembleBuildPass {
    machine: GraphAssemblerAssembleBuildPassStateMachine<GraphAssemblerAssembleBuildPassContext>,
}

impl Default for AssembleBuildPass {
    fn default() -> Self { Self::new() }
}

impl AssembleBuildPass {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerAssembleBuildPassStateMachine::new(Default::default()),
        }
    }

    /// Copies and dispatches one event synchronously.
    pub fn process_event(&mut self, event: AssemblerEventAssembleGraph) -> bool {
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAssemblerAssembleBuildPassEvents::AssemblerEventAssembleGraph)
            .is_ok()
    }

    /// Dispatches an explicit unexpected event.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine
            .process_event(GraphAssemblerAssembleBuildPassEvents::UnexpectedEvent)
            .is_ok()
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphAssemblerAssembleBuildPassStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphAssemblerAssembleBuildPassStates) -> bool {
        self.machine.is(&state)
    }

    /// Returns the retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerAssembleBuildPassContext { self.machine.context() }
}

/// Short actor alias matching the phase name.
pub type Actor = AssembleBuildPass;
