//! Source-aligned reserve-validation pass actor.
//!
//! The request is copied into bounded scalar fields before synchronous dispatch.  Pointer-like
//! C++ inputs are represented only by presence bits; this child never dereferences or retains
//! caller-owned resources.

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

/// Outcome of a reserve-validation phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PassOutcome {
    #[default]
    Unknown = 0,
    Done = 1,
    Failed = 2,
}

/// Error values matching the assembler error contract.
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

/// Bounded copy of the scalar reserve request fields used by source guards.
///
/// `has_model_topology` and `has_output_out` preserve the source pointer-presence checks without
/// exposing raw pointers or retaining caller-owned objects.
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
    /// Creates a bounded request from source-equivalent scalar values.
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

/// Runtime event shell retained by the existing generated machine name.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct AssemblerEventReserveGraph;

sml! {
    GraphAssemblerReserveValidatePass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventReserveGraph> [phase_done] / mark_done,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventReserveGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Context for `GraphAssemblerReserveValidatePass` containing only bounded copied state.
#[derive(Debug, Default)]
pub struct GraphAssemblerReserveValidatePassContext {
    /// Copied request consumed by guards.
    pub request: ReserveGraphRequest,
    /// Validation result consumed by the parent assembler.
    pub validate_outcome: PassOutcome,
    /// Error result consumed by the parent assembler.
    pub err: AssemblerError,
}

impl GraphAssemblerReserveValidatePassContext {
    /// Copies a request and resets transient phase state.
    pub fn set_request(&mut self, request: ReserveGraphRequest) {
        self.request = request;
        self.validate_outcome = PassOutcome::Unknown;
        self.err = AssemblerError::None;
    }

    /// Returns the retained phase outcome.
    #[must_use]
    pub const fn outcome(&self) -> PassOutcome {
        self.validate_outcome
    }

    /// Returns the retained assembler error.
    #[must_use]
    pub const fn error(&self) -> AssemblerError {
        self.err
    }
}

impl GraphAssemblerReserveValidatePassStateMachineContext
    for GraphAssemblerReserveValidatePassContext
{
    fn mark_done(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Done;
        self.err = AssemblerError::None;
        Ok(())
    }

    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Failed;
        self.err = AssemblerError::InvalidRequest;
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }

    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.validate_outcome = PassOutcome::Failed;
        self.err = AssemblerError::Internal;
        Ok(())
    }

    fn phase_done(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.request.has_model_topology
            && self.request.has_output_out
            && self.request.max_node_count != 0
            && self.request.max_tensor_count != 0
            && self.request.bytes_per_tensor != 0
            && self.request.workspace_capacity_bytes != 0)
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && (!self.request.has_model_topology
                || !self.request.has_output_out
                || self.request.max_node_count == 0
                || self.request.max_tensor_count == 0
                || self.request.bytes_per_tensor == 0
                || self.request.workspace_capacity_bytes == 0))
    }
}

/// Single-writer synchronous reserve-validation actor.
pub struct GraphAssemblerReserveValidatePass {
    machine: GraphAssemblerReserveValidatePassStateMachine<
        GraphAssemblerReserveValidatePassContext,
    >,
}

impl Default for GraphAssemblerReserveValidatePass {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAssemblerReserveValidatePass {
    /// Constructs an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerReserveValidatePassStateMachine::new(
                GraphAssemblerReserveValidatePassContext::default(),
            ),
        }
    }

    /// Copies a request and dispatches its completion synchronously.
    pub fn process_event(&mut self, request: ReserveGraphRequest) -> bool {
        self.machine.context_mut().set_request(request);
        self.machine
            .process_event(GraphAssemblerReserveValidatePassEvents::AssemblerEventReserveGraph)
            .is_ok()
    }

    /// Dispatches an explicit unexpected event synchronously.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine
            .process_event(GraphAssemblerReserveValidatePassEvents::UnexpectedEvent)
            .is_ok()
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphAssemblerReserveValidatePassStates {
        self.machine.state()
    }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphAssemblerReserveValidatePassStates) -> bool {
        self.machine.is(state)
    }

    /// Returns retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerReserveValidatePassContext {
        self.machine.context()
    }
}
