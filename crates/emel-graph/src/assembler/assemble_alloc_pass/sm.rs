//! Bounded graph-assembler assemble-allocation phase state machine.
//!
//! Source mapping: `emel.cpp/src/emel/graph/assembler/assemble_alloc_pass/{sm,context,events,actions,guards,errors}.hpp`.
//! Requests and phase results are copied into a single-writer context before synchronous
//! dispatch. Pointer-bearing C++ values are represented by bounded opaque handles or presence
//! bits; this actor never retains or dereferences caller-owned resources.

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

/// Outcome retained by the assemble-allocation phase.
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

/// Bounded allocation plan copied from the allocator child result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlan {
    pub tensor_count: u32,
    pub interval_count: u32,
    pub required_buffer_bytes: u64,
}

/// Copied assemble request and prerequisite results consumed by this phase.
///
/// `step_plan` is an opaque, non-dereferenced identity corresponding to the source topology
/// pointer. `output_out` preserves whether the caller supplied the destination for the plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventAssembleGraph {
    pub step_plan: u64,
    pub output_out: bool,
    pub lifecycle: u64,
    pub node_count_hint: u32,
    pub tensor_count_hint: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
    pub validate_outcome: PhaseOutcome,
    pub reuse_outcome: PhaseOutcome,
    pub build_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub reused_topology: bool,
    pub err: AssemblerError,
}

impl AssemblerEventAssembleGraph {
    /// Creates an event from the copied assemble request fields.
    #[must_use]
    pub const fn new(
        step_plan: u64,
        output_out: bool,
        lifecycle: u64,
        node_count_hint: u32,
        tensor_count_hint: u32,
        bytes_per_tensor: u64,
        workspace_capacity_bytes: u64,
    ) -> Self {
        Self {
            step_plan,
            output_out,
            lifecycle,
            node_count_hint,
            tensor_count_hint,
            bytes_per_tensor,
            workspace_capacity_bytes,
            validate_outcome: PhaseOutcome::Unknown,
            reuse_outcome: PhaseOutcome::Unknown,
            build_outcome: PhaseOutcome::Unknown,
            assembled_node_count: 0,
            assembled_tensor_count: 0,
            reused_topology: false,
            err: AssemblerError::None,
        }
    }

    /// Creates an event with copied prerequisite outcomes and assembled counts.
    #[must_use]
    pub const fn with_prerequisites(
        request: AssemblerEventAssembleGraph,
        build_outcome: PhaseOutcome,
        err: AssemblerError,
        assembled_node_count: u32,
        assembled_tensor_count: u32,
    ) -> Self {
        Self {
            build_outcome,
            err,
            assembled_node_count,
            assembled_tensor_count,
            ..request
        }
    }
}

sml! {
    GraphAssemblerAssembleAllocPass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_request_allocator] / request_allocator_plan,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Persistent bounded context for `GraphAssemblerAssembleAllocPass`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphAssemblerAssembleAllocPassContext {
    pub request: AssemblerEventAssembleGraph,
    pub validate_outcome: PhaseOutcome,
    pub reuse_outcome: PhaseOutcome,
    pub build_outcome: PhaseOutcome,
    pub assembled_node_count: u32,
    pub assembled_tensor_count: u32,
    pub reused_topology: bool,
    pub alloc_outcome: PhaseOutcome,
    pub alloc_plan: AllocationPlan,
    pub err: AssemblerError,
}

impl GraphAssemblerAssembleAllocPassContext {
    /// Copies one completion event and resets this phase's transient result.
    pub fn set_event(&mut self, event: AssemblerEventAssembleGraph) {
        self.request = event;
        self.validate_outcome = event.validate_outcome;
        self.reuse_outcome = event.reuse_outcome;
        self.build_outcome = event.build_outcome;
        self.assembled_node_count = event.assembled_node_count;
        self.assembled_tensor_count = event.assembled_tensor_count;
        self.reused_topology = event.reused_topology;
        self.alloc_outcome = PhaseOutcome::Unknown;
        self.alloc_plan = AllocationPlan::default();
        self.err = event.err;
    }

    #[must_use]
    pub const fn outcome(&self) -> PhaseOutcome { self.alloc_outcome }

    #[must_use]
    pub const fn error(&self) -> AssemblerError { self.err }

    #[must_use]
    pub const fn plan(&self) -> AllocationPlan { self.alloc_plan }
}

fn product_overflows(lhs: u32, rhs: u64) -> bool {
    lhs != 0 && rhs > u64::MAX / u64::from(lhs)
}

impl GraphAssemblerAssembleAllocPassStateMachineContext
    for GraphAssemblerAssembleAllocPassContext
{
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InvalidRequest;
        Ok(())
    }

    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.request.build_outcome == PhaseOutcome::Done
            && (self.request.step_plan == 0
                || !self.request.output_out
                || self.request.assembled_node_count == 0
                || self.request.assembled_tensor_count == 0))
    }

    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.request.build_outcome != PhaseOutcome::Done)
    }

    fn phase_request_allocator(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.request.build_outcome == PhaseOutcome::Done
            && self.request.step_plan != 0
            && self.request.output_out
            && self.request.assembled_node_count != 0
            && self.request.assembled_tensor_count != 0)
    }

    fn request_allocator_plan(&mut self) -> Result<(), ()> {
        self.alloc_outcome = PhaseOutcome::Failed;
        self.alloc_plan = AllocationPlan::default();
        self.err = AssemblerError::InternalError;
        if self.request.bytes_per_tensor == 0 || self.request.workspace_capacity_bytes == 0 {
            self.err = AssemblerError::InvalidRequest;
            return Ok(());
        }

        if product_overflows(
            self.request.assembled_tensor_count,
            self.request.bytes_per_tensor,
        ) {
            self.err = AssemblerError::Capacity;
            return Ok(());
        }

        let required_buffer_bytes = u64::from(self.request.assembled_tensor_count)
            * self.request.bytes_per_tensor;
        if required_buffer_bytes > self.request.workspace_capacity_bytes {
            self.err = AssemblerError::Capacity;
            return Ok(());
        }

        self.alloc_plan = AllocationPlan {
            tensor_count: self.request.assembled_tensor_count,
            interval_count: self.request.assembled_tensor_count,
            required_buffer_bytes,
        };
        self.alloc_outcome = PhaseOutcome::Done;
        self.err = AssemblerError::None;
        Ok(())
    }
}

/// Synchronous, single-writer assemble-allocation actor.
pub struct GraphAssemblerAssembleAllocPass {
    machine: GraphAssemblerAssembleAllocPassStateMachine<GraphAssemblerAssembleAllocPassContext>,
}

impl Default for GraphAssemblerAssembleAllocPass {
    fn default() -> Self { Self::new() }
}

impl GraphAssemblerAssembleAllocPass {
    /// Constructs the actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerAssembleAllocPassStateMachine::new(
                GraphAssemblerAssembleAllocPassContext::default(),
            ),
        }
    }

    /// Copies and dispatches one assemble-allocation completion synchronously.
    pub fn process_event(&mut self, event: AssemblerEventAssembleGraph) -> bool {
        self.machine.context_mut().set_event(event);
        self.machine
            .process_event(GraphAssemblerAssembleAllocPassEvents::AssemblerEventAssembleGraph)
            .is_ok()
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GraphAssemblerAssembleAllocPassStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GraphAssemblerAssembleAllocPassStates) -> bool {
        self.machine.is(&state)
    }

    /// Returns retained bounded context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerAssembleAllocPassContext { self.machine.context() }
}

/// Short actor alias matching the phase name.
pub type Actor = GraphAssemblerAssembleAllocPass;
