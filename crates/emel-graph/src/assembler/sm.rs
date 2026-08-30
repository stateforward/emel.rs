//! Safe, synchronous graph assembler actor.
//!
//! This is the Rust port of `emel.cpp/src/emel/graph/assembler/sm.hpp` and its
//! phase contracts.  Each reserve or assemble request carries its phase data
//! through RTC completion transitions; the actor context retains only the
//! successfully reserved topology needed by later assemble requests.

#![allow(clippy::module_name_repetitions)]
#![allow(private_interfaces)]
#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

/// Typed graph-assembly failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The request does not satisfy the public event contract.
    InvalidRequest,
    /// The requested graph does not fit the supplied workspace.
    Capacity,
    /// A phase failed because its prerequisite or lifecycle was invalid.
    Internal,
    /// An error code was not recognized by the actor.
    Untracked,
    /// An event was not valid for the current machine state.
    UnexpectedEvent,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => formatter.write_str("invalid graph assembler request"),
            Self::Capacity => formatter.write_str("graph assembler capacity exceeded"),
            Self::Internal => formatter.write_str("internal graph assembler error"),
            Self::Untracked => formatter.write_str("untracked graph assembler error"),
            Self::UnexpectedEvent => formatter.write_str("unexpected graph assembler event"),
        }
    }
}

impl std::error::Error for Error {}

/// Caller-owned opaque topology identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Topology(pub usize);

/// Caller-owned opaque lifecycle-manifest identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecycleManifest(pub usize);

/// Allocation plan produced by the allocator phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AllocationPlan {
    /// Number of tensors represented by the plan.
    pub tensor_count: u32,
    /// Number of allocation intervals represented by the plan.
    pub interval_count: u32,
    /// Required workspace bytes.
    pub required_buffer_bytes: u64,
}

/// Output published by reserve and assemble operations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblyOutput {
    /// Topology identity produced by the operation.
    pub graph_topology: Topology,
    /// Number of graph nodes.
    pub node_count: u32,
    /// Number of graph tensors.
    pub tensor_count: u32,
    /// Workspace bytes required by the graph.
    pub required_buffer_bytes: u64,
    /// Monotonically increasing successful-topology version.
    pub version: u32,
    /// Whether assemble reused the reserved topology.
    pub reused_topology: bool,
    /// Lifecycle-manifest identity associated with the graph.
    pub lifecycle: LifecycleManifest,
}

/// Reserve request corresponding to the C++ `event::reserve` event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reserve {
    /// Model topology identity.
    pub model_topology: Topology,
    /// Lifecycle manifest identity.
    pub lifecycle: LifecycleManifest,
    /// Maximum graph nodes.
    pub max_node_count: u32,
    /// Maximum graph tensors.
    pub max_tensor_count: u32,
    /// Bytes reserved per tensor.
    pub bytes_per_tensor: u64,
    /// Available workspace bytes.
    pub workspace_capacity_bytes: u64,
}

/// Assemble request corresponding to the C++ `event::assemble` event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Assemble {
    /// Step-plan identity.
    pub step_plan: Topology,
    /// Lifecycle manifest identity.
    pub lifecycle: LifecycleManifest,
    /// Expected node count used by the reuse decision.
    pub node_count_hint: u32,
    /// Expected tensor count used by the reuse decision.
    pub tensor_count_hint: u32,
    /// Bytes reserved per tensor.
    pub bytes_per_tensor: u64,
    /// Available workspace bytes.
    pub workspace_capacity_bytes: u64,
}

/// Public event set accepted by the actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    /// Reserve graph storage.
    Reserve(Reserve),
    /// Assemble a step plan, reusing a compatible reservation when possible.
    Assemble(Assemble),
}

/// Explicit event used to exercise unexpected-event handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

/// Successful public result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// A reserve operation completed.
    Reserved(AssemblyOutput),
    /// An assemble operation completed.
    Assembled(AssemblyOutput),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PhaseOutcome {
    #[default]
    Unknown,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ReuseOutcome {
    #[default]
    Unknown,
    Reused,
    Rebuild,
    Failed,
}

#[derive(Clone, Copy)]
struct ReserveRuntime<'dispatch> {
    request: Reserve,
    validate: &'dispatch Cell<PhaseOutcome>,
    build: &'dispatch Cell<PhaseOutcome>,
    alloc: &'dispatch Cell<PhaseOutcome>,
    err: &'dispatch Cell<Option<Error>>,
    plan: &'dispatch Cell<AllocationPlan>,
    output: &'dispatch Cell<AssemblyOutput>,
    result: &'dispatch RefCell<Result<Outcome, Error>>,
}

#[derive(Clone, Copy)]
struct AssembleRuntime<'dispatch> {
    request: Assemble,
    validate: &'dispatch Cell<PhaseOutcome>,
    reuse: &'dispatch Cell<ReuseOutcome>,
    build: &'dispatch Cell<PhaseOutcome>,
    alloc: &'dispatch Cell<PhaseOutcome>,
    err: &'dispatch Cell<Option<Error>>,
    plan: &'dispatch Cell<AllocationPlan>,
    output: &'dispatch Cell<AssemblyOutput>,
    result: &'dispatch RefCell<Result<Outcome, Error>>,
}

/// Persistent actor-owned reservation context.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphAssemblerContext {
    reserved_topology: Topology,
    reserved_node_count: u32,
    reserved_tensor_count: u32,
    reserved_required_buffer_bytes: u64,
    topology_version: u32,
    reserved_lifecycle: LifecycleManifest,
    has_reserved_topology: bool,
}

fn product_overflows(lhs: u32, rhs: u64) -> bool {
    lhs != 0 && rhs > u64::MAX / u64::from(lhs)
}

fn product(lhs: u32, rhs: u64) -> u64 {
    u64::from(lhs) * rhs
}

sml! {
    GraphAssembler<'dispatch> {
        // Reserve request and validation phases.
        "reserve_validate"_s <= *"uninitialized"_s
            + Reserve(ReserveRuntime<'dispatch>) [guard_reserve_valid] / effect_begin_reserve,
        "reserved"_s <= "reserved"_s
            + Reserve(ReserveRuntime<'dispatch>) [guard_reserve_valid] / effect_reject_second_reserve,
        "uninitialized"_s <= "uninitialized"_s
            + Reserve(ReserveRuntime<'dispatch>) [guard_reserve_invalid] / effect_invalid_reserve,

        "reserve_build_decision"_s <= "reserve_validate"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) / effect_reserve_validate_done,
        "reserve_build"_s <= "reserve_build_decision"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_build_valid]
            / effect_reserve_build_done,
        "reserve_error"_s <= "reserve_build_decision"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_build_capacity]
            / effect_reserve_build_capacity,
        "reserve_error"_s <= "reserve_build_decision"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_build_invalid]
            / effect_reserve_build_invalid,

        "reserve_alloc_decision"_s <= "reserve_build"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) / effect_reserve_build_phase_done,
        "reserve_alloc"_s <= "reserve_alloc_decision"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_alloc_valid]
            / effect_reserve_alloc_done,
        "reserve_error"_s <= "reserve_alloc_decision"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_alloc_invalid]
            / effect_reserve_alloc_invalid,

        "reserved"_s <= "reserve_alloc"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_success]
            / effect_commit_reserve,
        "uninitialized"_s <= "reserve_alloc"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) [guard_reserve_failure]
            / effect_publish_reserve_error,
        "uninitialized"_s <= "reserve_error"_s
            + completion<Reserve>(ReserveRuntime<'dispatch>) / effect_publish_reserve_error,

        // Assemble validation and explicit reuse/rebuild choice.
        "assemble_validate"_s <= "reserved"_s
            + Assemble(AssembleRuntime<'dispatch>) [guard_assemble_valid]
            / effect_begin_assemble,
        "reserved"_s <= "reserved"_s
            + Assemble(AssembleRuntime<'dispatch>) [guard_assemble_invalid]
            / effect_invalid_assemble,
        "uninitialized"_s <= "uninitialized"_s
            + Assemble(AssembleRuntime<'dispatch>) / effect_uninitialized_assemble,

        "assemble_reuse_decision"_s <= "assemble_validate"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) / effect_assemble_validate_done,
        "assemble_reuse"_s <= "assemble_reuse_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_reuse_selected]
            / effect_select_reuse,
        "assemble_build_decision"_s <= "assemble_reuse_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_rebuild_selected]
            / effect_select_rebuild,
        "assemble_error"_s <= "assemble_reuse_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_reuse_invalid]
            / effect_invalid_rebuild,

        "reserved"_s <= "assemble_reuse"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_reuse_success]
            / effect_commit_reuse,
        "reserved"_s <= "assemble_reuse"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_reuse_failure]
            / effect_publish_assemble_error,

        "assemble_build"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_valid]
            / effect_assemble_build_done,
        "assemble_error"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_capacity]
            / effect_assemble_build_capacity,
        "assemble_error"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_invalid]
            / effect_assemble_build_invalid,
        "assemble_alloc_decision"_s <= "assemble_build"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) / effect_assemble_build_phase_done,
        "assemble_alloc"_s <= "assemble_alloc_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_alloc_valid]
            / effect_assemble_alloc_done,
        "assemble_error"_s <= "assemble_alloc_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_alloc_invalid]
            / effect_assemble_alloc_invalid,
        "reserved"_s <= "assemble_alloc"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_success]
            / effect_commit_rebuild,
        "reserved"_s <= "assemble_alloc"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_failure]
            / effect_publish_assemble_error,
        "reserved"_s <= "assemble_error"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) / effect_publish_assemble_error,

        // Every non-ready state has an explicit recovery path for an unexpected event.
        "uninitialized"_s <= "uninitialized"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_uninitialized,
        "reserved"_s <= "reserved"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserved,
        "uninitialized"_s <= "reserve_validate"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "uninitialized"_s <= "reserve_build_decision"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "uninitialized"_s <= "reserve_build"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "uninitialized"_s <= "reserve_alloc_decision"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "uninitialized"_s <= "reserve_alloc"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "uninitialized"_s <= "reserve_error"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_reserve,
        "reserved"_s <= "assemble_validate"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_reuse_decision"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_reuse"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_build_decision"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_build"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_alloc_decision"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_alloc"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
        "reserved"_s <= "assemble_error"_s + Unexpected(UnexpectedEvent)
            / effect_unexpected_assemble,
    }
}

/// Single-writer, run-to-completion graph assembler actor.
pub struct Assembler {
    machine: GraphAssemblerStateMachine<GraphAssemblerContext>,
}

impl fmt::Debug for Assembler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Assembler")
            .field("uninitialized", &self.is_uninitialized())
            .field("reserved", &self.is_reserved())
            .finish()
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Assembler {
    /// Constructs an assembler with no reserved topology.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerStateMachine::new(GraphAssemblerContext::default()),
        }
    }

    /// Dispatches a public event synchronously.
    pub fn process_event(&mut self, event: Event) -> Result<Outcome, Error> {
        match event {
            Event::Reserve(request) => self.reserve(request),
            Event::Assemble(request) => self.assemble(request),
        }
    }

    /// Dispatches a reserve request through the explicit reserve phases.
    pub fn reserve(&mut self, request: Reserve) -> Result<Outcome, Error> {
        let validate = Cell::new(PhaseOutcome::Unknown);
        let build = Cell::new(PhaseOutcome::Unknown);
        let alloc = Cell::new(PhaseOutcome::Unknown);
        let err = Cell::new(None);
        let plan = Cell::new(AllocationPlan::default());
        let output = Cell::new(AssemblyOutput::default());
        let result = RefCell::new(Err(Error::Internal));
        self.machine
            .process_event(GraphAssemblerEvents::Reserve(ReserveRuntime {
                request,
                validate: &validate,
                build: &build,
                alloc: &alloc,
                err: &err,
                plan: &plan,
                output: &output,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }

    /// Dispatches an assemble request through validation, reuse/rebuild, and allocation phases.
    pub fn assemble(&mut self, request: Assemble) -> Result<Outcome, Error> {
        let validate = Cell::new(PhaseOutcome::Unknown);
        let reuse = Cell::new(ReuseOutcome::Unknown);
        let build = Cell::new(PhaseOutcome::Unknown);
        let alloc = Cell::new(PhaseOutcome::Unknown);
        let err = Cell::new(None);
        let plan = Cell::new(AllocationPlan::default());
        let output = Cell::new(AssemblyOutput::default());
        let result = RefCell::new(Err(Error::Internal));
        self.machine
            .process_event(GraphAssemblerEvents::Assemble(AssembleRuntime {
                request,
                validate: &validate,
                reuse: &reuse,
                build: &build,
                alloc: &alloc,
                err: &err,
                plan: &plan,
                output: &output,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        result.into_inner()
    }

    /// Dispatches an explicit unexpected event and recovers to the owning public state.
    pub fn process_unexpected(&mut self, event: UnexpectedEvent) -> Result<Outcome, Error> {
        self.machine
            .process_event(GraphAssemblerEvents::Unexpected(event))
            .map_err(|_| Error::Internal)?;
        Err(Error::UnexpectedEvent)
    }

    /// Reports whether the actor has no reservation.
    #[must_use]
    pub fn is_uninitialized(&self) -> bool {
        self.machine.is(&GraphAssemblerStates::Uninitialized)
    }

    /// Reports whether the actor has a successful reservation.
    #[must_use]
    pub fn is_reserved(&self) -> bool {
        self.machine.is(&GraphAssemblerStates::Reserved)
    }
}

impl GraphAssemblerStateMachineContext for GraphAssemblerContext {
    fn guard_reserve_valid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.model_topology.0 != 0
            && event.request.lifecycle.0 != 0
            && event.request.max_node_count != 0
            && event.request.max_tensor_count != 0
            && event.request.bytes_per_tensor != 0
            && event.request.workspace_capacity_bytes != 0)
    }

    fn guard_reserve_invalid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_reserve_valid(event)?)
    }

    fn guard_reserve_build_valid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && event.request.max_node_count != 0
            && event.request.max_tensor_count != 0
            && !product_overflows(
                event.request.max_tensor_count,
                event.request.bytes_per_tensor,
            )
            && product(
                event.request.max_tensor_count,
                event.request.bytes_per_tensor,
            ) <= event.request.workspace_capacity_bytes)
    }

    fn guard_reserve_build_capacity(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && event.request.max_tensor_count != 0
            && (product_overflows(
                event.request.max_tensor_count,
                event.request.bytes_per_tensor,
            ) || product(
                event.request.max_tensor_count,
                event.request.bytes_per_tensor,
            ) > event.request.workspace_capacity_bytes))
    }

    fn guard_reserve_build_invalid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && (event.request.max_node_count == 0
                || event.request.max_tensor_count == 0
                || event.request.bytes_per_tensor == 0))
    }

    fn guard_reserve_alloc_valid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.build.get() == PhaseOutcome::Done
            && event.request.model_topology.0 != 0
            && event.request.max_node_count != 0
            && event.request.max_tensor_count != 0)
    }

    fn guard_reserve_alloc_invalid(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.build.get() != PhaseOutcome::Done)
    }

    fn guard_reserve_success(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.alloc.get() == PhaseOutcome::Done && event.err.get().is_none())
    }

    fn guard_reserve_failure(&self, event: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_reserve_success(event)?)
    }

    fn guard_assemble_valid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.step_plan.0 != 0
            && event.request.lifecycle.0 != 0
            && event.request.node_count_hint != 0
            && event.request.tensor_count_hint != 0
            && event.request.bytes_per_tensor != 0
            && event.request.workspace_capacity_bytes != 0)
    }

    fn guard_assemble_invalid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_assemble_valid(event)?)
    }

    fn guard_reuse_selected(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && self.has_reserved_topology
            && event.request.node_count_hint == self.reserved_node_count
            && event.request.tensor_count_hint == self.reserved_tensor_count
            && event.request.lifecycle == self.reserved_lifecycle)
    }

    fn guard_rebuild_selected(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && !self.has_reserved_topology
            && event.request.node_count_hint != 0
            && event.request.tensor_count_hint != 0
            || event.validate.get() == PhaseOutcome::Done
                && self.has_reserved_topology
                && (event.request.node_count_hint != self.reserved_node_count
                    || event.request.tensor_count_hint != self.reserved_tensor_count
                    || event.request.lifecycle != self.reserved_lifecycle)
                && event.request.node_count_hint != 0
                && event.request.tensor_count_hint != 0)
    }

    fn guard_assemble_reuse_invalid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && event.request.node_count_hint == 0
            && event.request.tensor_count_hint == 0)
    }

    fn guard_reuse_success(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.reuse.get() == ReuseOutcome::Reused && event.err.get().is_none())
    }

    fn guard_reuse_failure(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_reuse_success(event)?)
    }

    fn guard_assemble_build_valid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.reuse.get() == ReuseOutcome::Rebuild
            && event.request.node_count_hint != 0
            && event.request.tensor_count_hint != 0
            && !product_overflows(
                event.request.tensor_count_hint,
                event.request.bytes_per_tensor,
            )
            && product(
                event.request.tensor_count_hint,
                event.request.bytes_per_tensor,
            ) <= event.request.workspace_capacity_bytes)
    }

    fn guard_assemble_build_capacity(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.reuse.get() == ReuseOutcome::Rebuild
            && event.request.tensor_count_hint != 0
            && (product_overflows(
                event.request.tensor_count_hint,
                event.request.bytes_per_tensor,
            ) || product(
                event.request.tensor_count_hint,
                event.request.bytes_per_tensor,
            ) > event.request.workspace_capacity_bytes))
    }

    fn guard_assemble_build_invalid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.reuse.get() == ReuseOutcome::Rebuild
            && (event.request.node_count_hint == 0
                || event.request.tensor_count_hint == 0
                || event.request.bytes_per_tensor == 0))
    }

    fn guard_assemble_alloc_valid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.build.get() == PhaseOutcome::Done && event.request.step_plan.0 != 0)
    }

    fn guard_assemble_alloc_invalid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.build.get() != PhaseOutcome::Done)
    }

    fn guard_assemble_success(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.alloc.get() == PhaseOutcome::Done && event.err.get().is_none())
    }

    fn guard_assemble_failure(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_assemble_success(event)?)
    }

    fn effect_begin_reserve(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.validate.set(PhaseOutcome::Unknown);
        event.build.set(PhaseOutcome::Unknown);
        event.alloc.set(PhaseOutcome::Unknown);
        event.err.set(None);
        event.plan.set(AllocationPlan::default());
        event.output.set(AssemblyOutput::default());
        *event.result.borrow_mut() = Err(Error::Internal);
        Ok(())
    }

    fn effect_reject_second_reserve(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::Internal));
        *event.result.borrow_mut() = Err(Error::Internal);
        Ok(())
    }

    fn effect_invalid_reserve(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::InvalidRequest));
        *event.result.borrow_mut() = Err(Error::InvalidRequest);
        Ok(())
    }

    fn effect_reserve_validate_done(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.validate.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_reserve_build_done(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_reserve_build_capacity(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::Capacity));
        Ok(())
    }

    fn effect_reserve_build_invalid(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::InvalidRequest));
        Ok(())
    }

    fn effect_reserve_build_phase_done(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.plan.set(AllocationPlan {
            tensor_count: event.request.max_tensor_count,
            interval_count: event.request.max_tensor_count,
            required_buffer_bytes: product(
                event.request.max_tensor_count,
                event.request.bytes_per_tensor,
            ),
        });
        Ok(())
    }

    fn effect_reserve_alloc_done(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.alloc.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_reserve_alloc_invalid(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.alloc.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::Internal));
        Ok(())
    }

    fn effect_commit_reserve(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        self.reserved_topology = event.request.model_topology;
        self.reserved_node_count = event.request.max_node_count;
        self.reserved_tensor_count = event.request.max_tensor_count;
        self.reserved_required_buffer_bytes = event.plan.get().required_buffer_bytes;
        self.topology_version = self.topology_version.saturating_add(1);
        self.reserved_lifecycle = event.request.lifecycle;
        self.has_reserved_topology = true;
        let output = AssemblyOutput {
            graph_topology: self.reserved_topology,
            node_count: self.reserved_node_count,
            tensor_count: self.reserved_tensor_count,
            required_buffer_bytes: self.reserved_required_buffer_bytes,
            version: self.topology_version,
            reused_topology: false,
            lifecycle: self.reserved_lifecycle,
        };
        event.output.set(output);
        *event.result.borrow_mut() = Ok(Outcome::Reserved(output));
        Ok(())
    }

    fn effect_publish_reserve_error(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(event.err.get().unwrap_or(Error::Internal));
        Ok(())
    }

    fn effect_begin_assemble(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.validate.set(PhaseOutcome::Unknown);
        event.reuse.set(ReuseOutcome::Unknown);
        event.build.set(PhaseOutcome::Unknown);
        event.alloc.set(PhaseOutcome::Unknown);
        event.err.set(None);
        event.plan.set(AllocationPlan::default());
        event.output.set(AssemblyOutput::default());
        *event.result.borrow_mut() = Err(Error::Internal);
        Ok(())
    }

    fn effect_invalid_assemble(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::InvalidRequest));
        *event.result.borrow_mut() = Err(Error::InvalidRequest);
        Ok(())
    }

    fn effect_uninitialized_assemble(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::InvalidRequest));
        *event.result.borrow_mut() = Err(Error::InvalidRequest);
        Ok(())
    }

    fn effect_assemble_validate_done(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.validate.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_select_reuse(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.reuse.set(ReuseOutcome::Reused);
        event.plan.set(AllocationPlan {
            tensor_count: self.reserved_tensor_count,
            interval_count: self.reserved_tensor_count,
            required_buffer_bytes: self.reserved_required_buffer_bytes,
        });
        Ok(())
    }

    fn effect_select_rebuild(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.reuse.set(ReuseOutcome::Rebuild);
        Ok(())
    }

    fn effect_invalid_rebuild(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.reuse.set(ReuseOutcome::Failed);
        event.err.set(Some(Error::InvalidRequest));
        Ok(())
    }

    fn effect_commit_reuse(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        let output = AssemblyOutput {
            graph_topology: self.reserved_topology,
            node_count: self.reserved_node_count,
            tensor_count: self.reserved_tensor_count,
            required_buffer_bytes: self.reserved_required_buffer_bytes,
            version: self.topology_version,
            reused_topology: true,
            lifecycle: event.request.lifecycle,
        };
        event.output.set(output);
        *event.result.borrow_mut() = Ok(Outcome::Assembled(output));
        Ok(())
    }

    fn effect_publish_assemble_error(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        *event.result.borrow_mut() = Err(event.err.get().unwrap_or(Error::Internal));
        Ok(())
    }

    fn effect_assemble_build_done(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_assemble_build_capacity(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::Capacity));
        Ok(())
    }

    fn effect_assemble_build_invalid(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.build.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::InvalidRequest));
        Ok(())
    }

    fn effect_assemble_build_phase_done(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.plan.set(AllocationPlan {
            tensor_count: event.request.tensor_count_hint,
            interval_count: event.request.tensor_count_hint,
            required_buffer_bytes: product(
                event.request.tensor_count_hint,
                event.request.bytes_per_tensor,
            ),
        });
        Ok(())
    }

    fn effect_assemble_alloc_done(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.alloc.set(PhaseOutcome::Done);
        Ok(())
    }

    fn effect_assemble_alloc_invalid(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.alloc.set(PhaseOutcome::Failed);
        event.err.set(Some(Error::Internal));
        Ok(())
    }

    fn effect_commit_rebuild(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        self.reserved_topology = event.request.step_plan;
        self.reserved_node_count = event.request.node_count_hint;
        self.reserved_tensor_count = event.request.tensor_count_hint;
        self.reserved_required_buffer_bytes = event.plan.get().required_buffer_bytes;
        self.topology_version = self.topology_version.saturating_add(1);
        self.reserved_lifecycle = event.request.lifecycle;
        self.has_reserved_topology = true;
        let output = AssemblyOutput {
            graph_topology: self.reserved_topology,
            node_count: self.reserved_node_count,
            tensor_count: self.reserved_tensor_count,
            required_buffer_bytes: self.reserved_required_buffer_bytes,
            version: self.topology_version,
            reused_topology: false,
            lifecycle: self.reserved_lifecycle,
        };
        event.output.set(output);
        *event.result.borrow_mut() = Ok(Outcome::Assembled(output));
        Ok(())
    }

    fn effect_unexpected_uninitialized(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_reserved(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_reserve(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_assemble(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        Ok(())
    }
}

/// Compatibility alias for code that names the actor `GraphAssembler`.
pub type GraphAssembler = Assembler;

#[cfg(test)]
mod tests {
    use super::*;

    const TOPOLOGY: Topology = Topology(1);
    const PLAN: Topology = Topology(2);
    const LIFECYCLE: LifecycleManifest = LifecycleManifest(3);

    const fn reserve(workspace_capacity_bytes: u64) -> Reserve {
        Reserve {
            model_topology: TOPOLOGY,
            lifecycle: LIFECYCLE,
            max_node_count: 4,
            max_tensor_count: 8,
            bytes_per_tensor: 16,
            workspace_capacity_bytes,
        }
    }

    const fn assemble(node_count_hint: u32, tensor_count_hint: u32, capacity: u64) -> Assemble {
        Assemble {
            step_plan: PLAN,
            lifecycle: LIFECYCLE,
            node_count_hint,
            tensor_count_hint,
            bytes_per_tensor: 16,
            workspace_capacity_bytes: capacity,
        }
    }

    #[test]
    fn reserve_success_publishes_allocation_and_enters_reserved_state() {
        let mut actor = Assembler::new();
        assert_eq!(
            actor.reserve(reserve(128)),
            Ok(Outcome::Reserved(AssemblyOutput {
                graph_topology: TOPOLOGY,
                node_count: 4,
                tensor_count: 8,
                required_buffer_bytes: 128,
                version: 1,
                reused_topology: false,
                lifecycle: LIFECYCLE,
            }))
        );
        assert!(actor.is_reserved());
    }

    #[test]
    fn reserve_capacity_failure_is_typed_and_does_not_reserve() {
        let mut actor = Assembler::new();
        assert_eq!(actor.reserve(reserve(127)), Err(Error::Capacity));
        assert!(actor.is_uninitialized());
    }

    #[test]
    fn duplicate_reserve_is_rejected_without_replacing_reservation() {
        let mut actor = Assembler::new();
        assert!(actor.reserve(reserve(128)).is_ok());
        assert_eq!(
            actor.reserve(Reserve {
                model_topology: Topology(9),
                ..reserve(256)
            }),
            Err(Error::Internal)
        );
        assert!(actor.is_reserved());
        assert_eq!(
            actor.assemble(assemble(4, 8, 128)),
            Ok(Outcome::Assembled(AssemblyOutput {
                graph_topology: TOPOLOGY,
                node_count: 4,
                tensor_count: 8,
                required_buffer_bytes: 128,
                version: 1,
                reused_topology: true,
                lifecycle: LIFECYCLE,
            }))
        );
    }

    #[test]
    fn assemble_reuses_matching_reservation() {
        let mut actor = Assembler::new();
        assert!(actor.reserve(reserve(128)).is_ok());
        let result = actor.assemble(assemble(4, 8, 128));
        assert_eq!(
            result,
            Ok(Outcome::Assembled(AssemblyOutput {
                graph_topology: TOPOLOGY,
                node_count: 4,
                tensor_count: 8,
                required_buffer_bytes: 128,
                version: 1,
                reused_topology: true,
                lifecycle: LIFECYCLE,
            }))
        );
    }

    #[test]
    fn assemble_rebuilds_with_new_shape_and_updates_reservation() {
        let mut actor = Assembler::new();
        assert!(actor.reserve(reserve(128)).is_ok());
        let result = actor.assemble(assemble(2, 4, 64));
        assert_eq!(
            result,
            Ok(Outcome::Assembled(AssemblyOutput {
                graph_topology: PLAN,
                node_count: 2,
                tensor_count: 4,
                required_buffer_bytes: 64,
                version: 2,
                reused_topology: false,
                lifecycle: LIFECYCLE,
            }))
        );
        assert!(actor.is_reserved());
    }

    #[test]
    fn assemble_rebuild_capacity_failure_preserves_previous_reservation() {
        let mut actor = Assembler::new();
        assert!(actor.reserve(reserve(128)).is_ok());
        assert_eq!(actor.assemble(assemble(2, 4, 63)), Err(Error::Capacity));
        assert!(actor.is_reserved());
    }

    #[test]
    fn invalid_requests_are_rejected_explicitly() {
        let mut actor = Assembler::new();
        assert_eq!(
            actor.reserve(Reserve {
                model_topology: Topology(0),
                ..reserve(128)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(
            actor.assemble(assemble(0, 0, 128)),
            Err(Error::InvalidRequest)
        );
        assert!(actor.is_uninitialized());
    }

    #[test]
    fn unexpected_event_is_typed_and_recovers_to_public_state() {
        let mut actor = Assembler::new();
        assert_eq!(
            actor.process_unexpected(UnexpectedEvent),
            Err(Error::UnexpectedEvent)
        );
        assert!(actor.is_uninitialized());
        assert!(actor.reserve(reserve(128)).is_ok());
        assert_eq!(
            actor.process_unexpected(UnexpectedEvent),
            Err(Error::UnexpectedEvent)
        );
        assert!(actor.is_reserved());
    }
}
