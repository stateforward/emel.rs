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

use crate::allocator::sm as allocator;
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
///
/// The value is copied into the optional output sink before the completion
/// callback is invoked. Both the sink and callback are borrowed only for this
/// synchronous run-to-completion dispatch; the callback's boolean return is
/// intentionally observational and does not alter the public [`Outcome`].
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

/// Synchronous assembler completion callback.
///
/// The callback runs after `output_out` has been updated, if supplied. It must
/// not retain references or re-enter the assembler; its return value is ignored
/// because callback observation is not a second success/failure channel.
pub type DispatchDoneFn = fn(&AssemblyOutput) -> bool;
/// Synchronous assembler error callback.
///
/// Errors publish the zero/default [`AssemblyOutput`] before this callback.
/// The callback must not retain references or re-enter the assembler, and its
/// boolean return is ignored.
pub type DispatchErrorFn = fn(&AssemblyOutput, Error) -> bool;

/// Reserve request corresponding to the C++ `event::reserve` event.
///
/// Request fields and callback function pointers are caller-owned for the
/// duration of one synchronous dispatch; the actor retains only successful
/// reservation facts in its persistent context.
#[derive(Clone, Copy, Debug)]
pub struct Reserve<'request> {
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
    pub output_out: Option<&'request Cell<AssemblyOutput>>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}
/// Assemble request corresponding to the C++ `event::assemble` event.
///
/// Like [`Reserve`], this request is consumed synchronously. Reuse or rebuild
/// completion is published before `assemble` returns, and no request callback
/// or transient phase result is retained by the actor.
#[derive(Clone, Copy, Debug)]
pub struct Assemble<'request> {
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
    pub output_out: Option<&'request Cell<AssemblyOutput>>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Public event set accepted by the actor.
#[derive(Clone, Copy, Debug)]
pub enum Event<'request> {
    /// Reserve graph storage.
    Reserve(Reserve<'request>),
    /// Assemble a step plan, reusing a compatible reservation when possible.
    Assemble(Assemble<'request>),
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
    request: Reserve<'dispatch>,
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
    request: Assemble<'dispatch>,
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
#[derive(Debug, Default)]
pub struct GraphAssemblerContext {
    reserved_topology: Topology,
    reserved_node_count: u32,
    reserved_tensor_count: u32,
    reserved_required_buffer_bytes: u64,
    topology_version: u32,
    reserved_lifecycle: LifecycleManifest,
    has_reserved_topology: bool,
    allocator: allocator::Allocator,
    last_error: Option<Error>,
}

const fn allocation_done(_: allocator::AllocationDone) -> bool {
    true
}
const fn allocation_error(_: allocator::AllocationErrorEvent) -> bool {
    true
}
const fn map_allocator_error(error: allocator::AllocationError) -> Option<Error> {
    match error {
        allocator::AllocationError::None => None,
        allocator::AllocationError::InvalidRequest => Some(Error::InvalidRequest),
        allocator::AllocationError::Capacity => Some(Error::Capacity),
        allocator::AllocationError::Internal => Some(Error::Internal),
        allocator::AllocationError::Untracked => Some(Error::Untracked),
    }
}
fn publish_done(
    sink: Option<&Cell<AssemblyOutput>>,
    cb: Option<DispatchDoneFn>,
    output: AssemblyOutput,
) {
    if let Some(sink) = sink {
        sink.set(output);
    }
    if let Some(cb) = cb {
        let _ = cb(&sink.map_or(output, Cell::get));
    }
}
fn publish_error(sink: Option<&Cell<AssemblyOutput>>, cb: Option<DispatchErrorFn>, error: Error) {
    let output = AssemblyOutput::default();
    if let Some(sink) = sink {
        sink.set(output);
    }
    if let Some(cb) = cb {
        let _ = cb(&sink.map_or(output, Cell::get), error);
    }
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
            + Reserve(ReserveRuntime<'dispatch>) [guard_reserve_valid] / effect_invalid_reserve,
        "reserved"_s <= "reserved"_s
            + Reserve(ReserveRuntime<'dispatch>) [guard_reserve_invalid] / effect_invalid_reserve,
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
        "assemble_dispatch_decision"_s <= "assemble_reuse_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_reuse_invalid]
            / effect_invalid_rebuild,

        "assemble_dispatch_decision"_s <= "assemble_reuse"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_reuse_success]
            / effect_commit_reuse,
        "assemble_dispatch_decision"_s <= "assemble_reuse"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_reuse_failure]
            / effect_dispatch_assemble_error,

        "assemble_build"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_valid]
            / effect_assemble_build_done,
        "assemble_dispatch_decision"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_capacity]
            / effect_assemble_build_capacity,
        "assemble_dispatch_decision"_s <= "assemble_build_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_build_invalid]
            / effect_assemble_build_invalid,
        "assemble_alloc_decision"_s <= "assemble_build"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) / effect_assemble_build_phase_done,
        "assemble_alloc"_s <= "assemble_alloc_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_alloc_valid]
            / effect_assemble_alloc_done,
        "assemble_dispatch_decision"_s <= "assemble_alloc_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_alloc_invalid]
            / effect_assemble_alloc_invalid,
        "assemble_dispatch_decision"_s <= "assemble_alloc"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_success]
            / effect_commit_rebuild,
        "assemble_dispatch_decision"_s <= "assemble_alloc"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_failure]
            / effect_dispatch_assemble_error,
        "reserved"_s <= "assemble_dispatch_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_error_none]
            / effect_dispatch_assemble_done,
        "reserved"_s <= "assemble_dispatch_decision"_s
            + completion<Assemble>(AssembleRuntime<'dispatch>) [guard_assemble_error]
            / effect_dispatch_assemble_error,

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
        "reserved"_s <= "assemble_dispatch_decision"_s + Unexpected(UnexpectedEvent)
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
    ///
    /// # Errors
    pub fn process_event<'request>(&mut self, event: Event<'request>) -> Result<Outcome, Error> {
        match event {
            Event::Reserve(request) => self.reserve(request),
            Event::Assemble(request) => self.assemble(request),
        }
    }

    /// Dispatches a reserve request through the explicit reserve phases.
    ///
    /// # Errors
    ///
    /// Returns an error when the request is invalid, exceeds capacity, or an internal phase fails.
    pub fn reserve<'request>(&mut self, request: Reserve<'request>) -> Result<Outcome, Error> {
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
    ///
    /// # Errors
    ///
    /// Returns an error when the request is invalid, exceeds capacity, or an internal phase fails.
    pub fn assemble<'request>(&mut self, request: Assemble<'request>) -> Result<Outcome, Error> {
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
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedEvent`] after recovery, or [`Error::Internal`] if recovery fails.
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
            && event.request.workspace_capacity_bytes != 0
            && event.request.output_out.is_some()
            && event.request.dispatch_done.is_some()
            && event.request.dispatch_error.is_some())
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
            && event.request.workspace_capacity_bytes != 0
            && event.request.output_out.is_some()
            && event.request.dispatch_done.is_some()
            && event.request.dispatch_error.is_some())
    }

    fn guard_assemble_invalid(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_assemble_valid(event)?)
    }

    fn guard_reuse_selected(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.validate.get() == PhaseOutcome::Done
            && self.has_reserved_topology
            && event.request.node_count_hint == self.reserved_node_count
            && event.request.tensor_count_hint == self.reserved_tensor_count)
    }

    fn guard_rebuild_selected(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok((event.validate.get() == PhaseOutcome::Done
            && !self.has_reserved_topology
            && event.request.node_count_hint != 0
            && event.request.tensor_count_hint != 0)
            || (event.validate.get() == PhaseOutcome::Done
                && self.has_reserved_topology
                && (event.request.node_count_hint != self.reserved_node_count
                    || event.request.tensor_count_hint != self.reserved_tensor_count)
                && event.request.node_count_hint != 0
                && event.request.tensor_count_hint != 0))
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

    fn guard_assemble_error_none(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.err.get().is_none())
    }

    fn guard_assemble_error(&self, event: &AssembleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.err.get().is_some())
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

    fn effect_invalid_reserve(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::InvalidRequest));
        publish_error(
            event.request.output_out,
            event.request.dispatch_error,
            Error::InvalidRequest,
        );
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
        let request = allocator::EventAllocateGraphPlan {
            request: allocator::AllocateGraph {
                graph_topology: event.request.model_topology.0,
                plan_out: true,
                node_count: event.request.max_node_count,
                tensor_count: event.request.max_tensor_count,
                tensor_capacity: event.request.max_tensor_count,
                interval_capacity: event.request.max_tensor_count,
                bytes_per_tensor: event.request.bytes_per_tensor,
                workspace_capacity_bytes: event.request.workspace_capacity_bytes,
                dispatch_done: Some(allocation_done),
                dispatch_error: Some(allocation_error),
            },
        };
        let accepted = self.allocator.process_event(request);
        let child_error = map_allocator_error(self.allocator.error());
        if accepted && child_error.is_none() {
            let plan = self.allocator.plan();
            event.plan.set(AllocationPlan {
                tensor_count: plan.tensor_count,
                interval_count: plan.interval_count,
                required_buffer_bytes: plan.required_buffer_bytes,
            });
            event.alloc.set(PhaseOutcome::Done);
        } else {
            event.alloc.set(PhaseOutcome::Failed);
            event.err.set(child_error.or(Some(Error::Internal)));
        }
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
        publish_done(
            event.request.output_out,
            event.request.dispatch_done,
            output,
        );
        *event.result.borrow_mut() = Ok(Outcome::Reserved(output));
        Ok(())
    }

    fn effect_publish_reserve_error(&mut self, event: ReserveRuntime<'_>) -> Result<(), ()> {
        let error = event.err.get().unwrap_or(Error::Internal);
        publish_error(
            event.request.output_out,
            event.request.dispatch_error,
            error,
        );
        *event.result.borrow_mut() = Err(error);
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
        publish_error(
            event.request.output_out,
            event.request.dispatch_error,
            Error::InvalidRequest,
        );
        *event.result.borrow_mut() = Err(Error::InvalidRequest);
        Ok(())
    }

    fn effect_uninitialized_assemble(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        event.err.set(Some(Error::InvalidRequest));
        publish_error(
            event.request.output_out,
            event.request.dispatch_error,
            Error::InvalidRequest,
        );
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
            lifecycle: self.reserved_lifecycle,
        };
        event.output.set(output);
        *event.result.borrow_mut() = Ok(Outcome::Assembled(output));
        Ok(())
    }

    fn effect_dispatch_assemble_error(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        let error = event.err.get().unwrap_or(Error::Internal);
        publish_error(
            event.request.output_out,
            event.request.dispatch_error,
            error,
        );
        *event.result.borrow_mut() = Err(error);
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
        let request = allocator::EventAllocateGraphPlan {
            request: allocator::AllocateGraph {
                graph_topology: event.request.step_plan.0,
                plan_out: true,
                node_count: event.request.node_count_hint,
                tensor_count: event.request.tensor_count_hint,
                tensor_capacity: event.request.tensor_count_hint,
                interval_capacity: event.request.tensor_count_hint,
                bytes_per_tensor: event.request.bytes_per_tensor,
                workspace_capacity_bytes: event.request.workspace_capacity_bytes,
                dispatch_done: Some(allocation_done),
                dispatch_error: Some(allocation_error),
            },
        };
        let accepted = self.allocator.process_event(request);
        let child_error = map_allocator_error(self.allocator.error());
        if accepted && child_error.is_none() {
            let plan = self.allocator.plan();
            event.plan.set(AllocationPlan {
                tensor_count: plan.tensor_count,
                interval_count: plan.interval_count,
                required_buffer_bytes: plan.required_buffer_bytes,
            });
            event.alloc.set(PhaseOutcome::Done);
        } else {
            event.alloc.set(PhaseOutcome::Failed);
            event.err.set(child_error.or(Some(Error::Internal)));
        }
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
        self.last_error = Some(Error::Internal);
        Ok(())
    }

    fn effect_unexpected_reserved(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        self.last_error = Some(Error::Internal);
        Ok(())
    }

    fn effect_unexpected_reserve(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        self.last_error = Some(Error::Internal);
        Ok(())
    }

    fn effect_unexpected_assemble(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        self.last_error = Some(Error::Internal);
        Ok(())
    }

    fn effect_dispatch_assemble_done(&mut self, event: AssembleRuntime<'_>) -> Result<(), ()> {
        let output = event.output.get();
        publish_done(
            event.request.output_out,
            event.request.dispatch_done,
            output,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    const TOPOLOGY: Topology = Topology(1);
    const PLAN: Topology = Topology(2);
    const OTHER_TOPOLOGY: Topology = Topology(9);
    const LIFECYCLE: LifecycleManifest = LifecycleManifest(3);
    const OTHER_LIFECYCLE: LifecycleManifest = LifecycleManifest(9);

    static ORDER_DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static DUPLICATE_VALID_ERRORS: AtomicUsize = AtomicUsize::new(0);
    static DUPLICATE_INVALID_ERRORS: AtomicUsize = AtomicUsize::new(0);
    static REBUILD_DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static REBUILD_CAPACITY_ERRORS: AtomicUsize = AtomicUsize::new(0);
    static INVALID_ERRORS: AtomicUsize = AtomicUsize::new(0);
    static LIFECYCLE_DONE_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn done_callback(output: &AssemblyOutput) -> bool {
        assert_ne!(*output, AssemblyOutput::default());
        true
    }

    fn error_callback(output: &AssemblyOutput, _: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        true
    }

    fn ordering_done(output: &AssemblyOutput) -> bool {
        assert_ne!(*output, AssemblyOutput::default());
        ORDER_DONE_CALLS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn duplicate_valid_error(output: &AssemblyOutput, error: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        assert_eq!(error, Error::InvalidRequest);
        DUPLICATE_VALID_ERRORS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn duplicate_invalid_error(output: &AssemblyOutput, error: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        assert_eq!(error, Error::InvalidRequest);
        DUPLICATE_INVALID_ERRORS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn rebuild_done(output: &AssemblyOutput) -> bool {
        assert_eq!(output.graph_topology, PLAN);
        REBUILD_DONE_CALLS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn rebuild_capacity_error(output: &AssemblyOutput, error: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        assert_eq!(error, Error::Capacity);
        REBUILD_CAPACITY_ERRORS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn invalid_error(output: &AssemblyOutput, error: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        assert_eq!(error, Error::InvalidRequest);
        INVALID_ERRORS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn lifecycle_done(output: &AssemblyOutput) -> bool {
        assert_eq!(output.lifecycle, LIFECYCLE);
        LIFECYCLE_DONE_CALLS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn callback_false(_: &AssemblyOutput) -> bool {
        false
    }

    fn error_callback_false(output: &AssemblyOutput, error: Error) -> bool {
        assert_eq!(*output, AssemblyOutput::default());
        assert_eq!(error, Error::InvalidRequest);
        false
    }

    fn reserve<'a>(capacity: u64, sink: &'a Cell<AssemblyOutput>) -> Reserve<'a> {
        Reserve {
            model_topology: TOPOLOGY,
            lifecycle: LIFECYCLE,
            max_node_count: 4,
            max_tensor_count: 8,
            bytes_per_tensor: 16,
            workspace_capacity_bytes: capacity,
            output_out: Some(sink),
            dispatch_done: Some(done_callback),
            dispatch_error: Some(error_callback),
        }
    }

    fn assemble<'a>(
        nodes: u32,
        tensors: u32,
        capacity: u64,
        sink: &'a Cell<AssemblyOutput>,
    ) -> Assemble<'a> {
        Assemble {
            step_plan: PLAN,
            lifecycle: LIFECYCLE,
            node_count_hint: nodes,
            tensor_count_hint: tensors,
            bytes_per_tensor: 16,
            workspace_capacity_bytes: capacity,
            output_out: Some(sink),
            dispatch_done: Some(done_callback),
            dispatch_error: Some(error_callback),
        }
    }

    fn reserved_output() -> AssemblyOutput {
        AssemblyOutput {
            graph_topology: TOPOLOGY,
            node_count: 4,
            tensor_count: 8,
            required_buffer_bytes: 128,
            version: 1,
            reused_topology: false,
            lifecycle: LIFECYCLE,
        }
    }

    fn reused_output() -> AssemblyOutput {
        AssemblyOutput {
            reused_topology: true,
            ..reserved_output()
        }
    }

    #[test]
    fn reserve_success_publishes_output_and_enters_reserved_state() {
        let mut actor = Assembler::new();
        let sink = Cell::new(AssemblyOutput::default());

        assert_eq!(
            actor.reserve(reserve(128, &sink)),
            Ok(Outcome::Reserved(reserved_output()))
        );
        assert_eq!(sink.get(), reserved_output());
        assert!(actor.is_reserved());
    }

    #[test]
    fn reserve_capacity_failure_resets_output_and_does_not_reserve() {
        let mut actor = Assembler::new();
        let sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        REBUILD_CAPACITY_ERRORS.store(0, Ordering::SeqCst);

        let result = actor.reserve(Reserve {
            dispatch_error: Some(rebuild_capacity_error),
            ..reserve(127, &sink)
        });

        assert_eq!(result, Err(Error::Capacity));
        assert_eq!(sink.get(), AssemblyOutput::default());
        assert_eq!(REBUILD_CAPACITY_ERRORS.load(Ordering::SeqCst), 1);
        assert!(actor.is_uninitialized());
    }

    #[test]
    fn missing_publication_fields_are_rejected_without_reserving() {
        let mut actor = Assembler::new();

        let sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        assert_eq!(
            actor.reserve(Reserve {
                output_out: None,
                ..reserve(128, &sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(sink.get().graph_topology, TOPOLOGY);

        let sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        assert_eq!(
            actor.reserve(Reserve {
                dispatch_done: None,
                ..reserve(128, &sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(sink.get(), AssemblyOutput::default());

        let sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        assert_eq!(
            actor.reserve(Reserve {
                dispatch_error: None,
                ..reserve(128, &sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(sink.get(), AssemblyOutput::default());
        assert!(actor.is_uninitialized());
    }

    #[test]
    fn valid_duplicate_reserve_rejects_and_preserves_original_reservation() {
        let mut actor = Assembler::new();
        let first_sink = Cell::new(AssemblyOutput::default());
        let second_sink = Cell::new(AssemblyOutput {
            graph_topology: OTHER_TOPOLOGY,
            ..AssemblyOutput::default()
        });
        DUPLICATE_VALID_ERRORS.store(0, Ordering::SeqCst);

        assert_eq!(
            actor.reserve(reserve(128, &first_sink)),
            Ok(Outcome::Reserved(reserved_output()))
        );
        assert_eq!(
            actor.reserve(Reserve {
                model_topology: OTHER_TOPOLOGY,
                lifecycle: OTHER_LIFECYCLE,
                dispatch_error: Some(duplicate_valid_error),
                ..reserve(256, &second_sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(second_sink.get(), AssemblyOutput::default());
        assert_eq!(DUPLICATE_VALID_ERRORS.load(Ordering::SeqCst), 1);
        assert!(actor.is_reserved());

        let reuse_sink = Cell::new(AssemblyOutput::default());
        assert_eq!(
            actor.assemble(assemble(4, 8, 128, &reuse_sink)),
            Ok(Outcome::Assembled(reused_output()))
        );
        assert_eq!(reuse_sink.get(), reused_output());
    }

    #[test]
    fn invalid_duplicate_reserve_rejects_and_preserves_original_reservation() {
        let mut actor = Assembler::new();
        let first_sink = Cell::new(AssemblyOutput::default());
        let second_sink = Cell::new(AssemblyOutput {
            graph_topology: OTHER_TOPOLOGY,
            ..AssemblyOutput::default()
        });
        DUPLICATE_INVALID_ERRORS.store(0, Ordering::SeqCst);

        assert!(actor.reserve(reserve(128, &first_sink)).is_ok());
        assert_eq!(
            actor.reserve(Reserve {
                model_topology: Topology(0),
                dispatch_error: Some(duplicate_invalid_error),
                ..reserve(128, &second_sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(second_sink.get(), AssemblyOutput::default());
        assert_eq!(DUPLICATE_INVALID_ERRORS.load(Ordering::SeqCst), 1);
        assert!(actor.is_reserved());

        let reuse_sink = Cell::new(AssemblyOutput::default());
        assert_eq!(
            actor.assemble(assemble(4, 8, 128, &reuse_sink)),
            Ok(Outcome::Assembled(reused_output()))
        );
    }

    #[test]
    fn assemble_reuses_matching_reservation() {
        let mut actor = Assembler::new();
        let reserve_sink = Cell::new(AssemblyOutput::default());
        let assemble_sink = Cell::new(AssemblyOutput::default());

        assert!(actor.reserve(reserve(128, &reserve_sink)).is_ok());
        assert_eq!(
            actor.assemble(assemble(4, 8, 128, &assemble_sink)),
            Ok(Outcome::Assembled(reused_output()))
        );
        assert_eq!(assemble_sink.get(), reused_output());
    }

    #[test]
    fn assemble_rebuilds_with_new_shape_and_updates_reservation() {
        let mut actor = Assembler::new();
        let reserve_sink = Cell::new(AssemblyOutput::default());
        let assemble_sink = Cell::new(AssemblyOutput::default());
        REBUILD_DONE_CALLS.store(0, Ordering::SeqCst);

        assert!(actor.reserve(reserve(128, &reserve_sink)).is_ok());
        assert_eq!(
            actor.assemble(Assemble {
                dispatch_done: Some(rebuild_done),
                ..assemble(2, 4, 64, &assemble_sink)
            }),
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
        assert_eq!(REBUILD_DONE_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(assemble_sink.get().graph_topology, PLAN);
        assert!(actor.is_reserved());
    }

    #[test]
    fn assemble_capacity_failure_preserves_old_reservation() {
        let mut actor = Assembler::new();
        let reserve_sink = Cell::new(AssemblyOutput::default());
        let failed_sink = Cell::new(AssemblyOutput {
            graph_topology: OTHER_TOPOLOGY,
            ..AssemblyOutput::default()
        });
        let reuse_sink = Cell::new(AssemblyOutput::default());
        REBUILD_CAPACITY_ERRORS.store(0, Ordering::SeqCst);

        assert!(actor.reserve(reserve(128, &reserve_sink)).is_ok());
        assert_eq!(
            actor.assemble(Assemble {
                dispatch_error: Some(rebuild_capacity_error),
                ..assemble(2, 4, 63, &failed_sink)
            }),
            Err(Error::Capacity)
        );
        assert_eq!(failed_sink.get(), AssemblyOutput::default());
        assert_eq!(REBUILD_CAPACITY_ERRORS.load(Ordering::SeqCst), 1);
        assert!(actor.is_reserved());
        assert_eq!(
            actor.assemble(assemble(4, 8, 128, &reuse_sink)),
            Ok(Outcome::Assembled(reused_output()))
        );
    }

    #[test]
    fn invalid_requests_are_rejected_explicitly() {
        let mut actor = Assembler::new();
        let reserve_sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        INVALID_ERRORS.store(0, Ordering::SeqCst);

        assert_eq!(
            actor.reserve(Reserve {
                model_topology: Topology(0),
                dispatch_error: Some(invalid_error),
                ..reserve(128, &reserve_sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(reserve_sink.get(), AssemblyOutput::default());

        let assemble_sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });
        assert_eq!(
            actor.assemble(Assemble {
                dispatch_error: Some(invalid_error),
                ..assemble(0, 0, 128, &assemble_sink)
            }),
            Err(Error::InvalidRequest)
        );
        assert_eq!(assemble_sink.get(), AssemblyOutput::default());
        assert_eq!(INVALID_ERRORS.load(Ordering::SeqCst), 2);
        assert!(actor.is_uninitialized());
    }

    #[test]
    fn unexpected_event_returns_error_and_recovers_to_public_state() {
        let mut actor = Assembler::new();
        assert_eq!(
            actor.process_unexpected(UnexpectedEvent),
            Err(Error::UnexpectedEvent)
        );
        assert!(actor.is_uninitialized());

        let reserve_sink = Cell::new(AssemblyOutput::default());
        assert!(actor.reserve(reserve(128, &reserve_sink)).is_ok());
        assert_eq!(
            actor.process_unexpected(UnexpectedEvent),
            Err(Error::UnexpectedEvent)
        );
        assert!(actor.is_reserved());
    }

    #[test]
    fn callbacks_are_synchronous_sink_first_observers() {
        let mut actor = Assembler::new();
        let sink = Cell::new(AssemblyOutput::default());
        ORDER_DONE_CALLS.store(0, Ordering::SeqCst);

        let result = actor.reserve(Reserve {
            dispatch_done: Some(ordering_done),
            ..reserve(128, &sink)
        });

        assert_eq!(result, Ok(Outcome::Reserved(reserved_output())));
        assert_eq!(sink.get(), reserved_output());
        assert_eq!(ORDER_DONE_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn error_callback_observes_default_output_and_cannot_change_result() {
        let mut actor = Assembler::new();
        let sink = Cell::new(AssemblyOutput {
            graph_topology: TOPOLOGY,
            ..AssemblyOutput::default()
        });

        let result = actor.reserve(Reserve {
            model_topology: Topology(0),
            dispatch_error: Some(error_callback_false),
            ..reserve(128, &sink)
        });

        assert_eq!(result, Err(Error::InvalidRequest));
        assert_eq!(sink.get(), AssemblyOutput::default());
    }

    #[test]
    fn same_shape_different_lifecycle_reuses_reserved_lifecycle() {
        let mut actor = Assembler::new();
        let reserve_sink = Cell::new(AssemblyOutput::default());
        let assemble_sink = Cell::new(AssemblyOutput::default());
        LIFECYCLE_DONE_CALLS.store(0, Ordering::SeqCst);

        assert!(actor.reserve(reserve(128, &reserve_sink)).is_ok());
        assert_eq!(
            actor.assemble(Assemble {
                lifecycle: OTHER_LIFECYCLE,
                dispatch_done: Some(lifecycle_done),
                ..assemble(4, 8, 128, &assemble_sink)
            }),
            Ok(Outcome::Assembled(reused_output()))
        );
        assert_eq!(assemble_sink.get().lifecycle, LIFECYCLE);
        assert_eq!(LIFECYCLE_DONE_CALLS.load(Ordering::SeqCst), 1);
    }
}
