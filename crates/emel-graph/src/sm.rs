//! Synchronous graph root actor composing assembly, tensor, and processor actors.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_value,
    clippy::needless_pass_by_ref_mut,
    clippy::unused_self,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs,
    // The SML macro must expose its generated event/context surface to the
    // enclosing crate while carrying private, synchronous runtime payloads.
    private_interfaces,
)]

use core::cell::{Cell, RefCell};
use core::fmt;
use sml::sml;

use super::{assembler::sm as assembler, processor::sm as processor, tensor::sm as tensor};

/// Root graph error classification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum RootError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    AssemblerFailed = 2,
    ProcessorFailed = 3,
    Busy = 4,
    InternalError = 5,
    Untracked = 6,
    Unknown = 255,
}

/// Result committed to caller-owned storage before a callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphOutput {
    pub graph_topology: usize,
    pub node_count: u32,
    pub tensor_count: u32,
    pub required_buffer_bytes: u64,
    pub version: u32,
    pub reused_topology: bool,
    pub outputs_produced: i32,
    pub graph_reused: bool,
    pub lifecycle: usize,
}

/// Tensor binding in the reserve lifecycle manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorBinding {
    pub tensor_id: usize,
    pub buffer: tensor::BufferHandle,
    pub buffer_bytes: u64,
    pub consumer_refs: u32,
    pub is_leaf: bool,
}

/// Synchronous graph completion callback.
pub type DispatchDoneFn = fn(&GraphOutput) -> bool;
/// Synchronous graph error callback.
pub type DispatchErrorFn = fn(&GraphOutput, RootError) -> bool;

/// Graph reservation request.
#[derive(Clone, Copy, Debug)]
pub struct ReserveRequest<'a> {
    pub model_topology: usize,
    pub output_out: Option<&'a Cell<GraphOutput>>,
    pub lifecycle: usize,
    pub tensor_manifest: Option<&'a [TensorBinding]>,
    pub max_node_count: u32,
    pub max_tensor_count: u32,
    pub bytes_per_tensor: u64,
    pub workspace_capacity_bytes: u64,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Full graph compute request.
#[derive(Clone, Copy, Debug)]
pub struct ComputeRequest<'a> {
    pub assemble: assembler::Assemble<'a>,
    pub execute: processor::ExecuteRequest,
    /// Borrowed lifecycle phase projected from the graph manifest.
    pub lifecycle_manifest: processor::LifecycleManifest<'a>,
    pub output_out: Option<&'a Cell<GraphOutput>>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Compute request using a previously reserved graph.
#[derive(Clone, Copy, Debug)]
pub struct ComputeReservedRequest<'a> {
    pub execute: processor::ExecuteRequest,
    /// Borrowed lifecycle phase projected from the graph manifest.
    pub lifecycle_manifest: processor::LifecycleManifest<'a>,
    pub output_out: Option<&'a Cell<GraphOutput>>,
    pub dispatch_done: Option<DispatchDoneFn>,
    pub dispatch_error: Option<DispatchErrorFn>,
}

/// Public graph event set.
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    Reserve(ReserveRequest<'a>),
    Compute(ComputeRequest<'a>),
    ComputeReserved(ComputeReservedRequest<'a>),
}

/// Explicit unexpected event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

/// Successful graph operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Reserved(GraphOutput),
    Computed(GraphOutput),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PhaseOutcome {
    #[default]
    Unknown,
    Done,
    Failed,
}

#[derive(Clone, Copy)]
struct ReserveRuntime<'d> {
    request: ReserveRequest<'d>,
    reserve: &'d Cell<PhaseOutcome>,
    tensors: &'d Cell<PhaseOutcome>,
    output: &'d Cell<GraphOutput>,
    error: &'d Cell<RootError>,
    result: &'d RefCell<Result<Outcome, RootError>>,
}
#[derive(Clone, Copy)]
struct ComputeRuntime<'d> {
    request: ComputeRequest<'d>,
    assemble: &'d Cell<PhaseOutcome>,
    execute: &'d Cell<PhaseOutcome>,
    output: &'d Cell<GraphOutput>,
    error: &'d Cell<RootError>,
    result: &'d RefCell<Result<Outcome, RootError>>,
}
#[derive(Clone, Copy)]
struct ReservedRuntime<'d> {
    request: ComputeReservedRequest<'d>,
    execute: &'d Cell<PhaseOutcome>,
    output: &'d Cell<GraphOutput>,
    error: &'d Cell<RootError>,
    result: &'d RefCell<Result<Outcome, RootError>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Reservation {
    output: GraphOutput,
    present: bool,
}

/// Actor-owned root context.
#[derive(Debug)]
pub struct GraphContext {
    assembler: assembler::Assembler,
    processor: processor::GraphProcessor,
    tensor: tensor::GraphTensor,
    reservation: Reservation,
    reserved_tensor_ids: Box<[bool]>,
    dispatch_generation: u64,
    last_error: RootError,
}
impl Default for GraphContext {
    fn default() -> Self {
        Self {
            assembler: assembler::Assembler::new(),
            processor: processor::GraphProcessor::new(),
            tensor: tensor::GraphTensor::new(),
            reservation: Reservation::default(),
            reserved_tensor_ids: vec![false; tensor::MAX_TENSORS].into_boxed_slice(),
            dispatch_generation: 0,
            last_error: RootError::None,
        }
    }
}

fn map_assembler_error(error: assembler::Error) -> RootError {
    match error {
        assembler::Error::InvalidRequest => RootError::InvalidRequest,
        assembler::Error::Untracked => RootError::Untracked,
        assembler::Error::Capacity | assembler::Error::Internal => RootError::AssemblerFailed,
        assembler::Error::UnexpectedEvent => RootError::InternalError,
    }
}
fn map_processor_error(error: processor::ProcessorError) -> RootError {
    match error {
        processor::ProcessorError::None => RootError::None,
        processor::ProcessorError::InvalidRequest => RootError::InvalidRequest,
        processor::ProcessorError::KernelFailed => RootError::ProcessorFailed,
        processor::ProcessorError::InternalError => RootError::InternalError,
        processor::ProcessorError::Untracked => RootError::Untracked,
        processor::ProcessorError::Callback | processor::ProcessorError::Unknown => {
            RootError::Unknown
        }
    }
}
fn valid_processor(r: &processor::ExecuteRequest) -> bool {
    r.step_plan != 0
        && r.lifecycle != 0
        && r.step_index >= 0
        && r.step_size > 0
        && r.kv_tokens >= 0
        && r.expected_outputs >= 0
        && r.positions_count >= 0
        && (r.positions_count == 0 || r.positions != 0)
        && r.seq_mask_words > 0
        && r.seq_masks_count >= 0
        && (r.seq_masks_count == 0 || r.seq_masks != 0)
        && r.seq_primary_ids_count >= 0
        && ((r.seq_primary_ids_count == 0 && r.positions_count == 0)
            || (r.seq_primary_ids_count > 0 && r.positions_count > 0))
        && r.validate.is_some()
        && r.prepare_graph.is_some()
        && r.alloc_graph.is_some()
        && r.bind_inputs.is_some()
        && r.run_kernel.is_some()
        && r.extract_outputs.is_some()
        && r.lifecycle_gate.is_some()
        && r.lifecycle_publish.is_some()
        && r.lifecycle_release.is_some()
}

fn valid_tensor_binding(binding: &TensorBinding, max_tensor_count: u32) -> bool {
    binding.tensor_id < tensor::MAX_TENSORS
        && binding.tensor_id < max_tensor_count as usize
        && binding.buffer.0 != 0
        && binding.buffer_bytes != 0
}
fn valid_phase_ids(ids: &[usize], reserved_tensor_ids: &[bool]) -> bool {
    ids.iter()
        .all(|&tensor_id| tensor_id < tensor::MAX_TENSORS && reserved_tensor_ids[tensor_id])
}

fn valid_unique_ids(ids: &[usize], reserved_tensor_ids: &[bool]) -> bool {
    valid_phase_ids(ids, reserved_tensor_ids)
        && ids
            .iter()
            .enumerate()
            .all(|(index, &tensor_id)| ids[..index].iter().all(|&previous| previous != tensor_id))
}

fn valid_lifecycle_manifest(
    manifest: processor::LifecycleManifest<'_>,
    reserved_tensor_ids: &[bool],
) -> bool {
    valid_unique_ids(manifest.phase.required_filled_ids, reserved_tensor_ids)
        && valid_unique_ids(manifest.phase.publish_ids, reserved_tensor_ids)
        && valid_unique_ids(manifest.phase.release_ids, reserved_tensor_ids)
}

fn valid_reserve(r: &ReserveRequest<'_>) -> bool {
    r.model_topology != 0
        && r.lifecycle != 0
        && r.output_out.is_some()
        && r.tensor_manifest.is_some_and(|manifest| {
            !manifest.is_empty()
                && manifest.len() <= r.max_tensor_count as usize
                && manifest
                    .iter()
                    .all(|binding| valid_tensor_binding(binding, r.max_tensor_count))
                && manifest.iter().enumerate().all(|(index, binding)| {
                    manifest[..index]
                        .iter()
                        .all(|previous| previous.tensor_id != binding.tensor_id)
                })
        })
        && r.max_node_count != 0
        && r.max_tensor_count != 0
        && r.bytes_per_tensor != 0
        && r.workspace_capacity_bytes != 0
        && r.dispatch_done.is_some()
        && r.dispatch_error.is_some()
}
fn valid_compute(c: &GraphContext, r: &ComputeRequest<'_>) -> bool {
    r.assemble.step_plan.0 != 0
        && r.assemble.lifecycle.0 != 0
        && r.execute.lifecycle == r.assemble.lifecycle.0
        && r.assemble.node_count_hint != 0
        && r.assemble.tensor_count_hint != 0
        && r.assemble.bytes_per_tensor != 0
        && r.assemble.workspace_capacity_bytes != 0
        && r.output_out.is_some()
        && r.dispatch_done.is_some()
        && r.dispatch_error.is_some()
        && valid_lifecycle_manifest(r.lifecycle_manifest, &c.reserved_tensor_ids)
        && valid_processor(&r.execute)
}
fn valid_reserved(c: &GraphContext, r: &ComputeReservedRequest<'_>) -> bool {
    c.reservation.present
        && r.output_out.is_some()
        && r.dispatch_done.is_some()
        && r.dispatch_error.is_some()
        && valid_lifecycle_manifest(r.lifecycle_manifest, &c.reserved_tensor_ids)
        && valid_processor(&r.execute)
        && r.execute.lifecycle == c.reservation.output.lifecycle
}
fn assembler_done(_: &assembler::AssemblyOutput) -> bool {
    true
}

fn assembler_error(_: &assembler::AssemblyOutput, _: assembler::Error) -> bool {
    true
}

fn processor_done(output: &processor::ExecutionOutput) -> bool {
    output.lifecycle != 0
}

fn processor_error(_: &processor::ExecutionOutput, error: i32) -> bool {
    error != processor::ProcessorError::None as i32
}

sml! {
    GraphRoot<'d> {
        "reserving"_s <= *"uninitialized"_s + Reserve(ReserveRuntime<'d>) [valid_reserve] / begin_reserve,
        "uninitialized"_s <= "uninitialized"_s + Reserve(ReserveRuntime<'d>) [invalid_reserve] / reject_reserve,
        "reserved"_s <= "reserved"_s + Reserve(ReserveRuntime<'d>) [valid_reserve] / reject_reserve,
        "reserved"_s <= "reserved"_s + Reserve(ReserveRuntime<'d>) [invalid_reserve] / reject_reserve,
        "reserve_decision"_s <= "reserving"_s + completion<Reserve>(ReserveRuntime<'d>) / request_reserve,
        "reserve_tensors"_s <= "reserve_decision"_s + completion<Reserve>(ReserveRuntime<'d>) [reserve_done] / request_tensors,
        "uninitialized"_s <= "reserve_decision"_s + completion<Reserve>(ReserveRuntime<'d>) [reserve_failed] / dispatch_reserve_error,
        "reserved"_s <= "reserve_tensors"_s + completion<Reserve>(ReserveRuntime<'d>) [tensor_done] / dispatch_reserve_done,
        "uninitialized"_s <= "reserve_tensors"_s + completion<Reserve>(ReserveRuntime<'d>) [tensor_failed] / dispatch_reserve_error,
        "assembling"_s <= "reserved"_s + Compute(ComputeRuntime<'d>) [valid_compute] / begin_compute,
        "reserved"_s <= "reserved"_s + Compute(ComputeRuntime<'d>) [invalid_compute] / reject_compute,
        "uninitialized"_s <= "uninitialized"_s + Compute(ComputeRuntime<'d>) [invalid_compute] / reject_compute,
        "uninitialized"_s <= "uninitialized"_s + Compute(ComputeRuntime<'d>) [valid_compute] / reject_compute,
        "assemble_decision"_s <= "assembling"_s + completion<Compute>(ComputeRuntime<'d>) / request_assemble,
        "executing"_s <= "assemble_decision"_s + completion<Compute>(ComputeRuntime<'d>) [assemble_done] / request_execute,
        "compute_decision"_s <= "assemble_decision"_s + completion<Compute>(ComputeRuntime<'d>) [assemble_failed],
        "execute_decision"_s <= "executing"_s + completion<Compute>(ComputeRuntime<'d>) / request_execute,
        "compute_decision"_s <= "execute_decision"_s + completion<Compute>(ComputeRuntime<'d>) [execute_done],
        "compute_decision"_s <= "execute_decision"_s + completion<Compute>(ComputeRuntime<'d>) [execute_failed],
        "reserved"_s <= "compute_decision"_s + completion<Compute>(ComputeRuntime<'d>) [compute_success] / dispatch_compute_done,
        "reserved"_s <= "compute_decision"_s + completion<Compute>(ComputeRuntime<'d>) [compute_failure] / dispatch_compute_error,
        "executing"_s <= "reserved"_s + ComputeReserved(ReservedRuntime<'d>) [valid_reserved] / begin_reserved,
        "reserved"_s <= "reserved"_s + ComputeReserved(ReservedRuntime<'d>) [invalid_reserved] / reject_reserved,
        "uninitialized"_s <= "uninitialized"_s + ComputeReserved(ReservedRuntime<'d>) [invalid_reserved] / reject_reserved,
        "uninitialized"_s <= "uninitialized"_s + ComputeReserved(ReservedRuntime<'d>) [valid_reserved] / reject_reserved,
        "execute_decision"_s <= "executing"_s + completion<ReservedCompute>(ReservedRuntime<'d>) / request_reserved_execute,
        "compute_decision"_s <= "execute_decision"_s + completion<ReservedCompute>(ReservedRuntime<'d>) [reserved_execute_done],
        "compute_decision"_s <= "execute_decision"_s + completion<ReservedCompute>(ReservedRuntime<'d>) [reserved_execute_failed],
        "reserved"_s <= "compute_decision"_s + completion<ReservedCompute>(ReservedRuntime<'d>) [reserved_success] / dispatch_reserved_done,
        "reserved"_s <= "compute_decision"_s + completion<ReservedCompute>(ReservedRuntime<'d>) [reserved_failure] / dispatch_reserved_error,
        "uninitialized"_s <= "uninitialized"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "reserved"_s + Unexpected(UnexpectedEvent) / unexpected,
        "uninitialized"_s <= "reserving"_s + Unexpected(UnexpectedEvent) / unexpected,
        "uninitialized"_s <= "reserve_decision"_s + Unexpected(UnexpectedEvent) / unexpected,
        "uninitialized"_s <= "reserve_tensors"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "assembling"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "assemble_decision"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "executing"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "execute_decision"_s + Unexpected(UnexpectedEvent) / unexpected,
        "reserved"_s <= "compute_decision"_s + Unexpected(UnexpectedEvent) / unexpected,
    }
}

/// Synchronous graph root actor.
pub struct Graph {
    machine: GraphRootStateMachine<GraphContext>,
}
impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph").finish()
    }
}
impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
impl Graph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphRootStateMachine::new(GraphContext::default()),
        }
    }
    pub fn reserve(&mut self, request: ReserveRequest<'_>) -> Result<Outcome, RootError> {
        let result = RefCell::new(Err(RootError::InternalError));
        let reserve = Cell::new(PhaseOutcome::Unknown);
        let tensors = Cell::new(PhaseOutcome::Unknown);
        let output = Cell::new(GraphOutput::default());
        let error = Cell::new(RootError::None);
        self.machine
            .process_event(GraphRootEvents::Reserve(ReserveRuntime {
                request,
                reserve: &reserve,
                tensors: &tensors,
                output: &output,
                error: &error,
                result: &result,
            }))
            .map_err(|_| RootError::InternalError)?;
        result.into_inner()
    }
    #[allow(clippy::large_types_passed_by_value)]
    pub fn compute(&mut self, request: ComputeRequest<'_>) -> Result<Outcome, RootError> {
        let result = RefCell::new(Err(RootError::InternalError));
        let assemble = Cell::new(PhaseOutcome::Unknown);
        let execute = Cell::new(PhaseOutcome::Unknown);
        let output = Cell::new(GraphOutput::default());
        let error = Cell::new(RootError::None);
        self.machine
            .process_event(GraphRootEvents::Compute(ComputeRuntime {
                request,
                assemble: &assemble,
                execute: &execute,
                output: &output,
                error: &error,
                result: &result,
            }))
            .map_err(|_| RootError::InternalError)?;
        result.into_inner()
    }
    pub fn compute_reserved(
        &mut self,
        request: ComputeReservedRequest<'_>,
    ) -> Result<Outcome, RootError> {
        let result = RefCell::new(Err(RootError::InternalError));
        let execute = Cell::new(PhaseOutcome::Unknown);
        let output = Cell::new(GraphOutput::default());
        let error = Cell::new(RootError::None);
        self.machine
            .process_event(GraphRootEvents::ComputeReserved(ReservedRuntime {
                request,
                execute: &execute,
                output: &output,
                error: &error,
                result: &result,
            }))
            .map_err(|_| RootError::InternalError)?;
        result.into_inner()
    }
    pub fn process_event(&mut self, event: Event<'_>) -> Result<Outcome, RootError> {
        match event {
            Event::Reserve(r) => self.reserve(r),
            Event::Compute(r) => self.compute(r),
            Event::ComputeReserved(r) => self.compute_reserved(r),
        }
    }
    pub fn process_unexpected(&mut self, event: UnexpectedEvent) -> Result<Outcome, RootError> {
        self.machine
            .process_event(GraphRootEvents::Unexpected(event))
            .map_err(|_| RootError::InternalError)?;
        Err(RootError::InternalError)
    }
    /// Returns the generated root state for inspection without mutating the actor.
    #[must_use]
    pub fn state(&self) -> &GraphRootStates {
        self.machine.state()
    }
    /// Tests the generated root state without bypassing the state machine.
    #[must_use]
    pub fn is(&self, state: &GraphRootStates) -> bool {
        self.machine.is(state)
    }
    pub fn context(&self) -> &GraphContext {
        self.machine.context()
    }
    #[must_use]
    pub fn is_reserved(&self) -> bool {
        self.machine.is(&GraphRootStates::Reserved)
    }
    #[must_use]
    pub fn is_uninitialized(&self) -> bool {
        self.machine.is(&GraphRootStates::Uninitialized)
    }
}

impl GraphRootStateMachineContext for GraphContext {
    fn valid_reserve(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_reserve(&e.request))
    }
    fn invalid_reserve(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_reserve(&e.request))
    }
    fn begin_reserve(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        e.reserve.set(PhaseOutcome::Unknown);
        e.tensors.set(PhaseOutcome::Unknown);
        e.output.set(GraphOutput::default());
        e.error.set(RootError::None);
        *e.result.borrow_mut() = Err(RootError::InternalError);
        self.dispatch_generation = self.dispatch_generation.wrapping_add(1);
        Ok(())
    }
    fn request_reserve(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        let output = Cell::new(assembler::AssemblyOutput::default());
        let r = assembler::Reserve {
            model_topology: assembler::Topology(e.request.model_topology),
            lifecycle: assembler::LifecycleManifest(e.request.lifecycle),
            max_node_count: e.request.max_node_count,
            max_tensor_count: e.request.max_tensor_count,
            bytes_per_tensor: e.request.bytes_per_tensor,
            workspace_capacity_bytes: e.request.workspace_capacity_bytes,
            output_out: Some(&output),
            dispatch_done: Some(assembler_done),
            dispatch_error: Some(assembler_error),
        };
        match self.assembler.reserve(r) {
            Ok(assembler::Outcome::Reserved(o)) => {
                e.reserve.set(PhaseOutcome::Done);
                e.output.set(GraphOutput {
                    graph_topology: o.graph_topology.0,
                    node_count: o.node_count,
                    tensor_count: o.tensor_count,
                    required_buffer_bytes: o.required_buffer_bytes,
                    version: o.version,
                    lifecycle: o.lifecycle.0,
                    ..GraphOutput::default()
                });
            }
            Ok(_) => {
                e.reserve.set(PhaseOutcome::Failed);
                e.error.set(RootError::AssemblerFailed);
            }
            Err(x) => {
                e.reserve.set(PhaseOutcome::Failed);
                e.error.set(map_assembler_error(x));
            }
        }
        Ok(())
    }
    fn reject_reserve(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        let error = RootError::InvalidRequest;
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
    fn reserve_done(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(e.reserve.get() == PhaseOutcome::Done && e.error.get() == RootError::None)
    }
    fn reserve_failed(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.reserve_done(e)?)
    }
    fn request_tensors(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        let mut ok = true;
        if let Some(manifest) = e.request.tensor_manifest {
            for b in manifest {
                if self
                    .tensor
                    .process_event(tensor::Event::Reserve(tensor::ReserveTensor {
                        tensor_id: b.tensor_id,
                        buffer: b.buffer,
                        buffer_bytes: b.buffer_bytes,
                        consumer_refs: b.consumer_refs,
                        is_leaf: b.is_leaf,
                    }))
                    .is_err()
                {
                    ok = false;
                }
            }
        } else {
            ok = false;
        }
        e.tensors.set(if ok {
            PhaseOutcome::Done
        } else {
            e.error.set(RootError::InternalError);
            PhaseOutcome::Failed
        });
        Ok(())
    }
    fn tensor_done(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(e.tensors.get() == PhaseOutcome::Done && e.error.get() == RootError::None)
    }
    fn tensor_failed(&self, e: &ReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tensor_done(e)?)
    }
    fn dispatch_reserve_done(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        let output = e.output.get();
        if let Some(out) = e.request.output_out {
            out.set(output);
            if let Some(manifest) = e.request.tensor_manifest {
                self.reserved_tensor_ids.fill(false);
                for binding in manifest {
                    self.reserved_tensor_ids[binding.tensor_id] = true;
                }
            }
            self.reservation = Reservation {
                output,
                present: true,
            };
            if let Some(f) = e.request.dispatch_done {
                let _ = f(&out.get());
            }
        }
        *e.result.borrow_mut() = Ok(Outcome::Reserved(output));
        Ok(())
    }
    fn dispatch_reserve_error(&mut self, e: ReserveRuntime<'_>) -> Result<(), ()> {
        let error = e.error.get();
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
    fn valid_compute(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_compute(self, &e.request))
    }
    fn begin_compute(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        e.assemble.set(PhaseOutcome::Unknown);
        e.execute.set(PhaseOutcome::Unknown);
        e.output.set(GraphOutput::default());
        e.error.set(RootError::None);
        *e.result.borrow_mut() = Err(RootError::InternalError);
        self.dispatch_generation = self.dispatch_generation.wrapping_add(1);
        Ok(())
    }
    fn request_assemble(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        let output = Cell::new(assembler::AssemblyOutput::default());
        let mut request = e.request.assemble;
        request.output_out = Some(&output);
        request.dispatch_done = Some(assembler_done);
        request.dispatch_error = Some(assembler_error);
        match self.assembler.assemble(request) {
            Ok(assembler::Outcome::Assembled(o)) => {
                e.assemble.set(PhaseOutcome::Done);
                e.output.set(GraphOutput {
                    graph_topology: o.graph_topology.0,
                    node_count: o.node_count,
                    tensor_count: o.tensor_count,
                    required_buffer_bytes: o.required_buffer_bytes,
                    version: o.version,
                    reused_topology: o.reused_topology,
                    lifecycle: o.lifecycle.0,
                    ..GraphOutput::default()
                });
            }
            Ok(_) => {
                e.assemble.set(PhaseOutcome::Failed);
                e.error.set(RootError::AssemblerFailed);
            }
            Err(x) => {
                e.assemble.set(PhaseOutcome::Failed);
                e.error.set(map_assembler_error(x));
            }
        }
        Ok(())
    }
    fn assemble_done(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.assemble.get() == PhaseOutcome::Done && e.error.get() == RootError::None)
    }
    fn assemble_failed(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.assemble_done(e)?)
    }
    fn request_execute(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        let mut r = e.request.execute;
        r.dispatch_done = Some(processor_done);
        r.dispatch_error = Some(processor_error);
        let ok = self.processor.process_event_with_driver(
            processor::EventExecuteStep::new(r),
            &mut self.tensor,
            e.request.lifecycle_manifest,
        );
        e.execute.set(if ok {
            PhaseOutcome::Done
        } else {
            PhaseOutcome::Failed
        });
        let child_error = map_processor_error(self.processor.error());
        if child_error != RootError::None {
            e.error.set(child_error);
        } else if !ok {
            e.error.set(RootError::ProcessorFailed);
        }
        let child = self.processor.output();
        let mut out = e.output.get();
        out.outputs_produced = child.outputs_produced;
        out.graph_reused = child.graph_reused != 0;
        out.lifecycle = child.lifecycle;
        e.output.set(out);
        Ok(())
    }
    fn execute_done(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.execute.get() == PhaseOutcome::Done && e.error.get() == RootError::None)
    }
    fn invalid_compute(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_compute(self, &e.request))
    }
    fn valid_reserved(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_reserved(self, &e.request))
    }
    fn invalid_reserved(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_reserved(self, &e.request))
    }
    fn begin_reserved(&mut self, e: ReservedRuntime<'_>) -> Result<(), ()> {
        e.execute.set(PhaseOutcome::Unknown);
        let mut output = self.reservation.output;
        output.reused_topology = true;
        e.output.set(output);
        e.error.set(RootError::None);
        *e.result.borrow_mut() = Err(RootError::InternalError);
        self.dispatch_generation = self.dispatch_generation.wrapping_add(1);
        Ok(())
    }
    fn unexpected(&mut self, _: UnexpectedEvent) -> Result<(), ()> {
        self.last_error = RootError::InternalError;
        Ok(())
    }
    fn execute_failed(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.execute_done(e)?)
    }
    fn compute_success(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.error.get() == RootError::None)
    }
    fn compute_failure(&self, e: &ComputeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.error.get() != RootError::None)
    }
    fn dispatch_compute_done(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        let output = e.output.get();
        if let Some(out) = e.request.output_out {
            out.set(output);
            if let Some(f) = e.request.dispatch_done {
                let _ = f(&out.get());
            }
        }
        *e.result.borrow_mut() = Ok(Outcome::Computed(output));
        Ok(())
    }
    fn dispatch_compute_error(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        let error = e.error.get();
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
    fn reject_compute(&mut self, e: ComputeRuntime<'_>) -> Result<(), ()> {
        let error = RootError::InvalidRequest;
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
    fn request_reserved_execute(&mut self, e: ReservedRuntime<'_>) -> Result<(), ()> {
        let mut r = e.request.execute;
        r.dispatch_done = Some(processor_done);
        r.dispatch_error = Some(processor_error);
        let ok = self.processor.process_event_with_driver(
            processor::EventExecuteStep::new(r),
            &mut self.tensor,
            e.request.lifecycle_manifest,
        );
        e.execute.set(if ok {
            PhaseOutcome::Done
        } else {
            PhaseOutcome::Failed
        });
        let child_error = map_processor_error(self.processor.error());
        if child_error != RootError::None {
            e.error.set(child_error);
        } else if !ok {
            e.error.set(RootError::ProcessorFailed);
        }
        let child = self.processor.output();
        let mut out = e.output.get();
        out.outputs_produced = child.outputs_produced;
        out.graph_reused = child.graph_reused != 0;
        out.lifecycle = child.lifecycle;
        e.output.set(out);
        Ok(())
    }
    fn reserved_execute_done(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(e.execute.get() == PhaseOutcome::Done && e.error.get() == RootError::None)
    }
    fn reserved_execute_failed(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.reserved_execute_done(e)?)
    }
    fn reserved_success(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(e.error.get() == RootError::None)
    }
    fn reserved_failure(&self, e: &ReservedRuntime<'_>) -> Result<bool, ()> {
        Ok(e.error.get() != RootError::None)
    }
    fn dispatch_reserved_done(&mut self, e: ReservedRuntime<'_>) -> Result<(), ()> {
        let output = e.output.get();
        if let Some(out) = e.request.output_out {
            out.set(output);
            if let Some(f) = e.request.dispatch_done {
                let _ = f(&out.get());
            }
        }
        *e.result.borrow_mut() = Ok(Outcome::Computed(output));
        Ok(())
    }
    fn dispatch_reserved_error(&mut self, e: ReservedRuntime<'_>) -> Result<(), ()> {
        let error = e.error.get();
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
    fn reject_reserved(&mut self, e: ReservedRuntime<'_>) -> Result<(), ()> {
        let error = RootError::InvalidRequest;
        if let Some(out) = e.request.output_out {
            out.set(GraphOutput::default());
            if let Some(f) = e.request.dispatch_error {
                let _ = f(&out.get(), error);
            }
        }
        *e.result.borrow_mut() = Err(error);
        self.last_error = error;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn dispatch_done(_: &GraphOutput) -> bool {
        true
    }

    fn dispatch_error(_: &GraphOutput, _: RootError) -> bool {
        true
    }

    static DUPLICATE_RESERVE_ERROR_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn duplicate_reserve_error(output: &GraphOutput, error: RootError) -> bool {
        assert_eq!(*output, GraphOutput::default());
        assert_eq!(error, RootError::InvalidRequest);
        DUPLICATE_RESERVE_ERROR_CALLS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn reserve_request<'a>(
        output_out: &'a Cell<GraphOutput>,
        manifest: &'a [TensorBinding],
    ) -> ReserveRequest<'a> {
        ReserveRequest {
            model_topology: 1,
            output_out: Some(output_out),
            lifecycle: 1,
            tensor_manifest: Some(manifest),
            max_node_count: 1,
            max_tensor_count: 1,
            bytes_per_tensor: 1,
            workspace_capacity_bytes: 1,
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }

    #[test]
    fn duplicate_valid_reserve_is_rejected_without_mutating_reservation() {
        DUPLICATE_RESERVE_ERROR_CALLS.store(0, Ordering::SeqCst);
        let mut graph = Graph::new();
        let prior_output = Cell::new(GraphOutput::default());
        let manifest = [TensorBinding {
            tensor_id: 0,
            buffer: tensor::BufferHandle(1),
            buffer_bytes: 1,
            consumer_refs: 0,
            is_leaf: true,
        }];

        assert!(
            graph
                .reserve(reserve_request(&prior_output, &manifest))
                .is_ok()
        );
        assert!(graph.is_reserved());
        let saved_output = prior_output.get();
        let saved_reservation = graph.context().reservation;

        let attempted_output = Cell::new(GraphOutput {
            graph_topology: 99,
            node_count: 99,
            ..GraphOutput::default()
        });
        let mut duplicate_request = reserve_request(&attempted_output, &manifest);
        duplicate_request.dispatch_error = Some(duplicate_reserve_error);

        assert_eq!(
            graph.reserve(duplicate_request),
            Err(RootError::InvalidRequest)
        );
        assert!(graph.is_reserved());
        assert_eq!(prior_output.get(), saved_output);
        assert_eq!(graph.context().reservation.output, saved_reservation.output);
        assert_eq!(
            graph.context().reservation.present,
            saved_reservation.present
        );
        assert_eq!(attempted_output.get(), GraphOutput::default());
        assert_eq!(DUPLICATE_RESERVE_ERROR_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unexpected_dispatches_generated_root_path_and_recovers_uninitialized() {
        let mut graph = Graph::new();

        assert_eq!(
            graph.process_unexpected(UnexpectedEvent),
            Err(RootError::InternalError)
        );
        assert!(graph.is_uninitialized());
        assert_eq!(graph.context().last_error, RootError::InternalError);

        let output = Cell::new(GraphOutput::default());
        let manifest = [TensorBinding {
            tensor_id: 0,
            buffer: tensor::BufferHandle(1),
            buffer_bytes: 1,
            consumer_refs: 0,
            is_leaf: true,
        }];
        assert!(graph.reserve(reserve_request(&output, &manifest)).is_ok());
        assert!(graph.is_reserved());
    }

    #[test]
    fn unexpected_dispatches_generated_root_path_and_recovers_reserved() {
        let mut graph = Graph::new();
        let output = Cell::new(GraphOutput::default());
        let manifest = [TensorBinding {
            tensor_id: 0,
            buffer: tensor::BufferHandle(1),
            buffer_bytes: 1,
            consumer_refs: 0,
            is_leaf: true,
        }];
        assert!(graph.reserve(reserve_request(&output, &manifest)).is_ok());
        assert!(graph.is_reserved());

        assert_eq!(
            graph.process_unexpected(UnexpectedEvent),
            Err(RootError::InternalError)
        );
        assert!(graph.is_reserved());
        assert_eq!(graph.context().last_error, RootError::InternalError);
        assert_eq!(
            graph.process_unexpected(UnexpectedEvent),
            Err(RootError::InternalError)
        );
        assert!(graph.is_reserved());
        assert_eq!(graph.context().last_error, RootError::InternalError);
    }

    #[test]
    fn begin_reserved_compute_marks_reserved_topology_reused() {
        let mut graph = Graph::new();
        let reserve_output = Cell::new(GraphOutput::default());
        let manifest = [TensorBinding {
            tensor_id: 0,
            buffer: tensor::BufferHandle(1),
            buffer_bytes: 1,
            consumer_refs: 0,
            is_leaf: true,
        }];
        assert!(
            graph
                .reserve(reserve_request(&reserve_output, &manifest))
                .is_ok()
        );

        let output = Cell::new(GraphOutput::default());
        let execute = Cell::new(PhaseOutcome::Unknown);

        let error = Cell::new(RootError::None);
        let result = RefCell::new(Err(RootError::InternalError));
        let request = ComputeReservedRequest {
            execute: processor::ExecuteRequest::default(),
            lifecycle_manifest: processor::LifecycleManifest::default(),
            output_out: Some(&output),
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        };
        <GraphContext as GraphRootStateMachineContext>::begin_reserved(
            graph.machine.context_mut(),
            ReservedRuntime {
                request,
                execute: &execute,
                output: &output,
                error: &error,
                result: &result,
            },
        )
        .expect("begin_reserved action succeeds");

        assert!(output.get().reused_topology);
        assert_eq!(
            output.get().graph_topology,
            reserve_output.get().graph_topology
        );
        assert_eq!(output.get().node_count, reserve_output.get().node_count);
        assert_eq!(output.get().tensor_count, reserve_output.get().tensor_count);
        assert_eq!(
            output.get().required_buffer_bytes,
            reserve_output.get().required_buffer_bytes
        );
        assert_eq!(output.get().version, reserve_output.get().version);
        assert_eq!(output.get().lifecycle, reserve_output.get().lifecycle);
    }

    #[test]
    fn manifest_guards_reject_duplicate_and_out_of_range_phase_ids() {
        let mut reserved = vec![false; tensor::MAX_TENSORS].into_boxed_slice();
        reserved[0] = true;
        reserved[1] = true;
        reserved[2] = true;
        assert!(!valid_unique_ids(&[0, 0], &reserved));
        assert!(!valid_unique_ids(&[tensor::MAX_TENSORS], &reserved));
        assert!(valid_unique_ids(&[0, 1, 2], &reserved));
    }

    #[test]
    fn reserve_guard_rejects_duplicate_or_oversized_manifest() {
        let output = Cell::new(GraphOutput::default());
        let duplicate = [
            TensorBinding {
                tensor_id: 0,
                buffer: tensor::BufferHandle(1),
                buffer_bytes: 1,
                consumer_refs: 0,
                is_leaf: true,
            },
            TensorBinding {
                tensor_id: 0,
                buffer: tensor::BufferHandle(2),
                buffer_bytes: 1,
                consumer_refs: 0,
                is_leaf: true,
            },
        ];
        let mut request = reserve_request(&output, &duplicate);
        assert!(!valid_reserve(&request));
        request.max_tensor_count = 1;
        assert!(!valid_reserve(&request));
    }

    #[test]
    fn compute_guard_rejects_lifecycle_mismatch() {
        let output = Cell::new(GraphOutput::default());
        let manifest = [];
        let mut request = ComputeRequest {
            assemble: assembler::Assemble {
                step_plan: assembler::Topology(1),
                lifecycle: assembler::LifecycleManifest(2),
                node_count_hint: 1,
                tensor_count_hint: 1,
                bytes_per_tensor: 1,
                workspace_capacity_bytes: 1,
                output_out: None,
                dispatch_done: None,
                dispatch_error: None,
            },
            execute: processor::ExecuteRequest::default(),
            lifecycle_manifest: processor::LifecycleManifest {
                phase: processor::LifecyclePhase {
                    required_filled_ids: &manifest,
                    publish_ids: &manifest,
                    release_ids: &manifest,
                },
            },
            output_out: Some(&output),
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        };
        request.execute.lifecycle = 3;
        assert!(!valid_compute(&GraphContext::default(), &request));
    }
    fn validate(_: &processor::ExecuteRequest, _: &mut i32) -> bool {
        true
    }

    fn prepare(_: &processor::ExecuteRequest, _: &mut bool, _: &mut i32) -> bool {
        true
    }

    fn phase(_: &processor::ExecuteRequest, _: &mut i32) -> bool {
        true
    }

    fn lifecycle(_: &processor::ExecuteRequest) -> bool {
        true
    }

    #[test]
    fn valid_compute_before_reservation_is_rejected_typed_and_stays_uninitialized() {
        let mut graph = Graph::new();
        let output = Cell::new(GraphOutput {
            graph_topology: 9,
            ..GraphOutput::default()
        });
        let execute = processor::ExecuteRequest {
            step_plan: 1,
            output_out: 1,
            lifecycle: 1,
            tensor_machine: 1,
            step_size: 1,
            seq_mask_words: 1,
            validate: Some(validate),
            prepare_graph: Some(prepare),
            alloc_graph: Some(phase),
            bind_inputs: Some(phase),
            run_kernel: Some(phase),
            extract_outputs: Some(|_, _, _| true),
            lifecycle_gate: Some(lifecycle),
            lifecycle_publish: Some(lifecycle),
            lifecycle_release: Some(lifecycle),
            ..processor::ExecuteRequest::default()
        };
        let request = ComputeRequest {
            assemble: assembler::Assemble {
                step_plan: assembler::Topology(1),
                lifecycle: assembler::LifecycleManifest(1),
                node_count_hint: 1,
                tensor_count_hint: 1,
                bytes_per_tensor: 1,
                workspace_capacity_bytes: 1,
                output_out: None,
                dispatch_done: None,
                dispatch_error: None,
            },
            execute,
            lifecycle_manifest: processor::LifecycleManifest::default(),
            output_out: Some(&output),
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        };
        assert_eq!(graph.compute(request), Err(RootError::InvalidRequest));
        assert!(graph.is_uninitialized());
        assert_eq!(output.get(), GraphOutput::default());
    }
}
