//! Execution-graph allocation, assembly, processing, and tensor binding.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// Identifies a node in an execution graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u32);

// Keep generated state-machine modules private to this crate. The exports
// below are the bounded actor boundary: callers receive typed requests,
// results, errors, and actor handles, but not generated machines, contexts,
// child actions, or guards.
pub(crate) mod allocator;
pub(crate) mod assembler;
pub(crate) mod processor;
pub(crate) mod sm;
pub(crate) mod tensor;

/// Root graph actor and its safe request/result contract.
pub use sm::{
    ComputeRequest, ComputeReservedRequest, DispatchDoneFn, DispatchErrorFn, Event, Graph,
    GraphOutput, Outcome, ReserveRequest, RootError, TensorBinding, UnexpectedEvent,
};

/// Processor actor and the request/lifecycle values consumed by graph callers.
pub use processor::sm::{
    AllocGraphFn, BindInputsFn, EventExecuteStep, ExecuteRequest, ExecutionOutput,
    ExtractOutputsFn, GraphProcessor, LifecycleDriver, LifecycleManifest, LifecycleOutcome,
    LifecyclePhase, PrepareGraphFn, ProcessorError, RunKernelFn, ValidateFn,
};

/// Tensor lifecycle values used to prepare graph reservations and manifests.
pub use tensor::sm::{
    BufferHandle, CaptureTensorState, Error as TensorError, Event as TensorEvent, GraphTensor,
    Lifecycle as TensorLifecycle, MAX_TENSORS, Outcome as TensorOutcome, PublishFilledTensor,
    ReleaseTensorRef, ReserveTensor, ResetTensorEpoch, Tensor as TensorActor, TensorState,
    UnexpectedEvent as TensorUnexpectedEvent,
};

/// Safe assembler actor and stable request/result values.
pub use assembler::sm::{
    AllocationPlan as AssemblerAllocationPlan, Assemble, Assembler, AssemblyOutput,
    Error as AssemblerError, Event as AssemblerEvent,
    LifecycleManifest as AssemblerLifecycleManifest, Outcome as AssemblerOutcome, Reserve,
    Topology,
};

/// Stable allocator result values and actor request contract.
pub use allocator::sm::{
    AllocateGraph, AllocationDone, AllocationError, AllocationErrorEvent, AllocationPlan,
    Allocator, DoneCallback as AllocationDoneCallback, ErrorCallback as AllocationErrorCallback,
    EventAllocateGraphPlan, PhaseOutcome as AllocationPhaseOutcome,
};

/// Request-event namespace for the maintained root graph actor.
pub mod event {
    pub use super::{ComputeRequest, ComputeReservedRequest, Event, ReserveRequest, TensorBinding};
}

/// Completion/result namespace for the maintained root graph actor.
pub mod events {
    pub use super::{GraphOutput, Outcome, RootError};
}
