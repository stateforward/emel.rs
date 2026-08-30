//! Safe, bounded graph-tensor lifecycle actor.
//!
//! This module keeps tensor lifecycle decisions in a generated
//! `stateforward-sml` transition table. The actor owns a fixed-capacity record
//! table; dispatch is synchronous and does not allocate.

#![allow(clippy::module_name_repetitions)]
#![allow(private_interfaces)]
#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Maximum number of tensor records retained by an actor.
pub const MAX_TENSORS: usize = 65_536;

/// Lifecycle of one graph tensor record.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum Lifecycle {
    /// No reservation exists for this id.
    #[default]
    Unallocated = 0,
    /// Reserved compute storage that is not currently filled.
    Empty = 1,
    /// Filled compute storage with live consumer references.
    Filled = 2,
    /// Filled leaf storage.
    LeafFilled = 3,
    /// A lifecycle operation failed after dispatch.
    InternalError = 4,
}

/// Opaque caller-owned storage identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferHandle(pub usize);

/// Observable snapshot of a tensor record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorState {
    /// Current lifecycle.
    pub lifecycle: Lifecycle,
    /// Whether this record represents a leaf tensor.
    pub is_leaf: bool,
    /// Consumer references assigned at reservation.
    pub seed_refs: u32,
    /// Consumer references still held by downstream operations.
    pub live_refs: u32,
    /// Caller-owned storage identity, when reserved.
    pub buffer: Option<BufferHandle>,
    /// Number of bytes supplied at reservation.
    pub buffer_bytes: u64,
}

/// Typed errors from tensor dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The request is malformed or outside the bounded actor contract.
    InvalidRequest,
    /// A request was well-formed but invalid for its current lifecycle.
    Internal,
    /// An event outside the actor's public event set was received.
    UnexpectedEvent,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => formatter.write_str("invalid tensor request"),
            Self::Internal => formatter.write_str("internal tensor lifecycle error"),
            Self::UnexpectedEvent => formatter.write_str("unexpected tensor event"),
        }
    }
}

impl std::error::Error for Error {}

/// Reservation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReserveTensor {
    /// Bounded tensor record index.
    pub tensor_id: usize,
    /// Caller-owned storage identity.
    pub buffer: BufferHandle,
    /// Number of bytes available to the tensor.
    pub buffer_bytes: u64,
    /// Number of downstream compute consumers.
    pub consumer_refs: u32,
    /// Whether this tensor is a leaf.
    pub is_leaf: bool,
}

/// Publish a reserved compute tensor as filled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublishFilledTensor {
    /// Bounded tensor record index.
    pub tensor_id: usize,
}

/// Release one compute consumer reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReleaseTensorRef {
    /// Bounded tensor record index.
    pub tensor_id: usize,
}

/// Reset one compute tensor for a new graph epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResetTensorEpoch {
    /// Bounded tensor record index.
    pub tensor_id: usize,
}

/// Capture one tensor's generated state inspection data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureTensorState {
    /// Bounded tensor record index.
    pub tensor_id: usize,
}

/// Typed public event set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    /// Reserve storage.
    Reserve(ReserveTensor),
    /// Publish a compute tensor as filled.
    PublishFilled(PublishFilledTensor),
    /// Release one consumer reference.
    Release(ReleaseTensorRef),
    /// Reset a tensor epoch.
    Reset(ResetTensorEpoch),
    /// Capture current state.
    Capture(CaptureTensorState),
}

/// Explicit event used to exercise unexpected-event handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

/// Successful operation result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// A mutating operation completed.
    Done,
    /// A state capture completed.
    State(TensorState),
}

struct Runtime<'a> {
    event: Event,
    result: &'a Cell<Result<Outcome, Error>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TensorRecord {
    lifecycle: Lifecycle,
    is_leaf: bool,
    seed_refs: u32,
    live_refs: u32,
    buffer: Option<BufferHandle>,
    buffer_bytes: u64,
}

impl TensorRecord {
    const fn snapshot(self) -> TensorState {
        TensorState {
            lifecycle: self.lifecycle,
            is_leaf: self.is_leaf,
            seed_refs: self.seed_refs,
            live_refs: self.live_refs,
            buffer: self.buffer,
            buffer_bytes: self.buffer_bytes,
        }
    }
}

/// Actor-owned bounded tensor storage.
pub struct GraphTensorContext {
    records: Box<[TensorRecord]>,
}

impl fmt::Debug for GraphTensorContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Context")
            .field("record_count", &self.records.len())
            .finish()
    }
}

impl Default for GraphTensorContext {
    fn default() -> Self {
        Self {
            records: vec![TensorRecord::default(); MAX_TENSORS].into_boxed_slice(),
        }
    }
}

const fn request_valid(_context: &GraphTensorContext, event: Event) -> bool {
    match event {
        Event::Reserve(request) => {
            request.tensor_id < MAX_TENSORS && request.buffer.0 != 0 && request.buffer_bytes != 0
        }
        Event::PublishFilled(request) => request.tensor_id < MAX_TENSORS,
        Event::Release(request) => request.tensor_id < MAX_TENSORS,
        Event::Reset(request) => request.tensor_id < MAX_TENSORS,
        Event::Capture(request) => request.tensor_id < MAX_TENSORS,
    }
}

sml! {
    GraphTensor<'dispatch> {
        "ready"_s <= *"ready"_s + Runtime(Runtime<'dispatch>)
            [request_valid] / execute_operation,
        "ready"_s <= "ready"_s + Runtime(Runtime<'dispatch>)
            [request_invalid] / reject_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedEvent)
            / reject_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / reject_generic_unexpected,
    }
}

/// Single-writer, run-to-completion graph tensor actor.
pub struct GraphTensor {
    machine: GraphTensorStateMachine<GraphTensorContext>,
}

impl fmt::Debug for GraphTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GraphTensor")
            .field("ready", &self.is_ready())
            .finish_non_exhaustive()
    }
}

impl Default for GraphTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphTensor {
    /// Constructs an actor with fixed-capacity tensor storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphTensorStateMachine::new(GraphTensorContext::default()),
        }
    }

    /// Dispatches an operation synchronously through the generated machine.
    ///
    /// # Errors
    ///
    /// Returns a typed request or lifecycle error.
    pub fn process_event(&mut self, event: Event) -> Result<Outcome, Error> {
        let result = Cell::new(Err(Error::Internal));
        self.machine
            .process_event(GraphTensorEvents::Runtime(Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        assert!(self.is_ready(), "graph tensor actor must return to ready");
        result.into_inner()
    }

    /// Dispatches an explicit unexpected event.
    pub fn process_unexpected(&mut self, _event: UnexpectedEvent) -> Result<Outcome, Error> {
        self.machine
            .process_event(GraphTensorEvents::Unexpected(UnexpectedEvent))
            .map_err(|_| Error::Internal)?;
        assert!(self.is_ready(), "graph tensor actor must return to ready");
        Err(Error::UnexpectedEvent)
    }

    /// Reports the generated machine's observable ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&GraphTensorStates::Ready)
    }
}

impl GraphTensorStateMachineContext for GraphTensorContext {
    fn request_valid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(request_valid(self, event.event))
    }

    fn request_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(self, event.event))
    }

    fn execute_operation(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        match event.event {
            Event::Reserve(request) => {
                let record = self.records.get_mut(request.tensor_id).ok_or(())?;
                if record.lifecycle != Lifecycle::Unallocated {
                    record.lifecycle = Lifecycle::InternalError;
                    event.result.set(Err(Error::Internal));
                    return Ok(());
                }
                record.lifecycle = if request.is_leaf {
                    Lifecycle::LeafFilled
                } else {
                    Lifecycle::Empty
                };
                record.is_leaf = request.is_leaf;
                record.seed_refs = if request.is_leaf {
                    0
                } else {
                    request.consumer_refs
                };
                record.live_refs = record.seed_refs;
                record.buffer = Some(request.buffer);
                record.buffer_bytes = request.buffer_bytes;
                event.result.set(Ok(Outcome::Done));
            }
            Event::PublishFilled(request) => {
                let record = self.records.get_mut(request.tensor_id).ok_or(())?;
                if record.lifecycle == Lifecycle::Empty && !record.is_leaf {
                    record.lifecycle = Lifecycle::Filled;
                    event.result.set(Ok(Outcome::Done));
                } else {
                    record.lifecycle = Lifecycle::InternalError;
                    event.result.set(Err(Error::Internal));
                }
            }
            Event::Release(request) => {
                let record = self.records.get_mut(request.tensor_id).ok_or(())?;
                match record.lifecycle {
                    Lifecycle::Filled if record.live_refs > 1 => {
                        record.live_refs -= 1;
                        event.result.set(Ok(Outcome::Done));
                    }
                    Lifecycle::Filled if record.live_refs == 1 => {
                        record.live_refs = 0;
                        record.lifecycle = Lifecycle::Empty;
                        event.result.set(Ok(Outcome::Done));
                    }
                    Lifecycle::LeafFilled => event.result.set(Ok(Outcome::Done)),
                    _ => {
                        record.lifecycle = Lifecycle::InternalError;
                        record.live_refs = 0;
                        event.result.set(Err(Error::Internal));
                    }
                }
            }
            Event::Reset(request) => {
                let record = self.records.get_mut(request.tensor_id).ok_or(())?;
                if record.lifecycle == Lifecycle::LeafFilled {
                    event.result.set(Ok(Outcome::Done));
                } else if !record.is_leaf
                    && matches!(record.lifecycle, Lifecycle::Empty | Lifecycle::Filled)
                {
                    record.lifecycle = Lifecycle::Empty;
                    record.live_refs = 0;
                    event.result.set(Ok(Outcome::Done));
                } else {
                    record.lifecycle = Lifecycle::InternalError;
                    record.live_refs = 0;
                    event.result.set(Err(Error::Internal));
                }
            }
            Event::Capture(request) => {
                let record = self.records.get(request.tensor_id).copied().ok_or(())?;
                event.result.set(Ok(Outcome::State(record.snapshot())));
            }
        }
        Ok(())
    }

    fn reject_invalid(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn reject_unexpected(&mut self, _event: UnexpectedEvent) -> Result<(), ()> {
        Ok(())
    }

    fn reject_generic_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// Compatibility alias matching the reference actor's public name.
pub type Tensor = GraphTensor;

#[cfg(test)]
mod tests {
    use super::*;

    const BUFFER: BufferHandle = BufferHandle(1);

    #[test]
    fn reserve_publish_release_and_capture_are_explicit() {
        let mut actor = GraphTensor::new();
        assert!(actor.is_ready());
        assert_eq!(
            actor.process_event(Event::Reserve(ReserveTensor {
                tensor_id: 3,
                buffer: BUFFER,
                buffer_bytes: 64,
                consumer_refs: 2,
                is_leaf: false,
            })),
            Ok(Outcome::Done)
        );
        assert_eq!(
            actor.process_event(Event::PublishFilled(PublishFilledTensor { tensor_id: 3 })),
            Ok(Outcome::Done)
        );
        assert_eq!(
            actor.process_event(Event::Release(ReleaseTensorRef { tensor_id: 3 })),
            Ok(Outcome::Done)
        );
        assert_eq!(
            actor.process_event(Event::Capture(CaptureTensorState { tensor_id: 3 })),
            Ok(Outcome::State(TensorState {
                lifecycle: Lifecycle::Filled,
                is_leaf: false,
                seed_refs: 2,
                live_refs: 1,
                buffer: Some(BUFFER),
                buffer_bytes: 64,
            }))
        );
    }

    #[test]
    fn invalid_request_is_rejected_without_mutation() {
        let mut actor = GraphTensor::new();
        assert_eq!(
            actor.process_event(Event::Reserve(ReserveTensor {
                tensor_id: MAX_TENSORS,
                buffer: BUFFER,
                buffer_bytes: 1,
                consumer_refs: 0,
                is_leaf: true,
            })),
            Err(Error::InvalidRequest)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn rejected_lifecycle_operation_is_reported_and_inspectable() {
        let mut actor = GraphTensor::new();
        let request = ReserveTensor {
            tensor_id: 0,
            buffer: BUFFER,
            buffer_bytes: 1,
            consumer_refs: 0,
            is_leaf: true,
        };
        assert_eq!(
            actor.process_event(Event::Reserve(request)),
            Ok(Outcome::Done)
        );
        assert_eq!(
            actor.process_event(Event::Reserve(request)),
            Err(Error::Internal)
        );
        assert_eq!(
            actor.process_event(Event::Capture(CaptureTensorState { tensor_id: 0 })),
            Ok(Outcome::State(TensorState {
                lifecycle: Lifecycle::InternalError,
                is_leaf: true,
                seed_refs: 0,
                live_refs: 0,
                buffer: Some(BUFFER),
                buffer_bytes: 1,
            }))
        );
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_returns_ready() {
        let mut actor = GraphTensor::new();
        assert_eq!(
            actor.process_unexpected(UnexpectedEvent),
            Err(Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }
}
