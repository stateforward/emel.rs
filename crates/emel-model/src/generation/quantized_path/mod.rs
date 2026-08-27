//! Contract-only generation quantized-path classification.
//!
//! Pinned C++ classifies these outcomes in
//! `emel/model/generation/any.hpp` via `build_quantized_path_audit`.
//! Outcomes describe the model-generation contract. They do not claim that a
//! particular operator, provider, architecture, or backend is available.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use emel_tensor::dtype::SerializedType;

use sm::{
    QuantizedPathMachineEvents, QuantizedPathMachineStateMachine,
    QuantizedPathMachineStateMachineContext, QuantizedPathMachineStates,
};

pub mod event;
mod sm;

/// The generation contract being queried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope {
    /// Token-embedding / vector dequant contract.
    VectorDequantContract,
    /// Output / matrix weight contract.
    MatrixWeightContract,
}

/// Contract-only classification for a serialized tensor type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// Native quantized execution is required by the generation contract.
    NativeQuantized,
    /// Dense F32 is the approved contract path.
    ApprovedDenseF32ByContract,
    /// The requested contract path is disallowed.
    DisallowedFallback,
    /// The type is outside the claimed generation-path set.
    ExplicitNoClaim,
}

/// Resolver dispatch failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to publish an exhaustive outcome.
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnexpectedEvent => "unexpected quantized-path resolver event",
            Self::Internal => "internal quantized-path resolver error",
        })
    }
}

impl std::error::Error for Error {}

struct Runtime<'a> {
    query: event::Query,
    result: &'a Cell<Result<Outcome, Error>>,
}

#[derive(Default)]
struct Context;

/// Single-writer, run-to-completion quantized-path resolver.
pub struct Resolver {
    machine: QuantizedPathMachineStateMachine<Context>,
}

impl fmt::Debug for Resolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Resolver").finish_non_exhaustive()
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver {
    /// Constructs a resolver with its own empty local context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: QuantizedPathMachineStateMachine::new(Context),
        }
    }

    /// Resolves one query within the synchronous dispatch boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] if the generated machine rejects a valid
    /// query or does not publish one of the exhaustive contract outcomes.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event(&mut self, query: event::Query) -> Result<Outcome, Error> {
        let result = Cell::new(Err(Error::UnexpectedEvent));
        self.machine
            .process_event(QuantizedPathMachineEvents::Query(Runtime {
                query,
                result: &result,
            }))
            .map_err(|_| Error::Internal)?;
        assert!(
            self.machine.is(&QuantizedPathMachineStates::Ready),
            "quantized-path machine must return to ready after dispatch"
        );
        result.get()
    }
}

const fn is_native(tensor_type: SerializedType) -> bool {
    matches!(
        tensor_type,
        SerializedType::Q4_0
            | SerializedType::Q4_1
            | SerializedType::Q5_0
            | SerializedType::Q8_0
            | SerializedType::Q2K
            | SerializedType::Q3K
            | SerializedType::Q4K
            | SerializedType::Q6K
    )
}

impl QuantizedPathMachineStateMachineContext for Context {
    fn guard_vector_f32(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::VectorDequantContract
            && event.query.tensor_type() == SerializedType::F32)
    }

    fn guard_vector_native(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::VectorDequantContract
            && is_native(event.query.tensor_type()))
    }

    fn guard_vector_explicit_no_claim(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::VectorDequantContract
            && event.query.tensor_type() != SerializedType::F32
            && !is_native(event.query.tensor_type()))
    }

    fn guard_matrix_native(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::MatrixWeightContract
            && is_native(event.query.tensor_type()))
    }

    fn guard_matrix_f32(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::MatrixWeightContract
            && event.query.tensor_type() == SerializedType::F32)
    }

    fn guard_matrix_explicit_no_claim(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.query.scope() == Scope::MatrixWeightContract
            && event.query.tensor_type() != SerializedType::F32
            && !is_native(event.query.tensor_type()))
    }

    fn effect_publish_native(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Ok(Outcome::NativeQuantized));
        Ok(())
    }

    fn effect_publish_dense_f32(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Ok(Outcome::ApprovedDenseF32ByContract));
        Ok(())
    }

    fn effect_publish_disallowed(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Ok(Outcome::DisallowedFallback));
        Ok(())
    }

    fn effect_publish_explicit_no_claim(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Ok(Outcome::ExplicitNoClaim));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Outcome, Resolver, Scope, event::Query};
    use emel_tensor::dtype::SerializedType;

    #[test]
    fn matrix_q4k_is_native_and_vector_native_is_dense_f32() {
        let mut resolver = Resolver::new();
        assert_eq!(
            resolver.process_event(Query::new(Scope::MatrixWeightContract, SerializedType::Q4K)),
            Ok(Outcome::NativeQuantized)
        );
        assert_eq!(
            resolver.process_event(Query::new(
                Scope::VectorDequantContract,
                SerializedType::Q4K
            )),
            Ok(Outcome::ApprovedDenseF32ByContract)
        );
        assert_eq!(
            Error::UnexpectedEvent.to_string(),
            "unexpected quantized-path resolver event"
        );
        let _ = format!("{:?}", Resolver::default());
    }
}
