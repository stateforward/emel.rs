//! Contract-only generation capability classification.
//!
//! Outcomes describe the model-generation contract. They do not claim that a
//! particular operator, provider, architecture, or backend is available.

use core::fmt;

use emel_tensor::dtype::SerializedType;

mod sm;

use sm::{CapabilityResolverStateMachine, CapabilityResolverStateMachineContext};

/// The generation contract being queried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope {
    VectorDequantContract,
    MatrixWeightContract,
}

/// Contract-only classification for a serialized tensor type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    NativeQuantized,
    ApprovedDenseF32ByContract,
    DisallowedFallback,
    ExplicitNoClaim,
}

/// Resolver dispatch failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    UnexpectedEvent,
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnexpectedEvent => "unexpected capability resolver event",
            Self::Internal => "internal capability resolver error",
        })
    }
}

impl std::error::Error for Error {}

/// Public capability events.
pub mod event {
    use emel_tensor::dtype::SerializedType;

    use super::Scope;

    /// Query one serialized type under one generation contract scope.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Query {
        pub(super) scope: Scope,
        pub(super) tensor_type: SerializedType,
    }

    impl Query {
        /// Creates an immutable allocation-free query.
        #[must_use]
        pub const fn new(scope: Scope, tensor_type: SerializedType) -> Self {
            Self { scope, tensor_type }
        }

        /// Returns the queried contract scope.
        #[must_use]
        pub const fn scope(self) -> Scope {
            self.scope
        }

        /// Returns the serialized tensor type under consideration.
        #[must_use]
        pub const fn tensor_type(self) -> SerializedType {
            self.tensor_type
        }
    }
}

struct QueryRuntime<'output> {
    query: event::Query,
    output: &'output mut Option<Outcome>,
}

#[derive(Clone, Copy, Debug)]
struct UnexpectedRuntime;

#[derive(Clone, Copy, Debug, Default)]
struct Context;

/// Single-writer, run-to-completion capability resolver actor.
pub struct Resolver {
    machine: CapabilityResolverStateMachine<Context>,
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
            machine: CapabilityResolverStateMachine::new(Context),
        }
    }

    /// Resolves one query within the synchronous dispatch boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] if the generated machine rejects a valid
    /// query or does not publish one of the exhaustive contract outcomes.
    pub fn process_event(&mut self, query: event::Query) -> Result<Outcome, Error> {
        let mut output = None;
        let mut runtime = QueryRuntime {
            query,
            output: &mut output,
        };
        self.machine
            .process_event(&mut runtime)
            .map_err(|_| Error::Internal)?;
        output.ok_or(Error::Internal)
    }
}

impl CapabilityResolverStateMachineContext for Context {
    fn guard_vector_f32<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::VectorDequantContract
            && event.query.tensor_type == SerializedType::F32)
    }

    fn guard_vector_native<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::VectorDequantContract
            && matches!(
                event.query.tensor_type,
                SerializedType::Q4_0
                    | SerializedType::Q4_1
                    | SerializedType::Q5_0
                    | SerializedType::Q8_0
                    | SerializedType::Q2K
                    | SerializedType::Q3K
                    | SerializedType::Q4K
                    | SerializedType::Q6K
            ))
    }

    fn guard_vector_explicit_no_claim<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::VectorDequantContract
            && event.query.tensor_type != SerializedType::F32
            && !matches!(
                event.query.tensor_type,
                SerializedType::Q4_0
                    | SerializedType::Q4_1
                    | SerializedType::Q5_0
                    | SerializedType::Q8_0
                    | SerializedType::Q2K
                    | SerializedType::Q3K
                    | SerializedType::Q4K
                    | SerializedType::Q6K
            ))
    }

    fn guard_matrix_native<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::MatrixWeightContract
            && matches!(
                event.query.tensor_type,
                SerializedType::Q4_0
                    | SerializedType::Q4_1
                    | SerializedType::Q5_0
                    | SerializedType::Q8_0
                    | SerializedType::Q2K
                    | SerializedType::Q3K
                    | SerializedType::Q4K
                    | SerializedType::Q6K
            ))
    }

    fn guard_matrix_f32<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::MatrixWeightContract
            && event.query.tensor_type == SerializedType::F32)
    }

    fn guard_matrix_explicit_no_claim<'event, 'output: 'event>(
        &self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<bool, ()> {
        Ok(event.query.scope == Scope::MatrixWeightContract
            && event.query.tensor_type != SerializedType::F32
            && !matches!(
                event.query.tensor_type,
                SerializedType::Q4_0
                    | SerializedType::Q4_1
                    | SerializedType::Q5_0
                    | SerializedType::Q8_0
                    | SerializedType::Q2K
                    | SerializedType::Q3K
                    | SerializedType::Q4K
                    | SerializedType::Q6K
            ))
    }

    fn guard_never(&self, _event: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }

    fn effect_publish_native<'event, 'output: 'event>(
        &mut self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<(), ()> {
        *event.output = Some(Outcome::NativeQuantized);
        Ok(())
    }

    fn effect_publish_dense_f32<'event, 'output: 'event>(
        &mut self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<(), ()> {
        *event.output = Some(Outcome::ApprovedDenseF32ByContract);
        Ok(())
    }

    fn effect_publish_disallowed<'event, 'output: 'event>(
        &mut self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<(), ()> {
        *event.output = Some(Outcome::DisallowedFallback);
        Ok(())
    }

    fn effect_publish_explicit_no_claim<'event, 'output: 'event>(
        &mut self,
        event: &'event mut QueryRuntime<'output>,
    ) -> Result<(), ()> {
        *event.output = Some(Outcome::ExplicitNoClaim);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unexpected_event_errors_and_actor_recovers() {
        let mut resolver = Resolver::new();
        assert!(resolver.machine.process_event(UnexpectedRuntime).is_err());
        assert_eq!(
            resolver.process_event(event::Query::new(
                Scope::MatrixWeightContract,
                SerializedType::Q4K,
            )),
            Ok(Outcome::NativeQuantized)
        );
    }
}
