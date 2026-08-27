//! Typed query for one quantized generation-path contract.

use emel_tensor::dtype::SerializedType;

use super::Scope;

/// Query one serialized type under one generation contract scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Query {
    scope: Scope,
    tensor_type: SerializedType,
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
