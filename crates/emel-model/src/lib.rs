//! Model architectures, family bindings, tensor ownership, and loading lifecycles.

#![forbid(unsafe_code)]

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);
