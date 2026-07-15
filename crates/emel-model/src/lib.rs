//! Model loading and reusable tensor lifecycle actors.
//!
//! The tensor actor is intentionally private until the complete model-domain
//! cutover is ready. External default-feature and internal proof-feature
//! boundaries are checked by `tests/tensor_privacy.rs`.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

pub(crate) mod loader;
pub(crate) mod tensor;

/// Internal parity, fuzz, and benchmark access to the tensor actor.
///
/// This non-default feature deliberately exposes only the actor and typed
/// events. It is not the production model-domain cutover.
#[cfg(feature = "model-tensor-proof")]
#[doc(hidden)]
pub mod model_tensor_proof {
    pub use crate::tensor::Store;
    pub use crate::tensor::event;
}

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
