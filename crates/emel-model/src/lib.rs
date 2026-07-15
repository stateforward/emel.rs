//! Model loading and reusable tensor lifecycle actors.
//!
//! The tensor actor is intentionally private until the complete model-domain
//! cutover is ready.
//!
//! ```compile_fail
//! use emel_model::tensor::Store;
//! ```

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

pub(crate) mod loader;
pub(crate) mod tensor;

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
