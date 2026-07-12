//! Model loading and tensor lifecycle state machines (scaffold).
//!
//! Generated SM modules are `pub(crate)` until actions/guards/contexts are ported.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

pub(crate) mod loader;
pub(crate) mod tensor;

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
