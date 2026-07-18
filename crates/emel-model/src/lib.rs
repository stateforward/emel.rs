//! Model loading and reusable tensor lifecycle actors.
//!
//! The tensor actor is the production ownership boundary for model bytes.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

pub mod catalog;
pub(crate) mod data;
pub mod generation;
pub mod generation_audit;
pub mod llama;
pub(crate) mod loader;
pub mod tensor;
pub mod vocabulary;

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
