//! Embedding-model components and embedding-generation orchestration.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// An embedding vector emitted by a model.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Box<[f32]>);
pub(crate) mod generator;
