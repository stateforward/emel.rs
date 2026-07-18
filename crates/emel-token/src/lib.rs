//! Token representations and batching contracts.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// A vocabulary index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenId(pub u32);
pub(crate) mod batcher;

/// Exact tokenizer model and pre-tokenizer profile resolution.
pub mod profile;
