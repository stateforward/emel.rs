//! KV, recurrent, streaming, and hybrid model-memory infrastructure.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The number of tokens currently retained by a memory strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetainedTokens(pub usize);
pub(crate) mod hybrid;
pub(crate) mod kv;
pub(crate) mod recurrent;
pub(crate) mod streaming;
