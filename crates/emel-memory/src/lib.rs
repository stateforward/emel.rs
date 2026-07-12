//! KV, recurrent, streaming, and hybrid model-memory infrastructure.

#![forbid(unsafe_code)]

/// The number of tokens currently retained by a memory strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetainedTokens(pub usize);
