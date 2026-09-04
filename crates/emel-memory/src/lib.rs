//! KV, recurrent, streaming, and hybrid model-memory infrastructure.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The number of tokens currently retained by a memory strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetainedTokens(pub usize);
/// Hybrid KV/recurrent memory state machine and actor binding boundary.
pub mod hybrid;
/// KV-cache memory state machine and ownership-safe actor boundary.
pub mod kv;
/// Recurrent memory state machine and ownership-safe actor boundary.
pub mod recurrent;
pub mod streaming;
/// Streaming memory state machine and ownership-safe actor boundary.
/// Source-aligned bounded geometry helpers and owned memory snapshots.
pub mod view;
