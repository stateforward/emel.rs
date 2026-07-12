//! Batch planning and deterministic request scheduling.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// An identifier assigned to a batch by the planner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchId(pub u64);
pub(crate) mod planner;
