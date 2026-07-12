//! Execution-graph allocation, assembly, processing, and tensor binding.

#![forbid(unsafe_code)]

/// Identifies a node in an execution graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u32);
