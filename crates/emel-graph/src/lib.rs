//! Execution-graph allocation, assembly, processing, and tensor binding.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// Identifies a node in an execution graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u32);
pub(crate) mod allocator;
pub(crate) mod assembler;
pub(crate) mod processor;
pub(crate) mod sm;
pub(crate) mod tensor;
