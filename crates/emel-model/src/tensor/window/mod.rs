//! Window SM scaffold (`emel.cpp/src/emel/model/tensor/window/`).

pub mod actor;
pub(crate) mod detail;
pub mod event;
// The maintained actor owns the lifecycle; the historical generated scaffold
// is intentionally not part of the public module surface until its actions are
// ported to the same event contract.
