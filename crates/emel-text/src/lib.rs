//! Text generation orchestration, expressed with `stateforward-sml` as the port grows.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The state-machine DSL used to express generation lifecycles.
pub use sml;

/// The lifecycle phase of a text-generation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GenerationPhase {
    /// Request validation and model binding.
    Initializing,
    /// Prompt tokens are being evaluated.
    Prefilling,
    /// New tokens are being decoded.
    Decoding,
    /// The request has completed.
    Complete,
}
pub(crate) mod conditioner;
pub(crate) mod detokenizer;
pub(crate) mod encoders;
pub(crate) mod formatter;
pub(crate) mod generator;
pub(crate) mod jinja;
pub(crate) mod renderer;
pub(crate) mod tokenizer;

#[cfg(test)]
mod detokenizer_port_tests;

#[cfg(test)]
#[path = "generator/layer_port_tests.rs"]
mod layer_port_tests;
