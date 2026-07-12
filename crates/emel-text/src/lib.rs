//! Text generation orchestration, expressed with `stateforward-sml` as the port grows.

#![forbid(unsafe_code)]

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
