//! Logit validation and deterministic sampling primitives.

#![forbid(unsafe_code)]

/// A token score before sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logit {
    /// The vocabulary index associated with the score.
    pub token: u32,
    /// The unnormalized score.
    pub value: f32,
}
