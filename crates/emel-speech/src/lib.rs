//! Speech-model components, including encoders and codec support.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The sampled audio format supplied to speech components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioFormat {
    /// Samples per second.
    pub sample_rate_hz: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}
pub(crate) mod codec;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod generator;
pub(crate) mod predictor;
pub(crate) mod tokenizer;
pub(crate) mod transcriber;

/// Public Mimi model-binding and prepared-runtime contracts.
pub use codec::mimi::binding as mimi;

#[cfg(test)]
mod generator_port_tests;
