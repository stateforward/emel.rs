//! Speech-model components, including encoders and codec support.

#![forbid(unsafe_code)]

/// The sampled audio format supplied to speech components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioFormat {
    /// Samples per second.
    pub sample_rate_hz: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}
