//! Speaker-diarization components, including Sortformer-oriented models.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// A speaker interval in an input recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerSegment {
    /// Segment start, in seconds.
    pub start_seconds: f32,
    /// Segment end, in seconds.
    pub end_seconds: f32,
    /// Model-assigned speaker index.
    pub speaker: u32,
}
pub(crate) mod sortformer;
