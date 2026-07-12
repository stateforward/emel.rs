//! Shared, dependency-free types used throughout `emel`.

#![forbid(unsafe_code)]

/// Identifies a family of models supported by emel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Meta Llama models.
    Llama,
    /// Alibaba Qwen models.
    Qwen,
    /// Whisper models.
    Whisper,
    /// An unrecognized model family.
    Unknown,
}

/// A crate-local result alias for domain operations.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Errors that can be produced by the port's shared domain layer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A requested capability is not available in the current build.
    Unsupported(&'static str),
    /// Model metadata was invalid or incomplete.
    InvalidModel(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported(capability) => {
                write!(formatter, "unsupported capability: {capability}")
            }
            Self::InvalidModel(reason) => write!(formatter, "invalid model: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
