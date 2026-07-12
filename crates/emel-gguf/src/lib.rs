//! GGUF container parsing and model metadata loading.

#![forbid(unsafe_code)]

/// The GGUF magic number in little-endian byte order.
pub const MAGIC: [u8; 4] = *b"GGUF";
