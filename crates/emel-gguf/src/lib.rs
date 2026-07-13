//! Bounds-checked GGUF v2/v3 probing and loading.

#![forbid(unsafe_code)]

/// Typed request and outcome events accepted or produced by GGUF actors.
pub mod event;

mod actor;
mod loader;

pub use actor::{Loader, State as LoaderState};

/// The GGUF file magic.
pub(crate) const MAGIC: [u8; 4] = *b"GGUF";

/// The newest supported GGUF format version.
pub(crate) const VERSION: u32 = 3;

/// The oldest supported GGUF format version.
pub(crate) const MIN_VERSION: u32 = 2;

/// The default tensor-data alignment when metadata does not override it.
pub(crate) const DEFAULT_ALIGNMENT: u32 = 32;

/// The maximum number of tensor dimensions supported by GGUF.
pub(crate) const MAX_TENSOR_DIMS: u32 = 4;
