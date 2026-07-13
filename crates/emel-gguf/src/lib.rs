//! Bounds-checked GGUF v2/v3 probing and loading.

#![forbid(unsafe_code)]

mod loader;

pub use loader::{Error, Gguf, KvEntry, Loader, LoaderState, Requirements, TensorInfo, load};

/// The GGUF file magic.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// The newest supported GGUF format version.
pub const VERSION: u32 = 3;

/// The oldest supported GGUF format version.
pub const MIN_VERSION: u32 = 2;

/// The default tensor-data alignment when metadata does not override it.
pub const DEFAULT_ALIGNMENT: u32 = 32;

/// The maximum number of tensor dimensions supported by GGUF.
pub const MAX_TENSOR_DIMS: u32 = 4;
