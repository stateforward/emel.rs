//! Bounds-checked GGUF v2/v3 probing and loading.
//!
//! The public runtime boundary is the [`Loader`] actor plus concrete types in
//! [`event`]. Parsed records, storage offsets, generated machine state, guards,
//! actions, and detail helpers remain private.
//!
//! ```compile_fail
//! use emel_gguf::loader::KvEntry;
//! ```
//!
//! ```compile_fail
//! let loader = emel_gguf::Loader::new();
//! let _ = loader.state();
//! ```

#![forbid(unsafe_code)]

#[cfg(test)]
use allocation_counter as _;

/// Typed request and outcome events accepted or produced by GGUF actors.
pub mod event;

mod actor;
mod loader;

pub use actor::Loader;

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
