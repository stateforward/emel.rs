//! Model input contracts and eventually memory-mapped model loading.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

use std::path::Path;

/// An input source for a model load operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSource {
    /// A local file path.
    File(std::path::PathBuf),
}

impl ModelSource {
    /// Builds a local-file model source.
    #[must_use]
    pub fn file(path: impl AsRef<Path>) -> Self {
        Self::File(path.as_ref().to_path_buf())
    }
}
pub(crate) mod loader;
pub(crate) mod mmap;
pub(crate) mod read;
pub(crate) mod staged_read;
