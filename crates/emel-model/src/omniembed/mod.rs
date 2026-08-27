//! `OmniEmbed` family boundary.

/// Explicit family operation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The model-family implementation is owned by `emel-embeddings`.
    OwnedByEmbeddingsCrate,
}

#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == b"omniembed"
}

/// Family detail boundary. Runtime behavior is provided by the owning
/// `emel-embeddings` actor; this type never panics or hides a route.
#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Returns the owning-crate boundary error for an unbound layer request.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedByEmbeddingsCrate`] until the owning actor
    /// is dispatched through the `emel-embeddings` public boundary.
    pub const fn bind_layers(&self) -> Result<(), Error> {
        Err(Error::OwnedByEmbeddingsCrate)
    }

    /// Returns the owning-crate boundary error for an unbound metadata request.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedByEmbeddingsCrate`] until the owning actor
    /// is dispatched through the `emel-embeddings` public boundary.
    pub const fn load_hparams(&self) -> Result<(), Error> {
        Err(Error::OwnedByEmbeddingsCrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unbound_operations_are_explicit_errors() {
        let detail = Detail;
        assert_eq!(detail.bind_layers(), Err(Error::OwnedByEmbeddingsCrate));
        assert_eq!(detail.load_hparams(), Err(Error::OwnedByEmbeddingsCrate));
    }

    #[test]
    fn architecture_predicate_is_exact_and_non_utf8_safe() {
        assert!(is_execution_architecture(b"omniembed"));
        assert!(!is_execution_architecture(b"omniembed.extra"));
        assert!(!is_execution_architecture(&[0xff]));
    }
}
