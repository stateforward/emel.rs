//! Moshi family ownership boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The maintained Moshi actor is owned by `emel-speech`.
    OwnedBySpeechCrate,
}

#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == b"moshi"
}

#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Attempts to bind family layers through the owning speech actor.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedBySpeechCrate`] at this model-domain
    /// boundary; callers must dispatch the speech owner actor.
    pub const fn bind_layers(&self) -> Result<(), Error> {
        Err(Error::OwnedBySpeechCrate)
    }

    /// Attempts to load family metadata through the owning speech actor.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedBySpeechCrate`] at this model-domain
    /// boundary; callers must dispatch the speech owner actor.
    pub const fn load_hparams(&self) -> Result<(), Error> {
        Err(Error::OwnedBySpeechCrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unbound_operations_are_explicit_errors() {
        let detail = Detail;
        assert_eq!(detail.bind_layers(), Err(Error::OwnedBySpeechCrate));
        assert_eq!(detail.load_hparams(), Err(Error::OwnedBySpeechCrate));
    }

    #[test]
    fn architecture_predicate_is_exact_and_non_utf8_safe() {
        assert!(is_execution_architecture(b"moshi"));
        assert!(!is_execution_architecture(b"Moshi"));
        assert!(!is_execution_architecture(&[0xff]));
    }
}
