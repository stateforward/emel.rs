//! Whisper family ownership boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The maintained Whisper implementation is owned by the speech crate.
    OwnedBySpeechCrate,
}

#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == b"whisper"
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

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Any;

impl Any {
    /// Constructs the speech-owned family actor boundary.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedBySpeechCrate`] until the speech crate
    /// exposes the maintained actor.
    pub const fn bind() -> Result<Self, Error> {
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
        assert_eq!(Any::bind(), Err(Error::OwnedBySpeechCrate));
    }

    #[test]
    fn architecture_predicate_is_exact_and_non_utf8_safe() {
        assert!(is_execution_architecture(b"whisper"));
        assert!(!is_execution_architecture(b"whisper.extra"));
        assert!(!is_execution_architecture(&[0xff]));
    }
}
