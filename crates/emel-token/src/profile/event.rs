//! Typed profile resolver requests and outcomes.

use core::fmt;

use super::Resolver;

mod sealed {
    pub trait Sealed {}
}

/// Event accepted by [`Resolver`].
///
/// The trait is sealed so generated machine details cannot escape the actor.
pub trait Event: sealed::Sealed {
    /// Result produced before the top-level dispatch returns.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Resolver) -> Self::Output;
}

/// Tokenizer algorithm selected by the model spelling.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Model {
    /// No vocabulary is present.
    None,
    /// `SentencePiece` model.
    SentencePiece,
    /// Byte-pair encoding model.
    Bpe,
    /// `WordPiece` model.
    WordPiece,
    /// Unigram model.
    Unigram,
    /// RWKV byte-trie model.
    Rwkv,
    /// `PLaMo2` model.
    Plamo2,
    /// A future or otherwise unknown model spelling.
    Unknown,
}

impl Model {
    /// Returns whether late model validation must classify this spelling.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// Opaque pre-tokenizer profile identity.
///
/// Consumers may compare or hash identities, but family-specific variants stay
/// private to the tokenizer owner.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct PreId(pub(super) u8);

impl fmt::Debug for PreId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PreId(..)")
    }
}

/// Source-compatible vocabulary defaults after model and pre-profile composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Defaults {
    pub(super) bos_id: i32,
    pub(super) eos_id: i32,
    pub(super) eot_id: i32,
    pub(super) eom_id: i32,
    pub(super) unk_id: i32,
    pub(super) sep_id: i32,
    pub(super) pad_id: i32,
    pub(super) cls_id: i32,
    pub(super) mask_id: i32,
    pub(super) prefix_id: i32,
    pub(super) suffix_id: i32,
    pub(super) middle_id: i32,
    pub(super) fim_pre_id: i32,
    pub(super) fim_suf_id: i32,
    pub(super) fim_mid_id: i32,
    pub(super) fim_pad_id: i32,
    pub(super) fim_rep_id: i32,
    pub(super) fim_sep_id: i32,
    pub(super) flags: u8,
}

impl Defaults {
    pub(super) const ADD_BOS: u8 = 1 << 0;
    pub(super) const ADD_EOS: u8 = 1 << 1;
    pub(super) const ADD_SEP: u8 = 1 << 2;
    pub(super) const ADD_SPACE_PREFIX: u8 = 1 << 3;
    pub(super) const REMOVE_EXTRA_WHITESPACES: u8 = 1 << 4;
    pub(super) const ESCAPE_WHITESPACES: u8 = 1 << 5;
    pub(super) const WHITESPACE_AS_SUFFIX: u8 = 1 << 6;
    pub(super) const IGNORE_MERGES: u8 = 1 << 7;

    pub(super) const SOURCE: Self = Self {
        bos_id: -1,
        eos_id: -1,
        eot_id: -1,
        eom_id: -1,
        unk_id: -1,
        sep_id: -1,
        pad_id: -1,
        cls_id: -1,
        mask_id: -1,
        prefix_id: -1,
        suffix_id: -1,
        middle_id: -1,
        fim_pre_id: -1,
        fim_suf_id: -1,
        fim_mid_id: -1,
        fim_pad_id: -1,
        fim_rep_id: -1,
        fim_sep_id: -1,
        flags: Self::ESCAPE_WHITESPACES,
    };

    /// Beginning-of-sequence token ID.
    #[must_use]
    pub const fn bos_id(self) -> i32 {
        self.bos_id
    }
    /// End-of-sequence token ID.
    #[must_use]
    pub const fn eos_id(self) -> i32 {
        self.eos_id
    }
    /// End-of-turn token ID.
    #[must_use]
    pub const fn eot_id(self) -> i32 {
        self.eot_id
    }
    /// End-of-message token ID.
    #[must_use]
    pub const fn eom_id(self) -> i32 {
        self.eom_id
    }
    /// Unknown token ID.
    #[must_use]
    pub const fn unk_id(self) -> i32 {
        self.unk_id
    }
    /// Separator token ID.
    #[must_use]
    pub const fn sep_id(self) -> i32 {
        self.sep_id
    }
    /// Padding token ID.
    #[must_use]
    pub const fn pad_id(self) -> i32 {
        self.pad_id
    }
    /// Classification token ID.
    #[must_use]
    pub const fn cls_id(self) -> i32 {
        self.cls_id
    }
    /// Mask token ID.
    #[must_use]
    pub const fn mask_id(self) -> i32 {
        self.mask_id
    }
    /// Prefix token ID.
    #[must_use]
    pub const fn prefix_id(self) -> i32 {
        self.prefix_id
    }
    /// Suffix token ID.
    #[must_use]
    pub const fn suffix_id(self) -> i32 {
        self.suffix_id
    }
    /// Middle token ID.
    #[must_use]
    pub const fn middle_id(self) -> i32 {
        self.middle_id
    }
    /// Fill-in-the-middle prefix token ID.
    #[must_use]
    pub const fn fim_pre_id(self) -> i32 {
        self.fim_pre_id
    }
    /// Fill-in-the-middle suffix token ID.
    #[must_use]
    pub const fn fim_suf_id(self) -> i32 {
        self.fim_suf_id
    }
    /// Fill-in-the-middle middle token ID.
    #[must_use]
    pub const fn fim_mid_id(self) -> i32 {
        self.fim_mid_id
    }
    /// Fill-in-the-middle padding token ID.
    #[must_use]
    pub const fn fim_pad_id(self) -> i32 {
        self.fim_pad_id
    }
    /// Fill-in-the-middle repository token ID.
    #[must_use]
    pub const fn fim_rep_id(self) -> i32 {
        self.fim_rep_id
    }
    /// Fill-in-the-middle separator token ID.
    #[must_use]
    pub const fn fim_sep_id(self) -> i32 {
        self.fim_sep_id
    }
    /// Whether to add a beginning-of-sequence token.
    #[must_use]
    pub const fn add_bos(self) -> bool {
        self.flags & Self::ADD_BOS != 0
    }
    /// Whether to add an end-of-sequence token.
    #[must_use]
    pub const fn add_eos(self) -> bool {
        self.flags & Self::ADD_EOS != 0
    }
    /// Whether to add a separator token.
    #[must_use]
    pub const fn add_sep(self) -> bool {
        self.flags & Self::ADD_SEP != 0
    }
    /// Whether to add a leading space.
    #[must_use]
    pub const fn add_space_prefix(self) -> bool {
        self.flags & Self::ADD_SPACE_PREFIX != 0
    }
    /// Whether to collapse extra whitespace.
    #[must_use]
    pub const fn remove_extra_whitespaces(self) -> bool {
        self.flags & Self::REMOVE_EXTRA_WHITESPACES != 0
    }
    /// Whether to escape whitespace.
    #[must_use]
    pub const fn escape_whitespaces(self) -> bool {
        self.flags & Self::ESCAPE_WHITESPACES != 0
    }
    /// Whether whitespace is a suffix.
    #[must_use]
    pub const fn treat_whitespace_as_suffix(self) -> bool {
        self.flags & Self::WHITESPACE_AS_SUFFIX != 0
    }
    /// Whether BPE merges are ignored.
    #[must_use]
    pub const fn ignore_merges(self) -> bool {
        self.flags & Self::IGNORE_MERGES != 0
    }
}

/// Successful profile classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resolved {
    pub(super) model: Model,
    pub(super) pre: PreId,
    pub(super) defaults: Defaults,
}

impl Resolved {
    /// Returns the tokenizer algorithm.
    #[must_use]
    pub const fn model(self) -> Model {
        self.model
    }
    /// Returns the opaque pre-tokenizer identity.
    #[must_use]
    pub const fn pre_id(self) -> PreId {
        self.pre
    }
    /// Returns the composed source defaults.
    #[must_use]
    pub const fn defaults(self) -> Defaults {
        self.defaults
    }
}

/// Resolver failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The actor encountered an internal or unexpected-event failure.
    Internal,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("internal tokenizer profile resolver error")
    }
}

impl std::error::Error for Error {}

/// Resolve a tokenizer model spelling and pre-tokenizer name together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resolve<'a> {
    pub(super) model: &'a str,
    pub(super) pre: &'a str,
}

impl<'a> Resolve<'a> {
    /// Creates one immutable resolution request.
    #[must_use]
    pub const fn new(model: &'a str, pre: &'a str) -> Self {
        Self { model, pre }
    }
}

impl sealed::Sealed for Resolve<'_> {}

impl Event for Resolve<'_> {
    type Output = Result<Resolved, Error>;

    fn dispatch(self, actor: &mut Resolver) -> Self::Output {
        actor.resolve(self)
    }
}
