//! Source-aligned, bounded text tokenizer state machine.
//!
#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::elidable_lifetime_names,
    clippy::if_not_else,
    clippy::option_if_let_else,
    clippy::unused_self,
    clippy::needless_pass_by_ref_mut,
    clippy::too_many_arguments,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

/// Maximum number of fragments retained for one request.
pub const MAX_FRAGMENTS: usize = 1024;

/// Supported preprocessor variants.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessorKind {
    Spm = 0,
    Bpe = 1,
    Wpm = 2,
    Ugm = 3,
    Rwkv = 4,
    Plamo2 = 5,
    #[default]
    Fallback = 6,
}

/// Supported encoder variants.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum EncoderKind {
    Spm = 0,
    Bpe = 1,
    Wpm = 2,
    Ugm = 3,
    Rwkv = 4,
    Plamo2 = 5,
    #[default]
    Fallback = 6,
}

/// Public tokenizer error. Numeric values mirror `text/tokenizer/errors.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum TokenizerError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ModelInvalid = 2,
    BackendError = 4,
    /// Internal sequencing failure; public result maps it to invalid request.
    Unexpected = 8,
}
impl TokenizerError {
    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// A fragment returned by a preprocessor child.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment {
    pub kind: FragmentKind,
    pub start: usize,
    pub end: usize,
    pub token: i32,
}

/// Fragment kind used by the tokenizer state machine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

/// Vocabulary information needed by prefix/suffix handling and fallback
/// child implementations. Implementations must return stable token slices.
pub trait VocabularyView {
    fn token_count(&self) -> usize {
        0
    }
    fn token(&self, _index: usize) -> Option<&[u8]> {
        None
    }
    fn bos_id(&self) -> i32 {
        -1
    }
    fn eos_id(&self) -> i32 {
        -1
    }
    fn sep_id(&self) -> i32 {
        -1
    }
    fn add_bos(&self) -> bool {
        false
    }
    fn add_eos(&self) -> bool {
        false
    }
    fn add_sep(&self) -> bool {
        false
    }
}

/// Result returned by a bounded child callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChildResult {
    pub accepted: bool,
    pub error: TokenizerError,
    pub count: usize,
}

/// Synchronous preprocessor child contract.
pub type PreprocessCallback = fn(&dyn VocabularyView, &[u8], bool, &mut [Fragment]) -> ChildResult;
/// Synchronous encoder child contract.
pub type EncodeCallback = fn(&dyn VocabularyView, &[u8], bool, &mut [i32]) -> ChildResult;
/// Synchronous child bind contract.
pub type BindCallback = fn(&dyn VocabularyView, u8) -> TokenizerError;

/// A selected preprocessor child owned by the tokenizer root.
///
/// The adapter is deliberately a small typed function-pointer boundary: it is
/// copied into the root context during the bind transition and used for every
/// subsequent preprocess dispatch. Completion callbacks are not adapters.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessorAdapter {
    pub kind: PreprocessorKind,
    pub bind: BindCallback,
    pub preprocess: PreprocessCallback,
}

/// A selected encoder child owned by the tokenizer root.
///
/// As with [`PreprocessorAdapter`], this is a non-allocating typed boundary
/// retained by the root context rather than a trait-object route in a hot path.
#[derive(Clone, Copy, Debug)]
pub struct EncoderAdapter {
    pub kind: EncoderKind,
    pub bind: BindCallback,
    pub encode: EncodeCallback,
}

/// Root-owned preprocessor selector. Every maintained family has an explicit
/// slot; no non-fallback family is selected without an installed adapter.
#[derive(Clone, Copy, Debug)]
pub enum PreprocessorSelector {
    Spm(PreprocessorAdapter),
    Bpe(PreprocessorAdapter),
    Wpm(PreprocessorAdapter),
    Ugm(PreprocessorAdapter),
    Rwkv(PreprocessorAdapter),
    Plamo2(PreprocessorAdapter),
    Fallback(PreprocessorAdapter),
}

impl PreprocessorSelector {
    #[must_use]
    pub const fn from_adapter(adapter: PreprocessorAdapter) -> Option<Self> {
        match adapter.kind {
            PreprocessorKind::Spm => Some(Self::Spm(adapter)),
            PreprocessorKind::Bpe => Some(Self::Bpe(adapter)),
            PreprocessorKind::Wpm => Some(Self::Wpm(adapter)),
            PreprocessorKind::Ugm => Some(Self::Ugm(adapter)),
            PreprocessorKind::Rwkv => Some(Self::Rwkv(adapter)),
            PreprocessorKind::Plamo2 => Some(Self::Plamo2(adapter)),
            PreprocessorKind::Fallback => Some(Self::Fallback(adapter)),
        }
    }
    #[must_use]
    pub const fn kind(self) -> PreprocessorKind {
        match self {
            Self::Spm(adapter)
            | Self::Bpe(adapter)
            | Self::Wpm(adapter)
            | Self::Ugm(adapter)
            | Self::Rwkv(adapter)
            | Self::Plamo2(adapter)
            | Self::Fallback(adapter) => adapter.kind,
        }
    }
}

/// Root-owned encoder selector. Every maintained family has an explicit slot;
/// no non-fallback family is selected without an installed adapter.
#[derive(Clone, Copy, Debug)]
pub enum EncoderSelector {
    Spm(EncoderAdapter),
    Bpe(EncoderAdapter),
    Wpm(EncoderAdapter),
    Ugm(EncoderAdapter),
    Rwkv(EncoderAdapter),
    Plamo2(EncoderAdapter),
    Fallback(EncoderAdapter),
}

impl EncoderSelector {
    #[must_use]
    pub const fn from_adapter(adapter: EncoderAdapter) -> Option<Self> {
        match adapter.kind {
            EncoderKind::Spm => Some(Self::Spm(adapter)),
            EncoderKind::Bpe => Some(Self::Bpe(adapter)),
            EncoderKind::Wpm => Some(Self::Wpm(adapter)),
            EncoderKind::Ugm => Some(Self::Ugm(adapter)),
            EncoderKind::Rwkv => Some(Self::Rwkv(adapter)),
            EncoderKind::Plamo2 => Some(Self::Plamo2(adapter)),
            EncoderKind::Fallback => Some(Self::Fallback(adapter)),
        }
    }
    #[must_use]
    pub const fn kind(self) -> EncoderKind {
        match self {
            Self::Spm(adapter)
            | Self::Bpe(adapter)
            | Self::Wpm(adapter)
            | Self::Ugm(adapter)
            | Self::Rwkv(adapter)
            | Self::Plamo2(adapter)
            | Self::Fallback(adapter) => adapter.kind,
        }
    }
}

const fn fallback_bind(_: &dyn VocabularyView, _: u8) -> TokenizerError {
    TokenizerError::None
}

#[allow(clippy::redundant_pub_crate)]
pub(crate) fn fallback_preprocess(
    _: &dyn VocabularyView,
    text: &[u8],
    _: bool,
    fragments: &mut [Fragment],
) -> ChildResult {
    if text.is_empty() {
        return ChildResult {
            accepted: true,
            error: TokenizerError::None,
            count: 0,
        };
    }
    if fragments.is_empty() {
        return ChildResult {
            accepted: false,
            error: TokenizerError::BackendError,
            count: 0,
        };
    }
    fragments[0] = Fragment {
        kind: FragmentKind::RawText,
        start: 0,
        end: text.len(),
        token: -1,
    };
    ChildResult {
        accepted: true,
        error: TokenizerError::None,
        count: 1,
    }
}

#[allow(clippy::redundant_pub_crate)]
pub(crate) fn fallback_encode(
    vocab: &dyn VocabularyView,
    text: &[u8],
    _: bool,
    output: &mut [i32],
) -> ChildResult {
    if text.len() > output.len() {
        return ChildResult {
            accepted: false,
            error: TokenizerError::BackendError,
            count: 0,
        };
    }
    for (index, byte) in text.iter().copied().enumerate() {
        let needle = [byte];
        let Some(token_id) = (0..vocab.token_count()).find_map(|id| {
            (vocab.token(id) == Some(needle.as_slice()))
                .then(|| i32::try_from(id).ok())
                .flatten()
        }) else {
            return ChildResult {
                accepted: false,
                error: TokenizerError::BackendError,
                count: 0,
            };
        };
        output[index] = token_id;
    }
    ChildResult {
        accepted: true,
        error: TokenizerError::None,
        count: text.len(),
    }
}

const FALLBACK_PREPROCESSOR: PreprocessorSelector =
    PreprocessorSelector::Fallback(PreprocessorAdapter {
        kind: PreprocessorKind::Fallback,
        bind: fallback_bind,
        preprocess: fallback_preprocess,
    });
const FALLBACK_ENCODER: EncoderSelector = EncoderSelector::Fallback(EncoderAdapter {
    kind: EncoderKind::Fallback,
    bind: fallback_bind,
    encode: fallback_encode,
});

/// Completion payload for a successful bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizerBindDone;
/// Completion payload for a failed bind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerBindError {
    pub error: TokenizerError,
}
/// Completion payload for successful tokenization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerDone {
    pub token_count: usize,
}
/// Completion payload for failed tokenization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerErrorEvent {
    pub error: TokenizerError,
}

/// Compatibility aliases matching the C++ event names.
pub type BindingDone = TokenizerBindDone;
pub type BindingError = TokenizerBindError;
pub type EventsTokenizerDone = TokenizerDone;
pub type EventsTokenizerError = TokenizerErrorEvent;

pub type BindDoneCallback = fn(TokenizerBindDone) -> bool;
pub type BindErrorCallback = fn(TokenizerBindError) -> bool;
pub type TokenizeDoneCallback = fn(TokenizerDone) -> bool;
pub type TokenizeErrorCallback = fn(TokenizerErrorEvent) -> bool;

/// Caller-owned bind request.
#[derive(Clone, Copy)]
pub struct BindRequest<'event> {
    pub vocab: &'event dyn VocabularyView,
    pub preprocessor_variant: PreprocessorKind,
    pub encoder_variant: EncoderKind,
    /// Explicit typed child selected for this bind. Legacy bind callbacks are
    /// retained below only for source compatibility and are not child routes.
    pub preprocessor_adapter: Option<PreprocessorAdapter>,
    pub encoder_adapter: Option<EncoderAdapter>,
    pub bind_preprocessor: Option<BindCallback>,
    pub bind_encoder: Option<BindCallback>,
    pub dispatch_done: Option<BindDoneCallback>,
    pub dispatch_error: Option<BindErrorCallback>,
}
impl<'event> BindRequest<'event> {
    #[must_use]
    pub const fn new(
        vocab: &'event dyn VocabularyView,
        preprocessor_variant: PreprocessorKind,
        encoder_variant: EncoderKind,
        dispatch_done: BindDoneCallback,
        dispatch_error: BindErrorCallback,
    ) -> Self {
        Self::with_adapters(
            vocab,
            preprocessor_variant,
            encoder_variant,
            None,
            None,
            dispatch_done,
            dispatch_error,
        )
    }
    #[must_use]
    pub const fn with_adapters(
        vocab: &'event dyn VocabularyView,
        preprocessor_variant: PreprocessorKind,
        encoder_variant: EncoderKind,
        preprocessor_adapter: Option<PreprocessorAdapter>,
        encoder_adapter: Option<EncoderAdapter>,
        dispatch_done: BindDoneCallback,
        dispatch_error: BindErrorCallback,
    ) -> Self {
        Self {
            vocab,
            preprocessor_variant,
            encoder_variant,
            preprocessor_adapter,
            encoder_adapter,
            bind_preprocessor: None,
            bind_encoder: None,
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }
    #[must_use]
    pub const fn with_callbacks(
        vocab: &'event dyn VocabularyView,
        preprocessor_variant: PreprocessorKind,
        encoder_variant: EncoderKind,
        bind_preprocessor: Option<BindCallback>,
        bind_encoder: Option<BindCallback>,
        dispatch_done: Option<BindDoneCallback>,
        dispatch_error: Option<BindErrorCallback>,
    ) -> Self {
        Self {
            vocab,
            preprocessor_variant,
            encoder_variant,
            preprocessor_adapter: None,
            encoder_adapter: None,
            bind_preprocessor,
            bind_encoder,
            dispatch_done,
            dispatch_error,
        }
    }
}
impl core::fmt::Debug for BindRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BindRequest")
            .field("preprocessor_variant", &self.preprocessor_variant)
            .field("encoder_variant", &self.encoder_variant)
            .field(
                "preprocessor_adapter",
                &self.preprocessor_adapter.map(|a| a.kind),
            )
            .field("encoder_adapter", &self.encoder_adapter.map(|a| a.kind))
            .finish_non_exhaustive()
    }
}

/// Caller-owned tokenization request.
#[derive(Clone, Copy)]
pub struct TokenizeRequest<'event> {
    pub vocab: &'event dyn VocabularyView,
    pub text: &'event [u8],
    pub add_special: bool,
    pub parse_special: bool,
    pub token_ids: &'event RefCell<&'event mut [i32]>,
    pub preprocess: Option<PreprocessCallback>,
    pub encode: Option<EncodeCallback>,
    pub dispatch_done: Option<TokenizeDoneCallback>,
    pub dispatch_error: Option<TokenizeErrorCallback>,
}
impl<'event> TokenizeRequest<'event> {
    #[must_use]
    pub const fn new(
        vocab: &'event dyn VocabularyView,
        text: &'event [u8],
        token_ids: &'event RefCell<&'event mut [i32]>,
        dispatch_done: TokenizeDoneCallback,
        dispatch_error: TokenizeErrorCallback,
    ) -> Self {
        Self {
            vocab,
            text,
            add_special: false,
            parse_special: false,
            token_ids,
            preprocess: None,
            encode: None,
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn with_callbacks(
        vocab: &'event dyn VocabularyView,
        text: &'event [u8],
        token_ids: &'event RefCell<&'event mut [i32]>,
        add_special: bool,
        parse_special: bool,
        preprocess: Option<PreprocessCallback>,
        encode: Option<EncodeCallback>,
        dispatch_done: Option<TokenizeDoneCallback>,
        dispatch_error: Option<TokenizeErrorCallback>,
    ) -> Self {
        Self {
            vocab,
            text,
            add_special,
            parse_special,
            token_ids,
            preprocess,
            encode,
            dispatch_done,
            dispatch_error,
        }
    }
}
impl core::fmt::Debug for TokenizeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TokenizeRequest")
            .field("text_length", &self.text.len())
            .field("add_special", &self.add_special)
            .field("parse_special", &self.parse_special)
            .field("token_capacity", &self.token_ids.borrow().len())
            .finish_non_exhaustive()
    }
}

/// Runtime bind event. The request and dispatch borrows have independent lifetimes.
#[derive(Clone, Copy, Debug)]
pub struct EventBindRuntime<'request, 'dispatch> {
    pub request: BindRequest<'request>,
    pub context: &'dispatch RefCell<BindContext>,
}
/// Runtime tokenization event. The request and dispatch borrows have independent lifetimes.
#[derive(Clone, Copy, Debug)]
pub struct EventTokenizeRuntime<'request, 'dispatch> {
    pub request: TokenizeRequest<'request>,
    pub context: &'dispatch RefCell<TokenizeContext>,
}

/// Bind operation context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindContext {
    pub err: TokenizerError,
    pub result: bool,
}
/// Tokenization operation context. Fragment and output accounting is bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizeContext {
    pub fragments: [Fragment; MAX_FRAGMENTS],
    pub fragment_count: usize,
    pub fragment_index: usize,
    pub preprocessed: bool,
    pub preprocess_accepted: bool,
    pub preprocess_err_code: TokenizerError,
    pub encode_accepted: bool,
    pub encode_err_code: TokenizerError,
    pub encode_token_count: usize,
    pub token_count: usize,
    pub err: TokenizerError,
    pub result: bool,
    pub unexpected: bool,
}
impl Default for TokenizeContext {
    fn default() -> Self {
        Self {
            fragments: [Fragment::default(); MAX_FRAGMENTS],
            fragment_count: 0,
            fragment_index: 0,
            preprocessed: false,
            preprocess_accepted: false,
            preprocess_err_code: TokenizerError::None,
            encode_accepted: false,
            encode_err_code: TokenizerError::None,
            encode_token_count: 0,
            token_count: 0,
            err: TokenizerError::None,
            result: false,
            unexpected: false,
        }
    }
}

sml! {
    TextTokenizer<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "binding_preprocessor"_s <= *"uninitialized"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) [can_bind] / begin_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) / reject_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / reject_invalid_from_uninitialized,
        "binding_preprocessor"_s <= "idle"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) [can_bind] / begin_bind_from_idle,
        "errored"_s <= "idle"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) / reject_bind_from_idle,
        "preprocessing"_s <= "idle"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [can_tokenize] / begin_tokenize_from_idle,
        "errored"_s <= "idle"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / reject_invalid_from_idle,
        "binding_preprocessor"_s <= "done"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) [can_bind] / begin_bind_from_done,
        "errored"_s <= "done"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) / reject_bind_from_done,
        "preprocessing"_s <= "done"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [can_tokenize] / begin_tokenize_from_done,
        "errored"_s <= "done"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / reject_invalid_from_done,
        "binding_preprocessor"_s <= "errored"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) [can_bind] / begin_bind_from_errored,
        "errored"_s <= "errored"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) / reject_bind_from_errored,
        "preprocessing"_s <= "errored"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [can_tokenize] / begin_tokenize_from_errored,
        "errored"_s <= "errored"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / reject_invalid_from_errored,
        "binding_preprocessor"_s <= "unexpected"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) [can_bind] / begin_bind_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventBindRuntime<'event, 'dispatch>>(&'dispatch EventBindRuntime<'event, 'dispatch>) / reject_bind_from_unexpected,
        "preprocessing"_s <= "unexpected"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [can_tokenize] / begin_tokenize_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventTokenizeRuntime<'event, 'dispatch>>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / reject_invalid_from_unexpected,
        "binding_preprocessor_decision"_s <= "binding_preprocessor"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) / bind_preprocessor,
        "binding_encoder"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_preprocessor_error_none],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_preprocessor_error_invalid_request],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_preprocessor_error_model_invalid],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_preprocessor_error_backend_error],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_preprocessor_error_unknown],
        "binding_encoder_decision"_s <= "binding_encoder"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) / bind_encoder,
        "idle"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_encoder_error_none] / mark_bind_success,
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_encoder_error_invalid_request],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_encoder_error_model_invalid],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_encoder_error_backend_error],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime>(&'dispatch EventBindRuntime<'event, 'dispatch>) [bind_encoder_error_unknown],
        "preprocess_decision"_s <= "preprocessing"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / dispatch_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [preprocess_rejected_no_error] / set_backend_error,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [preprocess_reported_error] / set_error_from_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [preprocess_fragment_count_invalid] / set_invalid_request_error_from_preprocess_decision,
        "prefix_decision"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [preprocess_success],
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [bos_ready] / append_bos,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [bos_no_capacity] / set_invalid_request_error_from_prefix_decision,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [bos_invalid_id] / set_invalid_id_error_from_prefix_decision,
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [no_prefix],
        "suffix_decision"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [no_more_fragments],
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [more_fragments_no_capacity] / set_invalid_request_error_from_encoding_ready,
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [more_fragments_token_invalid] / set_invalid_request_error_from_encoding_ready,
        "encoding_token_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [more_fragments_token_valid],
        "encoding_raw_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [more_fragments_raw],
        "encoding_ready"_s <= "encoding_token_fragment"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / append_fragment_token,
        "encoding_raw_decision"_s <= "encoding_raw_fragment"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / dispatch_encode_raw_fragment,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [encode_rejected_no_error] / set_invalid_id_error_from_encoding_raw_decision,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [encode_reported_error] / set_error_from_encode,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [encode_count_invalid] / set_invalid_request_error_from_encoding_raw_decision,
        "encoding_ready"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [encode_success] / commit_encoded_fragment,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [sep_ready] / append_sep,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [sep_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [sep_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [eos_ready] / append_eos,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [eos_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [eos_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) [no_suffix],
        "done"_s <= "finalizing"_s + completion<EventTokenizeRuntime>(&'dispatch EventTokenizeRuntime<'event, 'dispatch>) / finalize,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "binding_preprocessor"_s + unexpected_event<_> / on_unexpected_from_binding_preprocessor,
        "unexpected"_s <= "binding_preprocessor_decision"_s + unexpected_event<_> / on_unexpected_from_binding_preprocessor_decision,
        "unexpected"_s <= "binding_encoder"_s + unexpected_event<_> / on_unexpected_from_binding_encoder,
        "unexpected"_s <= "binding_encoder_decision"_s + unexpected_event<_> / on_unexpected_from_binding_encoder_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "preprocessing"_s + unexpected_event<_> / on_unexpected_from_preprocessing,
        "unexpected"_s <= "preprocess_decision"_s + unexpected_event<_> / on_unexpected_from_preprocess_decision,
        "unexpected"_s <= "prefix_decision"_s + unexpected_event<_> / on_unexpected_from_prefix_decision,
        "unexpected"_s <= "encoding_ready"_s + unexpected_event<_> / on_unexpected_from_encoding_ready,
        "unexpected"_s <= "encoding_token_fragment"_s + unexpected_event<_> / on_unexpected_from_encoding_token_fragment,
        "unexpected"_s <= "encoding_raw_fragment"_s + unexpected_event<_> / on_unexpected_from_encoding_raw_fragment,
        "unexpected"_s <= "encoding_raw_decision"_s + unexpected_event<_> / on_unexpected_from_encoding_raw_decision,
        "unexpected"_s <= "suffix_decision"_s + unexpected_event<_> / on_unexpected_from_suffix_decision,
        "unexpected"_s <= "finalizing"_s + unexpected_event<_> / on_unexpected_from_finalizing,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Persistent binding context owned by the tokenizer actor.
pub struct TextTokenizerContext<'event> {
    /// Identity of the vocabulary bound to this actor, without retaining its borrow.
    pub vocab_identity: Option<usize>,
    _vocab_lifetime: core::marker::PhantomData<&'event ()>,
    pub preprocess_kind: PreprocessorKind,
    pub model_kind: EncoderKind,
    /// Concrete child selected by the bind transition.
    pub preprocessor: Option<PreprocessorSelector>,
    /// Concrete child selected by the bind transition.
    pub encoder: Option<EncoderSelector>,
    pub is_bound: bool,
    pub last_error: TokenizerError,
    pub unexpected: bool,
}
impl<'event> Default for TextTokenizerContext<'event> {
    fn default() -> Self {
        Self {
            vocab_identity: None,
            _vocab_lifetime: core::marker::PhantomData,
            preprocess_kind: PreprocessorKind::Fallback,
            model_kind: EncoderKind::Fallback,
            preprocessor: Some(FALLBACK_PREPROCESSOR),
            encoder: Some(FALLBACK_ENCODER),
            is_bound: false,
            last_error: TokenizerError::None,
            unexpected: false,
        }
    }
}

fn set_token_error<'event, 'dispatch>(
    event: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    error: TokenizerError,
) {
    let mut c = event.context.borrow_mut();
    c.err = error;
    c.result = false;
    c.token_count = 0;
}
fn set_bind_error<'event, 'dispatch>(
    event: &'dispatch EventBindRuntime<'event, 'dispatch>,
    error: TokenizerError,
) {
    let mut c = event.context.borrow_mut();
    c.err = error;
    c.result = false;
}
fn valid_token_id(id: i32) -> bool {
    id >= 0
}
fn child_error(error: TokenizerError) -> bool {
    error != TokenizerError::None
}
fn supported_preprocessor(kind: PreprocessorKind) -> bool {
    matches!(
        kind,
        PreprocessorKind::Spm
            | PreprocessorKind::Bpe
            | PreprocessorKind::Wpm
            | PreprocessorKind::Ugm
            | PreprocessorKind::Rwkv
            | PreprocessorKind::Plamo2
            | PreprocessorKind::Fallback
    )
}
fn supported_encoder(kind: EncoderKind) -> bool {
    matches!(
        kind,
        EncoderKind::Spm
            | EncoderKind::Bpe
            | EncoderKind::Wpm
            | EncoderKind::Ugm
            | EncoderKind::Rwkv
            | EncoderKind::Plamo2
            | EncoderKind::Fallback
    )
}
impl<'ctx> TextTokenizerContext<'ctx> {
    fn begin_bind(
        &mut self,
        preprocessor_variant: PreprocessorKind,
        encoder_variant: EncoderKind,
        context: &RefCell<BindContext>,
    ) -> Result<(), ()> {
        self.preprocess_kind = preprocessor_variant;
        self.model_kind = encoder_variant;
        self.preprocessor = None;
        self.encoder = None;
        self.is_bound = false;
        self.unexpected = false;
        self.last_error = TokenizerError::None;
        context.borrow_mut().err = TokenizerError::None;
        Ok(())
    }
    fn begin_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = e.context.borrow_mut();
        *c = TokenizeContext::default();
        self.last_error = TokenizerError::None;
        Ok(())
    }
    fn reject_bind<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.is_bound = false;
        self.last_error = TokenizerError::InvalidRequest;
        set_bind_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn reject_invalid<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.last_error = TokenizerError::InvalidRequest;
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    #[allow(clippy::unused_self, clippy::needless_pass_by_ref_mut)]
    fn append_id<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
        id: i32,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = e.context.borrow_mut();
        let mut o = e.request.token_ids.borrow_mut();
        if c.token_count >= o.len() {
            c.err = TokenizerError::InvalidRequest;
            c.result = false;
            c.token_count = 0;
            return Ok(());
        }
        o[c.token_count] = id;
        c.token_count += 1;
        Ok(())
    }
    fn mark_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.is_bound = false;
        self.last_error = TokenizerError::Unexpected;
        Ok(())
    }
}
impl<'ctx> TextTokenizerStateMachineContext for TextTokenizerContext<'ctx> {
    fn begin_bind_from_uninitialized<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_bind(
            e.request.preprocessor_variant,
            e.request.encoder_variant,
            e.context,
        )
    }
    fn begin_bind_from_idle<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_bind(
            e.request.preprocessor_variant,
            e.request.encoder_variant,
            e.context,
        )
    }
    fn begin_bind_from_done<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_bind(
            e.request.preprocessor_variant,
            e.request.encoder_variant,
            e.context,
        )
    }
    fn begin_bind_from_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_bind(
            e.request.preprocessor_variant,
            e.request.encoder_variant,
            e.context,
        )
    }
    fn begin_bind_from_unexpected<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_bind(
            e.request.preprocessor_variant,
            e.request.encoder_variant,
            e.context,
        )
    }
    fn begin_tokenize_from_idle<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_tokenize(e)
    }
    fn begin_tokenize_from_done<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_tokenize(e)
    }
    fn begin_tokenize_from_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_tokenize(e)
    }
    fn begin_tokenize_from_unexpected<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_tokenize(e)
    }
    fn can_bind<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let preprocessor_ready = e
            .request
            .preprocessor_adapter
            .is_some_and(|adapter| adapter.kind == e.request.preprocessor_variant)
            || (e.request.preprocessor_variant == PreprocessorKind::Fallback
                && e.request.preprocessor_adapter.is_none());
        let encoder_ready = e
            .request
            .encoder_adapter
            .is_some_and(|adapter| adapter.kind == e.request.encoder_variant)
            || (e.request.encoder_variant == EncoderKind::Fallback
                && e.request.encoder_adapter.is_none());
        Ok(supported_preprocessor(e.request.preprocessor_variant)
            && supported_encoder(e.request.encoder_variant)
            && preprocessor_ready
            && encoder_ready)
    }
    fn can_tokenize<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.is_bound
            && self.preprocessor.is_some()
            && self.encoder.is_some()
            && self.vocab_identity
                == Some(core::ptr::from_ref(e.request.vocab).cast::<()>() as usize)
            && !e.request.token_ids.borrow().is_empty())
    }
    fn append_bos<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.append_id(e, e.request.vocab.bos_id())
    }
    fn append_sep<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.append_id(e, e.request.vocab.sep_id())
    }
    fn append_eos<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.append_id(e, e.request.vocab.eos_id())
    }
    fn bind_preprocessor<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let selected = e.request.preprocessor_adapter.or_else(|| {
            (e.request.preprocessor_variant == PreprocessorKind::Fallback).then_some(
                match FALLBACK_PREPROCESSOR {
                    PreprocessorSelector::Fallback(adapter) => adapter,
                    _ => unreachable!(),
                },
            )
        });
        self.preprocessor = selected.and_then(PreprocessorSelector::from_adapter);
        let error = selected.map_or(TokenizerError::InvalidRequest, |adapter| {
            (adapter.bind)(e.request.vocab, adapter.kind as u8)
        });
        e.context.borrow_mut().err = error;
        Ok(())
    }
    fn bind_encoder<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let selected = e.request.encoder_adapter.or_else(|| {
            (e.request.encoder_variant == EncoderKind::Fallback).then_some(match FALLBACK_ENCODER {
                EncoderSelector::Fallback(adapter) => adapter,
                _ => unreachable!(),
            })
        });
        self.encoder = selected.and_then(EncoderSelector::from_adapter);
        let error = selected.map_or(TokenizerError::InvalidRequest, |adapter| {
            (adapter.bind)(e.request.vocab, adapter.kind as u8)
        });
        e.context.borrow_mut().err = error;
        Ok(())
    }
    fn mark_bind_success<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.is_bound = true;
        self.last_error = TokenizerError::None;
        e.context.borrow_mut().result = true;
        Ok(())
    }
    fn reject_bind_from_uninitialized<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_bind(e)
    }
    fn reject_bind_from_idle<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_bind(e)
    }
    fn reject_bind_from_done<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_bind(e)
    }
    fn reject_bind_from_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_bind(e)
    }
    fn reject_bind_from_unexpected<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_bind(e)
    }
    fn reject_invalid_from_uninitialized<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_invalid(e)
    }
    fn reject_invalid_from_idle<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_invalid(e)
    }
    fn reject_invalid_from_done<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_invalid(e)
    }
    fn reject_invalid_from_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_invalid(e)
    }
    fn reject_invalid_from_unexpected<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_invalid(e)
    }
    fn dispatch_preprocess<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = e.context.borrow_mut();
        let Some(
            PreprocessorSelector::Spm(adapter)
            | PreprocessorSelector::Bpe(adapter)
            | PreprocessorSelector::Wpm(adapter)
            | PreprocessorSelector::Ugm(adapter)
            | PreprocessorSelector::Rwkv(adapter)
            | PreprocessorSelector::Plamo2(adapter)
            | PreprocessorSelector::Fallback(adapter),
        ) = self.preprocessor
        else {
            c.preprocess_accepted = false;
            c.preprocess_err_code = TokenizerError::BackendError;
            c.fragment_count = 0;
            c.fragment_index = 0;
            return Ok(());
        };
        let r = (adapter.preprocess)(
            e.request.vocab,
            e.request.text,
            e.request.parse_special,
            &mut c.fragments,
        );
        c.preprocess_accepted = r.accepted;
        c.preprocess_err_code = r.error;
        c.fragment_count = r.count;
        c.fragment_index = 0;
        c.preprocessed = r.accepted && r.error == TokenizerError::None;
        self.last_error = if r.accepted && r.error == TokenizerError::None {
            TokenizerError::None
        } else if r.error == TokenizerError::None {
            TokenizerError::BackendError
        } else {
            r.error
        };
        Ok(())
    }
    fn dispatch_encode_raw_fragment<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = e.context.borrow_mut();
        let preprocessed = c.preprocessed;
        if c.fragment_index >= c.fragment_count {
            c.encode_accepted = false;
            c.encode_err_code = TokenizerError::InvalidRequest;
            c.encode_token_count = 0;
            return Ok(());
        }
        let f = c.fragments[c.fragment_index];
        let bytes = if f.end <= e.request.text.len() && f.start <= f.end {
            &e.request.text[f.start..f.end]
        } else {
            c.encode_accepted = false;
            c.encode_err_code = TokenizerError::InvalidRequest;
            c.encode_token_count = 0;
            return Ok(());
        };
        let start = c.token_count;
        drop(c);
        let remaining = {
            let output = e.request.token_ids.borrow();
            output.len().saturating_sub(start)
        };
        let mut output = e.request.token_ids.borrow_mut();
        let target = &mut output[start..start + remaining];
        let r = match self.encoder {
            Some(
                EncoderSelector::Spm(adapter)
                | EncoderSelector::Bpe(adapter)
                | EncoderSelector::Wpm(adapter)
                | EncoderSelector::Ugm(adapter)
                | EncoderSelector::Rwkv(adapter)
                | EncoderSelector::Plamo2(adapter)
                | EncoderSelector::Fallback(adapter),
            ) => (adapter.encode)(e.request.vocab, bytes, preprocessed, target),
            None => ChildResult {
                accepted: false,
                error: TokenizerError::BackendError,
                count: 0,
            },
        };
        drop(output);
        let mut c = e.context.borrow_mut();
        c.encode_accepted = r.accepted;
        c.encode_err_code = r.error;
        c.encode_token_count = r.count;
        self.last_error = if r.accepted && r.error == TokenizerError::None {
            TokenizerError::None
        } else if r.error == TokenizerError::None {
            TokenizerError::BackendError
        } else {
            r.error
        };
        Ok(())
    }
    fn append_fragment_token<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let (index, id) = {
            let c = e.context.borrow();
            (c.token_count, c.fragments[c.fragment_index].token)
        };
        let mut out = e.request.token_ids.borrow_mut();
        if index >= out.len() {
            drop(out);
            set_token_error(e, TokenizerError::InvalidRequest);
            return Ok(());
        }
        out[index] = id;
        drop(out);
        let mut c = e.context.borrow_mut();
        c.token_count += 1;
        c.fragment_index += 1;
        Ok(())
    }
    fn commit_encoded_fragment<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = e.context.borrow_mut();
        let capacity = e.request.token_ids.borrow().len();
        if c.encode_token_count > capacity.saturating_sub(c.token_count) {
            c.err = TokenizerError::InvalidRequest;
            c.result = false;
            c.token_count = 0;
            return Ok(());
        }
        c.token_count += c.encode_token_count;
        c.fragment_index += 1;
        Ok(())
    }
    fn finalize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        if c.err != TokenizerError::None
            || !c.preprocess_accepted
            || c.fragment_index < c.fragment_count
        {
            drop(c);
            set_token_error(e, TokenizerError::BackendError);
            return Ok(());
        }
        drop(c);
        e.context.borrow_mut().result = true;
        self.last_error = TokenizerError::None;
        Ok(())
    }
    fn set_backend_error<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::BackendError);
        Ok(())
    }
    fn set_error_from_preprocess<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let x = e.context.borrow().preprocess_err_code;
        set_token_error(
            e,
            if x == TokenizerError::None {
                TokenizerError::BackendError
            } else {
                x
            },
        );
        Ok(())
    }
    fn set_error_from_encode<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let x = e.context.borrow().encode_err_code;
        set_token_error(
            e,
            if x == TokenizerError::None {
                TokenizerError::BackendError
            } else {
                x
            },
        );
        Ok(())
    }
    fn set_invalid_request_error_from_preprocess_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn set_invalid_request_error_from_prefix_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn set_invalid_request_error_from_encoding_ready<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn set_invalid_request_error_from_encoding_raw_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn set_invalid_request_error_from_suffix_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::InvalidRequest);
        Ok(())
    }
    fn set_invalid_id_error_from_prefix_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::ModelInvalid);
        Ok(())
    }
    fn set_invalid_id_error_from_encoding_raw_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::ModelInvalid);
        Ok(())
    }
    fn set_invalid_id_error_from_suffix_decision<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_token_error(e, TokenizerError::ModelInvalid);
        Ok(())
    }
    fn bind_preprocessor_error_none<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().err == TokenizerError::None)
    }
    fn bind_preprocessor_error_invalid_request<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().err == TokenizerError::InvalidRequest)
    }
    fn bind_preprocessor_error_model_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().err == TokenizerError::ModelInvalid)
    }
    fn bind_preprocessor_error_backend_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().err == TokenizerError::BackendError)
    }
    fn bind_preprocessor_error_unknown<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let error = e.context.borrow().err;
        Ok(child_error(error)
            && !matches!(
                error,
                TokenizerError::InvalidRequest
                    | TokenizerError::ModelInvalid
                    | TokenizerError::BackendError
            ))
    }
    fn bind_encoder_error_none<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        self.bind_preprocessor_error_none(e)
    }
    fn bind_encoder_error_invalid_request<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        self.bind_preprocessor_error_invalid_request(e)
    }
    fn bind_encoder_error_model_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        self.bind_preprocessor_error_model_invalid(e)
    }
    fn bind_encoder_error_backend_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        self.bind_preprocessor_error_backend_error(e)
    }
    fn bind_encoder_error_unknown<'dispatch, 'event>(
        &self,
        e: &'dispatch EventBindRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        self.bind_preprocessor_error_unknown(e)
    }
    fn preprocess_rejected_no_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(!c.preprocess_accepted && c.preprocess_err_code == TokenizerError::None)
    }
    fn preprocess_reported_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().preprocess_err_code != TokenizerError::None)
    }
    fn preprocess_fragment_count_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.preprocess_accepted
            && c.preprocess_err_code == TokenizerError::None
            && c.fragment_count > MAX_FRAGMENTS)
    }
    fn preprocess_success<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.preprocess_accepted
            && c.preprocess_err_code == TokenizerError::None
            && c.fragment_count <= MAX_FRAGMENTS)
    }
    fn bos_ready<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && e.request.vocab.add_bos()
            && valid_token_id(e.request.vocab.bos_id())
            && e.context.borrow().token_count < e.request.token_ids.borrow().len())
    }
    fn bos_no_capacity<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && e.request.vocab.add_bos()
            && valid_token_id(e.request.vocab.bos_id())
            && e.context.borrow().token_count >= e.request.token_ids.borrow().len())
    }
    fn bos_invalid_id<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && e.request.vocab.add_bos()
            && !valid_token_id(e.request.vocab.bos_id()))
    }
    fn no_prefix<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!e.request.add_special || !e.request.vocab.add_bos())
    }
    fn no_more_fragments<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.fragment_index >= c.fragment_count)
    }
    fn more_fragments_no_capacity<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.fragment_index < c.fragment_count
            && c.token_count >= e.request.token_ids.borrow().len())
    }
    fn more_fragments_token_valid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.fragment_index < c.fragment_count
            && c.fragments[c.fragment_index].kind == FragmentKind::Token
            && c.fragments[c.fragment_index].token >= 0)
    }
    fn more_fragments_token_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.fragment_index < c.fragment_count
            && c.fragments[c.fragment_index].kind == FragmentKind::Token
            && c.fragments[c.fragment_index].token < 0)
    }
    fn more_fragments_raw<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.fragment_index < c.fragment_count
            && c.fragments[c.fragment_index].kind == FragmentKind::RawText)
    }
    fn encode_rejected_no_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(!c.encode_accepted && c.encode_err_code == TokenizerError::None)
    }
    fn encode_reported_error<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.context.borrow().encode_err_code != TokenizerError::None)
    }
    fn encode_count_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.encode_accepted
            && c.encode_err_code == TokenizerError::None
            && c.encode_token_count
                > e.request
                    .token_ids
                    .borrow()
                    .len()
                    .saturating_sub(c.token_count))
    }
    fn encode_success<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = e.context.borrow();
        Ok(c.encode_accepted
            && c.encode_err_code == TokenizerError::None
            && c.encode_token_count
                <= e.request
                    .token_ids
                    .borrow()
                    .len()
                    .saturating_sub(c.token_count))
    }
    fn sep_ready<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind == EncoderKind::Wpm
            && e.request.vocab.add_sep()
            && valid_token_id(e.request.vocab.sep_id())
            && e.context.borrow().token_count < e.request.token_ids.borrow().len())
    }
    fn sep_no_capacity<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind == EncoderKind::Wpm
            && e.request.vocab.add_sep()
            && valid_token_id(e.request.vocab.sep_id())
            && e.context.borrow().token_count >= e.request.token_ids.borrow().len())
    }
    fn sep_invalid_id<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind == EncoderKind::Wpm
            && e.request.vocab.add_sep()
            && !valid_token_id(e.request.vocab.sep_id()))
    }
    fn eos_ready<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind != EncoderKind::Wpm
            && e.request.vocab.add_eos()
            && valid_token_id(e.request.vocab.eos_id())
            && e.context.borrow().token_count < e.request.token_ids.borrow().len())
    }
    fn eos_no_capacity<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind != EncoderKind::Wpm
            && e.request.vocab.add_eos()
            && valid_token_id(e.request.vocab.eos_id())
            && e.context.borrow().token_count >= e.request.token_ids.borrow().len())
    }
    fn eos_invalid_id<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.request.add_special
            && self.model_kind != EncoderKind::Wpm
            && e.request.vocab.add_eos()
            && !valid_token_id(e.request.vocab.eos_id()))
    }
    fn no_suffix<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenizeRuntime<'event, 'dispatch>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!((e.request.add_special
            && self.model_kind == EncoderKind::Wpm
            && e.request.vocab.add_sep())
            || (e.request.add_special
                && self.model_kind != EncoderKind::Wpm
                && e.request.vocab.add_eos())))
    }
    fn on_unexpected_from_binding_encoder_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_binding_encoder(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_binding_preprocessor_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_binding_preprocessor(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_prefix_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_preprocess_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_preprocessing(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_suffix_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_encoding_ready(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_encoding_token_fragment(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_encoding_raw_fragment(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_encoding_raw_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
    fn on_unexpected_from_finalizing(&mut self) -> Result<(), ()> {
        self.mark_unexpected()
    }
}

pub struct TextTokenizer<'event> {
    machine: TextTokenizerStateMachine<TextTokenizerContext<'event>>,
}
impl<'event> Default for TextTokenizer<'event> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'event> TextTokenizer<'event> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerStateMachine::new(TextTokenizerContext::default()),
        }
    }
    pub fn process_bind(
        &mut self,
        request: BindRequest<'_>,
    ) -> Result<TokenizerBindDone, TokenizerBindError> {
        self.machine.context_mut().vocab_identity =
            Some(core::ptr::from_ref(request.vocab).cast::<()>() as usize);
        let ctx = RefCell::new(BindContext::default());
        let event = EventBindRuntime {
            request,
            context: &ctx,
        };
        if self
            .machine
            .process_event(TextTokenizerEvents::EventBindRuntime(&event))
            .is_err()
        {
            self.machine.context_mut().unexpected = true;
            self.machine.context_mut().last_error = TokenizerError::Unexpected;
            self.machine.set_state(TextTokenizerStates::Unexpected);
            let error = TokenizerBindError {
                error: TokenizerError::Unexpected,
            };
            if let Some(cb) = request.dispatch_error {
                let _ = cb(error);
            }
            return Err(error);
        }
        let c = *ctx.borrow();
        if c.result && self.machine.is(&TextTokenizerStates::Idle) {
            if let Some(cb) = request.dispatch_done {
                let _ = cb(TokenizerBindDone);
            }
            Ok(TokenizerBindDone)
        } else {
            let error = TokenizerBindError {
                error: if c.err == TokenizerError::None {
                    TokenizerError::BackendError
                } else {
                    c.err
                },
            };
            self.machine.context_mut().last_error = error.error;
            self.machine.set_state(TextTokenizerStates::Errored);
            if let Some(cb) = request.dispatch_error {
                let _ = cb(error);
            }
            Err(error)
        }
    }
    pub fn process_tokenize(
        &mut self,
        request: TokenizeRequest<'_>,
    ) -> Result<TokenizerDone, TokenizerErrorEvent> {
        let ctx = RefCell::new(TokenizeContext::default());
        let event = EventTokenizeRuntime {
            request,
            context: &ctx,
        };
        if self
            .machine
            .process_event(TextTokenizerEvents::EventTokenizeRuntime(&event))
            .is_err()
        {
            self.machine.context_mut().last_error = TokenizerError::Unexpected;
            self.machine.set_state(TextTokenizerStates::Unexpected);
            let error = TokenizerErrorEvent {
                error: TokenizerError::Unexpected,
            };
            if let Some(cb) = request.dispatch_error {
                let _ = cb(error);
            }
            return Err(error);
        }
        let c = *ctx.borrow();
        if c.result && self.machine.is(&TextTokenizerStates::Done) {
            let d = TokenizerDone {
                token_count: c.token_count,
            };
            if let Some(cb) = request.dispatch_done {
                let _ = cb(d);
            }
            Ok(d)
        } else {
            let e = TokenizerErrorEvent {
                error: if self.machine.context().unexpected {
                    TokenizerError::Unexpected
                } else if c.err == TokenizerError::None {
                    TokenizerError::BackendError
                } else {
                    c.err
                },
            };
            if let Some(cb) = request.dispatch_error {
                let _ = cb(e);
            }
            Err(e)
        }
    }
    pub fn process_unexpected(&mut self) -> bool {
        self.machine.context_mut().unexpected = true;
        self.machine.context_mut().is_bound = false;
        self.machine.context_mut().last_error = TokenizerError::Unexpected;
        self.machine.set_state(TextTokenizerStates::Unexpected);
        false
    }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerContext<'event> {
        self.machine.context()
    }
    #[must_use]
    pub fn last_error(&self) -> TokenizerError {
        self.machine.context().last_error
    }
    /// Tokenizes caller-owned bytes and output synchronously after binding.
    pub fn tokenize_into(
        &mut self,
        vocab: &dyn VocabularyView,
        text: &[u8],
        add_special: bool,
        parse_special: bool,
        token_ids: &mut [i32],
    ) -> Result<usize, TokenizerError> {
        let ids = RefCell::new(token_ids);
        self.process_tokenize(TokenizeRequest::with_callbacks(
            vocab,
            text,
            &ids,
            add_special,
            parse_special,
            None,
            None,
            None,
            None,
        ))
        .map(|done| done.token_count)
        .map_err(|error| error.error)
    }
}
/// Compatibility actor alias.
pub type TextTokenizerActor<'event> = TextTokenizer<'event>;

#[cfg(test)]
mod tests {
    use super::*;

    struct Vocabulary;
    impl VocabularyView for Vocabulary {}

    fn bind_ok(_: &dyn VocabularyView, _: u8) -> TokenizerError {
        TokenizerError::None
    }

    fn selected_preprocess(
        _: &dyn VocabularyView,
        text: &[u8],
        _: bool,
        fragments: &mut [Fragment],
    ) -> ChildResult {
        if text.is_empty() || fragments.is_empty() {
            return ChildResult {
                accepted: false,
                error: TokenizerError::InvalidRequest,
                count: 0,
            };
        }
        fragments[0] = Fragment {
            kind: FragmentKind::RawText,
            start: 0,
            end: text.len(),
            token: -1,
        };
        ChildResult {
            accepted: true,
            error: TokenizerError::None,
            count: 1,
        }
    }

    fn selected_encode(
        _: &dyn VocabularyView,
        _: &[u8],
        _: bool,
        output: &mut [i32],
    ) -> ChildResult {
        if output.is_empty() {
            return ChildResult {
                accepted: false,
                error: TokenizerError::InvalidRequest,
                count: 0,
            };
        }
        output[0] = 77;
        ChildResult {
            accepted: true,
            error: TokenizerError::None,
            count: 1,
        }
    }

    fn done(_: TokenizerBindDone) -> bool {
        true
    }
    fn bind_error(_: TokenizerBindError) -> bool {
        true
    }

    #[test]
    fn non_fallback_kind_requires_maintained_adapter() {
        let vocab = Vocabulary;
        let mut tokenizer = TextTokenizer::new();
        let result = tokenizer.process_bind(BindRequest::with_callbacks(
            &vocab,
            PreprocessorKind::Bpe,
            EncoderKind::Bpe,
            None,
            None,
            None,
            None,
        ));
        assert_eq!(
            result,
            Err(TokenizerBindError {
                error: TokenizerError::InvalidRequest
            })
        );
        assert!(!tokenizer.context().is_bound);
    }

    #[test]
    fn selected_child_identity_survives_bind_and_tokenize() {
        let vocab = Vocabulary;
        let preprocessor = PreprocessorAdapter {
            kind: PreprocessorKind::Wpm,
            bind: bind_ok,
            preprocess: selected_preprocess,
        };
        let encoder = EncoderAdapter {
            kind: EncoderKind::Wpm,
            bind: bind_ok,
            encode: selected_encode,
        };
        let mut tokenizer = TextTokenizer::new();
        assert!(
            tokenizer
                .process_bind(BindRequest::with_adapters(
                    &vocab,
                    PreprocessorKind::Wpm,
                    EncoderKind::Wpm,
                    Some(preprocessor),
                    Some(encoder),
                    done,
                    bind_error,
                ))
                .is_ok()
        );
        assert_eq!(tokenizer.context().preprocess_kind, PreprocessorKind::Wpm);
        assert_eq!(tokenizer.context().model_kind, EncoderKind::Wpm);
        assert_eq!(
            tokenizer
                .context()
                .preprocessor
                .map(PreprocessorSelector::kind),
            Some(PreprocessorKind::Wpm)
        );
        assert_eq!(
            tokenizer.context().encoder.map(EncoderSelector::kind),
            Some(EncoderKind::Wpm)
        );
        let mut output = [0; 2];
        let ids = RefCell::new(&mut output[..]);
        let done = tokenizer
            .process_tokenize(TokenizeRequest::with_callbacks(
                &vocab, b"x", &ids, false, false, None, None, None, None,
            ))
            .expect("selected adapters dispatch synchronously");
        assert_eq!(done.token_count, 1);
        assert_eq!(output[0], 77);
    }
    #[test]
    fn fallback_resolves_bytes_through_bound_vocabulary_and_rejects_missing_bytes() {
        struct ByteVocabulary;
        impl VocabularyView for ByteVocabulary {
            fn token_count(&self) -> usize {
                2
            }

            fn token(&self, index: usize) -> Option<&[u8]> {
                [b"a".as_slice(), b"b".as_slice()].get(index).copied()
            }
        }

        let vocab = ByteVocabulary;

        let mut output = [0; 2];
        {
            let mut tokenizer = TextTokenizer::new();
            tokenizer
                .process_bind(BindRequest::with_callbacks(
                    &vocab,
                    PreprocessorKind::Fallback,
                    EncoderKind::Fallback,
                    None,
                    None,
                    None,
                    None,
                ))
                .expect("fallback bind succeeds with a bounded vocabulary");
            let ids = RefCell::new(&mut output[..]);
            let done = tokenizer
                .process_tokenize(TokenizeRequest::with_callbacks(
                    &vocab, b"ab", &ids, false, false, None, None, None, None,
                ))
                .expect("known bytes resolve through vocabulary");
            assert_eq!(done.token_count, 2);
        }
        assert_eq!(output, [0, 1]);

        let mut missing_output = [0; 1];
        let missing_result = {
            let mut tokenizer = TextTokenizer::new();
            tokenizer
                .process_bind(BindRequest::with_callbacks(
                    &vocab,
                    PreprocessorKind::Fallback,
                    EncoderKind::Fallback,
                    None,
                    None,
                    None,
                    None,
                ))
                .expect("fallback bind succeeds with a bounded vocabulary");
            let missing_ids = RefCell::new(&mut missing_output[..]);
            tokenizer.process_tokenize(TokenizeRequest::with_callbacks(
                &vocab,
                b"c",
                &missing_ids,
                false,
                false,
                None,
                None,
                None,
                None,
            ))
        };
        assert_eq!(
            missing_result,
            Err(TokenizerErrorEvent {
                error: TokenizerError::BackendError
            })
        );
    }
}
