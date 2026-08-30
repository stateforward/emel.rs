//! Source-aligned, bounded text tokenizer state machine.
//!
//! The machine follows the maintained tokenizer contract: binding selects the
//! preprocessor and encoder children, tokenization preprocesses one request,
//! emits the optional prefix, processes bounded fragments, emits the suffix,
//! and publishes a synchronous result.  Child calls are function-pointer
//! callbacks so the actor remains allocation-free and single-writer.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
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
pub enum PreprocessorKind { Spm = 0, Bpe = 1, Wpm = 2, Ugm = 3, Rwkv = 4, Plamo2 = 5, #[default] Fallback = 6 }

/// Supported encoder variants.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum EncoderKind { Spm = 0, Bpe = 1, Wpm = 2, Ugm = 3, Rwkv = 4, Plamo2 = 5, #[default] Fallback = 6 }

/// Public tokenizer error. Numeric values mirror `text/tokenizer/errors.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum TokenizerError {
    #[default] None = 0,
    InvalidRequest = 1,
    ModelInvalid = 2,
    BackendError = 4,
    /// Internal sequencing failure; public result maps it to invalid request.
    Unexpected = 8,
}
impl TokenizerError {
    #[must_use]
    pub const fn code(self) -> i32 { self as i32 }
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
pub enum FragmentKind { #[default] RawText = 0, Token = 1 }

/// Vocabulary information needed by prefix/suffix handling and fallback
/// child implementations. Implementations must return stable token slices.
pub trait VocabularyView {
    fn token_count(&self) -> usize { 0 }
    fn token(&self, _index: usize) -> Option<&[u8]> { None }
    fn bos_id(&self) -> i32 { -1 }
    fn eos_id(&self) -> i32 { -1 }
    fn sep_id(&self) -> i32 { -1 }
    fn add_bos(&self) -> bool { false }
    fn add_eos(&self) -> bool { false }
    fn add_sep(&self) -> bool { false }
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

/// Completion payload for a successful bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizerBindDone;
/// Completion payload for a failed bind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerBindError { pub error: TokenizerError }
/// Completion payload for successful tokenization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerDone { pub token_count: usize }
/// Completion payload for failed tokenization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerErrorEvent { pub error: TokenizerError }

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
    pub bind_preprocessor: Option<BindCallback>,
    pub bind_encoder: Option<BindCallback>,
    pub dispatch_done: Option<BindDoneCallback>,
    pub dispatch_error: Option<BindErrorCallback>,
}
impl<'event> BindRequest<'event> {
    #[must_use]
    pub const fn new(vocab: &'event dyn VocabularyView, preprocessor_variant: PreprocessorKind, encoder_variant: EncoderKind, dispatch_done: BindDoneCallback, dispatch_error: BindErrorCallback) -> Self {
        Self { vocab, preprocessor_variant, encoder_variant, bind_preprocessor: None, bind_encoder: None, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }
    #[must_use]
    pub const fn with_callbacks(vocab: &'event dyn VocabularyView, preprocessor_variant: PreprocessorKind, encoder_variant: EncoderKind, bind_preprocessor: Option<BindCallback>, bind_encoder: Option<BindCallback>, dispatch_done: Option<BindDoneCallback>, dispatch_error: Option<BindErrorCallback>) -> Self {
        Self { vocab, preprocessor_variant, encoder_variant, bind_preprocessor, bind_encoder, dispatch_done, dispatch_error }
    }
}
impl core::fmt::Debug for BindRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("BindRequest").field("preprocessor_variant", &self.preprocessor_variant).field("encoder_variant", &self.encoder_variant).finish_non_exhaustive() }
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
    pub const fn new(vocab: &'event dyn VocabularyView, text: &'event [u8], token_ids: &'event RefCell<&'event mut [i32]>, dispatch_done: TokenizeDoneCallback, dispatch_error: TokenizeErrorCallback) -> Self {
        Self { vocab, text, add_special: false, parse_special: false, token_ids, preprocess: None, encode: None, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }
    #[must_use]
    pub const fn with_callbacks(vocab: &'event dyn VocabularyView, text: &'event [u8], token_ids: &'event RefCell<&'event mut [i32]>, add_special: bool, parse_special: bool, preprocess: Option<PreprocessCallback>, encode: Option<EncodeCallback>, dispatch_done: Option<TokenizeDoneCallback>, dispatch_error: Option<TokenizeErrorCallback>) -> Self {
        Self { vocab, text, add_special, parse_special, token_ids, preprocess, encode, dispatch_done, dispatch_error }
    }
}
impl core::fmt::Debug for TokenizeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("TokenizeRequest").field("text_length", &self.text.len()).field("add_special", &self.add_special).field("parse_special", &self.parse_special).field("token_capacity", &self.token_ids.borrow().len()).finish_non_exhaustive() }
}

/// Runtime bind event. The alias remains usable without spelling its lifetime.
#[derive(Clone, Copy, Debug)]
pub struct EventBindRuntime<'event> { pub request: BindRequest<'event>, pub context: &'event RefCell<BindContext> }
/// Runtime tokenization event.
#[derive(Clone, Copy, Debug)]
pub struct EventTokenizeRuntime<'event> { pub request: TokenizeRequest<'event>, pub context: &'event RefCell<TokenizeContext> }

/// Bind operation context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindContext { pub err: TokenizerError, pub result: bool }
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
    fn default() -> Self { Self { fragments: [Fragment::default(); MAX_FRAGMENTS], fragment_count: 0, fragment_index: 0, preprocessed: false, preprocess_accepted: false, preprocess_err_code: TokenizerError::None, encode_accepted: false, encode_err_code: TokenizerError::None, encode_token_count: 0, token_count: 0, err: TokenizerError::None, result: false, unexpected: false } }
}

sml! {
    TextTokenizer<'event> {
        "binding_preprocessor"_s <= *"uninitialized"_s + event<EventBindRuntime<'event>> [can_bind] / begin_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventBindRuntime<'event>> / reject_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventTokenizeRuntime<'event>> / reject_invalid_from_uninitialized,
        "binding_preprocessor"_s <= "idle"_s + event<EventBindRuntime<'event>> [can_bind] / begin_bind_from_idle,
        "errored"_s <= "idle"_s + event<EventBindRuntime<'event>> / reject_bind_from_idle,
        "preprocessing"_s <= "idle"_s + event<EventTokenizeRuntime<'event>> [can_tokenize] / begin_tokenize_from_idle,
        "errored"_s <= "idle"_s + event<EventTokenizeRuntime<'event>> / reject_invalid_from_idle,
        "binding_preprocessor"_s <= "done"_s + event<EventBindRuntime<'event>> [can_bind] / begin_bind_from_done,
        "errored"_s <= "done"_s + event<EventBindRuntime<'event>> / reject_bind_from_done,
        "preprocessing"_s <= "done"_s + event<EventTokenizeRuntime<'event>> [can_tokenize] / begin_tokenize_from_done,
        "errored"_s <= "done"_s + event<EventTokenizeRuntime<'event>> / reject_invalid_from_done,
        "binding_preprocessor"_s <= "errored"_s + event<EventBindRuntime<'event>> [can_bind] / begin_bind_from_errored,
        "errored"_s <= "errored"_s + event<EventBindRuntime<'event>> / reject_bind_from_errored,
        "preprocessing"_s <= "errored"_s + event<EventTokenizeRuntime<'event>> [can_tokenize] / begin_tokenize_from_errored,
        "errored"_s <= "errored"_s + event<EventTokenizeRuntime<'event>> / reject_invalid_from_errored,
        "binding_preprocessor"_s <= "unexpected"_s + event<EventBindRuntime<'event>> [can_bind] / begin_bind_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventBindRuntime<'event>> / reject_bind_from_unexpected,
        "preprocessing"_s <= "unexpected"_s + event<EventTokenizeRuntime<'event>> [can_tokenize] / begin_tokenize_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventTokenizeRuntime<'event>> / reject_invalid_from_unexpected,
        "binding_preprocessor_decision"_s <= "binding_preprocessor"_s + completion<EventBindRuntime<'event>> / bind_preprocessor,
        "binding_encoder"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime<'event>> [bind_preprocessor_error_none],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime<'event>> [bind_preprocessor_error_invalid_request],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime<'event>> [bind_preprocessor_error_model_invalid],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime<'event>> [bind_preprocessor_error_backend_error],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime<'event>> [bind_preprocessor_error_unknown],
        "binding_encoder_decision"_s <= "binding_encoder"_s + completion<EventBindRuntime<'event>> / bind_encoder,
        "idle"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime<'event>> [bind_encoder_error_none] / mark_bind_success,
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime<'event>> [bind_encoder_error_invalid_request],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime<'event>> [bind_encoder_error_model_invalid],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime<'event>> [bind_encoder_error_backend_error],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime<'event>> [bind_encoder_error_unknown],
        "preprocess_decision"_s <= "preprocessing"_s + completion<EventTokenizeRuntime<'event>> / dispatch_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime<'event>> [preprocess_rejected_no_error] / set_backend_error,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime<'event>> [preprocess_reported_error] / set_error_from_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime<'event>> [preprocess_fragment_count_invalid] / set_invalid_request_error_from_preprocess_decision,
        "prefix_decision"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime<'event>> [preprocess_success],
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime<'event>> [bos_ready] / append_bos,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime<'event>> [bos_no_capacity] / set_invalid_request_error_from_prefix_decision,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime<'event>> [bos_invalid_id] / set_invalid_id_error_from_prefix_decision,
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime<'event>> [no_prefix],
        "suffix_decision"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime<'event>> [no_more_fragments],
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime<'event>> [more_fragments_no_capacity] / set_invalid_request_error_from_encoding_ready,
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime<'event>> [more_fragments_token_invalid] / set_invalid_request_error_from_encoding_ready,
        "encoding_token_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime<'event>> [more_fragments_token_valid],
        "encoding_raw_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime<'event>> [more_fragments_raw],
        "encoding_ready"_s <= "encoding_token_fragment"_s + completion<EventTokenizeRuntime<'event>> / append_fragment_token,
        "encoding_raw_decision"_s <= "encoding_raw_fragment"_s + completion<EventTokenizeRuntime<'event>> / dispatch_encode_raw_fragment,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime<'event>> [encode_rejected_no_error] / set_invalid_id_error_from_encoding_raw_decision,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime<'event>> [encode_reported_error] / set_error_from_encode,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime<'event>> [encode_count_invalid] / set_invalid_request_error_from_encoding_raw_decision,
        "encoding_ready"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime<'event>> [encode_success] / commit_encoded_fragment,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [sep_ready] / append_sep,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [sep_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [sep_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [eos_ready] / append_eos,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [eos_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [eos_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime<'event>> [no_suffix],
        "done"_s <= "finalizing"_s + completion<EventTokenizeRuntime<'event>> / finalize,
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
    pub vocab: Option<&'event dyn VocabularyView>,
    pub preprocess_kind: PreprocessorKind,
    pub model_kind: EncoderKind,
    pub is_bound: bool,
    pub last_error: TokenizerError,
    pub unexpected: bool,
}
impl<'event> Default for TextTokenizerContext<'event> {
    fn default() -> Self { Self { vocab: None, preprocess_kind: PreprocessorKind::Fallback, model_kind: EncoderKind::Fallback, is_bound: false, last_error: TokenizerError::None, unexpected: false } }
}

fn set_token_error(event: &EventTokenizeRuntime<'_>, error: TokenizerError) { let mut c = event.context.borrow_mut(); c.err = error; c.result = false; c.token_count = 0; }
fn set_bind_error(event: &EventBindRuntime<'_>, error: TokenizerError) { let mut c = event.context.borrow_mut(); c.err = error; c.result = false; }
fn valid_token_id(id: i32) -> bool { id >= 0 }
fn child_error(error: TokenizerError) -> bool { error != TokenizerError::None }

impl<'event> TextTokenizerStateMachineContext for TextTokenizerContext<'event> {
    fn begin_bind_from_uninitialized(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.begin_bind(e) }
    fn begin_bind_from_idle(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.begin_bind(e) }
    fn begin_bind_from_done(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.begin_bind(e) }
    fn begin_bind_from_errored(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.begin_bind(e) }
    fn begin_bind_from_unexpected(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.begin_bind(e) }
    fn begin_tokenize_from_idle(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.begin_tokenize(e) }
    fn begin_tokenize_from_done(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.begin_tokenize(e) }
    fn begin_tokenize_from_errored(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.begin_tokenize(e) }
    fn begin_tokenize_from_unexpected(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.begin_tokenize(e) }
    fn can_bind(&self, _e: &EventBindRuntime<'event>) -> Result<bool, ()> { Ok(true) }
    fn can_tokenize(&self, e: &EventTokenizeRuntime<'event>) -> Result<bool, ()> { Ok(self.is_bound && self.vocab.is_some_and(|v| core::ptr::eq(v, e.request.vocab)) && !e.request.token_ids.borrow().is_empty()) }
    fn bind_preprocessor(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { let error = e.request.bind_preprocessor.map_or(TokenizerError::None, |f| f(e.request.vocab, e.request.preprocessor_variant as u8)); e.context.borrow_mut().err = error; Ok(()) }
    fn bind_encoder(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { let error = e.request.bind_encoder.map_or(TokenizerError::None, |f| f(e.request.vocab, e.request.encoder_variant as u8)); e.context.borrow_mut().err = error; Ok(()) }
    fn mark_bind_success(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.is_bound = true; self.last_error = TokenizerError::None; e.context.borrow_mut().result = true; Ok(()) }
    fn reject_bind_from_uninitialized(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.reject_bind(e) }
    fn reject_bind_from_idle(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.reject_bind(e) }
    fn reject_bind_from_done(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.reject_bind(e) }
    fn reject_bind_from_errored(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.reject_bind(e) }
    fn reject_bind_from_unexpected(&mut self, e: &EventBindRuntime<'event>) -> Result<(), ()> { self.reject_bind(e) }
    fn reject_invalid_from_uninitialized(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.reject_invalid(e) }
    fn reject_invalid_from_idle(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.reject_invalid(e) }
    fn reject_invalid_from_done(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.reject_invalid(e) }
    fn reject_invalid_from_errored(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.reject_invalid(e) }
    fn reject_invalid_from_unexpected(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.reject_invalid(e) }
    fn dispatch_preprocess(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { let mut c = e.context.borrow_mut(); let out = &mut c.fragments[..]; let r = e.request.preprocess.map_or_else(|| { if e.request.text.is_empty() { ChildResult { accepted: true, ..ChildResult::default() } } else { out[0] = Fragment { kind: FragmentKind::RawText, start: 0, end: e.request.text.len(), token: -1 }; ChildResult { accepted: true, count: 1, ..ChildResult::default() } } }, |f| f(e.request.vocab, e.request.text, e.request.parse_special, out)); c.preprocess_accepted = r.accepted; c.preprocess_err_code = r.error; c.fragment_count = r.count; c.fragment_index = 0; c.preprocessed = r.accepted; Ok(()) }
    fn dispatch_encode_raw_fragment(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { let mut c = e.context.borrow_mut(); let f = c.fragments[c.fragment_index]; let bytes = if f.end <= e.request.text.len() && f.start <= f.end { &e.request.text[f.start..f.end] } else { c.encode_err_code = TokenizerError::InvalidRequest; return Ok(()) }; let mut output = e.request.token_ids.borrow_mut(); let remaining = output.len().saturating_sub(c.token_count); let r = e.request.encode.map_or_else(|| fallback_encode(e.request.vocab, bytes, c.preprocessed, &mut output[c.token_count..c.token_count + remaining]), |cb| cb(e.request.vocab, bytes, c.preprocessed, &mut output[c.token_count..c.token_count + remaining])); c.encode_accepted = r.accepted; c.encode_err_code = r.error; c.encode_token_count = r.count; Ok(()) }
    fn append_bos(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.append_id(e, e.request.vocab.bos_id()) }
    fn append_sep(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.append_id(e, e.request.vocab.sep_id()) }
    fn append_eos(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { self.append_id(e, e.request.vocab.eos_id()) }
    fn append_fragment_token(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { let mut c=e.context.borrow_mut(); let id=c.fragments[c.fragment_index].token; let mut out=e.request.token_ids.borrow_mut(); out[c.token_count]=id; c.token_count+=1; c.fragment_index+=1; Ok(()) }
    fn commit_encoded_fragment(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { let mut c=e.context.borrow_mut(); c.token_count += c.encode_token_count; c.fragment_index += 1; Ok(()) }
    fn finalize(&mut self, e: &EventTokenizeRuntime<'event>) -> Result<(), ()> { e.context.borrow_mut().result=true; self.last_error=TokenizerError::None; Ok(()) }
    fn set_backend_error(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::BackendError);Ok(())}
    fn set_error_from_preprocess(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{let x=e.context.borrow().preprocess_err_code;set_token_error(e,if x==TokenizerError::None{TokenizerError::BackendError}else{x});Ok(())}
    fn set_error_from_encode(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{let x=e.context.borrow().encode_err_code;set_token_error(e,if x==TokenizerError::None{TokenizerError::BackendError}else{x});Ok(())}
    fn set_invalid_request_error_from_preprocess_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn set_invalid_request_error_from_prefix_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn set_invalid_request_error_from_encoding_ready(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn set_invalid_request_error_from_encoding_raw_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn set_invalid_request_error_from_suffix_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn set_invalid_id_error_from_prefix_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::ModelInvalid);Ok(())}
    fn set_invalid_id_error_from_encoding_raw_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::ModelInvalid);Ok(())}
    fn set_invalid_id_error_from_suffix_decision(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{set_token_error(e,TokenizerError::ModelInvalid);Ok(())}
    fn bind_preprocessor_error_none(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().err==TokenizerError::None)}
    fn bind_preprocessor_error_invalid_request(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().err==TokenizerError::InvalidRequest)}
    fn bind_preprocessor_error_model_invalid(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().err==TokenizerError::ModelInvalid)}
    fn bind_preprocessor_error_backend_error(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().err==TokenizerError::BackendError)}
    fn bind_preprocessor_error_unknown(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{Ok(child_error(e.context.borrow().err)&&!matches!(e.context.borrow().err,TokenizerError::InvalidRequest|TokenizerError::ModelInvalid|TokenizerError::BackendError))}
    fn bind_encoder_error_none(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{self.bind_preprocessor_error_none(e)}
    fn bind_encoder_error_invalid_request(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{self.bind_preprocessor_error_invalid_request(e)}
    fn bind_encoder_error_model_invalid(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{self.bind_preprocessor_error_model_invalid(e)}
    fn bind_encoder_error_backend_error(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{self.bind_preprocessor_error_backend_error(e)}
    fn bind_encoder_error_unknown(&self,e:&EventBindRuntime<'event>)->Result<bool,()>{self.bind_preprocessor_error_unknown(e)}
    fn preprocess_rejected_no_error(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(!c.preprocess_accepted&&c.preprocess_err_code==TokenizerError::None)}
    fn preprocess_reported_error(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().preprocess_err_code!=TokenizerError::None)}
    fn preprocess_fragment_count_invalid(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().fragment_count>MAX_FRAGMENTS)}
    fn preprocess_success(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(c.preprocess_accepted&&c.preprocess_err_code==TokenizerError::None&&c.fragment_count<=MAX_FRAGMENTS)}
    fn bos_ready(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&e.request.vocab.add_bos()&&valid_token_id(e.request.vocab.bos_id())&&e.context.borrow().token_count<e.request.token_ids.borrow().len())}
    fn bos_no_capacity(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&e.request.vocab.add_bos()&&valid_token_id(e.request.vocab.bos_id())&&e.context.borrow().token_count>=e.request.token_ids.borrow().len())}
    fn bos_invalid_id(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&e.request.vocab.add_bos()&&!valid_token_id(e.request.vocab.bos_id()))}
    fn no_prefix(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(!e.request.add_special||!e.request.vocab.add_bos())}
    fn no_more_fragments(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().fragment_index>=e.context.borrow().fragment_count)}
    fn more_fragments_no_capacity(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().fragment_index<e.context.borrow().fragment_count&&e.context.borrow().token_count>=e.request.token_ids.borrow().len())}
    fn more_fragments_token_valid(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(c.fragment_index<c.fragment_count&&c.fragments[c.fragment_index].kind==FragmentKind::Token&&c.fragments[c.fragment_index].token>=0)}
    fn more_fragments_token_invalid(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(c.fragment_index<c.fragment_count&&c.fragments[c.fragment_index].kind==FragmentKind::Token&&c.fragments[c.fragment_index].token<0)}
    fn more_fragments_raw(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(c.fragment_index<c.fragment_count&&c.fragments[c.fragment_index].kind==FragmentKind::RawText)}
    fn encode_rejected_no_error(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(!c.encode_accepted&&c.encode_err_code==TokenizerError::None)}
    fn encode_reported_error(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.context.borrow().encode_err_code!=TokenizerError::None)}
    fn encode_count_invalid(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{let c=e.context.borrow();Ok(c.encode_accepted&&c.encode_err_code==TokenizerError::None&&c.encode_token_count>e.request.token_ids.borrow().len().saturating_sub(c.token_count))}
    fn encode_success(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok({let c=e.context.borrow();c.encode_accepted&&c.encode_err_code==TokenizerError::None&&c.encode_token_count<=e.request.token_ids.borrow().len().saturating_sub(c.token_count)})}
    fn sep_ready(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind==EncoderKind::Wpm&&e.request.vocab.add_sep()&&valid_token_id(e.request.vocab.sep_id())&&e.context.borrow().token_count<e.request.token_ids.borrow().len())}
    fn sep_no_capacity(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind==EncoderKind::Wpm&&e.request.vocab.add_sep()&&valid_token_id(e.request.vocab.sep_id())&&e.context.borrow().token_count>=e.request.token_ids.borrow().len())}
    fn sep_invalid_id(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind==EncoderKind::Wpm&&e.request.vocab.add_sep()&&!valid_token_id(e.request.vocab.sep_id()))}
    fn eos_ready(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind!=EncoderKind::Wpm&&e.request.vocab.add_eos()&&valid_token_id(e.request.vocab.eos_id())&&e.context.borrow().token_count<e.request.token_ids.borrow().len())}
    fn eos_no_capacity(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind!=EncoderKind::Wpm&&e.request.vocab.add_eos()&&valid_token_id(e.request.vocab.eos_id())&&e.context.borrow().token_count>=e.request.token_ids.borrow().len())}
    fn eos_invalid_id(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(e.request.add_special&&self.model_kind!=EncoderKind::Wpm&&e.request.vocab.add_eos()&&!valid_token_id(e.request.vocab.eos_id()))}
    fn no_suffix(&self,e:&EventTokenizeRuntime<'event>)->Result<bool,()>{Ok(!((e.request.add_special&&self.model_kind==EncoderKind::Wpm&&e.request.vocab.add_sep())||(e.request.add_special&&self.model_kind!=EncoderKind::Wpm&&e.request.vocab.add_eos())))}
    fn begin_bind(&mut self,e:&EventBindRuntime<'event>)->Result<(),()>{self.vocab=Some(e.request.vocab);self.preprocess_kind=e.request.preprocessor_variant;self.model_kind=e.request.encoder_variant;self.is_bound=false;self.last_error=TokenizerError::None;e.context.borrow_mut().err=TokenizerError::None;Ok(())}
    fn begin_tokenize(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{let mut c=e.context.borrow_mut();*c=TokenizeContext::default();self.last_error=TokenizerError::None;Ok(())}
    fn reject_bind(&mut self,e:&EventBindRuntime<'event>)->Result<(),()>{self.is_bound=false;self.last_error=TokenizerError::InvalidRequest;set_bind_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn reject_invalid(&mut self,e:&EventTokenizeRuntime<'event>)->Result<(),()>{self.last_error=TokenizerError::InvalidRequest;set_token_error(e,TokenizerError::InvalidRequest);Ok(())}
    fn append_id(&mut self,e:&EventTokenizeRuntime<'event>,id:i32)->Result<(),()>{let mut c=e.context.borrow_mut();let mut o=e.request.token_ids.borrow_mut();o[c.token_count]=id;c.token_count+=1;Ok(())}
    fn on_unexpected_from_binding_encoder_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_binding_encoder(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_binding_preprocessor_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_binding_preprocessor(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_done(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_errored(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_idle(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_prefix_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_preprocess_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_preprocessing(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_suffix_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_unexpected(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_uninitialized(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_encoding_ready(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_encoding_token_fragment(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_encoding_raw_fragment(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_encoding_raw_decision(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn on_unexpected_from_finalizing(&mut self)->Result<(),()>{self.mark_unexpected()}
    fn mark_unexpected(&mut self)->Result<(),()>{self.unexpected=true;self.is_bound=false;self.last_error=TokenizerError::Unexpected;Ok(())}
}

fn fallback_encode(vocab:&dyn VocabularyView,text:&[u8],_:bool,out:&mut [i32])->ChildResult { let mut count=0; for b in text.iter().copied(){ let mut found=-1; for i in 0..vocab.token_count(){ if vocab.token(i)==Some(core::slice::from_ref(&b)){found=i as i32;break;} } if found<0{return ChildResult{accepted:true,error:TokenizerError::BackendError,count}} if count>=out.len(){return ChildResult{accepted:true,error:TokenizerError::InvalidRequest,count}} out[count]=found;count+=1; } ChildResult{accepted:true,error:TokenizerError::None,count} }

/// Single-writer synchronous tokenizer actor.
pub struct TextTokenizer<'event> { machine: TextTokenizerStateMachine<'event, TextTokenizerContext<'event>> }
impl<'event> Default for TextTokenizer<'event> { fn default()->Self{Self::new()} }
impl<'event> TextTokenizer<'event> {
    #[must_use] pub fn new()->Self{Self{machine:TextTokenizerStateMachine::new(TextTokenizerContext::default())}}
    pub fn process_bind(&mut self,request:BindRequest<'event>)->Result<TokenizerBindDone,TokenizerBindError>{let ctx=RefCell::new(BindContext::default());let event=EventBindRuntime{request,context:&ctx};if self.machine.process_event(TextTokenizerEvents::EventBindRuntime(event)).is_err(){self.machine.set_state(TextTokenizerStates::Unexpected);return Err(TokenizerBindError{error:TokenizerError::Unexpected});}let c=*ctx.borrow();if c.result&&self.machine.is(&TextTokenizerStates::Idle){if let Some(cb)=request.dispatch_done{let _=cb(TokenizerBindDone);}Ok(TokenizerBindDone)}else{let e=TokenizerBindError{error:if c.err==TokenizerError::None{TokenizerError::BackendError}else{c.err}};if let Some(cb)=request.dispatch_error{let _=cb(e);}Err(e)}}
    pub fn process_tokenize(&mut self,request:TokenizeRequest<'event>)->Result<TokenizerDone,TokenizerErrorEvent>{let ctx=RefCell::new(TokenizeContext::default());let event=EventTokenizeRuntime{request,context:&ctx};if self.machine.process_event(TextTokenizerEvents::EventTokenizeRuntime(event)).is_err(){self.machine.set_state(TextTokenizerStates::Unexpected);}let c=*ctx.borrow();if c.result&&self.machine.is(&TextTokenizerStates::Done){let d=TokenizerDone{token_count:c.token_count};if let Some(cb)=request.dispatch_done{let _=cb(d);}Ok(d)}else{let e=TokenizerErrorEvent{error:if self.machine.context().unexpected{TokenizerError::Unexpected}else if c.err==TokenizerError::None{TokenizerError::BackendError}else{c.err}};if let Some(cb)=request.dispatch_error{let _=cb(e);}Err(e)}}
    pub fn process_unexpected(&mut self)->bool{self.machine.context_mut().unexpected=true;self.machine.set_state(TextTokenizerStates::Unexpected);false}
    #[must_use] pub fn state(&self)->&TextTokenizerStates{self.machine.state()}
    #[must_use] pub fn is(&self,state:&TextTokenizerStates)->bool{self.machine.is(state)}
    #[must_use] pub fn context(&self)->&TextTokenizerContext<'event>{self.machine.context()}
    #[must_use] pub fn last_error(&self)->TokenizerError{self.machine.context().last_error}
}
/// Compatibility actor alias.
pub type TextTokenizerActor<'event> = TextTokenizer<'event>;
