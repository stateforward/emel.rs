//! Source-aligned, bounded Plamo2 tokenizer preprocessor actor.
//!
//! The actor mirrors the pinned Plamo2 preprocessing state machine while keeping
//! all storage caller-owned or inline. A dispatch runs to completion and never
//! allocates or defers callback delivery.

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

use core::cell::{Cell, RefCell};
use sml::sml;

/// Maximum number of output fragments accepted by the pinned contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum number of cached special tokens accepted by the pinned contract.
pub const MAX_SPECIAL_TOKENS: usize = 1024;

/// Fragment kind emitted by preprocessing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

/// One bounded preprocessor output fragment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'text> {
    pub kind: FragmentKind,
    pub text: &'text str,
    pub token: i32,
}

/// One vocabulary entry used to discover special tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VocabEntry<'text> {
    pub text: &'text str,
    pub token: i32,
    pub token_type: i32,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Bounded vocabulary view supplied to one preprocessing request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Vocabulary<'text> {
    pub entries: &'text [VocabEntry<'text>],
}

/// Source-compatible preprocessing errors.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    BackendError = 2,
}

impl PreprocessError {
    #[must_use]
    pub const fn code(self) -> i32 { self as i32 }

    const fn resolve(runtime: Self) -> Self {
        if matches!(runtime, Self::None) { Self::BackendError } else { runtime }
    }
}

/// Successful completion callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessDone {
    pub fragment_count: usize,
}

/// Error completion callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessFailure {
    pub error: PreprocessError,
}

/// Synchronous successful completion callback.
pub type DoneCallback = fn(PreprocessDone) -> bool;
/// Synchronous error completion callback.
pub type ErrorCallback = fn(PreprocessFailure) -> bool;

/// Caller-owned request and output sinks.
#[derive(Debug)]
pub struct PreprocessRequest<'text, 'out> {
    pub vocab: Vocabulary<'text>,
    pub text: &'text str,
    pub parse_special: bool,
    pub fragments_out: &'out RefCell<&'out mut [Fragment<'text>]>,
    pub fragment_count_out: &'out Cell<usize>,
    pub preprocessed_out: Option<&'out Cell<bool>>,
    pub error_out: &'out Cell<i32>,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}

impl<'text, 'out> Copy for PreprocessRequest<'text, 'out> {}
impl<'text, 'out> Clone for PreprocessRequest<'text, 'out> {
    fn clone(&self) -> Self { *self }
}

impl<'text, 'out> PreprocessRequest<'text, 'out> {
    #[must_use]
    pub const fn new(
        vocab: Vocabulary<'text>,
        text: &'text str,
        parse_special: bool,
        fragments_out: &'out RefCell<&'out mut [Fragment<'text>]>,
        fragment_count_out: &'out Cell<usize>,
        error_out: &'out Cell<i32>,
    ) -> Self {
        Self {
            vocab,
            text,
            parse_special,
            fragments_out,
            fragment_count_out,
            preprocessed_out: None,
            error_out,
            dispatch_done: None,
            dispatch_error: None,
        }
    }

    #[must_use]
    pub const fn with_callbacks(
        mut self,
        preprocessed_out: Option<&'out Cell<bool>>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        self.preprocessed_out = preprocessed_out;
        self.dispatch_done = dispatch_done;
        self.dispatch_error = dispatch_error;
        self
    }
}

/// Runtime context carried by one state-machine dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreprocessRuntimeContext {
    pub fragment_count: usize,
    pub preprocessed: bool,
    pub phase_error: PreprocessError,
    pub error: PreprocessError,
    pub result: bool,
}

/// Runtime event accepted by the generated state machine.
#[derive(Clone, Copy, Debug)]
pub struct EventPreprocessRuntime<'text, 'out> {
    pub request: PreprocessRequest<'text, 'out>,
    pub context: &'out RefCell<PreprocessRuntimeContext>,
}

impl<'text, 'out> EventPreprocessRuntime<'text, 'out> {
    #[must_use]
    pub const fn new(
        request: PreprocessRequest<'text, 'out>,
        context: &'out RefCell<PreprocessRuntimeContext>,
    ) -> Self {
        Self { request, context }
    }
}

sml! {
    TextTokenizerPreprocessorPlamo2<'text, 'out> {
        "request_buffer_decision"_s <= *"idle"_s + event<EventPreprocessRuntime<'text, 'out>>,
        "request_buffer_decision"_s <= "done"_s + event<EventPreprocessRuntime<'text, 'out>>,
        "request_buffer_decision"_s <= "errored"_s + event<EventPreprocessRuntime<'text, 'out>>,
        "request_buffer_decision"_s <= "unexpected"_s + event<EventPreprocessRuntime<'text, 'out>>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / build_specials,
        "partition_specials_decision"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_no_specials_input_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [no_specials],
        "partition_parse_special_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [has_specials],
        "errored"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / ensure_last_error_from_partition_specials_decision,
        "partitioning_non_bpe_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [parse_special_enabled],
        "partitioning_non_bpe_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_empty] / set_empty_partition_result_from_partitioning_no_specials_input_decision,
        "partitioning_no_specials"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_nonempty],
        "errored"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / ensure_last_error_from_partitioning_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision,
        "partitioning_non_bpe_parse_special"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / ensure_last_error_from_partitioning_non_bpe_parse_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision,
        "partitioning_non_bpe_skip_special"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / ensure_last_error_from_partitioning_non_bpe_skip_input_decision,
        "partition_decision"_s <= "partitioning_no_specials"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / partition_no_specials,
        "partition_decision"_s <= "partitioning_non_bpe_parse_special"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / partition_non_bpe_parse_special,
        "partition_decision"_s <= "partitioning_non_bpe_skip_special"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) / partition_non_bpe_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'text, 'out>) [partition_unknown_error] / ensure_last_error_from_partition_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "request_buffer_decision"_s + unexpected_event<_> / on_unexpected_from_request_buffer_decision,
        "unexpected"_s <= "request_capacity_nonzero_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_nonzero_decision,
        "unexpected"_s <= "request_capacity_limit_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_limit_decision,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "build_specials_decision"_s + unexpected_event<_> / on_unexpected_from_build_specials_decision,
        "unexpected"_s <= "partition_specials_decision"_s + unexpected_event<_> / on_unexpected_from_partition_specials_decision,
        "unexpected"_s <= "partition_parse_special_decision"_s + unexpected_event<_> / on_unexpected_from_partition_parse_special_decision,
        "unexpected"_s <= "partitioning_no_specials_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_parse_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_skip_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_input_decision,
        "unexpected"_s <= "partitioning_no_specials"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials,
        "unexpected"_s <= "partitioning_non_bpe_parse_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_special,
        "unexpected"_s <= "partitioning_non_bpe_skip_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_special,
        "unexpected"_s <= "partition_decision"_s + unexpected_event<_> / on_unexpected_from_partition_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SpecialToken<'text> {
    text: &'text str,
    token: i32,
    token_type: i32,
    lstrip: bool,
    rstrip: bool,
}

/// Context retained by the generated state machine.
#[derive(Debug)]
pub struct TextTokenizerPreprocessorPlamo2Context<'text> {
    specials: [SpecialToken<'text>; MAX_SPECIAL_TOKENS],
    special_count: usize,
    pub unexpected: bool,
}

impl<'text> Default for TextTokenizerPreprocessorPlamo2Context<'text> {
    fn default() -> Self {
        Self {
            specials: [SpecialToken::default(); MAX_SPECIAL_TOKENS],
            special_count: 0,
            unexpected: false,
        }
    }
}

impl<'text> TextTokenizerPreprocessorPlamo2Context<'text> {
    fn build_specials(&mut self, vocab: Vocabulary<'text>) -> Result<bool, ()> {
        self.special_count = 0;
        for entry in vocab.entries {
            if !matches!(entry.token_type, 2 | 3 | 4) || entry.text.is_empty() { continue; }
            if self.special_count == MAX_SPECIAL_TOKENS { return Ok(false); }
            self.specials[self.special_count] = SpecialToken {
                text: entry.text,
                token: entry.token,
                token_type: entry.token_type,
                lstrip: entry.lstrip,
                rstrip: entry.rstrip,
            };
            self.special_count += 1;
        }
        self.specials[..self.special_count].sort_unstable_by(|a, b| b.text.len().cmp(&a.text.len()));
        Ok(true)
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
}

impl<'text, 'out> TextTokenizerPreprocessorPlamo2StateMachineContext
    for TextTokenizerPreprocessorPlamo2Context<'text>
{
    fn begin_preprocess(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
        self.unexpected = false;
        *event.context.borrow_mut() = PreprocessRuntimeContext::default();
        event.request.fragment_count_out.set(0);
        event.request.error_out.set(PreprocessError::None.code());
        if let Some(out) = event.request.preprocessed_out { out.set(false); }
        event.request.fragments_out.borrow_mut().fill(Fragment::default());
        self.special_count = 0;
        Ok(())
    }

    fn build_specials(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
        let ok = self.build_specials(event.request.vocab)?;
        let mut runtime = event.context.borrow_mut();
        runtime.phase_error = if ok { PreprocessError::None } else { PreprocessError::InvalidRequest };
        runtime.error = runtime.phase_error;
        runtime.result = false;
        Ok(())
    }

    fn build_specials_backend_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::BackendError) }
    fn build_specials_invalid_request_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::InvalidRequest) }
    fn build_specials_ok(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::None) }
    fn build_specials_unknown_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { let error = event.context.borrow().phase_error; Ok(!matches!(error, PreprocessError::None | PreprocessError::InvalidRequest | PreprocessError::BackendError)) }

    fn ensure_last_error_from_build_specials_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partition_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partition_parse_special_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partition_specials_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partitioning_no_specials_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { ensure_last_error(event) }

    fn fragments_buffer_missing(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.fragments_out.borrow().is_empty()) }
    fn fragments_buffer_present(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(!event.request.fragments_out.borrow().is_empty()) }
    fn fragments_capacity_exceeds_limit(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.fragments_out.borrow().len() > MAX_FRAGMENTS) }
    fn fragments_capacity_nonzero(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(!event.request.fragments_out.borrow().is_empty()) }
    fn fragments_capacity_within_limit(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.fragments_out.borrow().len() <= MAX_FRAGMENTS) }
    fn fragments_capacity_zero(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.fragments_out.borrow().is_empty()) }
    fn has_specials(&self, _event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(self.special_count != 0) }

    fn mark_done(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
        let mut runtime = event.context.borrow_mut();
        runtime.phase_error = PreprocessError::None;
        runtime.error = PreprocessError::None;
        runtime.result = true;
        event.request.fragment_count_out.set(runtime.fragment_count);
        if let Some(out) = event.request.preprocessed_out { out.set(runtime.preprocessed); }
        event.request.error_out.set(PreprocessError::None.code());
        Ok(())
    }

    fn no_specials(&self, _event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(self.special_count == 0) }

    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected() }

    fn parse_special_disabled(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(!event.request.parse_special) }
    fn parse_special_enabled(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.parse_special) }
    fn partition_backend_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::BackendError) }
    fn partition_invalid_request_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::InvalidRequest) }

    fn partition_no_specials(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
        let mut output = event.request.fragments_out.borrow_mut();
        let mut count = 0;
        let ok = push_raw(&mut output, &mut count, event.request.text);
        let mut runtime = event.context.borrow_mut();
        runtime.fragment_count = if ok { count } else { 0 };
        runtime.preprocessed = true;
        runtime.phase_error = if ok { PreprocessError::None } else { PreprocessError::InvalidRequest };
        runtime.error = runtime.phase_error;
        Ok(())
    }

    fn partition_non_bpe_parse_special(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { self.partition_with_specials(event, true) }
    fn partition_non_bpe_skip_special(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { self.partition_with_specials(event, false) }

    fn partition_ok(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.context.borrow().phase_error == PreprocessError::None) }
    fn partition_unknown_error(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { let error = event.context.borrow().phase_error; Ok(!matches!(error, PreprocessError::None | PreprocessError::InvalidRequest | PreprocessError::BackendError)) }

    fn reject_invalid_from_request_buffer_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_from_request_capacity_limit_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_from_request_capacity_nonzero_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { reject_invalid(event) }

    fn request_text_empty(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn request_text_nonempty(&self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }

    fn set_empty_partition_result_from_partitioning_no_specials_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { set_empty(event) }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { set_empty(event) }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision(&mut self, event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> { set_empty(event) }
}

impl<'text, 'out> TextTokenizerPreprocessorPlamo2Context<'text> {
    fn partition_with_specials(&mut self, event: &EventPreprocessRuntime<'text, 'out>, parse_special: bool) -> Result<(), ()> {
        let capacity = event.request.fragments_out.borrow().len();
        let mut current = [Fragment::default(); MAX_FRAGMENTS];
        let mut next = [Fragment::default(); MAX_FRAGMENTS];
        let mut count = usize::from(!event.request.text.is_empty());
        if count != 0 { current[0] = Fragment { kind: FragmentKind::RawText, text: event.request.text, token: -1 }; }
        let mut ok = true;
        for special in &self.specials[..self.special_count] {
            if !parse_special && matches!(special.token_type, 2 | 3) { continue; }
            let mut next_count = 0;
            for fragment in &current[..count] {
                if !ok { break; }
                if fragment.kind == FragmentKind::Token {
                    ok &= push_token(&mut next, capacity, &mut next_count, fragment.token);
                    continue;
                }
                let mut base = 0;
                while base < fragment.text.len() {
                    let Some(relative) = fragment.text[base..].find(special.text) else {
                        ok &= push_raw(&mut next[..capacity], &mut next_count, &fragment.text[base..]);
                        base = fragment.text.len();
                        break;
                    };
                    let match_at = base + relative;
                    let mut left = match_at - base;
                    if special.lstrip {
                        while left > 0 && fragment.text.as_bytes()[base + left - 1].is_ascii_whitespace() { left -= 1; }
                    }
                    ok &= push_raw(&mut next[..capacity], &mut next_count, &fragment.text[base..base + left]);
                    ok &= push_token(&mut next, capacity, &mut next_count, special.token);
                    let mut after = match_at + special.text.len();
                    if special.rstrip {
                        while after < fragment.text.len() && fragment.text.as_bytes()[after].is_ascii_whitespace() { after += 1; }
                    }
                    base = after;
                }
            }
            current[..next_count].copy_from_slice(&next[..next_count]);
            count = next_count;
        }
        let mut output = event.request.fragments_out.borrow_mut();
        if ok { output[..count].copy_from_slice(&current[..count]); }
        let mut runtime = event.context.borrow_mut();
        runtime.fragment_count = if ok { count } else { 0 };
        runtime.preprocessed = true;
        runtime.phase_error = if ok { PreprocessError::None } else { PreprocessError::InvalidRequest };
        runtime.error = runtime.phase_error;
        Ok(())
    }
}

fn push_raw(out: &mut [Fragment<'_>], count: &mut usize, text: &str) -> bool {
    if text.is_empty() { return true; }
    if *count >= out.len() { return false; }
    out[*count] = Fragment { kind: FragmentKind::RawText, text, token: -1 };
    *count += 1;
    true
}

fn push_token(out: &mut [Fragment<'_>; MAX_FRAGMENTS], capacity: usize, count: &mut usize, token: i32) -> bool {
    if token < 0 || *count >= capacity || *count >= MAX_FRAGMENTS { return false; }
    out[*count] = Fragment { kind: FragmentKind::Token, text: "", token };
    *count += 1;
    true
}

fn ensure_last_error<'text, 'out>(event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
    let mut runtime = event.context.borrow_mut();
    runtime.error = PreprocessError::resolve(runtime.phase_error);
    runtime.result = false;
    event.request.error_out.set(runtime.error.code());
    event.request.fragment_count_out.set(0);
    if let Some(out) = event.request.preprocessed_out { out.set(false); }
    Ok(())
}

fn reject_invalid<'text, 'out>(event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
    let mut runtime = event.context.borrow_mut();
    runtime.phase_error = PreprocessError::InvalidRequest;
    runtime.error = PreprocessError::InvalidRequest;
    runtime.result = false;
    runtime.fragment_count = 0;
    event.request.fragment_count_out.set(0);
    event.request.error_out.set(PreprocessError::InvalidRequest.code());
    if let Some(out) = event.request.preprocessed_out { out.set(false); }
    Ok(())
}

fn set_empty<'text, 'out>(event: &EventPreprocessRuntime<'text, 'out>) -> Result<(), ()> {
    let mut runtime = event.context.borrow_mut();
    runtime.fragment_count = 0;
    runtime.preprocessed = true;
    runtime.phase_error = PreprocessError::None;
    runtime.error = PreprocessError::None;
    runtime.result = false;
    event.request.fragment_count_out.set(0);
    Ok(())
}

/// Synchronous bounded Plamo2 preprocessor actor.
pub struct TextTokenizerPreprocessorPlamo2Actor<'text, 'out> {
    machine: TextTokenizerPreprocessorPlamo2StateMachine<'text, 'out, TextTokenizerPreprocessorPlamo2Context<'text>>,
}

impl<'text, 'out> Default for TextTokenizerPreprocessorPlamo2Actor<'text, 'out> {
    fn default() -> Self { Self::new() }
}

impl<'text, 'out> TextTokenizerPreprocessorPlamo2Actor<'text, 'out> {
    #[must_use]
    pub fn new() -> Self {
        Self { machine: TextTokenizerPreprocessorPlamo2StateMachine::new(TextTokenizerPreprocessorPlamo2Context::default()) }
    }

    pub fn process_event(&mut self, event: EventPreprocessRuntime<'text, 'out>) -> bool {
        let context = event.context;
        let request = event.request;
        let accepted = self.machine.process_event(TextTokenizerPreprocessorPlamo2Events::EventPreprocessRuntime(event)).is_ok();
        let runtime = context.borrow();
        let ok = accepted && runtime.result && runtime.error == PreprocessError::None;
        let error = if ok { PreprocessError::None } else { PreprocessError::resolve(runtime.error) };
        let count = runtime.fragment_count;
        request.fragment_count_out.set(count);
        request.error_out.set(error.code());
        if let Some(out) = request.preprocessed_out { out.set(runtime.preprocessed && ok); }
        drop(runtime);
        if ok {
            if let Some(callback) = request.dispatch_done { let _ = callback(PreprocessDone { fragment_count: count }); }
        } else if let Some(callback) = request.dispatch_error { let _ = callback(PreprocessFailure { error }); }
        ok
    }

    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.context_mut().unexpected = true;
        self.machine.set_state(TextTokenizerPreprocessorPlamo2States::Unexpected);
        false
    }

    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorPlamo2States { self.machine.state() }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorPlamo2States) -> bool { self.machine.is(state) }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorPlamo2Context<'text> { self.machine.context() }
}

/// Short public alias matching the pinned machine name.
pub type Actor<'text, 'out> = TextTokenizerPreprocessorPlamo2Actor<'text, 'out>;
