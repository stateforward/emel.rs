//! Source-aligned bounded WPM tokenizer preprocessor actor.
//!
//! WPM uses the non-BPE preprocessing contract: special tokens are cached in
//! vocabulary order (longest first), then the input is partitioned into
//! caller-owned raw/token fragments through the generated SML phases.
#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::elidable_lifetime_names,
    clippy::items_after_statements,
    clippy::needless_lifetimes,
    dead_code,
    missing_docs,
    private_interfaces,
    unused_imports
)]

use core::cell::RefCell;
use sml::sml;

pub const MAX_FRAGMENTS: usize = 1024;
pub const MAX_SPECIAL_TOKENS: usize = 1024;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'a> {
    pub kind: FragmentKind,
    pub text: &'a str,
    pub token: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VocabularyToken<'a> {
    pub text: &'a str,
    pub token: i32,
    pub token_type: i32,
    pub lstrip: bool,
    pub rstrip: bool,
}

pub trait VocabularyView {
    fn token_count(&self) -> usize;
    fn token(&self, index: usize) -> Option<VocabularyToken<'_>>;
}

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
    pub const fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::InvalidRequest => 1,
            Self::BackendError => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessDone {
    pub fragment_count: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessErrorEvent {
    pub err: PreprocessError,
}
pub type DoneCallback = fn(PreprocessDone) -> bool;
pub type ErrorCallback = fn(PreprocessErrorEvent) -> bool;

#[derive(Clone, Copy)]
pub struct PreprocessRequest<'a> {
    pub vocab: &'a dyn VocabularyView,
    pub text: &'a str,
    pub parse_special: bool,
    pub fragments_out: Option<&'a RefCell<&'a mut [Fragment<'a>]>>,
    pub fragment_count_out: &'a RefCell<usize>,
    pub preprocessed_out: Option<&'a RefCell<bool>>,
    pub error_out: &'a RefCell<i32>,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}
impl<'a> PreprocessRequest<'a> {
    #[must_use]
    pub fn new(
        vocab: &'a dyn VocabularyView,
        text: &'a str,
        parse_special: bool,
        fragments_out: &'a RefCell<&'a mut [Fragment<'a>]>,
        fragment_count_out: &'a RefCell<usize>,
        preprocessed_out: Option<&'a RefCell<bool>>,
        error_out: &'a RefCell<i32>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self::with_callbacks(
            vocab,
            text,
            parse_special,
            Some(fragments_out),
            fragment_count_out,
            preprocessed_out,
            error_out,
            Some(dispatch_done),
            Some(dispatch_error),
        )
    }

    #[must_use]
    pub const fn with_callbacks(
        vocab: &'a dyn VocabularyView,
        text: &'a str,
        parse_special: bool,
        fragments_out: Option<&'a RefCell<&'a mut [Fragment<'a>]>>,
        fragment_count_out: &'a RefCell<usize>,
        preprocessed_out: Option<&'a RefCell<bool>>,
        error_out: &'a RefCell<i32>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self {
            vocab,
            text,
            parse_special,
            fragments_out,
            fragment_count_out,
            preprocessed_out,
            error_out,
            dispatch_done,
            dispatch_error,
        }
    }
}
impl core::fmt::Debug for PreprocessRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreprocessRequest")
            .field("text", &self.text)
            .field("parse_special", &self.parse_special)
            .field(
                "fragment_capacity",
                &self
                    .fragments_out
                    .as_ref()
                    .map_or(0, |out| out.borrow().len()),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub struct EventPreprocessRuntime<'a> {
    pub request: PreprocessRequest<'a>,
    pub context: &'a RefCell<PreprocessContext>,
}
impl<'a> EventPreprocessRuntime<'a> {
    #[must_use]
    pub const fn new(
        request: PreprocessRequest<'a>,
        context: &'a RefCell<PreprocessContext>,
    ) -> Self {
        Self { request, context }
    }
}
impl core::fmt::Debug for EventPreprocessRuntime<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventPreprocessRuntime")
            .field("request", &self.request)
            .field("context", &self.context.borrow())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ScratchFragment {
    kind: FragmentKind,
    start: usize,
    end: usize,
    token: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessContext {
    special: [usize; MAX_SPECIAL_TOKENS],
    special_count: usize,
    scratch_a: [ScratchFragment; MAX_FRAGMENTS],
    scratch_b: [ScratchFragment; MAX_FRAGMENTS],
    pub fragment_count: usize,
    pub preprocessed: bool,
    pub phase_error: PreprocessError,
    pub err: PreprocessError,
    pub result: bool,
    pub unexpected: bool,
}

impl Default for PreprocessContext {
    fn default() -> Self {
        Self {
            special: [0; MAX_SPECIAL_TOKENS],
            special_count: 0,
            scratch_a: [ScratchFragment::default(); MAX_FRAGMENTS],
            scratch_b: [ScratchFragment::default(); MAX_FRAGMENTS],
            fragment_count: 0,
            preprocessed: false,
            phase_error: PreprocessError::None,
            err: PreprocessError::None,
            result: false,
            unexpected: false,
        }
    }
}

impl PreprocessContext {
    fn reset(&mut self) {
        self.special_count = 0;
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessError::None;
        self.err = PreprocessError::None;
        self.result = false;
        self.unexpected = false;
    }
    fn set_result(&mut self, ok: bool, count: usize, preprocessed: bool) {
        self.fragment_count = if ok { count } else { 0 };
        self.preprocessed = ok && preprocessed;
        self.phase_error = if ok {
            PreprocessError::None
        } else {
            PreprocessError::InvalidRequest
        };
        self.err = self.phase_error;
        self.result = false;
    }
    fn set_failure(&mut self, error: PreprocessError) {
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = error;
        self.err = error;
        self.result = false;
    }
    fn ensure_error(&mut self) {
        self.err = if self.phase_error == PreprocessError::None {
            PreprocessError::BackendError
        } else {
            self.phase_error
        };
        self.fragment_count = 0;
        self.preprocessed = false;
        self.result = false;
    }
    fn unexpected(&mut self) {
        self.unexpected = true;
        self.set_failure(PreprocessError::InvalidRequest);
    }
    fn push(
        target: &mut [ScratchFragment; MAX_FRAGMENTS],
        count: &mut usize,
        capacity: usize,
        value: ScratchFragment,
    ) -> bool {
        if *count >= capacity || *count >= MAX_FRAGMENTS {
            return false;
        }
        target[*count] = value;
        *count += 1;
        true
    }
    fn swap_scratch(&mut self) {
        core::mem::swap(&mut self.scratch_a, &mut self.scratch_b);
    }
}
fn source_is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}
fn trim_source_left(text: &str) -> &str {
    let mut end = text.len();
    while end != 0 && source_is_space(text.as_bytes()[end - 1]) {
        end -= 1;
    }
    &text[..end]
}
fn trim_source_right(text: &str) -> &str {
    let mut start = 0;
    while start < text.len() && source_is_space(text.as_bytes()[start]) {
        start += 1;
    }
    &text[start..]
}

sml! {
    TextTokenizerPreprocessorWpm<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "request_buffer_decision"_s <= *"idle"_s + event<&'dispatch EventPreprocessRuntime<'event>>,
        "request_buffer_decision"_s <= "done"_s + event<&'dispatch EventPreprocessRuntime<'event>>,
        "request_buffer_decision"_s <= "errored"_s + event<&'dispatch EventPreprocessRuntime<'event>>,
        "request_buffer_decision"_s <= "unexpected"_s + event<&'dispatch EventPreprocessRuntime<'event>>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / build_specials,
        "partition_specials_decision"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_no_specials_input_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [no_specials],
        "partition_parse_special_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [has_specials],
        "errored"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partition_specials_decision,
        "partitioning_non_bpe_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [parse_special_enabled],
        "partitioning_non_bpe_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_no_specials_input_decision,
        "partitioning_no_specials"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision,
        "partitioning_non_bpe_parse_special"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_non_bpe_parse_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision,
        "partitioning_non_bpe_skip_special"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_non_bpe_skip_input_decision,
        "partition_decision"_s <= "partitioning_no_specials"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_no_specials,
        "partition_decision"_s <= "partitioning_non_bpe_parse_special"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_non_bpe_parse_special,
        "partition_decision"_s <= "partitioning_non_bpe_skip_special"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_non_bpe_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [partition_unknown_error] / ensure_last_error_from_partition_decision,
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

impl TextTokenizerPreprocessorWpmStateMachineContext for PreprocessContext {
    fn begin_preprocess<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.context.borrow_mut().reset();
        *event.request.fragment_count_out.borrow_mut() = 0;
        *event.request.error_out.borrow_mut() = 0;
        if let Some(out) = event.request.preprocessed_out {
            *out.borrow_mut() = false;
        }
        Ok(())
    }
    fn build_specials<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut context = event.context.borrow_mut();
        context.special_count = 0;
        context.phase_error = PreprocessError::None;
        let count = event.request.vocab.token_count();
        for index in 0..count {
            let Some(token) = event.request.vocab.token(index) else {
                context.phase_error = PreprocessError::BackendError;
                return Ok(());
            };
            if token.text.is_empty()
                || !matches!(
                    token.token_type,
                    TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
                )
            {
                continue;
            }
            if context.special_count == MAX_SPECIAL_TOKENS {
                context.phase_error = PreprocessError::InvalidRequest;
                return Ok(());
            }
            let slot = context.special_count;
            context.special[slot] = index;
            context.special_count = slot + 1;
        }
        for index in 1..context.special_count {
            let id = context.special[index];
            let Some(current) = event.request.vocab.token(id) else {
                context.phase_error = PreprocessError::BackendError;
                return Ok(());
            };
            let len = current.text.len();
            let mut pos = index;
            while pos > 0 {
                let prior = context.special[pos - 1];
                let Some(previous) = event.request.vocab.token(prior) else {
                    context.phase_error = PreprocessError::BackendError;
                    return Ok(());
                };
                if previous.text.len() >= len {
                    break;
                }
                context.special[pos] = prior;
                pos -= 1;
            }
            context.special[pos] = id;
        }
        Ok(())
    }
    fn build_specials_ok<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::None)
    }
    fn build_specials_invalid_request_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::InvalidRequest)
    }
    fn build_specials_backend_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::BackendError)
    }
    fn build_specials_unknown_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let error = event.context.borrow().phase_error;
        Ok(!matches!(
            error,
            PreprocessError::None | PreprocessError::InvalidRequest | PreprocessError::BackendError
        ))
    }
    fn ensure_last_error_from_build_specials_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partition_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partition_parse_special_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partition_specials_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partitioning_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        ensure_last_error(event)
    }
    fn fragments_buffer_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.fragments_out.is_none())
    }
    fn fragments_buffer_present<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.fragments_out.is_some())
    }
    fn fragments_capacity_exceeds_limit<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .request
            .fragments_out
            .is_some_and(|out| out.borrow().len() > MAX_FRAGMENTS))
    }
    fn fragments_capacity_nonzero<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .request
            .fragments_out
            .is_some_and(|out| !out.borrow().is_empty()))
    }
    fn fragments_capacity_within_limit<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .request
            .fragments_out
            .is_some_and(|out| out.borrow().len() <= MAX_FRAGMENTS))
    }
    fn fragments_capacity_zero<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .request
            .fragments_out
            .is_some_and(|out| out.borrow().is_empty()))
    }
    fn has_specials<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().special_count != 0)
    }
    fn no_specials<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().special_count == 0)
    }
    fn parse_special_enabled<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.parse_special)
    }
    fn parse_special_disabled<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.request.parse_special)
    }
    fn partition_backend_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::BackendError)
    }
    fn partition_invalid_request_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::InvalidRequest)
    }
    fn partition_unknown_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let error = event.context.borrow().phase_error;
        Ok(!matches!(
            error,
            PreprocessError::None | PreprocessError::InvalidRequest | PreprocessError::BackendError
        ))
    }
    fn partition_ok<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessError::None)
    }
    fn request_text_empty<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.text.is_empty())
    }
    fn request_text_nonempty<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.request.text.is_empty())
    }
    fn reject_invalid_from_request_buffer_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        reject_invalid(event)
    }
    fn reject_invalid_from_request_capacity_limit_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        reject_invalid(event)
    }
    fn reject_invalid_from_request_capacity_nonzero_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        reject_invalid(event)
    }
    fn set_empty_partition_result_from_partitioning_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_empty(event);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_empty(event);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_empty(event);
        Ok(())
    }
    fn partition_no_specials<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(output) = event.request.fragments_out else {
            return reject_invalid(event);
        };
        let mut out = output.borrow_mut();
        if out.is_empty() {
            return reject_invalid(event);
        }
        out[0] = Fragment {
            kind: FragmentKind::RawText,
            text: event.request.text,
            token: -1,
        };
        set_result(event, 1, true);
        Ok(())
    }
    fn partition_non_bpe_parse_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        partition_with_specials(event, true)
    }
    fn partition_non_bpe_skip_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        partition_with_specials(event, false)
    }
    fn mark_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut context = event.context.borrow_mut();
        context.phase_error = PreprocessError::None;
        context.err = PreprocessError::None;
        context.result = true;
        *event.request.fragment_count_out.borrow_mut() = context.fragment_count;
        if let Some(out) = event.request.preprocessed_out {
            *out.borrow_mut() = context.preprocessed;
        }
        *event.request.error_out.borrow_mut() = 0;
        Ok(())
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
}

fn set_result<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
    count: usize,
    preprocessed: bool,
) where
    'event: 'dispatch,
{
    let mut context = event.context.borrow_mut();
    context.set_result(true, count, preprocessed);
}
fn set_empty<'dispatch, 'event>(event: &'dispatch EventPreprocessRuntime<'event>)
where
    'event: 'dispatch,
{
    let mut context = event.context.borrow_mut();
    context.set_result(true, 0, true);
}
fn reject_invalid<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
) -> Result<(), ()>
where
    'event: 'dispatch,
{
    let mut context = event.context.borrow_mut();
    context.fragment_count = 0;
    context.preprocessed = false;
    context.phase_error = PreprocessError::InvalidRequest;
    context.err = PreprocessError::InvalidRequest;
    context.result = false;
    *event.request.fragment_count_out.borrow_mut() = 0;
    *event.request.error_out.borrow_mut() = 1;
    if let Some(out) = event.request.preprocessed_out {
        *out.borrow_mut() = false;
    }
    Ok(())
}
fn ensure_last_error<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
) -> Result<(), ()>
where
    'event: 'dispatch,
{
    let mut context = event.context.borrow_mut();
    context.ensure_error();
    Ok(())
}
fn partition_with_specials<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
    parse_special: bool,
) -> Result<(), ()>
where
    'event: 'dispatch,
{
    let Some(output) = event.request.fragments_out else {
        return reject_invalid(event);
    };
    let capacity = output.borrow().len();
    let mut context = event.context.borrow_mut();
    let mut current_count = 1usize;
    context.scratch_a[0] = ScratchFragment {
        kind: FragmentKind::RawText,
        start: 0,
        end: event.request.text.len(),
        token: -1,
    };
    for special_index in 0..context.special_count {
        let id = context.special[special_index];
        let Some(token) = event.request.vocab.token(id) else {
            context.set_failure(PreprocessError::BackendError);
            return Ok(());
        };
        if token.text.is_empty()
            || (!parse_special
                && matches!(token.token_type, TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL))
        {
            continue;
        }
        let mut next_count = 0usize;
        for index in 0..current_count {
            let fragment = context.scratch_a[index];
            if fragment.kind == FragmentKind::Token {
                if fragment.token < 0
                    || !PreprocessContext::push(
                        &mut context.scratch_b,
                        &mut next_count,
                        capacity,
                        fragment,
                    )
                {
                    context.set_failure(PreprocessError::InvalidRequest);
                    return Ok(());
                }
                continue;
            }
            let raw = &event.request.text[fragment.start..fragment.end];
            let mut base = 0usize;
            while base < raw.len() {
                let Some(relative) = raw[base..].find(token.text) else {
                    if base < raw.len()
                        && !PreprocessContext::push(
                            &mut context.scratch_b,
                            &mut next_count,
                            capacity,
                            ScratchFragment {
                                kind: FragmentKind::RawText,
                                start: fragment.start + base,
                                end: fragment.end,
                                token: -1,
                            },
                        )
                    {
                        context.set_failure(PreprocessError::InvalidRequest);
                        return Ok(());
                    }
                    base = raw.len();
                    continue;
                };
                let match_at = base + relative;
                let mut left_len = match_at - base;
                if token.lstrip {
                    while left_len > 0 && source_is_space(raw.as_bytes()[base + left_len - 1]) {
                        left_len -= 1;
                    }
                }
                if left_len != 0
                    && !PreprocessContext::push(
                        &mut context.scratch_b,
                        &mut next_count,
                        capacity,
                        ScratchFragment {
                            kind: FragmentKind::RawText,
                            start: fragment.start + base,
                            end: fragment.start + base + left_len,
                            token: -1,
                        },
                    )
                {
                    context.set_failure(PreprocessError::InvalidRequest);
                    return Ok(());
                }
                if token.token < 0
                    || !PreprocessContext::push(
                        &mut context.scratch_b,
                        &mut next_count,
                        capacity,
                        ScratchFragment {
                            kind: FragmentKind::Token,
                            start: 0,
                            end: 0,
                            token: token.token,
                        },
                    )
                {
                    context.set_failure(PreprocessError::InvalidRequest);
                    return Ok(());
                }
                base = match_at + token.text.len();
                if token.rstrip {
                    while base < raw.len() && source_is_space(raw.as_bytes()[base]) {
                        base += 1;
                    }
                }
            }
        }
        context.swap_scratch();
        current_count = next_count;
    }
    let mut out = output.borrow_mut();
    for index in 0..current_count {
        let fragment = context.scratch_a[index];
        out[index] = Fragment {
            kind: fragment.kind,
            text: if fragment.kind == FragmentKind::RawText {
                &event.request.text[fragment.start..fragment.end]
            } else {
                ""
            },
            token: fragment.token,
        };
    }
    context.set_result(true, current_count, true);
    Ok(())
}

/// Synchronous bounded actor around the generated WPM machine.
pub struct TextTokenizerPreprocessorWpmActor<'a> {
    machine: TextTokenizerPreprocessorWpmStateMachine<PreprocessContext>,
    last_error: PreprocessError,
    fragment_count: usize,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> Default for TextTokenizerPreprocessorWpmActor<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> TextTokenizerPreprocessorWpmActor<'a> {
    /// Creates an actor in the generated `idle` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorWpmStateMachine::new(PreprocessContext::default()),
            last_error: PreprocessError::None,
            fragment_count: 0,
            _marker: core::marker::PhantomData,
        }
    }

    /// Processes one request synchronously through the complete state machine.
    pub fn process_event<'dispatch, 'event>(
        &mut self,
        event: EventPreprocessRuntime<'event>,
    ) -> bool
    where
        'event: 'dispatch,
    {
        // Actions mutate the runtime context carried by the event, not the
        // generated machine's private callback context.
        let runtime_context = event.context;
        let accepted = self
            .machine
            .process_event(TextTokenizerPreprocessorWpmEvents::EventPreprocessRuntime(
                &event,
            ))
            .is_ok();
        let done = self.machine.is(&TextTokenizerPreprocessorWpmStates::Done);
        let mut context = runtime_context.borrow_mut();
        let ok = accepted && done && context.result;
        let error = if ok {
            PreprocessError::None
        } else if context.err == PreprocessError::None {
            PreprocessError::BackendError
        } else {
            context.err
        };
        context.err = error;
        let count = if ok { context.fragment_count } else { 0 };
        let preprocessed = ok && context.preprocessed;
        self.last_error = error;
        self.fragment_count = count;
        drop(context);

        *event.request.fragment_count_out.borrow_mut() = count;
        *event.request.error_out.borrow_mut() = error.code();
        if let Some(out) = event.request.preprocessed_out {
            *out.borrow_mut() = preprocessed;
        }
        if ok {
            if let Some(callback) = event.request.dispatch_done {
                let _ = callback(PreprocessDone {
                    fragment_count: count,
                });
            }
        } else if let Some(callback) = event.request.dispatch_error {
            let _ = callback(PreprocessErrorEvent { err: error });
        }
        ok
    }

    /// Sends an explicit unexpected event through the machine's error path.
    pub fn process_unexpected(&mut self) -> bool {
        self.last_error = PreprocessError::InvalidRequest;
        self.fragment_count = 0;
        self.machine.context_mut().unexpected();
        self.machine
            .set_state(TextTokenizerPreprocessorWpmStates::Unexpected);
        false
    }

    /// Compatibility spelling used by sibling preprocessor actors.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.process_unexpected()
    }

    /// Returns the generated state.
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorWpmStates {
        self.machine.state()
    }

    /// Reports whether the actor is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorWpmStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the generated machine context.
    #[must_use]
    pub fn context(&self) -> &PreprocessContext {
        self.machine.context()
    }

    /// Returns the source-compatible error from the last request.
    #[must_use]
    pub fn last_error(&self) -> i32 {
        self.last_error.code()
    }

    /// Returns the fragment count from the last request.
    #[must_use]
    pub const fn fragment_count(&self) -> usize {
        self.fragment_count
    }
}

pub type Wpm<'a> = TextTokenizerPreprocessorWpmActor<'a>;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestVocabulary;

    impl VocabularyView for TestVocabulary {
        fn token_count(&self) -> usize {
            1
        }

        fn token(&self, index: usize) -> Option<VocabularyToken<'_>> {
            (index == 0).then_some(VocabularyToken {
                text: "@@",
                token: 42,
                token_type: TOKEN_TYPE_UNKNOWN,
                lstrip: false,
                rstrip: false,
            })
        }
    }

    fn done_callback(done: PreprocessDone) -> bool {
        assert_eq!(done.fragment_count, 3);
        true
    }

    fn error_callback(error: PreprocessErrorEvent) -> bool {
        panic!("unexpected WPM preprocessing error: {:?}", error.err)
    }

    #[test]
    fn public_actor_partitions_special_and_publishes_success() {
        let vocab = TestVocabulary;
        let text = "left @@ right";
        let mut fragments = [Fragment::default(); 8];
        let fragments_out = RefCell::new(&mut fragments[..]);
        let fragment_count = RefCell::new(usize::MAX);
        let preprocessed = RefCell::new(false);
        let error = RefCell::new(-1);
        let request = PreprocessRequest::new(
            &vocab,
            text,
            true,
            &fragments_out,
            &fragment_count,
            Some(&preprocessed),
            &error,
            done_callback,
            error_callback,
        );
        let context = RefCell::new(PreprocessContext::default());
        let mut actor = TextTokenizerPreprocessorWpmActor::new();

        assert!(actor.process_event(EventPreprocessRuntime::new(request, &context)));
        assert_eq!(*fragment_count.borrow(), 3);
        assert!(*preprocessed.borrow());
        assert_eq!(*error.borrow(), 0);
        assert_eq!(actor.last_error(), 0);
        assert_eq!(actor.fragment_count(), 3);
        assert_eq!(
            fragments_out.borrow()[0],
            Fragment {
                kind: FragmentKind::RawText,
                text: "left ",
                token: -1
            }
        );
        assert_eq!(
            fragments_out.borrow()[1],
            Fragment {
                kind: FragmentKind::Token,
                text: "",
                token: 42
            }
        );
        assert_eq!(
            fragments_out.borrow()[2],
            Fragment {
                kind: FragmentKind::RawText,
                text: " right",
                token: -1
            }
        );
    }

    #[test]
    fn longest_special_wins_and_parse_disabled_filters_unknown() {
        struct Entries;
        impl VocabularyView for Entries {
            fn token_count(&self) -> usize {
                3
            }

            fn token(&self, index: usize) -> Option<VocabularyToken<'_>> {
                [
                    VocabularyToken {
                        text: "@",
                        token: 7,
                        token_type: TOKEN_TYPE_USER_DEFINED,
                        lstrip: false,
                        rstrip: false,
                    },
                    VocabularyToken {
                        text: "@@",
                        token: 42,
                        token_type: TOKEN_TYPE_USER_DEFINED,
                        lstrip: false,
                        rstrip: false,
                    },
                    VocabularyToken {
                        text: "#",
                        token: 9,
                        token_type: TOKEN_TYPE_UNKNOWN,
                        lstrip: false,
                        rstrip: false,
                    },
                ]
                .get(index)
                .copied()
            }
        }

        let vocab = Entries;
        let mut fragments = [Fragment::default(); 8];
        let fragments_out = RefCell::new(&mut fragments[..]);
        let count = RefCell::new(0);
        let preprocessed = RefCell::new(false);
        let error = RefCell::new(-1);
        let request = PreprocessRequest::new(
            &vocab,
            "x @@ # @",
            false,
            &fragments_out,
            &count,
            Some(&preprocessed),
            &error,
            |_| true,
            |_| true,
        );
        let context = RefCell::new(PreprocessContext::default());
        let mut actor = TextTokenizerPreprocessorWpmActor::new();

        assert!(actor.process_event(EventPreprocessRuntime::new(request, &context)));
        assert_eq!(*count.borrow(), 4);
        assert_eq!(fragments_out.borrow()[0].text, "x ");
        assert_eq!(fragments_out.borrow()[1].token, 42);
        assert_eq!(fragments_out.borrow()[2].text, " # ");
        assert_eq!(fragments_out.borrow()[3].token, 7);
        assert!(*preprocessed.borrow());
        assert_eq!(*error.borrow(), 0);
    }

    #[test]
    fn lstrip_and_rstrip_remove_adjacent_source_whitespace() {
        struct Entries;
        impl VocabularyView for Entries {
            fn token_count(&self) -> usize {
                1
            }

            fn token(&self, index: usize) -> Option<VocabularyToken<'_>> {
                (index == 0).then_some(VocabularyToken {
                    text: "<x>",
                    token: 11,
                    token_type: TOKEN_TYPE_USER_DEFINED,
                    lstrip: true,
                    rstrip: true,
                })
            }
        }

        let vocab = Entries;
        let mut fragments = [Fragment::default(); 8];
        let fragments_out = RefCell::new(&mut fragments[..]);
        let count = RefCell::new(0);
        let preprocessed = RefCell::new(false);
        let error = RefCell::new(-1);
        let request = PreprocessRequest::new(
            &vocab,
            "left   <x>  right",
            true,
            &fragments_out,
            &count,
            Some(&preprocessed),
            &error,
            |_| true,
            |_| true,
        );
        let context = RefCell::new(PreprocessContext::default());
        let mut actor = TextTokenizerPreprocessorWpmActor::new();

        assert!(actor.process_event(EventPreprocessRuntime::new(request, &context)));
        assert_eq!(*count.borrow(), 3);
        assert_eq!(fragments_out.borrow()[0].text, "left");
        assert_eq!(fragments_out.borrow()[1].token, 11);
        assert_eq!(fragments_out.borrow()[2].text, "right");
    }

    #[test]
    fn missing_output_is_invalid_and_unexpected_state_is_explicit() {
        let vocab = TestVocabulary;
        let count = RefCell::new(usize::MAX);
        let preprocessed = RefCell::new(true);
        let error = RefCell::new(-1);
        let request = PreprocessRequest::with_callbacks(
            &vocab,
            "text",
            true,
            None,
            &count,
            Some(&preprocessed),
            &error,
            Some(|_| true),
            Some(|_| true),
        );
        let context = RefCell::new(PreprocessContext::default());
        let mut actor = TextTokenizerPreprocessorWpmActor::new();

        assert!(!actor.process_event(EventPreprocessRuntime::new(request, &context)));
        assert_eq!(*count.borrow(), 0);
        assert!(!*preprocessed.borrow());
        assert_eq!(*error.borrow(), 1);
        assert_eq!(actor.last_error(), 1);

        assert!(!actor.process_unexpected_event());
        assert!(actor.is(&TextTokenizerPreprocessorWpmStates::Unexpected));
        assert!(actor.context().unexpected);
    }
}
