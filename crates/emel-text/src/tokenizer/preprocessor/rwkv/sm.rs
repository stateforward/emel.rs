//! Source-aligned, bounded RWKV tokenizer preprocessor actor.
//!
//! The machine follows the pinned `preprocessor::rwkv` transition table.  All
//! request storage is caller-owned, processing is synchronous, and callbacks
//! are function pointers so a request never allocates or defers work.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::needless_lifetimes,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

/// Maximum number of output fragments accepted by the pinned contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum number of special-token entries retained by the bounded cache.
pub const MAX_SPECIAL_TOKENS: usize = 1024;

/// Token classes used by the vocabulary contract.
pub const TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const TOKEN_TYPE_CONTROL: i32 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// Errors reported by preprocessing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessorError {
    /// No error occurred.
    #[default]
    None = 0,
    /// The request or bounded destination is invalid.
    InvalidRequest = 1,
    /// The backend could not complete the operation.
    BackendError = 2,
}

impl PreprocessorError {
    /// Returns the source-compatible integer error code.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::InvalidRequest => 1,
            Self::BackendError => 2,
        }
    }
}

/// Output fragment kind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FragmentKind {
    /// A view into the original request text.
    #[default]
    RawText,
    /// A vocabulary token id.
    Token,
}

/// Caller-owned output fragment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'text> {
    /// Kind of this fragment.
    pub kind: FragmentKind,
    /// Raw text for [`FragmentKind::RawText`].
    pub text: &'text str,
    /// Token id for [`FragmentKind::Token`], or `-1` for raw fragments.
    pub token: i32,
}

/// One bounded vocabulary entry used while building the special-token cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VocabularyEntry<'text> {
    /// Token spelling.
    pub text: &'text str,
    /// Token id.
    pub token: i32,
    /// Source token class.
    pub token_type: i32,
    /// Remove whitespace immediately before a match.
    pub lstrip: bool,
    /// Remove whitespace immediately after a match.
    pub rstrip: bool,
}

/// Borrowed vocabulary view.  The caller owns entries and their spellings.
#[derive(Clone, Copy, Debug)]
pub struct Vocabulary<'text> {
    /// Vocabulary entries indexed by token id.
    pub entries: &'text [VocabularyEntry<'text>],
}

impl<'text> Vocabulary<'text> {
    /// Constructs a borrowed vocabulary view.
    #[must_use]
    pub const fn new(entries: &'text [VocabularyEntry<'text>]) -> Self {
        Self { entries }
    }
}

/// Successful completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessDone {
    /// Number of output fragments.
    pub fragment_count: usize,
}

/// Failed completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessErrorEvent {
    /// Source-compatible error.
    pub error: PreprocessorError,
}
/// Compatibility alias used by sibling preprocessor actors.
pub type PreprocessError = PreprocessorError;
/// Compatibility vocabulary-entry name.
pub type VocabEntry<'text> = VocabularyEntry<'text>;
/// Compatibility failure payload name.
pub type PreprocessFailure = PreprocessErrorEvent;

/// Synchronous successful completion callback.
pub type DoneCallback = fn(PreprocessDone) -> bool;
/// Synchronous failed completion callback.
pub type ErrorCallback = fn(PreprocessErrorEvent) -> bool;

/// Caller-owned preprocessing request.
#[derive(Clone, Copy, Debug)]
pub struct PreprocessRequest<'event> {
    /// Vocabulary used to identify special tokens.
    pub vocab: Vocabulary<'event>,
    /// Text to partition.
    pub text: &'event str,
    /// Whether control/user-defined/unknown tokens are parsed as specials.
    pub parse_special: bool,
    /// Caller-owned bounded destination. `None` models a null destination.
    pub fragments_out: Option<&'event RefCell<&'event mut [Fragment<'event>]>>,
    /// Caller-owned completion outputs.
    pub fragment_count_out: &'event RefCell<usize>,
    pub preprocessed_out: Option<&'event RefCell<bool>>,
    pub error_out: &'event RefCell<i32>,
    /// Optional successful completion callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional failed completion callback.
    pub dispatch_error: Option<ErrorCallback>,
}

impl<'event> PreprocessRequest<'event> {
    /// Constructs a request with both callbacks installed.
    #[must_use]
    pub const fn new(
        vocab: Vocabulary<'event>,
        text: &'event str,
        parse_special: bool,
        fragments_out: &'event RefCell<&'event mut [Fragment<'event>]>,
        fragment_count_out: &'event RefCell<usize>,
        preprocessed_out: Option<&'event RefCell<bool>>,
        error_out: &'event RefCell<i32>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self {
            vocab,
            text,
            parse_special,
            fragments_out: Some(fragments_out),
            fragment_count_out,
            preprocessed_out,
            error_out,
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }

    /// Constructs a request with independently optional destination/callbacks.
    #[must_use]
    pub const fn with_callbacks(
        vocab: Vocabulary<'event>,
        text: &'event str,
        parse_special: bool,
        fragments_out: Option<&'event RefCell<&'event mut [Fragment<'event>]>>,
        fragment_count_out: &'event RefCell<usize>,
        preprocessed_out: Option<&'event RefCell<bool>>,
        error_out: &'event RefCell<i32>,
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

/// Mutable result context corresponding to the source `preprocess_ctx`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessContext {
    /// Number of produced fragments.
    pub fragment_count: usize,
    /// Whether the text was successfully preprocessed.
    pub preprocessed: bool,
    /// Error from the current phase.
    pub phase_error: PreprocessorError,
    /// Final request error.
    pub error: PreprocessorError,
    /// Final request result.
    pub result: bool,
    /// Set after an unexpected event.
    pub unexpected: bool,
    special_ids: [usize; MAX_SPECIAL_TOKENS],
    special_count: usize,
}

impl Default for PreprocessContext {
    fn default() -> Self {
        Self {
            fragment_count: 0,
            preprocessed: false,
            phase_error: PreprocessorError::None,
            error: PreprocessorError::None,
            result: false,
            unexpected: false,
            special_ids: [0; MAX_SPECIAL_TOKENS],
            special_count: 0,
        }
    }
}

impl PreprocessContext {
    fn reset(&mut self) {
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::None;
        self.error = PreprocessorError::None;
        self.result = false;
        self.unexpected = false;
        self.special_count = 0;
    }
}

/// Runtime event carrying one borrowed request and result context.
#[derive(Clone, Copy, Debug)]
pub struct EventPreprocessRuntime<'event> {
    /// Request fields.
    pub request: PreprocessRequest<'event>,
    /// Mutable result context.
    pub context: &'event RefCell<PreprocessContext>,
}

impl<'event> EventPreprocessRuntime<'event> {
    /// Constructs one runtime event.
    #[must_use]
    pub const fn new(
        request: PreprocessRequest<'event>,
        context: &'event RefCell<PreprocessContext>,
    ) -> Self {
        Self { request, context }
    }
}

sml! {
    TextTokenizerPreprocessorRwkv<'dispatch, 'event>
    where
        'event: 'dispatch, {
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

/// Compatibility context name emitted by the state-machine generator.
pub type TextTokenizerPreprocessorRwkvContext = PreprocessContext;

impl TextTokenizerPreprocessorRwkvStateMachineContext for TextTokenizerPreprocessorRwkvContext {
    fn begin_preprocess<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.context.borrow_mut().reset();
        *event.request.fragment_count_out.borrow_mut() = 0;
        if let Some(out) = event.request.preprocessed_out {
            *out.borrow_mut() = false;
        }
        *event.request.error_out.borrow_mut() = 0;
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
        context.phase_error = PreprocessorError::None;
        for (index, entry) in event.request.vocab.entries.iter().enumerate() {
            let special = matches!(
                entry.token_type,
                TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
            ) && !entry.text.is_empty();
            if !special {
                continue;
            }
            if context.special_count == MAX_SPECIAL_TOKENS {
                context.phase_error = PreprocessorError::InvalidRequest;
                return Ok(());
            }
            let slot = context.special_count;
            context.special_ids[slot] = index;
            context.special_count = slot + 1;
        }
        // The source cache is ordered longest-first to ensure deterministic
        // handling when one special token is a prefix of another.
        let count = context.special_count;
        let mut i = 1;
        while i < count {
            let id = context.special_ids[i];
            let len = event.request.vocab.entries[id].text.len();
            let mut j = i;
            while j > 0 {
                let previous = context.special_ids[j - 1];
                if event.request.vocab.entries[previous].text.len() >= len {
                    break;
                }
                context.special_ids[j] = previous;
                j -= 1;
            }
            context.special_ids[j] = id;
            i += 1;
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
        Ok(event.context.borrow().phase_error == PreprocessorError::None)
    }
    fn build_specials_invalid_request_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessorError::InvalidRequest)
    }
    fn build_specials_backend_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessorError::BackendError)
    }
    fn build_specials_unknown_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let e = event.context.borrow().phase_error;
        Ok(e != PreprocessorError::None
            && e != PreprocessorError::InvalidRequest
            && e != PreprocessorError::BackendError)
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
        Ok(event.context.borrow().phase_error == PreprocessorError::BackendError)
    }
    fn partition_invalid_request_error<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessorError::InvalidRequest)
    }
    fn partition_unknown_error<'dispatch, 'event>(
        &self,
        _event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(false)
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
        out[0] = Fragment {
            kind: FragmentKind::RawText,
            text: event.request.text,
            token: -1,
        };
        set_phase_result(event, 1, true);
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
    fn partition_ok<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.context.borrow().phase_error == PreprocessorError::None)
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
    fn set_empty_partition_result_from_partitioning_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_phase_result(event, 0, true);
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
        set_phase_result(event, 0, true);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        set_phase_result(event, 0, true);
        Ok(())
    }

    fn mark_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut c = event.context.borrow_mut();
        c.phase_error = PreprocessorError::None;
        c.error = PreprocessorError::None;
        c.result = true;
        *event.request.fragment_count_out.borrow_mut() = c.fragment_count;
        if let Some(out) = event.request.preprocessed_out {
            *out.borrow_mut() = c.preprocessed;
        }
        *event.request.error_out.borrow_mut() = 0;
        Ok(())
    }

    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        mark_unexpected(self)
    }
}

fn mark_unexpected(context: &mut PreprocessContext) -> Result<(), ()> {
    context.unexpected = true;
    context.fragment_count = 0;
    context.preprocessed = false;
    context.phase_error = PreprocessorError::InvalidRequest;
    context.error = PreprocessorError::InvalidRequest;
    context.result = false;
    Ok(())
}

fn set_phase_result(event: &EventPreprocessRuntime<'_>, count: usize, preprocessed: bool) {
    let mut c = event.context.borrow_mut();
    c.fragment_count = count;
    c.preprocessed = preprocessed;
    c.phase_error = PreprocessorError::None;
    c.error = PreprocessorError::None;
    c.result = false;
}

fn reject_invalid<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
) -> Result<(), ()>
where
    'event: 'dispatch,
{
    let mut c = event.context.borrow_mut();
    c.fragment_count = 0;
    c.preprocessed = false;
    c.phase_error = PreprocessorError::InvalidRequest;
    c.error = PreprocessorError::InvalidRequest;
    c.result = false;
    *event.request.fragment_count_out.borrow_mut() = 0;
    if let Some(out) = event.request.preprocessed_out {
        *out.borrow_mut() = false;
    }
    *event.request.error_out.borrow_mut() = c.error.code();
    Ok(())
}

fn ensure_last_error<'dispatch, 'event>(
    event: &'dispatch EventPreprocessRuntime<'event>,
) -> Result<(), ()>
where
    'event: 'dispatch,
{
    let mut c = event.context.borrow_mut();
    c.error = if c.phase_error == PreprocessorError::None {
        PreprocessorError::BackendError
    } else {
        c.phase_error
    };
    c.result = false;
    *event.request.fragment_count_out.borrow_mut() = 0;
    if let Some(out) = event.request.preprocessed_out {
        *out.borrow_mut() = false;
    }
    *event.request.error_out.borrow_mut() = c.error.code();
    Ok(())
}

fn allowed(entry: VocabularyEntry<'_>, parse_special: bool) -> bool {
    !entry.text.is_empty()
        && (parse_special || !matches!(entry.token_type, TOKEN_TYPE_CONTROL | TOKEN_TYPE_UNKNOWN))
}

/// Matches the byte-oriented `std::isspace(unsigned char)` checks in the source.
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
    let mut current = [Fragment::default(); MAX_FRAGMENTS];
    let mut next = [Fragment::default(); MAX_FRAGMENTS];
    let mut current_count = 1usize;
    current[0] = Fragment {
        kind: FragmentKind::RawText,
        text: event.request.text,
        token: -1,
    };
    let context = event.context.borrow();
    let ids = context.special_ids;
    let special_count = context.special_count;
    drop(context);
    for id_index in 0..special_count {
        let entry = event.request.vocab.entries[ids[id_index]];
        if !allowed(entry, parse_special) {
            continue;
        }
        let mut next_count = 0usize;
        for fragment in current[..current_count].iter().copied() {
            if fragment.kind == FragmentKind::Token {
                if next_count == capacity || next_count == MAX_FRAGMENTS {
                    return reject_invalid(event);
                }
                next[next_count] = fragment;
                next_count += 1;
                continue;
            }
            let mut rest = fragment.text;
            loop {
                let Some(match_at) = rest.find(entry.text) else {
                    if !rest.is_empty() {
                        if next_count == capacity || next_count == MAX_FRAGMENTS {
                            return reject_invalid(event);
                        }
                        next[next_count] = Fragment {
                            kind: FragmentKind::RawText,
                            text: rest,
                            token: -1,
                        };
                        next_count += 1;
                    }
                    break;
                };
                let mut left = &rest[..match_at];
                if entry.lstrip {
                    left = trim_source_left(left);
                }
                if !left.is_empty() {
                    if next_count == capacity || next_count == MAX_FRAGMENTS {
                        return reject_invalid(event);
                    }
                    next[next_count] = Fragment {
                        kind: FragmentKind::RawText,
                        text: left,
                        token: -1,
                    };
                    next_count += 1;
                }
                if next_count == capacity || next_count == MAX_FRAGMENTS || entry.token < 0 {
                    return reject_invalid(event);
                }
                let token_id = entry.token;
                next[next_count] = Fragment {
                    kind: FragmentKind::Token,
                    text: "",
                    token: token_id,
                };
                next_count += 1;
                let mut after = &rest[match_at + entry.text.len()..];
                if entry.rstrip {
                    after = trim_source_right(after);
                }
                rest = after;
            }
        }
        current[..next_count].copy_from_slice(&next[..next_count]);
        current_count = next_count;
    }
    let mut out = output.borrow_mut();
    out[..current_count].copy_from_slice(&current[..current_count]);
    set_phase_result(event, current_count, true);
    Ok(())
}

/// Synchronous bounded actor around the generated RWKV machine.
pub struct TextTokenizerPreprocessorRwkvActor {
    machine: TextTokenizerPreprocessorRwkvStateMachine<TextTokenizerPreprocessorRwkvContext>,
    last_error: PreprocessorError,
    fragment_count: usize,
}

impl Default for TextTokenizerPreprocessorRwkvActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTokenizerPreprocessorRwkvActor {
    /// Creates an actor in the generated `idle` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorRwkvStateMachine::new(
                TextTokenizerPreprocessorRwkvContext::default(),
            ),
            last_error: PreprocessorError::None,
            fragment_count: 0,
        }
    }

    /// Processes one bounded request to run-to-completion.
    pub fn process_event<'dispatch, 'event>(
        &mut self,
        event: EventPreprocessRuntime<'event>,
    ) -> bool
    where
        'event: 'dispatch,
    {
        let context = event.context;
        let accepted = self
            .machine
            .process_event(TextTokenizerPreprocessorRwkvEvents::EventPreprocessRuntime(
                &event,
            ))
            .is_ok();
        let ok = accepted
            && context.borrow().result
            && self.machine.is(&TextTokenizerPreprocessorRwkvStates::Done);
        let mut result = context.borrow_mut();
        result.error = if ok {
            PreprocessorError::None
        } else if result.error == PreprocessorError::None {
            PreprocessorError::BackendError
        } else {
            result.error
        };
        let error = result.error;
        let fragment_count = result.fragment_count;
        self.last_error = error;
        self.fragment_count = fragment_count;
        drop(result);
        if !ok {
            *event.request.fragment_count_out.borrow_mut() = 0;
            if let Some(out) = event.request.preprocessed_out {
                *out.borrow_mut() = false;
            }
            *event.request.error_out.borrow_mut() = error.code();
        }
        if ok {
            if let Some(callback) = event.request.dispatch_done {
                let _ = callback(PreprocessDone { fragment_count });
            }
        } else if let Some(callback) = event.request.dispatch_error {
            let _ = callback(PreprocessErrorEvent { error });
        }
        ok
    }

    /// Sends an explicit unexpected event through the machine's error path.
    pub fn process_unexpected(&mut self) -> bool {
        self.last_error = PreprocessorError::InvalidRequest;
        self.fragment_count = 0;
        let context = self.machine.context_mut();
        context.unexpected = true;
        context.phase_error = PreprocessorError::InvalidRequest;
        context.error = PreprocessorError::InvalidRequest;
        context.result = false;
        context.preprocessed = false;
        context.fragment_count = 0;
        self.machine
            .set_state(TextTokenizerPreprocessorRwkvStates::Unexpected);
        false
    }

    /// Returns the generated state.
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorRwkvStates {
        self.machine.state()
    }
    /// Reports whether the actor is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorRwkvStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the machine context.
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorRwkvContext {
        self.machine.context()
    }
    /// Returns the source-compatible numeric error code from the last request.
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

/// Short public alias matching the pinned machine name.
pub type Actor = TextTokenizerPreprocessorRwkvActor;
