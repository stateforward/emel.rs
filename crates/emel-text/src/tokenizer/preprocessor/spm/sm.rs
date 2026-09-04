//! Source-aligned, bounded SentencePiece-style tokenizer preprocessor actor.
//!
//! The machine keeps all intermediate fragments in fixed-size caller-independent
//! scratch buffers. Requests borrow the vocabulary, source text, and output
//! fragment storage; dispatch and optional completion callbacks are synchronous.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unused_self,
    clippy::needless_lifetimes,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

use core::cell::RefCell;
use sml::sml;

/// Maximum number of output fragments accepted by the pinned contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum number of special-token entries retained by the pinned contract.
pub const MAX_SPECIAL_TOKENS: usize = 1024;

/// Token classes used by the preprocessor's special-token policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenType {
    #[default]
    Normal = 1,
    /// SentencePiece/control token class skipped when special parsing is off.
    Unknown = 2,
    /// Control token class skipped when special parsing is off.
    Control = 3,
    /// User-defined token class retained when special parsing is off.
    UserDefined = 4,
}

/// One bounded vocabulary entry borrowed by a request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VocabularyToken<'a> {
    pub text: &'a str,
    pub token: i32,
    pub token_type: TokenType,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Vocabulary view used by [`EventPreprocessRuntime`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Vocabulary<'a> {
    pub tokens: &'a [VocabularyToken<'a>],
}

/// Concise alias matching the model-layer terminology.
pub type Vocab<'a> = Vocabulary<'a>;

/// Kind of a produced fragment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

/// Caller-owned output fragment. Raw fragments borrow the request text; token
/// fragments have an empty text view and a non-negative token identifier.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'a> {
    pub kind: FragmentKind,
    pub text: &'a str,
    pub token: i32,
}

/// Errors represented by the pinned preprocessor boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessorError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    BackendError = 2,
}

impl PreprocessorError {
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::InvalidRequest => 1,
            Self::BackendError => 2,
        }
    }
}

/// Successful synchronous completion payload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreprocessDone {
    pub fragment_count: usize,
}

/// Error synchronous completion payload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreprocessError {
    pub error: PreprocessorError,
}

/// Runtime request over caller-owned input and output storage.
///
/// Callback function pointers are deliberately non-capturing and optional: no
/// heap allocation, deferred task, or unbounded callback list is introduced.
pub struct EventPreprocessRuntime<'a> {
    pub vocab: &'a Vocabulary<'a>,
    pub text: &'a str,
    pub parse_special: bool,
    pub fragments_out: &'a RefCell<&'a mut [Fragment<'a>]>,
    pub fragment_count_out: &'a RefCell<usize>,
    pub preprocessed_out: Option<&'a RefCell<bool>>,
    pub error_out: &'a RefCell<PreprocessorError>,
    pub on_done: Option<fn(&PreprocessDone)>,
    pub on_error: Option<fn(&PreprocessError)>,
}

impl<'a> EventPreprocessRuntime<'a> {
    #[must_use]
    pub fn new(
        vocab: &'a Vocabulary<'a>,
        text: &'a str,
        parse_special: bool,
        fragments_out: &'a RefCell<&'a mut [Fragment<'a>]>,
        fragment_count_out: &'a RefCell<usize>,
        preprocessed_out: Option<&'a RefCell<bool>>,
        error_out: &'a RefCell<PreprocessorError>,
    ) -> Self {
        Self {
            vocab,
            text,
            parse_special,
            fragments_out,
            fragment_count_out,
            preprocessed_out,
            error_out,
            on_done: None,
            on_error: None,
        }
    }
}

sml! {
    TextTokenizerPreprocessorSpm<'dispatch, 'event>
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CachedSpecial {
    id: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ScratchFragment {
    kind: FragmentKind,
    start: usize,
    end: usize,
    token: i32,
}

/// Persistent bounded context for the SPM preprocessor machine.
#[derive(Debug)]
pub struct TextTokenizerPreprocessorSpmContext {
    pub fragment_count: usize,
    pub preprocessed: bool,
    pub phase_error: PreprocessorError,
    pub err: PreprocessorError,
    pub result: bool,
    pub special_count: usize,
    specials: [CachedSpecial; MAX_SPECIAL_TOKENS],
    scratch_a: [ScratchFragment; MAX_FRAGMENTS],
    scratch_b: [ScratchFragment; MAX_FRAGMENTS],
}

impl Default for TextTokenizerPreprocessorSpmContext {
    fn default() -> Self {
        Self {
            fragment_count: 0,
            preprocessed: false,
            phase_error: PreprocessorError::None,
            err: PreprocessorError::None,
            result: false,
            special_count: 0,
            specials: [CachedSpecial::default(); MAX_SPECIAL_TOKENS],
            scratch_a: [ScratchFragment::default(); MAX_FRAGMENTS],
            scratch_b: [ScratchFragment::default(); MAX_FRAGMENTS],
        }
    }
}

impl TextTokenizerPreprocessorSpmContext {
    fn clear(&mut self, event: &EventPreprocessRuntime<'_>) {
        *event.fragment_count_out.borrow_mut() = 0;
        if let Some(preprocessed) = event.preprocessed_out {
            *preprocessed.borrow_mut() = false;
        }
        *event.error_out.borrow_mut() = PreprocessorError::None;
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::None;
        self.err = PreprocessorError::None;
        self.result = false;
        self.special_count = 0;
    }

    fn set_phase_result(&mut self, ok: bool, count: usize, preprocessed: bool) {
        self.fragment_count = if ok { count } else { 0 };
        self.preprocessed = ok && preprocessed;
        self.phase_error = if ok {
            PreprocessorError::None
        } else {
            PreprocessorError::InvalidRequest
        };
        self.err = self.phase_error;
        self.result = false;
    }

    fn ensure_error(&mut self) {
        self.err = if self.phase_error == PreprocessorError::None {
            PreprocessorError::BackendError
        } else {
            self.phase_error
        };
        self.result = false;
    }

    fn unexpected(&mut self) {
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::InvalidRequest;
        self.err = PreprocessorError::InvalidRequest;
        self.result = false;
    }

    fn token<'a>(
        &self,
        event: &EventPreprocessRuntime<'a>,
        id: usize,
    ) -> Option<VocabularyToken<'a>> {
        event.vocab.tokens.get(id).copied()
    }

    fn push_scratch(
        target: &mut [ScratchFragment; MAX_FRAGMENTS],
        count: &mut usize,
        value: ScratchFragment,
        capacity: usize,
    ) -> bool {
        if *count >= capacity || *count >= MAX_FRAGMENTS {
            return false;
        }
        target[*count] = value;
        *count += 1;
        true
    }

    fn materialize<'a>(
        source: &'a str,
        scratch: &[ScratchFragment; MAX_FRAGMENTS],
        count: usize,
        output: &mut [Fragment<'a>],
    ) -> bool {
        if count > output.len() || count > MAX_FRAGMENTS {
            return false;
        }
        for index in 0..count {
            let fragment = scratch[index];
            if fragment.kind == FragmentKind::Token {
                if fragment.token < 0 {
                    return false;
                }
                output[index] = Fragment {
                    kind: FragmentKind::Token,
                    text: "",
                    token: fragment.token,
                };
            } else {
                let Some(text) = source.get(fragment.start..fragment.end) else {
                    return false;
                };
                output[index] = Fragment {
                    kind: FragmentKind::RawText,
                    text,
                    token: -1,
                };
            }
        }
        true
    }

    fn partition(&mut self, event: &EventPreprocessRuntime<'_>, allow_skipped: bool) -> bool {
        let mut output = event.fragments_out.borrow_mut();
        let capacity = output.len();
        let mut current_count = 0usize;
        if !Self::push_scratch(
            &mut self.scratch_a,
            &mut current_count,
            ScratchFragment {
                kind: FragmentKind::RawText,
                start: 0,
                end: event.text.len(),
                token: -1,
            },
            capacity,
        ) {
            self.set_phase_result(false, 0, false);
            return false;
        }
        for special_index in 0..self.special_count {
            let id = self.specials[special_index].id;
            let Some(token) = self.token(event, id) else {
                self.set_phase_result(false, 0, false);
                return false;
            };
            let token_text = token.text;
            let token_id = token.token;
            let token_lstrip = token.lstrip;
            let token_rstrip = token.rstrip;
            if token_text.is_empty()
                || (!allow_skipped
                    && matches!(token.token_type, TokenType::Control | TokenType::Unknown))
            {
                continue;
            }
            let mut next_count = 0usize;
            let mut ok = true;
            for fragment_index in 0..current_count {
                let fragment = self.scratch_a[fragment_index];
                if fragment.kind == FragmentKind::Token {
                    ok = ok
                        && Self::push_scratch(
                            &mut self.scratch_b,
                            &mut next_count,
                            fragment,
                            capacity,
                        );
                    continue;
                }
                let raw = &event.text[fragment.start..fragment.end];
                let mut base = 0usize;
                while base < raw.len() {
                    let Some(relative) = raw[base..].find(token_text) else {
                        ok = ok
                            && Self::push_scratch(
                                &mut self.scratch_b,
                                &mut next_count,
                                ScratchFragment {
                                    kind: FragmentKind::RawText,
                                    start: fragment.start + base,
                                    end: fragment.end,
                                    token: -1,
                                },
                                capacity,
                            );
                        base = raw.len();
                        continue;
                    };
                    let match_at = base + relative;
                    let mut left_len = match_at - base;
                    if token_lstrip {
                        while left_len > 0
                            && raw.as_bytes()[base + left_len - 1].is_ascii_whitespace()
                        {
                            left_len -= 1;
                        }
                    }
                    if left_len != 0 {
                        ok = ok
                            && Self::push_scratch(
                                &mut self.scratch_b,
                                &mut next_count,
                                ScratchFragment {
                                    kind: FragmentKind::RawText,
                                    start: fragment.start + base,
                                    end: fragment.start + base + left_len,
                                    token: -1,
                                },
                                capacity,
                            );
                    }
                    if token_id < 0
                        || !Self::push_scratch(
                            &mut self.scratch_b,
                            &mut next_count,
                            ScratchFragment {
                                kind: FragmentKind::Token,
                                start: 0,
                                end: 0,
                                token: token_id,
                            },
                            capacity,
                        )
                    {
                        ok = false;
                    }
                    base = match_at + token_text.len();
                    if token_rstrip {
                        while base < raw.len() && raw.as_bytes()[base].is_ascii_whitespace() {
                            base += 1;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                self.set_phase_result(false, 0, false);
                return false;
            }
            core::mem::swap(&mut self.scratch_a, &mut self.scratch_b);
            current_count = next_count;
        }
        if !Self::materialize(event.text, &self.scratch_a, current_count, &mut output) {
            self.set_phase_result(false, 0, false);
            return false;
        }
        self.set_phase_result(true, current_count, true);
        true
    }
}

impl TextTokenizerPreprocessorSpmStateMachineContext for TextTokenizerPreprocessorSpmContext {
    fn begin_preprocess<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.clear(event);
        Ok(())
    }
    fn build_specials<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.special_count = 0;
        for (id, token) in event.vocab.tokens.iter().enumerate() {
            if !token.text.is_empty()
                && matches!(
                    token.token_type,
                    TokenType::Unknown | TokenType::Control | TokenType::UserDefined
                )
            {
                if self.special_count >= MAX_SPECIAL_TOKENS {
                    self.set_phase_result(false, 0, false);
                    return Ok(());
                }
                self.specials[self.special_count] = CachedSpecial { id };
                self.special_count += 1;
            }
        }
        for index in 1..self.special_count {
            let mut position = index;
            while position > 0 {
                let left = self.specials[position - 1].id;
                let right = self.specials[position].id;
                if event.vocab.tokens[left].text.len() >= event.vocab.tokens[right].text.len() {
                    break;
                }
                self.specials.swap(position - 1, position);
                position -= 1;
            }
        }
        self.set_phase_result(true, 0, false);
        Ok(())
    }
    fn build_specials_backend_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::BackendError)
    }
    fn build_specials_invalid_request_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::InvalidRequest)
    }
    fn build_specials_ok<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::None)
    }
    fn build_specials_unknown_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(false)
    }
    fn ensure_last_error_from_build_specials_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_parse_special_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_specials_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn fragments_buffer_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.fragments_out.borrow().is_empty())
    }
    fn fragments_buffer_present<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.fragments_out.borrow().is_empty())
    }
    fn fragments_capacity_exceeds_limit<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.fragments_out.borrow().len() > MAX_FRAGMENTS)
    }
    fn fragments_capacity_nonzero<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.fragments_out.borrow().is_empty())
    }
    fn fragments_capacity_within_limit<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.fragments_out.borrow().len() <= MAX_FRAGMENTS)
    }
    fn fragments_capacity_zero<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.fragments_out.borrow().is_empty())
    }
    fn has_specials<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.special_count != 0)
    }
    fn no_specials<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.special_count == 0)
    }
    fn partition_invalid_request_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::InvalidRequest)
    }
    fn partition_ok<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::None)
    }
    fn partition_unknown_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(false)
    }
    fn parse_special_disabled<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.parse_special)
    }
    fn parse_special_enabled<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.parse_special)
    }
    fn reject_invalid_from_request_buffer_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_error = PreprocessorError::InvalidRequest;
        self.ensure_error();
        Ok(())
    }
    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn mark_done<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_error = PreprocessorError::None;
        self.err = PreprocessorError::None;
        self.result = true;
        Ok(())
    }
    fn partition_backend_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::BackendError)
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
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
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> {
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
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> {
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
    fn partition_no_specials<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.partition(event, true);
        Ok(())
    }
    fn partition_non_bpe_parse_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.partition(event, true);
        Ok(())
    }
    fn partition_non_bpe_skip_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.partition(event, false);
        Ok(())
    }
    fn reject_invalid_from_request_capacity_limit_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_error = PreprocessorError::InvalidRequest;
        self.ensure_error();
        Ok(())
    }
    fn reject_invalid_from_request_capacity_nonzero_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_error = PreprocessorError::InvalidRequest;
        self.ensure_error();
        Ok(())
    }
    fn request_text_empty<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.text.is_empty())
    }
    fn request_text_nonempty<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.text.is_empty())
    }
    fn set_empty_partition_result_from_partitioning_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.set_phase_result(true, 0, true);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.set_phase_result(true, 0, true);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.set_phase_result(true, 0, true);
        Ok(())
    }
}

/// Single-writer, synchronous SPM preprocessor actor.
pub struct TextTokenizerPreprocessorSpm {
    machine: TextTokenizerPreprocessorSpmStateMachine<TextTokenizerPreprocessorSpmContext>,
    last_error: PreprocessorError,
    fragment_count: usize,
}

impl Default for TextTokenizerPreprocessorSpm {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTokenizerPreprocessorSpm {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorSpmStateMachine::new(
                TextTokenizerPreprocessorSpmContext::default(),
            ),
            last_error: PreprocessorError::None,
            fragment_count: 0,
        }
    }

    pub fn process_event(
        &mut self,
        event: EventPreprocessRuntime<'_>,
    ) -> Result<PreprocessDone, PreprocessorError> {
        let accepted = self
            .machine
            .process_event(TextTokenizerPreprocessorSpmEvents::EventPreprocessRuntime(
                &event,
            ))
            .is_ok();
        let done = self.machine.is(&TextTokenizerPreprocessorSpmStates::Done);
        let context = self.machine.context_mut();
        let ok = accepted && done && context.result;
        let error = if ok {
            PreprocessorError::None
        } else if context.err == PreprocessorError::None {
            PreprocessorError::BackendError
        } else {
            context.err
        };
        context.err = error;
        self.last_error = error;
        self.fragment_count = if ok { context.fragment_count } else { 0 };
        *event.fragment_count_out.borrow_mut() = self.fragment_count;
        if let Some(preprocessed) = event.preprocessed_out {
            *preprocessed.borrow_mut() = ok && context.preprocessed;
        }
        *event.error_out.borrow_mut() = error;
        if ok {
            let done = PreprocessDone {
                fragment_count: self.fragment_count,
            };
            if let Some(callback) = event.on_done {
                callback(&done);
            }
            Ok(done)
        } else {
            let failure = PreprocessError { error };
            if let Some(callback) = event.on_error {
                callback(&failure);
            }
            Err(error)
        }
    }

    #[must_use]
    pub fn last_error(&self) -> PreprocessorError {
        self.last_error
    }
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.fragment_count
    }
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorSpmStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorSpmStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorSpmContext {
        self.machine.context()
    }
}

pub type SpmPreprocessor = TextTokenizerPreprocessorSpm;
