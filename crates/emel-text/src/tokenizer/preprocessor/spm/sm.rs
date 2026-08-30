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
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

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
    pub fragments_out: &'a mut [Fragment<'a>],
    pub fragment_count_out: &'a mut usize,
    pub preprocessed_out: Option<&'a mut bool>,
    pub error_out: &'a mut PreprocessorError,
    pub on_done: Option<fn(&PreprocessDone)>,
    pub on_error: Option<fn(&PreprocessError)>,
}

impl<'a> EventPreprocessRuntime<'a> {
    #[must_use]
    pub fn new(
        vocab: &'a Vocabulary<'a>,
        text: &'a str,
        parse_special: bool,
        fragments_out: &'a mut [Fragment<'a>],
        fragment_count_out: &'a mut usize,
        preprocessed_out: Option<&'a mut bool>,
        error_out: &'a mut PreprocessorError,
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
    TextTokenizerPreprocessorSpm {
        "request_buffer_decision"_s <= *"idle"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "done"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "errored"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "unexpected"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime<'dispatch>> / build_specials,
        "partition_specials_decision"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_no_specials_input_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [no_specials],
        "partition_parse_special_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [has_specials],
        "errored"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / ensure_last_error_from_partition_specials_decision,
        "partitioning_non_bpe_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [parse_special_enabled],
        "partitioning_non_bpe_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_empty] / set_empty_partition_result_from_partitioning_no_specials_input_decision,
        "partitioning_no_specials"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_nonempty],
        "errored"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / ensure_last_error_from_partitioning_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision,
        "partitioning_non_bpe_parse_special"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / ensure_last_error_from_partitioning_non_bpe_parse_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision,
        "partitioning_non_bpe_skip_special"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime<'dispatch>> / ensure_last_error_from_partitioning_non_bpe_skip_input_decision,
        "partition_decision"_s <= "partitioning_no_specials"_s + completion<EventPreprocessRuntime<'dispatch>> / partition_no_specials,
        "partition_decision"_s <= "partitioning_non_bpe_parse_special"_s + completion<EventPreprocessRuntime<'dispatch>> / partition_non_bpe_parse_special,
        "partition_decision"_s <= "partitioning_non_bpe_skip_special"_s + completion<EventPreprocessRuntime<'dispatch>> / partition_non_bpe_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime<'dispatch>> [partition_unknown_error] / ensure_last_error_from_partition_decision,
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
        *event.fragment_count_out = 0;
        if let Some(preprocessed) = event.preprocessed_out.as_deref_mut() {
            *preprocessed = false;
        }
        *event.error_out = PreprocessorError::None;
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
        self.phase_error = if ok { PreprocessorError::None } else { PreprocessorError::InvalidRequest };
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

    fn token(&self, event: &EventPreprocessRuntime<'_>, id: usize) -> Option<VocabularyToken<'_>> {
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

    fn partition(&mut self, event: &EventPreprocessRuntime<'_>, allow_skipped: bool) -> bool {
        let capacity = event.fragments_out.len();
        let mut current_count = 0usize;
        if !Self::push_scratch(
            &mut self.scratch_a,
            &mut current_count,
            ScratchFragment { kind: FragmentKind::RawText, start: 0, end: event.text.len(), token: -1 },
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
            if token.text.is_empty() || (!allow_skipped && matches!(token.token_type, TokenType::Control | TokenType::Unknown)) {
                continue;
            }
            let mut next_count = 0usize;
            let mut ok = true;
            for fragment_index in 0..current_count {
                let fragment = self.scratch_a[fragment_index];
                if fragment.kind == FragmentKind::Token {
                    ok = ok && Self::push_scratch(
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
                    let Some(relative) = raw[base..].find(token.text) else {
                        ok = ok && Self::push_scratch(
                            &mut self.scratch_b,
                            &mut next_count,
                            ScratchFragment { kind: FragmentKind::RawText, start: fragment.start + base, end: fragment.end, token: -1 },
                            capacity,
                        );
                        base = raw.len();
                        continue;
                    };
                    let match_at = base + relative;
                    let mut left_len = match_at - base;
                    if token.lstrip {
                        while left_len > 0 && raw.as_bytes()[base + left_len - 1].is_ascii_whitespace() {
                            left_len -= 1;
                        }
                    }
                    ok = ok && Self::push_scratch(
                        &mut self.scratch_b,
                        &mut next_count,
                        ScratchFragment { kind: FragmentKind::RawText, start: fragment.start + base, end: fragment.start + base + left_len, token: -1 },
                        capacity,
                    );
                    ok = ok && Self::push_scratch(
                        &mut self.scratch_b,
                        &mut next_count,
                        ScratchFragment { kind: FragmentKind::Token, start: 0, end: 0, token: token.token },
                        capacity,
                    );
                    base = match_at + token.text.len();
                    if token.rstrip {
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

        if current_count > capacity {
            self.set_phase_result(false, 0, false);
            return false;
        }
        for index in 0..current_count {
            let fragment = self.scratch_a[index];
            let output = &mut event.fragments_out[index];
            output.kind = fragment.kind;
            output.token = fragment.token;
            output.text = if fragment.kind == FragmentKind::RawText {
                &event.text[fragment.start..fragment.end]
            } else {
                ""
            };
        }
        self.set_phase_result(true, current_count, true);
        true
    }
}

impl TextTokenizerPreprocessorSpmStateMachineContext for TextTokenizerPreprocessorSpmContext {
    fn begin_preprocess(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> {
        self.clear(event);
        Ok(())
    }

    fn build_specials(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> {
        self.special_count = 0;
        for (id, token) in event.vocab.tokens.iter().enumerate() {
            let special = !token.text.is_empty()
                && matches!(token.token_type, TokenType::Unknown | TokenType::Control | TokenType::UserDefined);
            if !special {
                continue;
            }
            if self.special_count >= MAX_SPECIAL_TOKENS {
                self.set_phase_result(false, 0, false);
                return Ok(());
            }
            self.specials[self.special_count] = CachedSpecial { id };
            self.special_count += 1;
        }
        for index in 1..self.special_count {
            let mut position = index;
            while position > 0 {
                let left = self.specials[position - 1].id;
                let right = self.specials[position].id;
                let left_len = event.vocab.tokens[left].text.len();
                let right_len = event.vocab.tokens[right].text.len();
                if left_len >= right_len {
                    break;
                }
                self.specials.swap(position - 1, position);
                position -= 1;
            }
        }
        self.set_phase_result(true, 0, false);
        Ok(())
    }

    fn build_specials_backend_error(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::BackendError) }
    fn build_specials_invalid_request_error(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::InvalidRequest) }
    fn build_specials_ok(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::None) }
    fn build_specials_unknown_error(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(false) }

    fn ensure_last_error_from_build_specials_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partition_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partition_parse_special_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partition_specials_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_no_specials_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }

    fn fragments_buffer_missing(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.is_empty()) }
    fn fragments_buffer_present(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.fragments_out.is_empty()) }
    fn fragments_capacity_exceeds_limit(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.len() > MAX_FRAGMENTS) }
    fn fragments_capacity_nonzero(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.fragments_out.is_empty()) }
    fn fragments_capacity_within_limit(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.len() <= MAX_FRAGMENTS) }
    fn fragments_capacity_zero(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.is_empty()) }
    fn has_specials(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.special_count != 0) }

    fn mark_done(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> {
        self.phase_error = PreprocessorError::None;
        self.err = PreprocessorError::None;
        self.result = true;
        Ok(())
    }

    fn no_specials(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.special_count == 0) }

    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }

    fn parse_special_disabled(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.parse_special) }
    fn parse_special_enabled(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.parse_special) }
    fn partition_backend_error(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::BackendError) }
    fn partition_invalid_request_error(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::InvalidRequest) }
    fn partition_no_specials(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.partition(event, true); Ok(()) }
    fn partition_non_bpe_parse_special(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.partition(event, true); Ok(()) }
    fn partition_non_bpe_skip_special(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.partition(event, false); Ok(()) }
    fn partition_ok(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::None) }
    fn partition_unknown_error(&self, _event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(false) }

    fn reject_invalid_from_request_buffer_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }
    fn reject_invalid_from_request_capacity_limit_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }
    fn reject_invalid_from_request_capacity_nonzero_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }

    fn request_text_empty(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.text.is_empty()) }
    fn request_text_nonempty(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.text.is_empty()) }

    fn set_empty_partition_result_from_partitioning_no_specials_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.set_phase_result(true, 0, true); Ok(()) }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.set_phase_result(true, 0, true); Ok(()) }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision(&mut self, _event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.set_phase_result(true, 0, true); Ok(()) }
}

/// Single-writer, synchronous SPM preprocessor actor.
pub struct TextTokenizerPreprocessorSpm<'event> {
    machine: TextTokenizerPreprocessorSpmStateMachine<'event, TextTokenizerPreprocessorSpmContext>,
    last_error: PreprocessorError,
    fragment_count: usize,
}

impl<'event> Default for TextTokenizerPreprocessorSpm<'event> {
    fn default() -> Self { Self::new() }
}

impl<'event> TextTokenizerPreprocessorSpm<'event> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorSpmStateMachine::new(TextTokenizerPreprocessorSpmContext::default()),
            last_error: PreprocessorError::None,
            fragment_count: 0,
        }
    }

    /// Dispatches one borrowed request through the complete preprocessing flow.
    pub fn process_event(&mut self, mut event: EventPreprocessRuntime<'event>) -> Result<PreprocessDone, PreprocessorError> {
        let accepted = self.machine.process_event(TextTokenizerPreprocessorSpmEvents::EventPreprocessRuntime(&event)).is_ok();
        let context = self.machine.context_mut();
        let ok = accepted && context.result;
        let error = if ok { PreprocessorError::None } else if context.err == PreprocessorError::None { PreprocessorError::BackendError } else { context.err };
        context.err = error;
        self.last_error = error;
        self.fragment_count = if ok { context.fragment_count } else { 0 };
        *event.fragment_count_out = self.fragment_count;
        if let Some(preprocessed) = event.preprocessed_out.as_deref_mut() {
            *preprocessed = ok && context.preprocessed;
        }
        *event.error_out = error;
        if ok {
            let done = PreprocessDone { fragment_count: self.fragment_count };
            if let Some(callback) = event.on_done { callback(&done); }
            Ok(done)
        } else {
            let failure = PreprocessError { error };
            if let Some(callback) = event.on_error { callback(&failure); }
            Err(error)
        }
    }

    #[must_use]
    pub fn last_error(&self) -> PreprocessorError { self.last_error }
    #[must_use]
    pub fn fragment_count(&self) -> usize { self.fragment_count }
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorSpmStates { self.machine.state() }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorSpmStates) -> bool { self.machine.is(state) }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorSpmContext { self.machine.context() }
}

/// Concise actor alias matching the pinned machine name.
pub type SpmPreprocessor<'event> = TextTokenizerPreprocessorSpm<'event>;
