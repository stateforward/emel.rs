//! Source-aligned bounded BPE tokenizer preprocessor actor.
//!
//! The actor mirrors the pinned BPE preprocessor lifecycle while keeping all
//! request data and intermediate output in caller-owned bounded storage.

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

/// Maximum output fragments accepted by the pinned preprocessor contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum special-token entries retained by the bounded context.
pub const MAX_SPECIAL_TOKENS: usize = 1024;
/// Maximum bytes used by one BPE-preprocessed request.
pub const MAX_ENCODED_BYTES: usize = 65_536;

/// Token classes used by special-token filtering.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenType {
    #[default]
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
}

/// One caller-supplied vocabulary token.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VocabularyToken<'a> {
    pub text: &'a str,
    pub token: i32,
    pub token_type: TokenType,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Borrowed bounded vocabulary view.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vocabulary<'a> {
    pub tokens: &'a [VocabularyToken<'a>],
}

/// Alias matching model-layer terminology.
pub type Vocab<'a> = Vocabulary<'a>;

/// Kind of produced fragment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

/// Caller-owned output fragment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'a> {
    pub kind: FragmentKind,
    pub text: &'a str,
    pub token: i32,
}

/// Bounded preprocessor error values.
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
pub struct PreprocessErrorEvent {
    pub error: PreprocessorError,
}

/// Borrowed preprocessing request. `encoded_out` is scratch owned by the
/// caller and is used for BPE byte-to-unicode pieces.
pub struct EventPreprocessRuntime<'a> {
    pub vocab: &'a Vocabulary<'a>,
    pub text: &'a str,
    pub parse_special: bool,
    pub fragments_out: &'a mut [Fragment<'a>],
    pub encoded_out: &'a mut [u8],
    pub fragment_count_out: &'a mut usize,
    pub preprocessed_out: Option<&'a mut bool>,
    pub error_out: &'a mut PreprocessorError,
    pub on_done: Option<fn(&PreprocessDone)>,
    pub on_error: Option<fn(&PreprocessErrorEvent)>,
}

impl<'a> EventPreprocessRuntime<'a> {
    #[must_use]
    pub fn new(
        vocab: &'a Vocabulary<'a>,
        text: &'a str,
        parse_special: bool,
        fragments_out: &'a mut [Fragment<'a>],
        encoded_out: &'a mut [u8],
        fragment_count_out: &'a mut usize,
        preprocessed_out: Option<&'a mut bool>,
        error_out: &'a mut PreprocessorError,
    ) -> Self {
        Self { vocab, text, parse_special, fragments_out, encoded_out, fragment_count_out, preprocessed_out, error_out, on_done: None, on_error: None }
    }
}

sml! {
    TextTokenizerPreprocessorBpe {
        "request_buffer_decision"_s <= *"idle"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "done"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "errored"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_buffer_decision"_s <= "unexpected"_s + event<EventPreprocessRuntime<'dispatch>>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / build_specials,
        "partitioning_select"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_bpe_no_specials_input_decision"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [no_specials],
        "partition_parse_special_decision"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [has_specials],
        "errored"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / ensure_last_error_from_partitioning_select,
        "partitioning_bpe_with_specials_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [parse_special_enabled],
        "partitioning_bpe_with_specials_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_no_specials_input_decision,
        "partitioning_bpe_no_specials"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / ensure_last_error_from_partitioning_bpe_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_with_specials_parse_input_decision,
        "partitioning_bpe_with_specials_parse_special"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / ensure_last_error_from_partitioning_bpe_with_specials_parse_input_decision,
        "partition_decision"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_with_specials_skip_input_decision,
        "partitioning_bpe_with_specials_skip_special"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / ensure_last_error_from_partitioning_bpe_with_specials_skip_input_decision,
        "partition_decision"_s <= "partitioning_bpe_no_specials"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / partition_bpe_no_specials,
        "partition_decision"_s <= "partitioning_bpe_with_specials_parse_special"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / partition_bpe_with_specials_parse_special,
        "partition_decision"_s <= "partitioning_bpe_with_specials_skip_special"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) / partition_bpe_with_specials_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime>(EventPreprocessRuntime<'dispatch>) [partition_unknown_error] / ensure_last_error_from_partition_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "request_buffer_decision"_s + unexpected_event<_> / on_unexpected_from_request_buffer_decision,
        "unexpected"_s <= "request_capacity_nonzero_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_nonzero_decision,
        "unexpected"_s <= "request_capacity_limit_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_limit_decision,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "build_specials_decision"_s + unexpected_event<_> / on_unexpected_from_build_specials_decision,
        "unexpected"_s <= "partitioning_select"_s + unexpected_event<_> / on_unexpected_from_partitioning_select,
        "unexpected"_s <= "partition_parse_special_decision"_s + unexpected_event<_> / on_unexpected_from_partition_parse_special_decision,
        "unexpected"_s <= "partitioning_bpe_no_specials_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_no_specials_input_decision,
        "unexpected"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_with_specials_parse_input_decision,
        "unexpected"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_with_specials_skip_input_decision,
        "unexpected"_s <= "partitioning_bpe_no_specials"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_no_specials,
        "unexpected"_s <= "partitioning_bpe_with_specials_parse_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_with_specials_parse_special,
        "unexpected"_s <= "partitioning_bpe_with_specials_skip_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_bpe_with_specials_skip_special,
        "unexpected"_s <= "partition_decision"_s + unexpected_event<_> / on_unexpected_from_partition_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CachedSpecial { index: usize }

/// Persistent bounded BPE context.
#[derive(Debug)]
pub struct TextTokenizerPreprocessorBpeContext {
    pub fragment_count: usize,
    pub preprocessed: bool,
    pub phase_error: PreprocessorError,
    pub err: PreprocessorError,
    pub result: bool,
    pub special_count: usize,
    specials: [CachedSpecial; MAX_SPECIAL_TOKENS],
    encoded_offset: usize,
}

impl Default for TextTokenizerPreprocessorBpeContext {
    fn default() -> Self {
        Self { fragment_count: 0, preprocessed: false, phase_error: PreprocessorError::None, err: PreprocessorError::None, result: false, special_count: 0, specials: [CachedSpecial::default(); MAX_SPECIAL_TOKENS], encoded_offset: 0 }
    }
}

impl TextTokenizerPreprocessorBpeContext {
    fn clear(&mut self, event: &EventPreprocessRuntime<'_>) {
        *event.fragment_count_out = 0;
        if let Some(value) = event.preprocessed_out.as_deref_mut() { *value = false; }
        *event.error_out = PreprocessorError::None;
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::None;
        self.err = PreprocessorError::None;
        self.result = false;
        self.special_count = 0;
        self.encoded_offset = 0;
    }

    fn phase_result(&mut self, ok: bool, count: usize, preprocessed: bool) {
        self.fragment_count = if ok { count } else { 0 };
        self.preprocessed = ok && preprocessed;
        self.phase_error = if ok { PreprocessorError::None } else { PreprocessorError::InvalidRequest };
        self.err = self.phase_error;
        self.result = false;
    }

    fn ensure_error(&mut self) {
        self.err = if self.phase_error == PreprocessorError::None { PreprocessorError::BackendError } else { self.phase_error };
        self.result = false;
    }

    fn unexpected(&mut self) { self.fragment_count = 0; self.preprocessed = false; self.phase_error = PreprocessorError::InvalidRequest; self.err = PreprocessorError::InvalidRequest; self.result = false; }

    fn push<'a>(out: &mut [Fragment<'a>], count: &mut usize, value: Fragment<'a>) -> bool {
        if *count >= out.len() || *count >= MAX_FRAGMENTS { return false; }
        out[*count] = value;
        *count += 1;
        true
    }

    fn encoded_piece<'a>(&mut self, event: &'a EventPreprocessRuntime<'a>, raw: &[u8]) -> Option<&'a str> {
        let start = self.encoded_offset;
        for &byte in raw {
            let mapped = if (0x21..=0x7e).contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae { u32::from(byte) } else { 256 + u32::from(byte) };
            let mut utf8 = [0u8; 4];
            let ch = char::from_u32(mapped)?;
            let text = ch.encode_utf8(&mut utf8);
            if self.encoded_offset + text.len() > event.encoded_out.len() || self.encoded_offset + text.len() > MAX_ENCODED_BYTES { return None; }
            event.encoded_out[self.encoded_offset..self.encoded_offset + text.len()].copy_from_slice(text.as_bytes());
            self.encoded_offset += text.len();
        }
        core::str::from_utf8(&event.encoded_out[start..self.encoded_offset]).ok()
    }

    fn emit_bpe<'a>(&mut self, event: &'a EventPreprocessRuntime<'a>, text: &str, count: &mut usize) -> bool {
        let bytes = text.as_bytes();
        if bytes.is_empty() { return true; }
        let mut start = 0usize;
        while start < bytes.len() {
            let mut end = start;
            if bytes[start].is_ascii_whitespace() {
                while end < bytes.len() && bytes[end].is_ascii_whitespace() { end += 1; }
                while end < bytes.len() && !bytes[end].is_ascii_whitespace() { end += 1; }
            } else {
                while end < bytes.len() && !bytes[end].is_ascii_whitespace() { end += 1; }
                if end < bytes.len() {
                    while end < bytes.len() && bytes[end].is_ascii_whitespace() { end += 1; }
                    while end < bytes.len() && !bytes[end].is_ascii_whitespace() { end += 1; }
                }
            }
            let piece = self.encoded_piece(event, &bytes[start..end]);
            let Some(piece) = piece else { return false; };
            if !Self::push(event.fragments_out, count, Fragment { kind: FragmentKind::RawText, text: piece, token: -1 }) { return false; }
            start = end;
        }
        true
    }

    fn emit_special_partition<'a>(&mut self, event: &'a EventPreprocessRuntime<'a>, allow_control: bool) -> bool {
        let capacity = event.fragments_out.len();
        let mut count = 0usize;
        let mut cursor = 0usize;
        while cursor < event.text.len() {
            let mut best: Option<(usize, usize)> = None;
            for idx in 0..self.special_count {
                let token_index = self.specials[idx].index;
                let token = event.vocab.tokens.get(token_index).copied();
                let Some(token) = token else { return false; };
                if token.text.is_empty() || (!allow_control && matches!(token.token_type, TokenType::Control | TokenType::Unknown)) { continue; }
                if let Some(found) = event.text[cursor..].find(token.text) {
                    let at = cursor + found;
                    if best.is_none_or(|(old, _)| at < old) { best = Some((at, token_index)); }
                }
            }
            let Some((at, token_index)) = best else {
                if !self.emit_bpe(event, &event.text[cursor..], &mut count) { return false; }
                cursor = event.text.len();
                continue;
            };
            let token = event.vocab.tokens[token_index];
            let mut left_end = at;
            if token.lstrip { while left_end > cursor && event.text.as_bytes()[left_end - 1].is_ascii_whitespace() { left_end -= 1; } }
            if !self.emit_bpe(event, &event.text[cursor..left_end], &mut count) { return false; }
            if count >= capacity || !Self::push(event.fragments_out, &mut count, Fragment { kind: FragmentKind::Token, text: "", token: token.token }) { return false; }
            cursor = at + token.text.len();
            if token.rstrip { while cursor < event.text.len() && event.text.as_bytes()[cursor].is_ascii_whitespace() { cursor += 1; } }
        }
        self.phase_result(true, count, true);
        true
    }
}


impl TextTokenizerPreprocessorBpeStateMachineContext for TextTokenizerPreprocessorBpeContext {
    fn begin_preprocess(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.fragment_count = 0; self.preprocessed = false; self.phase_error = PreprocessorError::None; self.err = PreprocessorError::None; self.result = false; self.special_count = 0; self.encoded_offset = 0; *event.fragment_count_out = 0; if let Some(value) = event.preprocessed_out.as_deref_mut() { *value = false; } *event.error_out = PreprocessorError::None; Ok(()) }
    fn build_specials(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> {
        self.special_count = 0;
        for (index, token) in event.vocab.tokens.iter().copied().enumerate() {
            if !token.text.is_empty() && matches!(token.token_type, TokenType::Unknown | TokenType::Control | TokenType::UserDefined) {
                if self.special_count >= MAX_SPECIAL_TOKENS { self.phase_result(false, 0, false); return Ok(()); }
                self.specials[self.special_count] = CachedSpecial { index };
                self.special_count += 1;
            }
        }
        for index in 1..self.special_count { let mut pos = index; while pos > 0 && event.vocab.tokens[self.specials[pos - 1].index].text.len() < event.vocab.tokens[self.specials[pos].index].text.len() { self.specials.swap(pos - 1, pos); pos -= 1; } }
        self.phase_result(true, 0, false);
        Ok(())
    }
    fn build_specials_backend_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::BackendError) }
    fn build_specials_invalid_request_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::InvalidRequest) }
    fn build_specials_ok(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::None) }
    fn build_specials_unknown_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(false) }
    fn ensure_last_error_from_build_specials_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partition_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partition_parse_special_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_bpe_no_specials_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_bpe_with_specials_parse_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_bpe_with_specials_skip_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn ensure_last_error_from_partitioning_select(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.ensure_error(); Ok(()) }
    fn fragments_buffer_missing(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.is_empty()) }
    fn fragments_buffer_present(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.fragments_out.is_empty()) }
    fn fragments_capacity_exceeds_limit(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.len() > MAX_FRAGMENTS) }
    fn fragments_capacity_nonzero(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.fragments_out.is_empty()) }
    fn fragments_capacity_within_limit(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.len() <= MAX_FRAGMENTS) }
    fn fragments_capacity_zero(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.fragments_out.is_empty()) }
    fn has_specials(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.special_count != 0) }
    fn mark_done(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::None; self.err = PreprocessorError::None; self.result = true; Ok(()) }
    fn no_specials(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.special_count == 0) }
    fn parse_special_disabled(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.parse_special) }
    fn parse_special_enabled(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.parse_special) }
    fn partition_backend_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::BackendError) }
    fn partition_invalid_request_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::InvalidRequest) }
    fn partition_ok(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(self.phase_error == PreprocessorError::None) }
    fn partition_unknown_error(&self, _: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(false) }
    fn request_text_empty(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(event.text.is_empty()) }
    fn request_text_nonempty(&self, event: &EventPreprocessRuntime<'_>) -> Result<bool, ()> { Ok(!event.text.is_empty()) }
    fn partition_bpe_no_specials(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { let mut count = 0; let ok = self.emit_bpe(event, event.text, &mut count); self.phase_result(ok, count, true); Ok(()) }
    fn partition_bpe_with_specials_parse_special(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.emit_special_partition(event, true); Ok(()) }
    fn partition_bpe_with_specials_skip_special(&mut self, event: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.emit_special_partition(event, false); Ok(()) }
    fn set_empty_partition_result_from_partitioning_bpe_no_specials_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_result(true, 0, true); Ok(()) }
    fn set_empty_partition_result_from_partitioning_bpe_with_specials_parse_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_result(true, 0, true); Ok(()) }
    fn set_empty_partition_result_from_partitioning_bpe_with_specials_skip_input_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_result(true, 0, true); Ok(()) }
    fn reject_invalid_from_request_buffer_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }
    fn reject_invalid_from_request_capacity_limit_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }
    fn reject_invalid_from_request_capacity_nonzero_decision(&mut self, _: &EventPreprocessRuntime<'_>) -> Result<(), ()> { self.phase_error = PreprocessorError::InvalidRequest; self.ensure_error(); Ok(()) }
    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_no_specials(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_no_specials_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_with_specials_parse_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_with_specials_parse_special(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_with_specials_skip_input_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_bpe_with_specials_skip_special(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_partitioning_select(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
}

/// Single-writer synchronous BPE preprocessor actor.
pub struct TextTokenizerPreprocessorBpe {
    machine: TextTokenizerPreprocessorBpeStateMachine<TextTokenizerPreprocessorBpeContext>,
    last_error: PreprocessorError,
    fragment_count: usize,
}

impl Default for TextTokenizerPreprocessorBpe { fn default() -> Self { Self::new() } }

impl TextTokenizerPreprocessorBpe {
    #[must_use]
    pub fn new() -> Self { Self { machine: TextTokenizerPreprocessorBpeStateMachine::new(TextTokenizerPreprocessorBpeContext::default()), last_error: PreprocessorError::None, fragment_count: 0 } }

    /// Dispatches one borrowed request through the complete preprocessing flow.
    pub fn process_event(&mut self, mut event: EventPreprocessRuntime<'_>) -> Result<PreprocessDone, PreprocessorError> {
        let accepted = self.machine.process_event(TextTokenizerPreprocessorBpeEvents::EventPreprocessRuntime(&event)).is_ok();
        let context = self.machine.context_mut();
        let ok = accepted && context.result;
        let error = if ok { PreprocessorError::None } else if context.err == PreprocessorError::None { PreprocessorError::BackendError } else { context.err };
        context.err = error;
        self.last_error = error;
        self.fragment_count = if ok { context.fragment_count } else { 0 };
        *event.fragment_count_out = self.fragment_count;
        if let Some(value) = event.preprocessed_out.as_deref_mut() { *value = ok && context.preprocessed; }
        *event.error_out = error;
        if ok {
            let done = PreprocessDone { fragment_count: self.fragment_count };
            if let Some(callback) = event.on_done { callback(&done); }
            Ok(done)
        } else {
            let failure = PreprocessErrorEvent { error };
            if let Some(callback) = event.on_error { callback(&failure); }
            Err(error)
        }
    }

    #[must_use]
    pub fn last_error(&self) -> PreprocessorError { self.last_error }
    #[must_use]
    pub fn fragment_count(&self) -> usize { self.fragment_count }
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorBpeStates { self.machine.state() }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorBpeStates) -> bool { self.machine.is(state) }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorBpeContext { self.machine.context() }
}

/// Concise actor alias.
pub type BpePreprocessor = TextTokenizerPreprocessorBpe;
