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
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::elidable_lifetime_names,
    clippy::needless_borrow,
    clippy::unnecessary_map_or,
    clippy::nonminimal_bool,
    clippy::if_same_then_else,
    clippy::manual_let_else,
    clippy::option_if_let_else,
    clippy::sliced_string_as_bytes,
    clippy::items_after_statements,
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

use core::cell::RefCell;
use sml::sml;

/// Maximum output fragments accepted by the pinned preprocessor contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum special-token entries retained by the bounded context.
pub const MAX_SPECIAL_TOKENS: usize = 1024;
/// Maximum byte-to-unicode encoded bytes retained for one request.
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreprocessErrorEvent {
    pub error: PreprocessorError,
}

/// Borrowed preprocessing request. Raw fragments borrow `text`; BPE-encoded
/// fragments borrow the caller-owned `encoded_out` scratch for this dispatch.
pub struct EventPreprocessRuntime<'a> {
    pub vocab: &'a Vocabulary<'a>,
    pub text: &'a str,
    pub parse_special: bool,
    pub fragments_out: &'a RefCell<&'a mut [Fragment<'a>]>,
    pub encoded_out: &'a mut [u8],
    pub fragment_count_out: &'a RefCell<usize>,
    pub preprocessed_out: Option<&'a RefCell<bool>>,
    pub error_out: &'a RefCell<PreprocessorError>,
    pub on_done: Option<fn(&PreprocessDone)>,
    pub on_error: Option<fn(&PreprocessErrorEvent)>,
}

impl<'a> EventPreprocessRuntime<'a> {
    #[must_use]
    pub fn new(
        vocab: &'a Vocabulary<'a>,
        text: &'a str,
        parse_special: bool,
        fragments_out: &'a RefCell<&'a mut [Fragment<'a>]>,
        encoded_out: &'a mut [u8],
        fragment_count_out: &'a RefCell<usize>,
        preprocessed_out: Option<&'a RefCell<bool>>,
        error_out: &'a RefCell<PreprocessorError>,
    ) -> Self {
        Self {
            vocab,
            text,
            parse_special,
            fragments_out,
            encoded_out,
            fragment_count_out,
            preprocessed_out,
            error_out,
            on_done: None,
            on_error: None,
        }
    }
}

sml! {
    TextTokenizerPreprocessorBpe<'dispatch, 'event>
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
        "partitioning_select"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_bpe_no_specials_input_decision"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [no_specials],
        "partition_parse_special_decision"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [has_specials],
        "errored"_s <= "partitioning_select"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_select,
        "partitioning_bpe_with_specials_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [parse_special_enabled],
        "partitioning_bpe_with_specials_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_no_specials_input_decision,
        "partitioning_bpe_no_specials"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_no_specials_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_bpe_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_with_specials_parse_input_decision,
        "partitioning_bpe_with_specials_parse_special"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_with_specials_parse_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_bpe_with_specials_parse_input_decision,
        "partition_decision"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_empty] / set_empty_partition_result_from_partitioning_bpe_with_specials_skip_input_decision,
        "partitioning_bpe_with_specials_skip_special"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) [request_text_nonempty],
        "errored"_s <= "partitioning_bpe_with_specials_skip_input_decision"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / ensure_last_error_from_partitioning_bpe_with_specials_skip_input_decision,
        "partition_decision"_s <= "partitioning_bpe_no_specials"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_bpe_no_specials,
        "partition_decision"_s <= "partitioning_bpe_with_specials_parse_special"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_bpe_with_specials_parse_special,
        "partition_decision"_s <= "partitioning_bpe_with_specials_skip_special"_s + completion<EventPreprocessRuntime>(&'dispatch EventPreprocessRuntime<'event>) / partition_bpe_with_specials_skip_special,
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
struct CachedSpecial {
    index: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ScratchFragment {
    kind: FragmentKind,
    start: usize,
    end: usize,
    token: i32,
}

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
    scratch_a: [ScratchFragment; MAX_FRAGMENTS],
    scratch_b: [ScratchFragment; MAX_FRAGMENTS],
    encoded_offset: usize,
}

impl Default for TextTokenizerPreprocessorBpeContext {
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
            encoded_offset: 0,
        }
    }
}

impl TextTokenizerPreprocessorBpeContext {
    fn clear(&mut self, event: &EventPreprocessRuntime<'_>) {
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::None;
        self.err = PreprocessorError::None;
        self.result = false;
        self.special_count = 0;
        self.encoded_offset = 0;
        *event.fragment_count_out.borrow_mut() = 0;
        if let Some(value) = event.preprocessed_out {
            *value.borrow_mut() = false;
        }
        *event.error_out.borrow_mut() = PreprocessorError::None;
    }

    fn phase_result(&mut self, ok: bool, count: usize, preprocessed: bool) {
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
        self.fragment_count = 0;
        self.preprocessed = false;
        self.result = false;
    }

    fn unexpected(&mut self) {
        self.fragment_count = 0;
        self.preprocessed = false;
        self.phase_error = PreprocessorError::InvalidRequest;
        self.err = PreprocessorError::InvalidRequest;
        self.result = false;
    }

    fn push<'a>(out: &mut [Fragment<'a>], count: &mut usize, value: Fragment<'a>) -> bool {
        if *count >= out.len() || *count >= MAX_FRAGMENTS {
            return false;
        }
        out[*count] = value;
        *count += 1;
        true
    }

    fn push_scratch_to(
        out: &mut [ScratchFragment],
        count: &mut usize,
        value: ScratchFragment,
        capacity: usize,
    ) -> bool {
        if *count >= capacity || *count >= MAX_FRAGMENTS {
            return false;
        }
        out[*count] = value;
        *count += 1;
        true
    }

    fn push_scratch(
        &mut self,
        target_b: bool,
        count: &mut usize,
        value: ScratchFragment,
        capacity: usize,
    ) -> bool {
        let out = if target_b {
            &mut self.scratch_b
        } else {
            &mut self.scratch_a
        };
        Self::push_scratch_to(out, count, value, capacity)
    }

    fn emit_bpe(
        fragments_out: &mut [ScratchFragment],
        text: &str,
        source_base: usize,
        count: &mut usize,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        let mut pos = 0usize;
        while pos < text.len() {
            let start = pos;
            let Some(cpt) = text[pos..].chars().next() else {
                return false;
            };
            let next = pos + cpt.len_utf8();
            let next_cpt = text[next..].chars().next();
            let contraction_end = if cpt == '\'' {
                let first = next_cpt.and_then(|ch| ch.to_lowercase().next());
                if matches!(first, Some('s' | 't' | 'm' | 'd')) {
                    Some(next + next_cpt.map_or(0, char::len_utf8))
                } else if matches!(first, Some('r' | 'v' | 'l')) {
                    next_cpt.and_then(|ch| {
                        let second = text[next + ch.len_utf8()..].chars().next()?;
                        let first_lower = ch.to_lowercase().next().unwrap_or(ch);
                        let second_lower = second.to_lowercase().next().unwrap_or(second);
                        ((first_lower == 'r' && second_lower == 'e')
                            || (first_lower == 'v' && second_lower == 'e')
                            || (first_lower == 'l' && second_lower == 'l'))
                            .then_some(next + ch.len_utf8() + second.len_utf8())
                    })
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(end) = contraction_end {
                pos = end;
            } else if cpt != '\r'
                && cpt != '\n'
                && !cpt.is_numeric()
                && (cpt.is_alphabetic() || next_cpt.is_some_and(char::is_alphabetic))
            {
                pos = next;
                while let Some(ch) = text[pos..].chars().next() {
                    if !ch.is_alphabetic() {
                        break;
                    }
                    pos += ch.len_utf8();
                }
            } else if cpt.is_numeric() {
                let mut digits = 0usize;
                while let Some(ch) = text[pos..].chars().next() {
                    if !ch.is_numeric() {
                        break;
                    }
                    pos += ch.len_utf8();
                    digits += 1;
                    if digits == 3 {
                        if !Self::push_scratch_to(
                            fragments_out,
                            count,
                            ScratchFragment {
                                kind: FragmentKind::RawText,
                                start: source_base + start,
                                end: source_base + pos,
                                token: -1,
                            },
                            fragments_out.len(),
                        ) {
                            return false;
                        }
                        return Self::emit_bpe(
                            fragments_out,
                            &text[pos..],
                            source_base + pos,
                            count,
                        );
                    }
                }
            } else if cpt == ' '
                && next_cpt.is_some_and(|ch| {
                    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
                })
            {
                pos = next;
                while let Some(ch) = text[pos..].chars().next() {
                    if ch.is_whitespace() || ch.is_alphabetic() || ch.is_numeric() {
                        break;
                    }
                    pos += ch.len_utf8();
                }
                while let Some(ch) = text[pos..].chars().next() {
                    if ch != '\r' && ch != '\n' {
                        break;
                    }
                    pos += ch.len_utf8();
                }
            } else if !cpt.is_whitespace() && !cpt.is_alphabetic() && !cpt.is_numeric() {
                pos = next;
                while let Some(ch) = text[pos..].chars().next() {
                    if ch.is_whitespace() || ch.is_alphabetic() || ch.is_numeric() {
                        break;
                    }
                    pos += ch.len_utf8();
                }
                while let Some(ch) = text[pos..].chars().next() {
                    if ch != '\r' && ch != '\n' {
                        break;
                    }
                    pos += ch.len_utf8();
                }
            } else if cpt.is_whitespace() {
                let mut whitespace_end = pos;
                let mut whitespace_count = 0usize;
                let mut last_newline_end = None;
                while let Some(ch) = text[whitespace_end..].chars().next() {
                    if !ch.is_whitespace() {
                        break;
                    }
                    whitespace_end += ch.len_utf8();
                    whitespace_count += 1;
                    if ch == '\r' || ch == '\n' {
                        last_newline_end = Some(whitespace_end);
                    }
                }
                pos = if let Some(end) = last_newline_end {
                    end
                } else if whitespace_end != text.len() && whitespace_count > 1 {
                    text[..whitespace_end]
                        .char_indices()
                        .next_back()
                        .map_or(whitespace_end, |(at, _)| at)
                } else {
                    whitespace_end
                };
            } else {
                pos = next;
            }
            if pos <= start
                || !Self::push_scratch_to(
                    fragments_out,
                    count,
                    ScratchFragment {
                        kind: FragmentKind::RawText,
                        start: source_base + start,
                        end: source_base + pos,
                        token: -1,
                    },
                    fragments_out.len(),
                )
            {
                return false;
            }
        }
        true
    }

    fn expand_scratch(&mut self, text: &str, scratch_count: &mut usize) -> bool {
        let initial_count = *scratch_count;
        let mut final_count = 0usize;
        for idx in 0..initial_count {
            let fragment = self.scratch_a[idx];
            if fragment.kind == FragmentKind::Token {
                if !Self::push_scratch_to(
                    &mut self.scratch_b,
                    &mut final_count,
                    fragment,
                    MAX_FRAGMENTS,
                ) {
                    return false;
                }
            } else if !Self::emit_bpe(
                &mut self.scratch_b,
                &text[fragment.start..fragment.end],
                fragment.start,
                &mut final_count,
            ) {
                return false;
            }
        }
        core::mem::swap(&mut self.scratch_a, &mut self.scratch_b);
        *scratch_count = final_count;
        true
    }

    fn encode_piece(
        encoded_out: &mut [u8],
        offset: &mut usize,
        text: &str,
    ) -> Option<(usize, usize)> {
        let start = *offset;
        for &byte in text.as_bytes() {
            let mapped =
                if (0x21..=0x7e).contains(&byte) || (0xa1..=0xac).contains(&byte) || byte >= 0xae {
                    u32::from(byte)
                } else {
                    256 + u32::from(byte)
                };
            let ch = char::from_u32(mapped)?;
            let mut utf8 = [0u8; 4];
            let encoded = ch.encode_utf8(&mut utf8);
            let end = offset.checked_add(encoded.len())?;
            if end > encoded_out.len() || end > MAX_ENCODED_BYTES {
                return None;
            }
            encoded_out[*offset..end].copy_from_slice(encoded.as_bytes());
            *offset = end;
        }
        Some((start, *offset))
    }
    fn materialize<'text>(
        &mut self,
        text: &'text str,
        encoded_out: &'text mut [u8],
        fragments_out: &mut [Fragment<'text>],
        scratch_count: usize,
    ) -> bool {
        let mut encoded_offset = 0usize;
        for idx in 0..scratch_count {
            let fragment = self.scratch_a[idx];
            if fragment.kind == FragmentKind::RawText {
                let Some(raw) = text.get(fragment.start..fragment.end) else {
                    return false;
                };
                let Some((start, end)) = Self::encode_piece(encoded_out, &mut encoded_offset, raw)
                else {
                    return false;
                };
                self.scratch_b[idx] = ScratchFragment {
                    kind: fragment.kind,
                    start,
                    end,
                    token: fragment.token,
                };
            } else {
                self.scratch_b[idx] = fragment;
            }
        }
        let encoded = &encoded_out[..encoded_offset];
        let mut count = 0usize;
        for idx in 0..scratch_count {
            let fragment = self.scratch_b[idx];
            let piece = encoded
                .get(fragment.start..fragment.end)
                .and_then(|bytes| core::str::from_utf8(bytes).ok())
                .unwrap_or("");
            if !Self::push(
                fragments_out,
                &mut count,
                Fragment {
                    kind: fragment.kind,
                    text: piece,
                    token: fragment.token,
                },
            ) {
                return false;
            }
        }
        self.encoded_offset = encoded_offset;
        true
    }

    fn emit_special_partition<'text>(
        &mut self,
        vocab: &Vocabulary<'text>,
        text: &'text str,
        allow_control: bool,
    ) -> bool {
        let capacity = MAX_FRAGMENTS;
        let mut current_count = 0usize;
        if !text.is_empty()
            && !self.push_scratch(
                false,
                &mut current_count,
                ScratchFragment {
                    kind: FragmentKind::RawText,
                    start: 0,
                    end: text.len(),
                    token: -1,
                },
                capacity,
            )
        {
            return false;
        }
        for special_idx in 0..self.special_count {
            let token_index = self.specials[special_idx].index;
            let Some(token) = vocab.tokens.get(token_index).copied() else {
                return false;
            };
            if token.text.is_empty()
                || (!allow_control
                    && matches!(token.token_type, TokenType::Control | TokenType::Unknown))
            {
                continue;
            }
            let mut next_count = 0usize;
            for fragment_idx in 0..current_count {
                let fragment = self.scratch_a[fragment_idx];
                if fragment.kind == FragmentKind::Token {
                    if !self.push_scratch(true, &mut next_count, fragment, capacity) {
                        return false;
                    }
                    continue;
                }
                let raw = &text[fragment.start..fragment.end];
                let mut base = 0usize;
                while base < raw.len() {
                    let Some(relative) = raw[base..].find(token.text) else {
                        if base < raw.len()
                            && !self.push_scratch(
                                true,
                                &mut next_count,
                                ScratchFragment {
                                    kind: FragmentKind::RawText,
                                    start: fragment.start + base,
                                    end: fragment.end,
                                    token: -1,
                                },
                                capacity,
                            )
                        {
                            return false;
                        }
                        break;
                    };
                    let match_at = base + relative;
                    let mut left_len = match_at - base;
                    if token.lstrip {
                        while left_len > 0
                            && raw.as_bytes()[base + left_len - 1].is_ascii_whitespace()
                        {
                            left_len -= 1;
                        }
                    }
                    if left_len != 0
                        && !self.push_scratch(
                            true,
                            &mut next_count,
                            ScratchFragment {
                                kind: FragmentKind::RawText,
                                start: fragment.start + base,
                                end: fragment.start + base + left_len,
                                token: -1,
                            },
                            capacity,
                        )
                    {
                        return false;
                    }
                    if !self.push_scratch(
                        true,
                        &mut next_count,
                        ScratchFragment {
                            kind: FragmentKind::Token,
                            start: 0,
                            end: 0,
                            token: token.token,
                        },
                        capacity,
                    ) {
                        return false;
                    }
                    base = match_at + token.text.len();
                    if token.rstrip {
                        while base < raw.len() && raw.as_bytes()[base].is_ascii_whitespace() {
                            base += 1;
                        }
                    }
                }
            }
            core::mem::swap(&mut self.scratch_a, &mut self.scratch_b);
            current_count = next_count;
        }
        self.phase_result(true, current_count, true);
        true
    }
}

impl TextTokenizerPreprocessorBpeStateMachineContext for TextTokenizerPreprocessorBpeContext {
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
        for (index, token) in event.vocab.tokens.iter().copied().enumerate() {
            if !token.text.is_empty()
                && matches!(
                    token.token_type,
                    TokenType::Unknown | TokenType::Control | TokenType::UserDefined
                )
            {
                if self.special_count >= MAX_SPECIAL_TOKENS {
                    self.phase_result(false, 0, false);
                    return Ok(());
                }
                self.specials[self.special_count] = CachedSpecial { index };
                self.special_count += 1;
            }
        }
        for index in 1..self.special_count {
            let mut pos = index;
            while pos > 0
                && event.vocab.tokens[self.specials[pos - 1].index].text.len()
                    < event.vocab.tokens[self.specials[pos].index].text.len()
            {
                self.specials.swap(pos - 1, pos);
                pos -= 1;
            }
        }
        self.phase_result(true, 0, false);
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
    fn ensure_last_error_from_partitioning_bpe_no_specials_input_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_bpe_with_specials_parse_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_bpe_with_specials_skip_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_select<'dispatch, 'event>(
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
    fn no_specials<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.special_count == 0)
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
    fn partition_backend_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.phase_error == PreprocessorError::BackendError)
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
    fn set_empty_partition_result_from_partitioning_bpe_no_specials_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_result(true, 0, true);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_bpe_with_specials_parse_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_result(true, 0, true);
        Ok(())
    }
    fn partition_bpe_no_specials<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut scratch_count = 0usize;
        let ok = Self::emit_bpe(&mut self.scratch_a, event.text, 0, &mut scratch_count);
        self.phase_result(ok, if ok { scratch_count } else { 0 }, true);
        Ok(())
    }
    fn partition_bpe_with_specials_parse_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let ok = self.emit_special_partition(event.vocab, event.text, true);
        if !ok {
            self.phase_result(false, 0, false);
        }
        Ok(())
    }
    fn partition_bpe_with_specials_skip_special<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let ok = self.emit_special_partition(event.vocab, event.text, false);
        if !ok {
            self.phase_result(false, 0, false);
        }
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_bpe_with_specials_skip_input_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        _: &'dispatch EventPreprocessRuntime<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.phase_result(true, 0, true);
        Ok(())
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
    fn on_unexpected_from_partitioning_bpe_no_specials(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_bpe_no_specials_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_bpe_with_specials_parse_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_bpe_with_specials_parse_special(
        &mut self,
    ) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_bpe_with_specials_skip_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_bpe_with_specials_skip_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_select(&mut self) -> Result<(), ()> {
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

/// Single-writer synchronous BPE preprocessor actor.
pub struct TextTokenizerPreprocessorBpe {
    machine: TextTokenizerPreprocessorBpeStateMachine<TextTokenizerPreprocessorBpeContext>,
    last_error: PreprocessorError,
    fragment_count: usize,
}

impl Default for TextTokenizerPreprocessorBpe {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTokenizerPreprocessorBpe {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorBpeStateMachine::new(
                TextTokenizerPreprocessorBpeContext::default(),
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
            .process_event(TextTokenizerPreprocessorBpeEvents::EventPreprocessRuntime(
                &event,
            ))
            .is_ok();
        let context = self.machine.context_mut();
        let mut ok = accepted && context.result;
        if ok {
            let scratch_count = context.fragment_count;
            let mut fragments = event.fragments_out.borrow_mut();
            ok = context.materialize(event.text, event.encoded_out, &mut fragments, scratch_count);
            if !ok {
                context.phase_error = PreprocessorError::BackendError;
                context.err = PreprocessorError::BackendError;
                context.result = false;
            }
        }
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
        if let Some(value) = event.preprocessed_out {
            *value.borrow_mut() = ok && context.preprocessed;
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
            let failure = PreprocessErrorEvent { error };
            if let Some(callback) = event.on_error {
                callback(&failure);
            }
            Err(error)
        }
    }
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.fragment_count
    }
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorBpeStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorBpeStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextTokenizerPreprocessorBpeContext {
        self.machine.context()
    }
}

/// Concise actor alias.
pub type BpePreprocessor = TextTokenizerPreprocessorBpe;
