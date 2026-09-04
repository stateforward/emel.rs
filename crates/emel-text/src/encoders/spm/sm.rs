//! Source-aligned bounded SentencePiece encoder state machine.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    clippy::large_stack_frames,
    clippy::items_after_statements,
    clippy::ref_as_ptr,
    clippy::elidable_lifetime_names,
    clippy::too_many_lines,
    clippy::unused_self,
    clippy::doc_markdown,
    clippy::needless_lifetimes,
    dead_code,
    missing_docs
)]

use sml::sml;

pub const MAX_ENCODE_BYTES: usize = 16_384;
pub const MAX_ENCODE_SYMBOLS: usize = 4_096;
pub const MAX_ENCODE_TOKENS: usize = 16_384;
pub const MAX_VOCAB_ENTRIES: usize = 320_000;

pub const ERROR_OK: i32 = 0;
pub const ERROR_INVALID_ARGUMENT: i32 = 1;
pub const ERROR_BACKEND: i32 = 2;
pub const ERROR_MODEL_INVALID: i32 = 4;
pub const ERROR_UNEXPECTED: i32 = 8;

/// Bounded vocabulary consumed by the SPM encoder.
#[derive(Clone, Copy, Debug)]
pub struct SpmVocabulary<'a> {
    pub tokens: &'a [&'a str],
    pub scores: &'a [f32],
    pub generation: u64,
    pub add_space_prefix: bool,
    pub treat_whitespace_as_suffix: bool,
    pub escape_whitespaces: bool,
}

impl<'a> SpmVocabulary<'a> {
    #[must_use]
    pub const fn new(tokens: &'a [&'a str], scores: &'a [f32]) -> Self {
        Self {
            tokens,
            scores,
            generation: 0,
            add_space_prefix: false,
            treat_whitespace_as_suffix: false,
            escape_whitespaces: true,
        }
    }

    #[must_use]
    pub const fn with_options(
        mut self,
        generation: u64,
        add_space_prefix: bool,
        treat_whitespace_as_suffix: bool,
        escape_whitespaces: bool,
    ) -> Self {
        self.generation = generation;
        self.add_space_prefix = add_space_prefix;
        self.treat_whitespace_as_suffix = treat_whitespace_as_suffix;
        self.escape_whitespaces = escape_whitespaces;
        self
    }
}

/// Borrowed encode request. All result storage belongs to the caller.
pub struct EncodeRequest<'a> {
    pub vocab: &'a SpmVocabulary<'a>,
    pub text: &'a str,
    pub token_ids: &'a mut [i32],
    pub token_count_out: Option<&'a mut i32>,
    pub error_out: Option<&'a mut i32>,
    pub on_done: Option<fn(EventsEncodingDone<'a>)>,
    pub on_error: Option<fn(EventsEncodingError<'a>)>,
}

impl core::fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeRequest")
            .field("text", &self.text)
            .field("token_capacity", &self.token_ids.len())
            .finish_non_exhaustive()
    }
}

/// Source-compatible alias for callers that use the shorter request name.
pub type Encode<'a> = EncodeRequest<'a>;

#[derive(Clone, Copy, Debug, Default)]
pub struct EventsEncodingDone<'a> {
    pub request: Option<&'a EncodeRequest<'a>>,
    pub token_count: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EventsEncodingError<'a> {
    pub request: Option<&'a EncodeRequest<'a>>,
    pub err: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct EventEncodeRuntime<'a> {
    pub vocab: &'a SpmVocabulary<'a>,
    pub text: &'a str,
    pub token_capacity: usize,
}
impl<'a> EventEncodeRuntime<'a> {
    #[must_use]
    pub fn new(request: &EncodeRequest<'a>) -> Self {
        Self {
            vocab: request.vocab,
            text: request.text,
            token_capacity: request.token_ids.len(),
        }
    }
}
sml! {
    TextEncodersSpm<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "encode_validity_decision"_s <= *"initialized"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "done"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "errored"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "unexpected"_s + event<EventEncodeRuntime<'event>>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [text_empty] / mark_done_from_encode_precheck_decision,
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [tables_missing],
        "encode_prepare_exec"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / sync_tables,
        "encode_prepare_exec"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "encode_prepare_result_decision"_s <= "encode_prepare_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / run_prepare,
        "encode_merge_input_capacity_decision"_s <= "encode_prepare_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prepare_result_ok],
        "errored"_s <= "encode_prepare_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prepare_result_invalid_argument_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prepare_result_backend_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prepare_result_model_invalid_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prepare_result_unclassified_error_code] / ensure_last_error_from_encode_prepare_result_decision,
        "encode_merge_exec"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_symbol_capacity_within_limit],
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_symbol_capacity_exceeded] / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "encode_merge_result_decision"_s <= "encode_merge_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / run_merge,
        "encode_emit_input_decision"_s <= "encode_merge_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_result_ok],
        "errored"_s <= "encode_merge_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_result_invalid_argument_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_result_backend_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_result_model_invalid_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [merge_result_unclassified_error_code] / ensure_last_error_from_encode_merge_result_decision,
        "encode_exec"_s <= "encode_emit_input_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [symbols_present],
        "encode_result_decision"_s <= "encode_emit_input_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [symbols_absent] / set_emit_result_empty,
        "errored"_s <= "encode_emit_input_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / ensure_last_error_from_encode_emit_input_decision,
        "emit_result_decision"_s <= "encode_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / run_encode,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [emit_result_ok] / apply_emit_result_ok,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [emit_result_failed] / apply_emit_result_failed,
        "errored"_s <= "emit_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / ensure_last_error_from_emit_result_decision,
        "done"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_policy_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_exec"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_result_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_exec"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_result_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_emit_input_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_result_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<EventEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone<'event>>(&'dispatch EventsEncodingDone<'event>) / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError<'event>>(&'dispatch EventsEncodingError<'event>) / on_unexpected_events_encoding_error,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone<'event>>(&'dispatch EventsEncodingDone<'event>) / on_unexpected_events_encoding_done,
        "unexpected"_s <= "done"_s + event<EventsEncodingError<'event>>(&'dispatch EventsEncodingError<'event>) / on_unexpected_events_encoding_error,
        "unexpected"_s <= "errored"_s + event<EventsEncodingDone<'event>>(&'dispatch EventsEncodingDone<'event>) / on_unexpected_events_encoding_done,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError<'event>>(&'dispatch EventsEncodingError<'event>) / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingDone<'event>>(&'dispatch EventsEncodingDone<'event>) / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingError<'event>>(&'dispatch EventsEncodingError<'event>) / on_unexpected_events_encoding_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_prepare_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_prepare_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_emit_input_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "emit_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }

}
#[derive(Clone, Copy, Debug, Default)]
struct Symbol {
    start: usize,
    len: usize,
    alive: bool,
}

#[derive(Debug)]
pub struct TextEncodersSpmContext {
    pub err: i32,
    pub token_count: i32,
    pub tables_ready: bool,
    vocab_key: usize,
    symbols: [Symbol; MAX_ENCODE_SYMBOLS],
    output: [i32; MAX_ENCODE_TOKENS],
    transformed: [u8; MAX_ENCODE_BYTES],
    transformed_len: usize,
}

impl Default for TextEncodersSpmContext {
    fn default() -> Self {
        Self {
            err: ERROR_OK,
            token_count: 0,
            tables_ready: false,
            vocab_key: 0,
            symbols: [Symbol::default(); MAX_ENCODE_SYMBOLS],
            output: [0; MAX_ENCODE_TOKENS],
            transformed: [0; MAX_ENCODE_BYTES],
            transformed_len: 0,
        }
    }
}

impl TextEncodersSpmContext {
    fn reset(&mut self) {
        self.err = ERROR_OK;
        self.token_count = 0;
        self.transformed_len = 0;
        self.output.fill(0);
        self.symbols.fill(Symbol::default());
    }
    fn lookup(vocab: &SpmVocabulary<'_>, text: &str) -> Option<i32> {
        vocab
            .tokens
            .iter()
            .take(MAX_VOCAB_ENTRIES)
            .position(|token| *token == text)
            .and_then(|id| i32::try_from(id).ok())
    }
    fn push(&mut self, token_capacity: usize, token: i32) -> bool {
        let index = usize::try_from(self.token_count).unwrap_or(MAX_ENCODE_TOKENS);
        if index >= MAX_ENCODE_TOKENS || index >= token_capacity {
            return false;
        }
        self.output[index] = token;
        self.token_count += 1;
        true
    }
    fn prepare(&mut self, event: &EventEncodeRuntime<'_>) {
        let text = event.text;
        self.transformed_len = 0;
        if text.len() > MAX_ENCODE_BYTES {
            self.err = ERROR_INVALID_ARGUMENT;
            return;
        }
        let vocab = event.vocab;
        let prefix = vocab.add_space_prefix && !vocab.treat_whitespace_as_suffix;
        let suffix = vocab.add_space_prefix && vocab.treat_whitespace_as_suffix;
        let mut out = 0usize;
        let mut prefixed = false;
        for byte in text.bytes() {
            if prefix && !prefixed && byte != b' ' {
                if !Self::append_space(&mut self.transformed, &mut out, vocab.escape_whitespaces) {
                    self.err = ERROR_INVALID_ARGUMENT;
                    return;
                }
                prefixed = true;
            }
            if byte == b' ' {
                if !Self::append_space(&mut self.transformed, &mut out, vocab.escape_whitespaces) {
                    self.err = ERROR_INVALID_ARGUMENT;
                    return;
                }
            } else if out == MAX_ENCODE_BYTES {
                self.err = ERROR_INVALID_ARGUMENT;
                return;
            } else {
                self.transformed[out] = byte;
                out += 1;
            }
        }
        if suffix && !Self::append_space(&mut self.transformed, &mut out, vocab.escape_whitespaces)
        {
            self.err = ERROR_INVALID_ARGUMENT;
            return;
        }
        self.transformed_len = out;
        let mut offset = 0usize;
        while offset < out {
            let Some(index) = self.symbols.iter().position(|symbol| !symbol.alive) else {
                self.err = ERROR_INVALID_ARGUMENT;
                return;
            };
            let len = utf8_len(self.transformed[offset]).min(out - offset);
            self.symbols[index] = Symbol {
                start: offset,
                len,
                alive: true,
            };
            offset += len;
        }
    }
    fn append_space(buffer: &mut [u8; MAX_ENCODE_BYTES], out: &mut usize, escaped: bool) -> bool {
        let marker: &[u8] = if escaped { b"\xE2\x96\x81" } else { b" " };
        if *out + marker.len() > buffer.len() {
            return false;
        }
        buffer[*out..*out + marker.len()].copy_from_slice(marker);
        *out += marker.len();
        true
    }
    fn merge(&mut self, event: &EventEncodeRuntime<'_>) {
        let text = core::str::from_utf8(&self.transformed[..self.transformed_len]).unwrap_or("");
        let count = self.symbols.iter().filter(|symbol| symbol.alive).count();
        for _ in 0..count.saturating_sub(1) {
            let mut best: Option<(usize, f32)> = None;
            for left in 0..MAX_ENCODE_SYMBOLS {
                if !self.symbols[left].alive {
                    continue;
                }
                let Some(right) =
                    (left + 1..MAX_ENCODE_SYMBOLS).find(|index| self.symbols[*index].alive)
                else {
                    continue;
                };
                let a = symbol_bytes(&self.symbols[left], text);
                let b = symbol_bytes(&self.symbols[right], text);
                let mut joined = [0u8; MAX_ENCODE_BYTES];
                if a.len() + b.len() > joined.len() {
                    continue;
                }
                joined[..a.len()].copy_from_slice(a);
                joined[a.len()..a.len() + b.len()].copy_from_slice(b);
                let Ok(candidate) = core::str::from_utf8(&joined[..a.len() + b.len()]) else {
                    continue;
                };
                if let Some(id) = Self::lookup(event.vocab, candidate) {
                    let score = event
                        .vocab
                        .scores
                        .get(id as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY);
                    if best.is_none_or(|(_, old)| score > old) {
                        best = Some((left, score));
                    }
                }
            }
            let Some((left, _)) = best else {
                break;
            };
            let Some(right) =
                (left + 1..MAX_ENCODE_SYMBOLS).find(|index| self.symbols[*index].alive)
            else {
                break;
            };
            self.symbols[left].len += self.symbols[right].len;
            self.symbols[right].alive = false;
        }
    }
    fn encode(&mut self, event: &EventEncodeRuntime<'_>) {
        for index in 0..MAX_ENCODE_SYMBOLS {
            if !self.symbols[index].alive {
                continue;
            }
            let start = self.symbols[index].start;
            let end = start
                .saturating_add(self.symbols[index].len)
                .min(self.transformed_len);
            let direct = core::str::from_utf8(&self.transformed[start..end])
                .ok()
                .and_then(|piece| Self::lookup(event.vocab, piece));
            if let Some(id) = direct {
                if !self.push(event.token_capacity, id) {
                    self.err = ERROR_INVALID_ARGUMENT;
                    return;
                }
            } else {
                for offset in start..end {
                    let byte = self.transformed[offset];
                    let mut repr = [0u8; 6];
                    repr.copy_from_slice(b"<0x00>");
                    const HEX: &[u8; 16] = b"0123456789ABCDEF";
                    repr[3] = HEX[(byte >> 4) as usize];
                    repr[4] = HEX[(byte & 15) as usize];
                    let hex_id = core::str::from_utf8(&repr)
                        .ok()
                        .and_then(|piece| Self::lookup(event.vocab, piece));
                    let id = hex_id.or_else(|| {
                        let single = [byte];
                        core::str::from_utf8(&single)
                            .ok()
                            .and_then(|piece| Self::lookup(event.vocab, piece))
                    });
                    let Some(id) = id else {
                        self.err = ERROR_BACKEND;
                        return;
                    };
                    if !self.push(event.token_capacity, id) {
                        self.err = ERROR_INVALID_ARGUMENT;
                        return;
                    }
                }
            }
        }
    }
    fn append_result(&self, request: &mut EncodeRequest<'_>) {
        let success = self.err == ERROR_OK;
        let count = if success {
            usize::try_from(self.token_count)
                .unwrap_or(0)
                .min(request.token_ids.len())
        } else {
            0
        };
        request.token_ids[..count].copy_from_slice(&self.output[..count]);
        if let Some(out) = request.token_count_out.as_deref_mut() {
            *out = if success { self.token_count } else { 0 };
        }
        if let Some(out) = request.error_out.as_deref_mut() {
            *out = self.err;
        }
    }
}

fn symbol_bytes<'a>(symbol: &Symbol, text: &'a str) -> &'a [u8] {
    let bytes = text.as_bytes();
    let start = symbol.start.min(bytes.len());
    let end = start.saturating_add(symbol.len).min(bytes.len());
    &bytes[start..end]
}
fn utf8_len(byte: u8) -> usize {
    if byte < 0x80 {
        1
    } else if byte & 0xE0 == 0xC0 {
        2
    } else if byte & 0xF0 == 0xE0 {
        3
    } else if byte & 0xF8 == 0xF0 {
        4
    } else {
        1
    }
}
fn is_other(error: i32) -> bool {
    !matches!(
        error,
        ERROR_OK | ERROR_INVALID_ARGUMENT | ERROR_BACKEND | ERROR_MODEL_INVALID
    )
}

impl TextEncodersSpmStateMachineContext for TextEncodersSpmContext {
    fn apply_emit_result_failed(&mut self, _: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.token_count = 0;
        Ok(())
    }
    fn apply_emit_result_ok(&mut self, _: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.err = ERROR_OK;
        Ok(())
    }
    fn begin_encode(&mut self, _: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn begin_encode_sync_vocab(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.reset();
        self.vocab_key = event.vocab as *const _ as usize;
        self.tables_ready = false;
        Ok(())
    }
    fn emit_result_failed(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err != ERROR_OK)
    }
    fn emit_result_ok(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_OK)
    }
    fn encode_result_backend_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_BACKEND)
    }
    fn encode_result_invalid_argument_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_INVALID_ARGUMENT)
    }
    fn encode_result_model_invalid_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_MODEL_INVALID)
    }
    fn encode_result_ok(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_OK)
    }
    fn encode_result_unclassified_error_code(
        &self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(is_other(self.err))
    }
    fn ensure_last_error_from_emit_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_encode_emit_input_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_encode_merge_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_encode_prepare_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_table_policy_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn ensure_last_error_from_table_sync_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        if self.err == ERROR_OK {
            self.err = ERROR_BACKEND;
        }
        Ok(())
    }
    fn invalid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_encode(event)?)
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_OK;
        Ok(())
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_OK;
        Ok(())
    }
    fn merge_result_backend_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_BACKEND)
    }
    fn merge_result_invalid_argument_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_INVALID_ARGUMENT)
    }
    fn merge_result_model_invalid_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_MODEL_INVALID)
    }
    fn merge_result_ok(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_OK)
    }
    fn merge_result_unclassified_error_code(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(is_other(self.err))
    }
    fn merge_symbol_capacity_exceeded(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.text.len() > MAX_ENCODE_SYMBOLS)
    }
    fn merge_symbol_capacity_within_limit(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!self.merge_symbol_capacity_exceeded(event)?)
    }
    fn on_unexpected_events_encoding_done<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventsEncodingDone<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = ERROR_UNEXPECTED;
        Ok(())
    }
    fn on_unexpected_events_encoding_error<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventsEncodingError<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = ERROR_UNEXPECTED;
        Ok(())
    }
    fn on_unexpected_runtime_encode_runtime(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_UNEXPECTED;
        Ok(())
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        self.err = ERROR_UNEXPECTED;
        Ok(())
    }
    fn prepare_result_backend_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_BACKEND)
    }
    fn prepare_result_invalid_argument_error(
        &self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(self.err == ERROR_INVALID_ARGUMENT)
    }
    fn prepare_result_model_invalid_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_MODEL_INVALID)
    }
    fn prepare_result_ok(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_OK)
    }
    fn prepare_result_unclassified_error_code(
        &self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(is_other(self.err))
    }
    fn reject_invalid_encode_from_encode_merge_input_capacity_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_INVALID_ARGUMENT;
        self.token_count = 0;
        Ok(())
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_INVALID_ARGUMENT;
        self.token_count = 0;
        Ok(())
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        _: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        self.err = ERROR_INVALID_ARGUMENT;
        self.token_count = 0;
        Ok(())
    }
    fn run_encode(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.encode(event);
        Ok(())
    }
    fn run_merge(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.merge(event);
        Ok(())
    }
    fn run_prepare(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.prepare(event);
        Ok(())
    }
    fn set_emit_result_empty(&mut self, _: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.token_count = 0;
        self.err = ERROR_OK;
        Ok(())
    }
    fn symbols_absent(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.symbols.iter().any(|symbol| symbol.alive))
    }
    fn symbols_present(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.symbols.iter().any(|symbol| symbol.alive))
    }
    fn sync_tables(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        self.tables_ready = event.vocab.tokens.len() <= MAX_VOCAB_ENTRIES
            && event.vocab.tokens.len() == event.vocab.scores.len();
        self.vocab_key = event.vocab as *const _ as usize;
        self.err = if self.tables_ready {
            ERROR_OK
        } else {
            ERROR_MODEL_INVALID
        };
        Ok(())
    }
    fn table_sync_backend_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_BACKEND)
    }
    fn table_sync_invalid_argument_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_INVALID_ARGUMENT)
    }
    fn table_sync_model_invalid_error(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_MODEL_INVALID)
    }
    fn table_sync_ok(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == ERROR_OK)
    }
    fn table_sync_unclassified_error_code(&self, _: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(is_other(self.err))
    }
    fn tables_missing(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tables_ready || self.vocab_key != event.vocab as *const _ as usize)
    }
    fn tables_ready(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.tables_ready && self.vocab_key == event.vocab as *const _ as usize)
    }
    fn text_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.text.is_empty())
    }
    fn text_non_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.text.is_empty())
    }
    fn valid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.token_capacity != 0 && event.token_capacity <= MAX_ENCODE_TOKENS)
    }
    fn vocab_changed(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.vocab_key != event.vocab as *const _ as usize)
    }
    fn vocab_unchanged(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.vocab_changed(event)?)
    }
}

/// Synchronous bounded actor around the generated SPM machine.
pub struct TextEncodersSpm {
    machine: TextEncodersSpmStateMachine<TextEncodersSpmContext>,
}
impl Default for TextEncodersSpm {
    fn default() -> Self {
        Self::new()
    }
}
impl TextEncodersSpm {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextEncodersSpmStateMachine::new(TextEncodersSpmContext::default()),
        }
    }
    pub fn process<'event>(&mut self, mut request: EncodeRequest<'event>) -> bool {
        let accepted = {
            let event = EventEncodeRuntime::new(&request);
            self.machine
                .process_event(TextEncodersSpmEvents::EventEncodeRuntime(event))
                .is_ok()
        };
        let (err, token_count) = {
            let ctx = self.machine.context();
            let err = if accepted {
                ctx.err
            } else if ctx.err == ERROR_OK {
                ERROR_INVALID_ARGUMENT
            } else {
                ctx.err
            };
            (err, ctx.token_count)
        };
        self.machine.context().append_result(&mut request);
        if let Some(out) = request.error_out.as_deref_mut() {
            *out = err;
        }
        if err == ERROR_OK {
            if let Some(callback) = request.on_done {
                callback(EventsEncodingDone {
                    request: None,
                    token_count,
                });
            }
        } else if let Some(callback) = request.on_error {
            callback(EventsEncodingError { request: None, err });
        }
        err == ERROR_OK
    }
    pub fn encode<'event>(&mut self, request: EncodeRequest<'event>) -> bool {
        self.process(request)
    }
    #[must_use]
    pub fn state(&self) -> &TextEncodersSpmStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextEncodersSpmStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextEncodersSpmContext {
        self.machine.context()
    }
    #[must_use]
    pub fn last_error(&self) -> i32 {
        self.machine.context().err
    }
    pub fn process_unexpected(&mut self) {
        self.machine.set_state(TextEncodersSpmStates::Unexpected);
    }
}

pub type Spm = TextEncodersSpm;
pub type Machine = TextEncodersSpmStateMachine<TextEncodersSpmContext>;

#[cfg(test)]
mod tests {
    use super::*;

    fn run(vocab: &SpmVocabulary<'_>, text: &str, output: &mut [i32]) -> (bool, i32, i32) {
        let mut count = -1;
        let mut error = -1;
        let request = EncodeRequest {
            vocab,
            text,
            token_ids: output,
            token_count_out: Some(&mut count),
            error_out: Some(&mut error),
            on_done: None,
            on_error: None,
        };
        let ok = TextEncodersSpm::new().process(request);
        (ok, count, error)
    }

    #[test]
    fn merge_uses_transformed_space_offsets() {
        let tokens = ["a", "b", "ab"];
        let scores = [0.0, 0.0, 1.0];
        let vocab = SpmVocabulary::new(&tokens, &scores);
        let mut output = [-1; 2];
        let (ok, count, error) = run(&vocab, "ab", &mut output);
        assert!(ok);
        assert_eq!((count, error), (1, ERROR_OK));
        assert_eq!(output[0], 2);
    }

    #[test]
    fn normalization_prefix_is_used_for_lookup() {
        let tokens = ["▁a"];
        let scores = [1.0];
        let vocab = SpmVocabulary::new(&tokens, &scores).with_options(1, true, false, true);
        let mut output = [-1; 1];
        let (ok, count, error) = run(&vocab, "a", &mut output);
        assert!(ok);
        assert_eq!((count, error, output[0]), (1, ERROR_OK, 0));
    }

    #[test]
    fn failed_emit_publishes_no_partial_output() {
        let tokens = ["a"];
        let scores = [1.0];
        let vocab = SpmVocabulary::new(&tokens, &scores);
        let mut output = [41, 42];
        let (ok, count, error) = run(&vocab, "ab", &mut output);
        assert!(!ok);
        assert_eq!((count, error), (0, ERROR_BACKEND));
        assert_eq!(output, [41, 42]);
    }

    #[test]
    fn transformed_buffer_overflow_is_invalid() {
        let tokens: [&str; 0] = [];
        let scores: [f32; 0] = [];
        let vocab = SpmVocabulary::new(&tokens, &scores);
        let text = " ".repeat(MAX_ENCODE_SYMBOLS * 2);
        let mut output = [7; 1];
        let (ok, count, error) = run(&vocab, &text, &mut output);
        assert!(!ok);
        assert_eq!((count, error), (0, ERROR_INVALID_ARGUMENT));
    }

    #[test]
    fn empty_input_publishes_empty_success() {
        let tokens: [&str; 0] = [];
        let scores: [f32; 0] = [];
        let vocab = SpmVocabulary::new(&tokens, &scores);
        let mut output = [9; 1];
        let (ok, count, error) = run(&vocab, "", &mut output);
        assert!(ok);
        assert_eq!((count, error, output[0]), (0, ERROR_OK, 9));
    }
}
