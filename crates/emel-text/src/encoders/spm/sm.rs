//! Source-aligned bounded SentencePiece encoder state machine.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
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

/// Source-compatible alias for callers that use the shorter request name.
pub type Encode<'a> = EncodeRequest<'a>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingDone<'a> {
    pub request: Option<&'a EncodeRequest<'a>>,
    pub token_count: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingError<'a> {
    pub request: Option<&'a EncodeRequest<'a>>,
    pub err: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeEncodeRuntime<'a> {
    pub request: &'a EncodeRequest<'a>,
}

sml! {
    TextEncodersSpm<'event> {
        "encode_validity_decision"_s <= *"initialized"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "done"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "errored"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "unexpected"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> [text_empty] / mark_done_from_encode_precheck_decision,
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [tables_missing],
        "encode_prepare_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime<'event>> / sync_tables,
        "encode_prepare_exec"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "encode_prepare_result_decision"_s <= "encode_prepare_exec"_s + completion<RuntimeEncodeRuntime<'event>> / run_prepare,
        "encode_merge_input_capacity_decision"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [prepare_result_ok],
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [prepare_result_invalid_argument_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [prepare_result_backend_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [prepare_result_model_invalid_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [prepare_result_unclassified_error_code] / ensure_last_error_from_encode_prepare_result_decision,
        "encode_merge_exec"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_symbol_capacity_within_limit],
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_symbol_capacity_exceeded] / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "encode_merge_result_decision"_s <= "encode_merge_exec"_s + completion<RuntimeEncodeRuntime<'event>> / run_merge,
        "encode_emit_input_decision"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_result_ok],
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_result_invalid_argument_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_result_backend_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_result_model_invalid_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_result_unclassified_error_code] / ensure_last_error_from_encode_merge_result_decision,
        "encode_exec"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime<'event>> [symbols_present],
        "encode_result_decision"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime<'event>> [symbols_absent] / set_emit_result_empty,
        "errored"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error_from_encode_emit_input_decision,
        "emit_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime<'event>> / run_encode,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [emit_result_ok] / apply_emit_result_ok,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [emit_result_failed] / apply_emit_result_failed,
        "errored"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error_from_emit_result_decision,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_policy_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_emit_input_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone<'event>> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError<'event>> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone<'event>> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "done"_s + event<EventsEncodingError<'event>> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "errored"_s + event<EventsEncodingDone<'event>> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError<'event>> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingDone<'event>> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingError<'event>> / on_unexpected_events_encoding_error,
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
struct Symbol { start: usize, len: usize, alive: bool }

#[derive(Debug)]
pub struct TextEncodersSpmContext {
    pub err: i32,
    pub token_count: i32,
    pub tables_ready: bool,
    vocab_key: usize,
    symbols: [Symbol; MAX_ENCODE_SYMBOLS],
    output: [i32; MAX_ENCODE_TOKENS],
}

impl Default for TextEncodersSpmContext {
    fn default() -> Self { Self { err: ERROR_OK, token_count: 0, tables_ready: false, vocab_key: 0, symbols: [Symbol::default(); MAX_ENCODE_SYMBOLS], output: [0; MAX_ENCODE_TOKENS] } }
}

impl TextEncodersSpmContext {
    fn reset(&mut self) { self.err = ERROR_OK; self.token_count = 0; self.output.fill(0); self.symbols.fill(Symbol::default()); }
    fn lookup(vocab: &SpmVocabulary<'_>, text: &str) -> Option<i32> { vocab.tokens.iter().take(MAX_VOCAB_ENTRIES).position(|token| *token == text).and_then(|id| i32::try_from(id).ok()) }
    fn push(&mut self, request: &EncodeRequest<'_>, token: i32) -> bool { let index = usize::try_from(self.token_count).unwrap_or(MAX_ENCODE_TOKENS); if index >= MAX_ENCODE_TOKENS || index >= request.token_ids.len() { return false; } self.output[index] = token; self.token_count += 1; true }
    fn prepare(&mut self, event: &EventEncodeRuntime<'_>) { let text = event.request.text; if text.len() > MAX_ENCODE_BYTES { self.err = ERROR_INVALID_ARGUMENT; return; } let vocab = event.request.vocab; let prefix = vocab.add_space_prefix && !vocab.treat_whitespace_as_suffix; let suffix = vocab.add_space_prefix && vocab.treat_whitespace_as_suffix; let mut escaped = [0u8; MAX_ENCODE_BYTES]; let mut out = 0usize; let mut prefixed = false; for byte in text.bytes() { if prefix && !prefixed && byte != b' ' { if !Self::append_space(&mut escaped, &mut out, vocab.escape_whitespaces) { self.err = ERROR_INVALID_ARGUMENT; return; } prefixed = true; } if byte == b' ' { if !Self::append_space(&mut escaped, &mut out, vocab.escape_whitespaces) { self.err = ERROR_INVALID_ARGUMENT; return; } } else if out == MAX_ENCODE_BYTES { self.err = ERROR_INVALID_ARGUMENT; return; } else { escaped[out] = byte; out += 1; } } if suffix && !Self::append_space(&mut escaped, &mut out, vocab.escape_whitespaces) { self.err = ERROR_INVALID_ARGUMENT; return; } let mut offset = 0usize; while offset < out { if self.symbols.iter().all(|symbol| symbol.alive) { self.err = ERROR_INVALID_ARGUMENT; return; } let index = self.symbols.iter().position(|symbol| !symbol.alive).unwrap_or(MAX_ENCODE_SYMBOLS); if index == MAX_ENCODE_SYMBOLS { self.err = ERROR_INVALID_ARGUMENT; return; } let len = utf8_len(escaped[offset]).min(out - offset); self.symbols[index] = Symbol { start: offset, len, alive: true }; offset += len; } }
    fn append_space(buffer: &mut [u8; MAX_ENCODE_BYTES], out: &mut usize, escaped: bool) -> bool { let marker: &[u8] = if escaped { b"\xE2\x96\x81" } else { b" " }; if *out + marker.len() > buffer.len() { return false; } buffer[*out..*out + marker.len()].copy_from_slice(marker); *out += marker.len(); true }
    fn merge(&mut self, event: &EventEncodeRuntime<'_>) { let text = event.request.text; let mut count = 0usize; for symbol in self.symbols.iter().copied() { if symbol.alive { count += 1; } } for _ in 0..count.saturating_sub(1) { let mut best: Option<(usize, f32)> = None; for left in 0..MAX_ENCODE_SYMBOLS { if !self.symbols[left].alive { continue; } let right = (left + 1..MAX_ENCODE_SYMBOLS).find(|index| self.symbols[*index].alive); let Some(right) = right else { continue; }; let mut joined = [0u8; MAX_ENCODE_BYTES]; let a = symbol_bytes(&self.symbols[left], text); let b = symbol_bytes(&self.symbols[right], text); if a.len() + b.len() > joined.len() { continue; } joined[..a.len()].copy_from_slice(a); joined[a.len()..a.len() + b.len()].copy_from_slice(b); let Ok(candidate) = core::str::from_utf8(&joined[..a.len() + b.len()]) else { continue; }; if let Some(id) = Self::lookup(event.request.vocab, candidate) { let score = event.request.vocab.scores.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY); if best.is_none_or(|(_, old)| score > old) { best = Some((left, score)); } } } let Some((left, _)) = best else { break; }; let Some(right) = (left + 1..MAX_ENCODE_SYMBOLS).find(|index| self.symbols[*index].alive) else { break; }; self.symbols[left].len += self.symbols[right].len; self.symbols[right].alive = false; } }
    fn encode(&mut self, event: &EventEncodeRuntime<'_>) { let text = event.request.text; for index in 0..MAX_ENCODE_SYMBOLS { if !self.symbols[index].alive { continue; } let symbol = symbol_bytes(&self.symbols[index], text); let Ok(piece) = core::str::from_utf8(symbol) else { self.err = ERROR_BACKEND; return; }; if let Some(id) = Self::lookup(event.request.vocab, piece) { if !self.push(event.request, id) { self.err = ERROR_INVALID_ARGUMENT; return; } } else { for byte in symbol { let mut repr = [0u8; 6]; repr.copy_from_slice(b"<0x00>"); const HEX: &[u8; 16] = b"0123456789ABCDEF"; repr[3] = HEX[(byte >> 4) as usize]; repr[4] = HEX[(byte & 15) as usize]; let Ok(hex_piece) = core::str::from_utf8(&repr) else { self.err = ERROR_BACKEND; return; }; let id = Self::lookup(event.request.vocab, hex_piece).or_else(|| Self::lookup(event.request.vocab, core::str::from_utf8(&[byte]).ok()?)); let Some(id) = id else { self.err = ERROR_BACKEND; return; }; if !self.push(event.request, id) { self.err = ERROR_INVALID_ARGUMENT; return; } } } } }
    fn append_result(&self, request: &mut EncodeRequest<'_>) { let count = usize::try_from(self.token_count).unwrap_or(0).min(request.token_ids.len()); request.token_ids[..count].copy_from_slice(&self.output[..count]); if let Some(out) = request.token_count_out.as_deref_mut() { *out = self.token_count; } if let Some(out) = request.error_out.as_deref_mut() { *out = self.err; } }
}

fn symbol_bytes<'a>(symbol: &Symbol, text: &'a str) -> &'a [u8] { let bytes = text.as_bytes(); let start = symbol.start.min(bytes.len()); let end = start.saturating_add(symbol.len).min(bytes.len()); &bytes[start..end] }
fn utf8_len(byte: u8) -> usize { if byte < 0x80 { 1 } else if byte & 0xE0 == 0xC0 { 2 } else if byte & 0xF0 == 0xE0 { 3 } else if byte & 0xF8 == 0xF0 { 4 } else { 1 } }
fn is_other(error: i32) -> bool { !matches!(error, ERROR_OK | ERROR_INVALID_ARGUMENT | ERROR_BACKEND | ERROR_MODEL_INVALID) }

impl<'event> TextEncodersSpmStateMachineContext for TextEncodersSpmContext {
    fn apply_emit_result_failed(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { Ok(()) }
    fn apply_emit_result_ok(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_OK; Ok(()) }
    fn begin_encode(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.reset(); Ok(()) }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.reset(); self.vocab_key = event.request.vocab as *const _ as usize; self.tables_ready = false; Ok(()) }
    fn emit_result_failed(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err != ERROR_OK) }
    fn emit_result_ok(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_OK) }
    fn encode_result_backend_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_BACKEND) }
    fn encode_result_invalid_argument_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_INVALID_ARGUMENT) }
    fn encode_result_model_invalid_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_MODEL_INVALID) }
    fn encode_result_ok(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_OK) }
    fn encode_result_unclassified_error_code(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(is_other(self.err)) }
    fn ensure_last_error_from_emit_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_encode_emit_input_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_encode_merge_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_encode_precheck_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_encode_prepare_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_encode_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_table_policy_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn ensure_last_error_from_table_sync_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { if self.err == ERROR_OK { self.err = ERROR_BACKEND; } Ok(()) }
    fn invalid_encode(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!self.valid_encode(event)?) }
    fn mark_done_from_encode_precheck_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_OK; Ok(()) }
    fn mark_done_from_encode_result_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_OK; Ok(()) }
    fn merge_result_backend_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_BACKEND) }
    fn merge_result_invalid_argument_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_INVALID_ARGUMENT) }
    fn merge_result_model_invalid_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_MODEL_INVALID) }
    fn merge_result_ok(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_OK) }
    fn merge_result_unclassified_error_code(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(is_other(self.err)) }
    fn merge_symbol_capacity_exceeded(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(event.request.text.chars().count() > MAX_ENCODE_SYMBOLS) }
    fn merge_symbol_capacity_within_limit(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!self.merge_symbol_capacity_exceeded(event)?) }
    fn on_unexpected_events_encoding_done(&mut self, _: &EventsEncodingDone<'event>) -> Result<(), ()> { self.err = ERROR_UNEXPECTED; Ok(()) }
    fn on_unexpected_events_encoding_error(&mut self, _: &EventsEncodingError<'event>) -> Result<(), ()> { self.err = ERROR_UNEXPECTED; Ok(()) }
    fn on_unexpected_runtime_encode_runtime(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_UNEXPECTED; Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.err = ERROR_UNEXPECTED; Ok(()) }
    fn prepare_result_backend_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_BACKEND) }
    fn prepare_result_invalid_argument_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_INVALID_ARGUMENT) }
    fn prepare_result_model_invalid_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_MODEL_INVALID) }
    fn prepare_result_ok(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_OK) }
    fn prepare_result_unclassified_error_code(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(is_other(self.err)) }
    fn reject_invalid_encode_from_encode_merge_input_capacity_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_INVALID_ARGUMENT; self.token_count = 0; Ok(()) }
    fn reject_invalid_encode_from_encode_validity_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_INVALID_ARGUMENT; self.token_count = 0; Ok(()) }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.err = ERROR_INVALID_ARGUMENT; self.token_count = 0; Ok(()) }
    fn run_encode(&mut self, event: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.encode(event); Ok(()) }
    fn run_merge(&mut self, event: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.merge(event); Ok(()) }
    fn run_prepare(&mut self, event: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.prepare(event); Ok(()) }
    fn set_emit_result_empty(&mut self, _: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.token_count = 0; self.err = ERROR_OK; Ok(()) }
    fn symbols_absent(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!self.symbols.iter().any(|symbol| symbol.alive)) }
    fn symbols_present(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.symbols.iter().any(|symbol| symbol.alive)) }
    fn sync_tables(&mut self, event: &RuntimeEncodeRuntime<'event>) -> Result<(), ()> { self.tables_ready = event.request.vocab.tokens.len() <= MAX_VOCAB_ENTRIES && event.request.vocab.tokens.len() == event.request.vocab.scores.len(); self.vocab_key = event.request.vocab as *const _ as usize; self.err = if self.tables_ready { ERROR_OK } else { ERROR_MODEL_INVALID }; Ok(()) }
    fn table_sync_backend_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_BACKEND) }
    fn table_sync_invalid_argument_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_INVALID_ARGUMENT) }
    fn table_sync_model_invalid_error(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_MODEL_INVALID) }
    fn table_sync_ok(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.err == ERROR_OK) }
    fn table_sync_unclassified_error_code(&self, _: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(is_other(self.err)) }
    fn tables_missing(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!self.tables_ready || self.vocab_key != event.request.vocab as *const _ as usize) }
    fn tables_ready(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.tables_ready && self.vocab_key == event.request.vocab as *const _ as usize) }
    fn text_empty(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn valid_encode(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!event.request.token_ids.is_empty() && event.request.token_ids.len() <= MAX_ENCODE_TOKENS) }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(self.vocab_key != event.request.vocab as *const _ as usize) }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime<'event>) -> Result<bool, ()> { Ok(!self.vocab_changed(event)?) }
}

/// Synchronous bounded actor around the generated SPM machine.
pub struct TextEncodersSpm<'event> { machine: TextEncodersSpmStateMachine<'event, TextEncodersSpmContext> }
impl<'event> Default for TextEncodersSpm<'event> { fn default() -> Self { Self::new() } }
impl<'event> TextEncodersSpm<'event> {
    #[must_use]
    pub fn new() -> Self { Self { machine: TextEncodersSpmStateMachine::new(TextEncodersSpmContext::default()) } }
    pub fn process(&mut self, mut request: EncodeRequest<'event>) -> bool { let event = RuntimeEncodeRuntime { request: &request }; let accepted = self.machine.process_event(event).is_ok(); let ctx = self.machine.context(); ctx.append_result(&mut request); let err = if accepted { ctx.err } else if ctx.err == ERROR_OK { ERROR_INVALID_ARGUMENT } else { ctx.err }; if let Some(out) = request.error_out.as_deref_mut() { *out = err; } if err == ERROR_OK { if let Some(callback) = request.on_done { callback(EventsEncodingDone { request: None, token_count: ctx.token_count }); } } else if let Some(callback) = request.on_error { callback(EventsEncodingError { request: None, err }); } err == ERROR_OK }
    pub fn encode(&mut self, request: EncodeRequest<'event>) -> bool { self.process(request) }
    #[must_use] pub fn state(&self) -> &TextEncodersSpmStates { self.machine.state() }
    #[must_use] pub fn is(&self, state: &TextEncodersSpmStates) -> bool { self.machine.is(state) }
    #[must_use] pub fn context(&self) -> &TextEncodersSpmContext { self.machine.context() }
    #[must_use] pub fn last_error(&self) -> i32 { self.machine.context().err }
    pub fn process_unexpected(&mut self) { self.machine.set_state(TextEncodersSpmStates::Unexpected); }
}
pub type Spm<'event> = TextEncodersSpm<'event>;
pub type Machine<'event> = TextEncodersSpmStateMachine<'event, TextEncodersSpmContext>;
