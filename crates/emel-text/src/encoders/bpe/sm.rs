//! Source-aligned, bounded BPE text encoder actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs
)]

use core::cell::{Cell, RefCell};
use sml::sml;

/// Maximum UTF-8 symbols accepted by the merge scratch space.
pub const MAX_ENCODE_SYMBOLS: usize = 16_384;
/// Maximum token IDs accepted and emitted by one request.
pub const MAX_ENCODE_TOKENS: usize = 16_384;
/// Maximum vocabulary entries and merge records in the bounded view.
pub const MAX_VOCAB_ENTRIES: usize = 320_000;
pub const MAX_VOCAB_MERGES: usize = 600_000;

/// Encoder error values matching the maintained encoder error contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum EncoderError {
    #[default]
    None = 0,
    InvalidArgument = 1,
    Backend = 2,
    ModelInvalid = 3,
    Unexpected = 4,
}

/// Read-only bounded vocabulary contract consumed by the encoder.
pub trait VocabularyView {
    fn token_count(&self) -> usize;
    fn token(&self, index: usize) -> Option<&[u8]>;
    fn merge_count(&self) -> usize { 0 }
    fn merge(&self, _index: usize) -> Option<(&[u8], &[u8])> { None }
    fn ignore_merges(&self) -> bool { false }
}

/// Successful synchronous completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingDone { pub token_count: usize }
/// Failed synchronous completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingError { pub error: EncoderError }
pub type DoneCallback = fn(EncodingDone) -> bool;
pub type ErrorCallback = fn(EncodingError) -> bool;

/// Caller-owned bounded encode request.
#[derive(Clone, Copy)]
pub struct EncodeRequest<'event> {
    pub vocabulary: &'event dyn VocabularyView,
    pub text: &'event [u8],
    pub preprocessed: bool,
    pub token_ids: &'event RefCell<&'event mut [i32]>,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}

impl core::fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeRequest")
            .field("text_length", &self.text.len())
            .field("preprocessed", &self.preprocessed)
            .field("token_capacity", &self.token_ids.borrow().len())
            .finish_non_exhaustive()
    }
}

impl<'event> EncodeRequest<'event> {
    #[must_use]
    pub const fn new(
        vocabulary: &'event dyn VocabularyView,
        text: &'event [u8],
        token_ids: &'event RefCell<&'event mut [i32]>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self { vocabulary, text, preprocessed: false, token_ids, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }

    #[must_use]
    pub const fn with_callbacks(
        vocabulary: &'event dyn VocabularyView,
        text: &'event [u8],
        token_ids: &'event RefCell<&'event mut [i32]>,
        preprocessed: bool,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self { vocabulary, text, preprocessed, token_ids, dispatch_done, dispatch_error }
    }
}

/// Runtime event carrying a request, result context, and phase outputs.
pub struct RuntimeEncodeRuntime<'event> {
    pub request: EncodeRequest<'event>,
    pub context: &'event RefCell<EncodeContext>,
    pub encode_result_error: Cell<EncoderError>,
    pub encode_result_token_count: Cell<usize>,
}

impl core::fmt::Debug for RuntimeEncodeRuntime<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RuntimeEncodeRuntime")
            .field("request", &self.request)
            .field("context", &self.context.borrow())
            .field("encode_result_error", &self.encode_result_error.get())
            .field("encode_result_token_count", &self.encode_result_token_count.get())
            .finish()
    }
}

/// Compatibility alias retained for generated event naming.
pub type EventEncodeRuntime<'event> = RuntimeEncodeRuntime<'event>;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingDone;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Symbol { start: usize, len: usize, next: usize, alive: bool }

sml! {
    TextEncodersBpe<'event> {
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
        "encode_input_policy_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error_from_encode_precheck_decision,
        "encode_table_prepare"_s <= "encode_input_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [preprocessed] / prepare_tables,
        "errored"_s <= "encode_input_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [not_preprocessed] / reject_invalid_encode_from_encode_input_policy_decision,
        "errored"_s <= "encode_input_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode_from_encode_input_policy_decision,
        "encode_path_decision"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime<'event>> [table_prepare_ok],
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime<'event>> [table_prepare_backend_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime<'event>> [table_prepare_invalid_argument_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime<'event>> [table_prepare_model_invalid_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime<'event>> [table_prepare_unclassified_error_code] / ensure_last_error_from_encode_table_prepare,
        "encode_direct_word_policy_decision"_s <= "encode_path_decision"_s + completion<RuntimeEncodeRuntime<'event>> [ignore_merges_enabled],
        "encode_exec"_s <= "encode_path_decision"_s + completion<RuntimeEncodeRuntime<'event>>,
        "encode_result_decision"_s <= "encode_direct_word_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [direct_word_token_available] / run_encode_ignore_merges,
        "encode_merge_input_capacity_decision"_s <= "encode_direct_word_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>>,
        "encode_exec"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_symbol_capacity_within_limit],
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [merge_symbol_capacity_exceeded] / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "encode_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime<'event>> / run_encode_merge_path,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_table_prepare"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_path_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_path_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_path_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "done"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "errored"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_input_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_table_prepare"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_path_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

/// Mutable result context for one dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeContext { pub error: EncoderError, pub token_count: usize, pub unexpected: bool }
impl Default for EncodeContext { fn default() -> Self { Self { error: EncoderError::None, token_count: 0, unexpected: false } } }
impl EncodeContext {
    fn reset(&mut self) { *self = Self::default(); }
    fn reject(&mut self, error: EncoderError) { self.token_count = 0; self.error = error; }
}

#[derive(Debug)]
pub struct TextEncodersBpeContext {
    vocabulary_identity: usize,
    tables_ready: bool,
    unexpected: bool,
    symbols: [Symbol; MAX_ENCODE_SYMBOLS],
}
impl Default for TextEncodersBpeContext {
    fn default() -> Self {
        Self { vocabulary_identity: 0, tables_ready: false, unexpected: false, symbols: [Symbol::default(); MAX_ENCODE_SYMBOLS] }
    }
}
impl TextEncodersBpeContext {
    fn clear(&mut self, identity: usize) { self.vocabulary_identity = identity; self.tables_ready = false; }
    fn mark_unexpected(&mut self) { self.unexpected = true; }
    fn lookup(vocabulary: &dyn VocabularyView, needle: &[u8]) -> Option<i32> {
        (0..vocabulary.token_count().min(MAX_VOCAB_ENTRIES)).find_map(|i| vocabulary.token(i).filter(|token| *token == needle).and_then(|_| i32::try_from(i).ok()))
    }
    fn merge_rank(vocabulary: &dyn VocabularyView, left: &[u8], right: &[u8]) -> Option<usize> {
        (0..vocabulary.merge_count().min(MAX_VOCAB_MERGES)).find(|&i| vocabulary.merge(i).is_some_and(|(a, b)| a == left && b == right))
    }
    fn push(event: &RuntimeEncodeRuntime<'_>, id: i32, count: &mut usize) -> bool {
        let mut output = event.request.token_ids.borrow_mut();
        if *count >= output.len() || *count >= MAX_ENCODE_TOKENS { return false; }
        output[*count] = id; *count += 1; true
    }
    fn encode_ignore(&mut self, event: &RuntimeEncodeRuntime<'_>) {
        let mut count = 0;
        if let Some(id) = Self::lookup(event.request.vocabulary, event.request.text) {
            if !Self::push(event, id, &mut count) { event.context.borrow_mut().error = EncoderError::InvalidArgument; }
        } else { event.context.borrow_mut().error = EncoderError::Backend; }
        event.context.borrow_mut().token_count = count;
        event.encode_result_token_count.set(count);
        event.encode_result_error.set(event.context.borrow().error);
    }
    fn encode_merge(&mut self, event: &RuntimeEncodeRuntime<'_>) {
        let text = event.request.text;
        let mut output_count = 0usize;
        let mut symbol_count = 0usize;
        for (start, _) in text.iter().enumerate() {
            if symbol_count >= self.symbols.len() { event.context.borrow_mut().reject(EncoderError::InvalidArgument); return; }
            self.symbols[symbol_count] = Symbol { start, len: 1, next: symbol_count + 1, alive: true };
            symbol_count += 1;
        }
        if symbol_count > 0 { self.symbols[symbol_count - 1].next = symbol_count; }
        for _ in 0..symbol_count.saturating_sub(1) {
            let mut best: Option<(usize, usize, usize)> = None;
            let mut left_index = 0usize;
            while left_index < symbol_count {
                let right_index = self.symbols[left_index].next;
                if right_index >= symbol_count { break; }
                if self.symbols[left_index].alive && self.symbols[right_index].alive {
                    let left = &text[self.symbols[left_index].start..self.symbols[left_index].start + self.symbols[left_index].len];
                    let right = &text[self.symbols[right_index].start..self.symbols[right_index].start + self.symbols[right_index].len];
                    if let Some(rank) = Self::merge_rank(event.request.vocabulary, left, right) && best.is_none_or(|(_, _, current)| rank < current) { best = Some((left_index, right_index, rank)); }
                }
                left_index = right_index;
            }
            let Some((left, right, _)) = best else { break; };
            self.symbols[left].len = self.symbols[right].start + self.symbols[right].len - self.symbols[left].start;
            self.symbols[left].next = self.symbols[right].next;
            self.symbols[right].alive = false;
        }
        let mut index = 0usize;
        while index < symbol_count {
            if self.symbols[index].alive {
                let piece = &text[self.symbols[index].start..self.symbols[index].start + self.symbols[index].len];
                if let Some(id) = Self::lookup(event.request.vocabulary, piece) {
                    if !Self::push(event, id, &mut output_count) { event.context.borrow_mut().reject(EncoderError::InvalidArgument); return; }
                } else {
                    for unit in piece.chunks(1) {
                        let Some(id) = Self::lookup(event.request.vocabulary, unit) else { event.context.borrow_mut().error = EncoderError::Backend; return; };
                        if !Self::push(event, id, &mut output_count) { event.context.borrow_mut().reject(EncoderError::InvalidArgument); return; }
                    }
                }
            }
            let next = self.symbols[index].next;
            if next >= symbol_count || next == index { break; }
            index = next;
        }
        event.context.borrow_mut().token_count = output_count;
        event.encode_result_token_count.set(output_count);
        event.encode_result_error.set(event.context.borrow().error);
    }
}

impl TextEncodersBpeStateMachineContext for TextEncodersBpeContext {
    fn begin_encode(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().reset(); event.encode_result_error.set(EncoderError::None); event.encode_result_token_count.set(0); self.unexpected = false; Ok(()) }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.begin_encode(event)?; self.clear(core::ptr::from_ref(event.request.vocabulary) as *const () as usize); Ok(()) }
    fn direct_word_token_available(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(Self::lookup(event.request.vocabulary, event.request.text).is_some()) }
    fn encode_result_backend_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::Backend) }
    fn encode_result_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::InvalidArgument) }
    fn encode_result_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::ModelInvalid) }
    fn encode_result_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::None) }
    fn encode_result_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(matches!(event.context.borrow().error, EncoderError::Unexpected)) }
    fn ensure_last_error_from_encode_precheck_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_encode_table_prepare(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_last_error(event) }
    fn ignore_merges_enabled(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.vocabulary.ignore_merges()) }
    fn invalid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!valid_request(event)) }
    fn mark_done_from_encode_precheck_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().error = EncoderError::None; Ok(()) }
    fn mark_done_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().error = EncoderError::None; Ok(()) }
    fn merge_symbol_capacity_exceeded(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.text.len() > MAX_ENCODE_SYMBOLS) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.mark_unexpected(); Ok(()) }
    fn on_unexpected_runtime_encode_runtime(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().reject(EncoderError::Unexpected); self.mark_unexpected(); Ok(()) }
    fn on_unexpected_events_encoding_done(&mut self, _: &EventsEncodingDone) -> Result<(), ()> { self.mark_unexpected(); Ok(()) }
    fn on_unexpected_events_encoding_error(&mut self, _: &EventsEncodingError) -> Result<(), ()> { self.mark_unexpected(); Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.mark_unexpected(); Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { Ok(()) }
    fn prepare_tables(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let vocabulary = event.request.vocabulary; if vocabulary.token_count() > MAX_VOCAB_ENTRIES || vocabulary.merge_count() > MAX_VOCAB_MERGES { event.context.borrow_mut().error = EncoderError::ModelInvalid; } else { self.tables_ready = true; event.context.borrow_mut().error = EncoderError::None; } Ok(()) }
    fn preprocessed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.preprocessed) }
    fn reject_invalid_encode_from_encode_input_policy_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_encode_from_encode_merge_input_capacity_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_encode_from_encode_validity_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject_invalid(event) }
    fn run_encode_ignore_merges(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.encode_ignore(event); Ok(()) }
    fn run_encode_merge_path(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.encode_merge(event); Ok(()) }
    fn table_prepare_backend_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::Backend) }
    fn table_prepare_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::InvalidArgument) }
    fn table_prepare_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncoderError::ModelInvalid) }
    fn table_prepare_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.tables_ready && event.context.borrow().error == EncoderError::None) }
    fn table_prepare_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(matches!(event.context.borrow().error, EncoderError::Unexpected)) }
    fn text_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn valid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(valid_request(event)) }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocabulary_identity != core::ptr::from_ref(event.request.vocabulary) as *const () as usize) }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocabulary_identity == core::ptr::from_ref(event.request.vocabulary) as *const () as usize) }
}

fn valid_request(event: &RuntimeEncodeRuntime<'_>) -> bool { !event.request.token_ids.borrow().is_empty() && event.request.text.len() <= MAX_ENCODE_SYMBOLS * 4 && event.request.vocabulary.token_count() <= MAX_VOCAB_ENTRIES && event.request.vocabulary.merge_count() <= MAX_VOCAB_MERGES }
fn reject_invalid(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().reject(EncoderError::InvalidArgument); Ok(()) }
fn ensure_last_error(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut context = event.context.borrow_mut(); if context.error == EncoderError::None { context.error = EncoderError::Backend; } Ok(()) }

/// Synchronous bounded actor around the generated BPE machine.
pub struct TextEncodersBpeActor<'event> { machine: TextEncodersBpeStateMachine<'event, TextEncodersBpeContext> }
impl<'event> Default for TextEncodersBpeActor<'event> { fn default() -> Self { Self::new() } }
impl<'event> TextEncodersBpeActor<'event> {
    #[must_use] pub fn new() -> Self { Self { machine: TextEncodersBpeStateMachine::new(TextEncodersBpeContext::default()) } }
    pub fn process_event(&mut self, request: EncodeRequest<'event>) -> Result<EncodingDone, EncodingError> {
        let result = RefCell::new(EncodeContext::default());
        let runtime = RuntimeEncodeRuntime { request, context: &result, encode_result_error: Cell::new(EncoderError::None), encode_result_token_count: Cell::new(0) };
        if self.machine.process_event(TextEncodersBpeEvents::RuntimeEncodeRuntime(runtime)).is_err() {
            result.borrow_mut().reject(EncoderError::Unexpected);
            self.machine.set_state(TextEncodersBpeStates::Unexpected);
        }
        let context = *result.borrow();
        if context.unexpected || self.machine.context().unexpected {
            let error = EncodingError { error: EncoderError::Unexpected };
            if let Some(callback) = request.dispatch_error { let _ = callback(error); }
            return Err(error);
        }
        if context.error == EncoderError::None {
            let done = EncodingDone { token_count: context.token_count };
            if let Some(callback) = request.dispatch_done { let _ = callback(done); }
            Ok(done)
        } else {
            let error = EncodingError { error: context.error };
            if let Some(callback) = request.dispatch_error { let _ = callback(error); }
            Err(error)
        }
    }
    #[must_use] pub fn state(&self) -> &TextEncodersBpeStates { self.machine.state() }
    #[must_use] pub fn is(&self, state: &TextEncodersBpeStates) -> bool { self.machine.is(state) }
    #[must_use] pub fn context(&self) -> &TextEncodersBpeContext { self.machine.context() }
}

pub type Bpe<'event> = TextEncodersBpeActor<'event>;
