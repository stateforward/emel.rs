//! Source-aligned bounded fallback text encoder actor.

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

use core::cell::Cell;
use sml::sml;

/// Maximum vocabulary entries retained by the bounded fallback table.
pub const MAX_FALLBACK_TOKENS: usize = 2048;
/// Number of fixed open-addressing slots used by the fallback table.
pub const MAX_FALLBACK_TABLE_SLOTS: usize = 4096;
/// Maximum input bytes accepted by one request.
pub const MAX_FALLBACK_TEXT_BYTES: usize = 16_384;

/// Read-only vocabulary contract for fallback encoding.
pub trait VocabularyView {
    fn token_count(&self) -> usize;
    fn token(&self, index: usize) -> Option<&[u8]>;
}

/// Source-compatible encoder error classes.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingDone { pub token_count: usize }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingError { pub error: EncoderError }
pub type DoneCallback = fn(EncodingDone) -> bool;
pub type ErrorCallback = fn(EncodingError) -> bool;

/// Caller-owned bounded fallback request.
pub struct EncodeRequest<'a> {
    pub vocabulary: &'a dyn VocabularyView,
    pub text: &'a [u8],
    pub preprocessed: bool,
    pub token_ids: &'a mut [i32],
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}

impl core::fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeRequest")
            .field("text_length", &self.text.len())
            .field("preprocessed", &self.preprocessed)
            .field("token_capacity", &self.token_ids.len())
            .finish_non_exhaustive()
    }
}

impl<'a> EncodeRequest<'a> {
    #[must_use]
    pub const fn new(
        vocabulary: &'a dyn VocabularyView,
        text: &'a [u8],
        token_ids: &'a mut [i32],
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self { vocabulary, text, preprocessed: false, token_ids, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }

    #[must_use]
    pub const fn with_callbacks(
        vocabulary: &'a dyn VocabularyView,
        text: &'a [u8],
        token_ids: &'a mut [i32],
        preprocessed: bool,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self { vocabulary, text, preprocessed, token_ids, dispatch_done, dispatch_error }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingDone;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventsEncodingError;

/// Runtime event. Cells carry completion state without heap allocation.
pub struct RuntimeEncodeRuntime<'a> {
    pub request: &'a mut EncodeRequest<'a>,
    pub error: Cell<EncoderError>,
    pub token_count: Cell<usize>,
    pub emit_error: Cell<EncoderError>,
    pub emit_count: Cell<usize>,
}

sml! {
    TextEncodersFallback<'a> {
        "encode_validity_decision"_s <= *"initialized"_s + event<RuntimeEncodeRuntime<'a>>,
        "encode_validity_decision"_s <= "done"_s + event<RuntimeEncodeRuntime<'a>>,
        "encode_validity_decision"_s <= "errored"_s + event<RuntimeEncodeRuntime<'a>>,
        "encode_validity_decision"_s <= "unexpected"_s + event<RuntimeEncodeRuntime<'a>>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [text_empty] / mark_done_from_encode_precheck_decision,
        "encode_table_prepare"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [text_non_empty] / prepare_tables,
        "encode_exec"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_prepare_ok],
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_prepare_invalid_argument_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_prepare_backend_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_prepare_model_invalid_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_prepare_unclassified_error_code] / ensure_last_error_from_encode_table_prepare,
        "emit_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / run_encode_exec,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [emit_result_ok] / apply_emit_result_ok,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [emit_result_failed] / apply_emit_result_failed,
        "errored"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_emit_result_decision,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_table_prepare"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "emit_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "emit_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
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
        "unexpected"_s <= "encode_table_prepare"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "emit_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

#[derive(Debug)]
pub struct TextEncodersFallbackContext {
    hashes: [u32; MAX_FALLBACK_TABLE_SLOTS],
    values: [i32; MAX_FALLBACK_TABLE_SLOTS],
    vocabulary_identity: usize,
    tables_ready: bool,
    pub unexpected: bool,
}

impl Default for TextEncodersFallbackContext {
    fn default() -> Self { Self { hashes: [0; MAX_FALLBACK_TABLE_SLOTS], values: [-1; MAX_FALLBACK_TABLE_SLOTS], vocabulary_identity: 0, tables_ready: false, unexpected: false } }
}

impl TextEncodersFallbackContext {
    fn reset_table(&mut self, identity: usize) { self.hashes.fill(0); self.values.fill(-1); self.vocabulary_identity = identity; self.tables_ready = false; }
    fn unexpected(&mut self) { self.unexpected = true; }
    fn insert(&mut self, vocab: &dyn VocabularyView, text: &[u8], id: i32) -> bool {
        let hash = hash_bytes(text); let mut slot = (hash as usize) & (MAX_FALLBACK_TABLE_SLOTS - 1);
        for _ in 0..MAX_FALLBACK_TABLE_SLOTS {
            if self.hashes[slot] == 0 { self.hashes[slot] = hash; self.values[slot] = id; return true; }
            if self.hashes[slot] == hash && vocab.token(self.values[slot] as usize).is_some_and(|value| value == text) { return true; }
            slot = (slot + 1) & (MAX_FALLBACK_TABLE_SLOTS - 1);
        }
        false
    }
    fn lookup(&self, vocab: &dyn VocabularyView, byte: u8) -> i32 {
        let hash = hash_bytes(&[byte]); let mut slot = (hash as usize) & (MAX_FALLBACK_TABLE_SLOTS - 1);
        for _ in 0..MAX_FALLBACK_TABLE_SLOTS {
            if self.hashes[slot] == 0 { return -1; }
            if self.hashes[slot] == hash { let id = self.values[slot]; if vocab.token(id as usize).is_some_and(|text| text == [byte]) { return id; } }
            slot = (slot + 1) & (MAX_FALLBACK_TABLE_SLOTS - 1);
        }
        -1
    }
}

impl TextEncodersFallbackStateMachineContext for TextEncodersFallbackContext {
    fn apply_emit_result_failed(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.token_count.set(0); event.error.set(event.emit_error.get()); Ok(()) }
    fn apply_emit_result_ok(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.token_count.set(event.emit_count.get()); event.error.set(EncoderError::None); Ok(()) }
    fn begin_encode(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.error.set(EncoderError::None); event.token_count.set(0); event.emit_error.set(EncoderError::None); event.emit_count.set(0); self.unexpected = false; Ok(()) }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.begin_encode(event)?; self.reset_table(identity(event.request.vocabulary)); Ok(()) }
    fn emit_result_failed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.emit_error.get() != EncoderError::None) }
    fn emit_result_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.emit_error.get() == EncoderError::None) }
    fn encode_result_backend_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::Backend) }
    fn encode_result_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::InvalidArgument) }
    fn encode_result_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::ModelInvalid) }
    fn encode_result_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::None) }
    fn encode_result_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::Unexpected) }
    fn ensure_last_error_from_emit_result_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event) }
    fn ensure_last_error_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event) }
    fn ensure_last_error_from_encode_table_prepare(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event) }
    fn invalid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!valid(event)) }
    fn mark_done_from_encode_precheck_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.error.set(EncoderError::None); Ok(()) }
    fn mark_done_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.error.set(EncoderError::None); Ok(()) }
    fn on_unexpected_events_encoding_done(&mut self, _event: &EventsEncodingDone) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_events_encoding_error(&mut self, _event: &EventsEncodingError) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_runtime_encode_runtime(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.error.set(EncoderError::Unexpected); self.unexpected(); Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn prepare_tables(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> {
        let vocab = event.request.vocabulary; let count = vocab.token_count(); self.reset_table(identity(vocab));
        if count > MAX_FALLBACK_TOKENS { event.error.set(EncoderError::InvalidArgument); return Ok(()); }
        for index in 0..count { let Some(text) = vocab.token(index) else { event.error.set(EncoderError::ModelInvalid); return Ok(()); }; if !text.is_empty() && !self.insert(vocab, text, index as i32) { event.error.set(EncoderError::InvalidArgument); return Ok(()); } }
        self.tables_ready = true; event.error.set(EncoderError::None); Ok(())
    }
    fn reject_invalid_encode_from_encode_validity_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject(event) }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject(event) }
    fn run_encode_exec(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> {
        let mut count = 0; let mut error = EncoderError::None;
        for byte in event.request.text.iter().copied() { let token = self.lookup(event.request.vocabulary, byte); if token < 0 || count >= event.request.token_ids.len() { error = EncoderError::Backend; break; } event.request.token_ids[count] = token; count += 1; }
        event.emit_count.set(count); event.emit_error.set(error); Ok(())
    }
    fn table_prepare_backend_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::Backend) }
    fn table_prepare_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::InvalidArgument) }
    fn table_prepare_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::ModelInvalid) }
    fn table_prepare_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.tables_ready && event.error.get() == EncoderError::None) }
    fn table_prepare_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.error.get() == EncoderError::Unexpected) }
    fn text_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn valid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(valid(event)) }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocabulary_identity != identity(event.request.vocabulary)) }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocabulary_identity == identity(event.request.vocabulary)) }
}

fn hash_bytes(bytes: &[u8]) -> u32 { let mut hash = 2_166_136_261u32; for byte in bytes { hash = hash.wrapping_mul(16_777_619) ^ u32::from(*byte); } if hash == 0 { 1 } else { hash } }
fn identity(vocab: &dyn VocabularyView) -> usize { core::ptr::from_ref(vocab) as *const () as usize }
fn valid(event: &RuntimeEncodeRuntime<'_>) -> bool { !event.request.token_ids.is_empty() && event.request.text.len() <= MAX_FALLBACK_TEXT_BYTES && event.request.vocabulary.token_count() <= MAX_FALLBACK_TOKENS }
fn reject(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.error.set(EncoderError::InvalidArgument); event.token_count.set(0); Ok(()) }
fn ensure_error(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { if event.error.get() == EncoderError::None { event.error.set(EncoderError::Backend); } Ok(()) }

/// Single-writer synchronous fallback actor.
pub struct TextEncodersFallbackActor<'a> { machine: TextEncodersFallbackStateMachine<'a, TextEncodersFallbackContext> }
impl<'a> Default for TextEncodersFallbackActor<'a> { fn default() -> Self { Self::new() } }
impl<'a> TextEncodersFallbackActor<'a> {
    #[must_use]
    pub fn new() -> Self { Self { machine: TextEncodersFallbackStateMachine::new(TextEncodersFallbackContext::default()) } }
    pub fn process_event(&mut self, mut request: EncodeRequest<'a>) -> Result<EncodingDone, EncodingError> {
        let event = RuntimeEncodeRuntime { request: &mut request, error: Cell::new(EncoderError::None), token_count: Cell::new(0), emit_error: Cell::new(EncoderError::None), emit_count: Cell::new(0) };
        let accepted = self.machine.process_event(TextEncodersFallbackEvents::EventRuntimeEncodeRuntime(event));
        let event = match accepted {
            Ok(event) => event,
            Err(_) => {
                self.machine.set_state(TextEncodersFallbackStates::Unexpected);
                let failure = EncodingError { error: EncoderError::Unexpected };
                if let Some(callback) = request.dispatch_error { let _ = callback(failure); }
                return Err(failure);
            }
        };
        let error = event.error.get();
        let count = event.token_count.get();
        if error == EncoderError::None {
            let done = EncodingDone { token_count: count };
            if let Some(callback) = event.request.dispatch_done { let _ = callback(done); }
            Ok(done)
        } else {
            let failure = EncodingError { error };
            if let Some(callback) = event.request.dispatch_error { let _ = callback(failure); }
            Err(failure)
        }
    }
}
pub type Fallback<'a> = TextEncodersFallbackActor<'a>;
