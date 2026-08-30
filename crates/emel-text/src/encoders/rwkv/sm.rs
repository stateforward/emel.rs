//! Source-aligned bounded RWKV text encoder state machine.

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

use core::cell::RefCell;
use sml::sml;

/// Maximum decoded bytes held for one vocabulary token.
pub const MAX_RWKV_TOKEN_BYTES: usize = 256;

/// Errors returned by a bounded encode request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum EncodeError {
    #[default]
    None = 0,
    InvalidArgument = 1,
    Backend = 2,
    ModelInvalid = 3,
    Unclassified = -1,
}
impl EncodeError { #[must_use] pub const fn code(self) -> i32 { self as i32 } }

/// Borrowed vocabulary contract required by RWKV.
pub trait RwkvVocabulary {
    fn identity(&self) -> usize;
    fn token_count(&self) -> usize;
    fn token_text(&self, index: usize) -> &[u8];
    fn unk_id(&self) -> i32;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingDone { pub token_count: usize }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingError { pub error: EncodeError }
pub type DoneCallback = fn(EventsEncodingDone) -> bool;
pub type ErrorCallback = fn(EventsEncodingError) -> bool;

/// Caller-owned request and synchronous callback handoff.
pub struct EncodeRequest<'event> {
    pub vocabulary: &'event dyn RwkvVocabulary,
    pub text: &'event [u8],
    pub token_ids: &'event RefCell<&'event mut [i32]>,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}
impl core::fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeRequest").field("text_length", &self.text.len()).field("output_capacity", &self.token_ids.borrow().len()).finish()
    }
}
impl<'event> EncodeRequest<'event> {
    #[must_use]
    pub const fn new(vocabulary: &'event dyn RwkvVocabulary, text: &'event [u8], token_ids: &'event RefCell<&'event mut [i32]>, dispatch_done: DoneCallback, dispatch_error: ErrorCallback) -> Self {
        Self { vocabulary, text, token_ids, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }
    #[must_use]
    pub const fn with_callbacks(vocabulary: &'event dyn RwkvVocabulary, text: &'event [u8], token_ids: &'event RefCell<&'event mut [i32]>, dispatch_done: Option<DoneCallback>, dispatch_error: Option<ErrorCallback>) -> Self {
        Self { vocabulary, text, token_ids, dispatch_done, dispatch_error }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EncodeContext {
    pub token_count: usize,
    pub error: EncodeError,
    pub unexpected: bool,
}

#[derive(Debug)]
pub struct RuntimeEncodeRuntime<'event> {
    pub request: EncodeRequest<'event>,
    pub context: &'event RefCell<EncodeContext>,
    unk_id: RefCell<i32>,
    unk_lookup_found: RefCell<bool>,
    encode_push_failed: RefCell<bool>,
}
impl<'event> RuntimeEncodeRuntime<'event> {
    #[must_use]
    pub fn new(request: EncodeRequest<'event>, context: &'event RefCell<EncodeContext>) -> Self {
        Self { request, context, unk_id: RefCell::new(-1), unk_lookup_found: RefCell::new(false), encode_push_failed: RefCell::new(false) }
    }
}

sml! {
    TextEncodersRwkv<'event> {
        "encode_validity_decision"_s <= *"initialized"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "done"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "errored"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "unexpected"_s + event<RuntimeEncodeRuntime<'event>>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [invalid_encode] / reject_invalid_encode,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode,
        "done"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> [text_empty] / mark_done,
        "encode_capacity_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error,
        "table_policy_decision"_s <= "encode_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [output_capacity_covers_text],
        "errored"_s <= "encode_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> [output_capacity_short] / reject_invalid_encode,
        "errored"_s <= "encode_capacity_decision"_s + completion<RuntimeEncodeRuntime<'event>> / reject_invalid_encode,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [tables_missing],
        "unk_resolution_decision"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime<'event>> / sync_tables,
        "unk_resolution_decision"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_invalid_argument_error] / ensure_last_error,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_backend_error] / ensure_last_error,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_model_invalid_error] / ensure_last_error,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [table_sync_unclassified_error_code] / ensure_last_error,
        "encode_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_unk_present] / resolve_vocab_unk,
        "unk_lookup_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime<'event>> [vocab_unk_missing],
        "unk_lookup_result_decision"_s <= "unk_lookup_exec"_s + completion<RuntimeEncodeRuntime<'event>> / lookup_unk_candidate,
        "encode_exec"_s <= "unk_lookup_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [unk_lookup_found] / set_unk_from_lookup,
        "encode_exec"_s <= "unk_lookup_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [unk_lookup_missing] / set_unk_missing,
        "encode_emit_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime<'event>> / run_encode,
        "errored"_s <= "encode_emit_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_push_failed] / mark_encode_push_failed,
        "encode_result_decision"_s <= "encode_emit_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_push_ok],
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_ok] / mark_done,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_invalid_argument_error] / ensure_last_error,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_backend_error] / ensure_last_error,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_model_invalid_error] / ensure_last_error,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> [encode_result_unclassified_error_code] / ensure_last_error,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime<'event>> / ensure_last_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_capacity_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "table_policy_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "unk_resolution_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "unk_lookup_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "unk_lookup_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_exec"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_emit_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_capacity_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "unk_resolution_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "unk_lookup_exec"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "unk_lookup_result_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_emit_result_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_wild,
    }
}

#[derive(Debug, Default)]
pub struct TextEncodersRwkvContext { tables_ready: bool, vocabulary_identity: usize }

impl TextEncodersRwkvStateMachineContext for TextEncodersRwkvContext {
    fn begin_encode(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut c = event.context.borrow_mut(); c.token_count = 0; c.error = EncodeError::None; c.unexpected = false; *event.unk_id.borrow_mut() = -1; *event.unk_lookup_found.borrow_mut() = false; *event.encode_push_failed.borrow_mut() = false; Ok(()) }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.begin_encode(event)?; self.tables_ready = false; self.vocabulary_identity = 0; Ok(()) }
    fn valid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(valid(event)) }
    fn invalid_encode(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!valid(event)) }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocabulary_identity != event.request.vocabulary.identity()) }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!self.vocab_changed(event)?) }
    fn text_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn output_capacity_covers_text(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.token_ids.borrow().len() >= event.request.text.len()) }
    fn tables_missing(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!self.tables_ready || self.vocabulary_identity != event.request.vocabulary.identity()) }
    fn tables_ready(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.tables_ready && self.vocabulary_identity == event.request.vocabulary.identity()) }
    fn sync_tables(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { self.tables_ready = false; self.vocabulary_identity = event.request.vocabulary.identity(); let mut decoded = [0; MAX_RWKV_TOKEN_BYTES]; for i in 0..event.request.vocabulary.token_count() { if decode_token(event.request.vocabulary.token_text(i), &mut decoded).is_none() { event.context.borrow_mut().error = EncodeError::InvalidArgument; return Ok(()); } } self.tables_ready = true; event.context.borrow_mut().error = EncodeError::None; Ok(()) }
    fn table_sync_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None) }
    fn vocab_unk_present(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.vocabulary.unk_id() >= 0) }
    fn vocab_unk_missing(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.vocabulary.unk_id() < 0) }
    fn resolve_vocab_unk(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let id = event.request.vocabulary.unk_id(); *event.unk_id.borrow_mut() = id; *event.unk_lookup_found.borrow_mut() = id >= 0; Ok(()) }
    fn lookup_unk_candidate(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut decoded = [0; MAX_RWKV_TOKEN_BYTES]; let mut found = -1; for i in 0..event.request.vocabulary.token_count() { if let Some(n) = decode_token(event.request.vocabulary.token_text(i), &mut decoded) { if decoded[..n] == *b"<unk>" { found = i as i32; break; } } } *event.unk_id.borrow_mut() = found; *event.unk_lookup_found.borrow_mut() = found >= 0; Ok(()) }
    fn unk_lookup_found(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(*event.unk_lookup_found.borrow()) }
    fn set_unk_from_lookup(&mut self, _event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { Ok(()) }
    fn set_unk_missing(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { *event.unk_id.borrow_mut() = -1; Ok(()) }
    fn run_encode(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut output = event.request.token_ids.borrow_mut(); let mut pos = 0; let mut count = 0; let unknown = *event.unk_id.borrow(); let mut failed = false; while pos < event.request.text.len() { let mut best_id = -1; let mut best_len = 0; let mut decoded = [0; MAX_RWKV_TOKEN_BYTES]; for i in 0..event.request.vocabulary.token_count() { if let Some(n) = decode_token(event.request.vocabulary.token_text(i), &mut decoded) { if n > best_len && pos + n <= event.request.text.len() && event.request.text[pos..pos+n] == decoded[..n] { best_id = i as i32; best_len = n; } } } if best_len == 0 { best_len = 1; best_id = unknown; } if best_id >= 0 { if count == output.len() { failed = true; break; } output[count] = best_id; count += 1; } pos += best_len; } *event.encode_push_failed.borrow_mut() = failed; let mut c = event.context.borrow_mut(); c.token_count = if failed { 0 } else { count }; if failed { c.error = EncodeError::InvalidArgument; } Ok(()) }
    fn encode_push_failed(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(*event.encode_push_failed.borrow()) }
    fn encode_push_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!*event.encode_push_failed.borrow()) }
    fn encode_result_ok(&self, event: &RuntimeEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None) }
    fn reject_invalid_encode(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject(event) }
    fn mark_encode_push_failed(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { reject(event) }
    fn mark_done(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { event.context.borrow_mut().error = EncodeError::None; Ok(()) }
    fn ensure_last_error(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut c = event.context.borrow_mut(); if c.error == EncodeError::None { c.error = EncodeError::Backend; } Ok(()) }
    fn on_unexpected_runtime(&mut self, event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut c = event.context.borrow_mut(); c.token_count = 0; c.error = EncodeError::InvalidArgument; c.unexpected = true; Ok(()) }
    fn on_unexpected_wild(&mut self) -> Result<(), ()> { Ok(()) }
}

fn valid(event: &RuntimeEncodeRuntime<'_>) -> bool { !event.request.token_ids.borrow().is_empty() && event.request.dispatch_done.is_some() && event.request.dispatch_error.is_some() }
fn reject(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> { let mut c = event.context.borrow_mut(); c.token_count = 0; c.error = EncodeError::InvalidArgument; Ok(()) }
fn hex_digit(b: u8) -> Option<u8> { match b { b'0'..=b'9' => Some(b - b'0'), b'a'..=b'f' => Some(b - b'a' + 10), b'A'..=b'F' => Some(b - b'A' + 10), _ => None } }
fn decode_token(input: &[u8], output: &mut [u8; MAX_RWKV_TOKEN_BYTES]) -> Option<usize> { let mut i = 0; let mut n = 0; while i < input.len() { if n == output.len() { return None; } if input[i] != b'\\' { output[n] = input[i]; n += 1; i += 1; continue; } i += 1; if i == input.len() { return None; } let value = match input[i] { b't' => b'\t', b'n' => b'\n', b'r' => b'\r', b'\\' => b'\\', b'x' => { if i + 2 >= input.len() { return None; } let h = hex_digit(input[i + 1])?; let l = hex_digit(input[i + 2])?; i += 2; (h << 4) | l }, x => x }; output[n] = value; n += 1; i += 1; } Some(n) }

/// Synchronous single-writer process wrapper.
pub struct TextEncodersRwkvActor<'event> { machine: TextEncodersRwkvStateMachine<'event, TextEncodersRwkvContext> }
impl<'event> Default for TextEncodersRwkvActor<'event> { fn default() -> Self { Self::new() } }
impl<'event> TextEncodersRwkvActor<'event> {
    #[must_use] pub fn new() -> Self { Self { machine: TextEncodersRwkvStateMachine::new(TextEncodersRwkvContext::default()) } }
    pub fn process_event(&mut self, request: EncodeRequest<'event>, context: &'event RefCell<EncodeContext>) -> EncodeError { let done = request.dispatch_done; let error = request.dispatch_error; let runtime = RuntimeEncodeRuntime::new(request, context); let accepted = self.machine.process_event(TextEncodersRwkvEvents::EventRuntimeEncodeRuntime(runtime)).is_ok(); let result = if accepted { context.borrow().error } else { EncodeError::Backend }; if result == EncodeError::None { if let Some(cb) = done { let _ = cb(EventsEncodingDone { token_count: context.borrow().token_count }); } } else if let Some(cb) = error { let _ = cb(EventsEncodingError { error: result }); } result }
    pub fn process_unexpected(&mut self, context: &RefCell<EncodeContext>) -> EncodeError { let mut c = context.borrow_mut(); c.token_count = 0; c.error = EncodeError::InvalidArgument; c.unexpected = true; self.machine.set_state(TextEncodersRwkvStates::Unexpected); EncodeError::InvalidArgument }
    #[must_use] pub fn state(&self) -> &TextEncodersRwkvStates { self.machine.state() }
    #[must_use] pub fn machine_context(&self) -> &TextEncodersRwkvContext { self.machine.context() }
}

pub type Rwkv = TextEncodersRwkvActor<'static>;
