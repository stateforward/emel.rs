//! Source-aligned bounded WPM text encoder actor.

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

use sml::sml;

/// Maximum bytes inspected in one WPM request.  The source scratch buffers are
/// fixed-size; requests beyond this bound take the invalid-argument path.
pub const MAX_INPUT_BYTES: usize = 4096;
/// Maximum vocabulary entries inspected while synchronizing and looking up a
/// piece.  This is deliberately bounded even for malformed model metadata.
pub const MAX_VOCAB_ENTRIES: usize = 32_768;

/// A vocabulary view used by the encoder.  Implementations own the storage;
/// the actor only borrows it for the duration of one synchronous request.
pub trait WpmVocabularyView {
    /// Returns a stable identity for cache invalidation.
    fn identity(&self) -> usize;
    /// Returns the number of active entries.
    fn token_count(&self) -> u32;
    /// Returns the UTF-8 bytes for an active token.
    fn token(&self, index: u32) -> Option<&[u8]>;
    /// Returns the model's unknown-token id, or `-1` when absent.
    fn unk_id(&self) -> i32 { -1 }
}

/// Successful completion notification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingDone {
    /// Number of token ids written.
    pub token_count: i32,
}
/// Failed completion notification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingError {
    /// Source-compatible error code.
    pub error: EncodeError,
}
/// Synchronous successful callback.
pub type DoneCallback = fn(EventsEncodingDone) -> bool;
/// Synchronous failed callback.
pub type ErrorCallback = fn(EventsEncodingError) -> bool;

/// Source-compatible encoder error code.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum EncodeError {
    /// The request completed successfully.
    #[default]
    Ok = 0,
    /// Request, capacity, or output storage was invalid.
    InvalidArgument = 1,
    /// A backend operation failed.
    Backend = 2,
    /// Model metadata could not be used.
    ModelInvalid = 4,
}
impl EncodeError {
    /// Returns the source numeric representation.
    #[must_use]
    pub const fn code(self) -> i32 { self as i32 }
}

/// Caller-owned bounded encode request.
pub struct EncodeRequest<'a> {
    pub vocab: &'a dyn WpmVocabularyView,
    pub text: &'a str,
    pub token_ids: &'a RefCell<&'a mut [i32]>,
    pub token_count_out: Option<&'a RefCell<i32>>,
    pub error_out: Option<&'a RefCell<i32>>,
    pub dispatch_done: Option<DoneCallback>,
    pub dispatch_error: Option<ErrorCallback>,
}
impl<'a> EncodeRequest<'a> {
    pub fn new(vocab: &'a dyn WpmVocabularyView, text: &'a str, token_ids: &'a RefCell<&'a mut [i32]>, token_count_out: &'a RefCell<i32>, error_out: &'a RefCell<i32>, dispatch_done: DoneCallback, dispatch_error: ErrorCallback) -> Self {
        Self { vocab, text, token_ids, token_count_out: Some(token_count_out), error_out: Some(error_out), dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }
    pub const fn with_callbacks(vocab: &'a dyn WpmVocabularyView, text: &'a str, token_ids: &'a RefCell<&'a mut [i32]>, token_count_out: Option<&'a RefCell<i32>>, error_out: Option<&'a RefCell<i32>>, dispatch_done: Option<DoneCallback>, dispatch_error: Option<ErrorCallback>) -> Self {
        Self { vocab, text, token_ids, token_count_out, error_out, dispatch_done, dispatch_error }
    }
}

/// Mutable result context corresponding to the source `encode_ctx`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EncodeContext {
    /// Number of ids produced by the current request.
    pub token_count: i32,
    /// Current request error.
    pub error: EncodeError,
    /// Whether vocabulary-derived tables are ready.
    pub tables_ready: bool,
    /// Identity of the vocabulary used to build tables.
    pub vocab_identity: usize,
    /// Longest token byte length observed during synchronization.
    pub max_token_len: usize,
    /// Set after an unexpected event.
    pub unexpected: bool,
}

/// Runtime event carrying a borrowed request and its result context.
pub struct EventEncodeRuntime<'a> {
    /// Request being processed.
    pub request: &'a mut EncodeRequest<'a>,
    /// Mutable runtime result/context.
    pub context: &'a mut EncodeContext,
}
impl<'a> EventEncodeRuntime<'a> {
    /// Constructs one runtime event.
    pub const fn new(request: &'a mut EncodeRequest<'a>, context: &'a mut EncodeContext) -> Self { Self { request, context } }
}
impl core::fmt::Debug for EventEncodeRuntime<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("EventEncodeRuntime").field("text", &self.request.text).finish_non_exhaustive() }
}

sml! {
    TextEncodersWpm<'event> {
        "encode_validity_decision"_s <= *"initialized"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "done"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "errored"_s + event<EventEncodeRuntime<'event>>,
        "encode_validity_decision"_s <= "unexpected"_s + event<EventEncodeRuntime<'event>>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime<'event>> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime<'event>> [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime<'event>> / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime<'event>> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime<'event>> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime<'event>> / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime<'event>> [text_empty] / mark_done_from_encode_precheck_decision,
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime<'event>> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime<'event>> / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime<'event>> [tables_missing],
        "encode_input_capacity_decision"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime<'event>> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime<'event>> / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<EventEncodeRuntime<'event>> / sync_tables,
        "encode_input_capacity_decision"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime<'event>> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime<'event>> [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime<'event>> [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime<'event>> [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime<'event>> [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "encode_exec"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime<'event>> [prefix_buffer_capacity_within_limit],
        "errored"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime<'event>> [prefix_buffer_capacity_exceeded] / reject_invalid_encode_from_encode_input_capacity_decision,
        "errored"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime<'event>> / reject_invalid_encode_from_encode_input_capacity_decision,
        "encode_result_decision"_s <= "encode_exec"_s + completion<EventEncodeRuntime<'event>> / run_encode,
        "done"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime<'event>> [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime<'event>> [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime<'event>> [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime<'event>> [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime<'event>> [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingError>,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_input_capacity_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_input_capacity_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingError>,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingError>,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "done"_s + event<EventsEncodingError>,
        "unexpected"_s <= "errored"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError>,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingDone>,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingError>,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_input_capacity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

impl TextEncodersWpmStateMachineContext for EncodeContext {
    fn begin_encode(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { reset_runtime(event); Ok(()) }
    fn begin_encode_sync_vocab(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { reset_runtime(event); self.vocab_identity = event.request.vocab.identity(); self.tables_ready = false; Ok(()) }
    fn encode_result_backend_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::Backend) }
    fn encode_result_invalid_argument_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::InvalidArgument) }
    fn encode_result_model_invalid_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::ModelInvalid) }
    fn encode_result_ok(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::Ok) }
    fn encode_result_unclassified_error_code(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!matches!(event.context.error, EncodeError::Ok | EncodeError::InvalidArgument | EncodeError::Backend | EncodeError::ModelInvalid)) }
    fn ensure_last_error_from_encode_precheck_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event); Ok(()) }
    fn ensure_last_error_from_encode_result_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event); Ok(()) }
    fn ensure_last_error_from_table_policy_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event); Ok(()) }
    fn ensure_last_error_from_table_sync_result_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { ensure_error(event); Ok(()) }
    fn invalid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!valid_encode(event)) }
    fn mark_done_from_encode_precheck_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { finish_done(event); Ok(()) }
    fn mark_done_from_encode_result_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { finish_done(event); Ok(()) }
    fn on_unexpected_event_encode_runtime(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { unexpected(event); Ok(()) }
    fn on_unexpected_events_encoding_done(&mut self, _event: &EventsEncodingDone) -> Result<(), ()> { self.unexpected = true; self.error = EncodeError::InvalidArgument; Ok(()) }
    fn on_unexpected_events_encoding_error(&mut self, _event: &EventsEncodingError) -> Result<(), ()> { self.unexpected = true; self.error = EncodeError::InvalidArgument; Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.unexpected = true; self.error = EncodeError::InvalidArgument; Ok(()) }
    fn prefix_buffer_capacity_exceeded(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!prefix_capacity(event)) }
    fn prefix_buffer_capacity_within_limit(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(prefix_capacity(event)) }
    fn reject_invalid_encode_from_encode_input_capacity_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { reject(event); Ok(()) }
    fn reject_invalid_encode_from_encode_validity_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { reject(event); Ok(()) }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { reject(event); Ok(()) }
    fn run_encode(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { encode_wpm(event); Ok(()) }
    fn sync_tables(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> { sync_tables(event); Ok(()) }
    fn table_sync_backend_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::Backend) }
    fn table_sync_invalid_argument_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::InvalidArgument) }
    fn table_sync_model_invalid_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::ModelInvalid) }
    fn table_sync_ok(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.context.error == EncodeError::Ok) }
    fn tables_missing(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!tables_ready(event)) }
    fn tables_ready(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(tables_ready(event)) }
    fn text_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn valid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(valid_encode(event)) }
    fn vocab_changed(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocab_identity != event.request.vocab.identity()) }
    fn vocab_unchanged(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> { Ok(self.vocab_identity == event.request.vocab.identity()) }
}

fn reset_runtime(event: &EventEncodeRuntime<'_>) { event.context.token_count = 0; event.context.error = EncodeError::Ok; event.context.unexpected = false; }
fn valid_encode(event: &EventEncodeRuntime<'_>) -> bool { !event.request.token_ids.borrow().is_empty() && event.request.text.len() <= MAX_INPUT_BYTES }
fn prefix_capacity(event: &EventEncodeRuntime<'_>) -> bool { event.request.text.len() <= MAX_INPUT_BYTES.saturating_sub(3) }
fn tables_ready(event: &EventEncodeRuntime<'_>) -> bool { event.context.tables_ready && event.context.vocab_identity == event.request.vocab.identity() }
fn reject(event: &EventEncodeRuntime<'_>) { event.context.token_count = 0; event.context.error = EncodeError::InvalidArgument; publish_error(event); }
fn ensure_error(event: &EventEncodeRuntime<'_>) { if event.context.error == EncodeError::Ok { event.context.error = EncodeError::Backend; } publish_error(event); }
fn unexpected(event: &EventEncodeRuntime<'_>) { event.context.unexpected = true; event.context.token_count = 0; event.context.error = EncodeError::InvalidArgument; publish_error(event); }

fn sync_tables(event: &EventEncodeRuntime<'_>) {
    let count = (event.request.vocab.token_count() as usize).min(MAX_VOCAB_ENTRIES);
    if count == 0 { event.context.error = EncodeError::ModelInvalid; event.context.tables_ready = false; return; }
    let mut max_len = 0usize;
    for index in 0..count {
        let Some(token) = event.request.vocab.token(index as u32) else { event.context.error = EncodeError::ModelInvalid; event.context.tables_ready = false; return; };
        max_len = max_len.max(token.len());
    }
    event.context.max_token_len = max_len;
    event.context.tables_ready = true;
    event.context.vocab_identity = event.request.vocab.identity();
    event.context.error = EncodeError::Ok;
}

fn lookup(vocab: &dyn WpmVocabularyView, candidate: &[u8]) -> Option<i32> {
    if candidate.is_empty() { return None; }
    let count = (vocab.token_count() as usize).min(MAX_VOCAB_ENTRIES);
    for index in 0..count { if vocab.token(index as u32).is_some_and(|token| token == candidate) { return Some(index as i32); } }
    None
}
fn publish_error(event: &EventEncodeRuntime<'_>) { if let Some(out) = event.request.token_count_out { *out.borrow_mut() = 0; } if let Some(out) = event.request.error_out { *out.borrow_mut() = event.context.error.code(); } if let Some(callback) = event.request.dispatch_error { let _ = callback(EventsEncodingError { error: event.context.error }); } }
fn finish_done(event: &EventEncodeRuntime<'_>) { event.context.error = EncodeError::Ok; if let Some(out) = event.request.token_count_out { *out.borrow_mut() = event.context.token_count; } if let Some(out) = event.request.error_out { *out.borrow_mut() = EncodeError::Ok.code(); } if let Some(callback) = event.request.dispatch_done { let _ = callback(EventsEncodingDone { token_count: event.context.token_count }); } }

fn encode_wpm(event: &EventEncodeRuntime<'_>) {
    let text = event.request.text.as_bytes();
    let mut pos = 0usize;
    let mut count = 0i32;
    let mut word = [0u8; MAX_INPUT_BYTES];
    while pos < text.len() {
        while pos < text.len() && text[pos].is_ascii_whitespace() { pos += 1; }
        if pos >= text.len() { break; }
        let start = pos;
        let punctuation = text[pos].is_ascii_punctuation();
        if punctuation { pos += 1; } else { while pos < text.len() && !text[pos].is_ascii_whitespace() && !text[pos].is_ascii_punctuation() { pos += 1; } }
        let piece = &text[start..pos];
        if piece.len() > word.len().saturating_sub(3) { event.context.error = EncodeError::InvalidArgument; return; }
        for (dst, src) in word.iter_mut().zip(piece.iter().copied()) { *dst = src.to_ascii_lowercase(); }
        if !encode_piece(event.request, &word[..piece.len()], &mut count) { event.context.error = EncodeError::InvalidArgument; return; }
    }
    event.context.token_count = count;
    event.context.error = EncodeError::Ok;
}

fn encode_piece(request: &EncodeRequest<'_>, piece: &[u8], count: &mut i32) -> bool {
    let mut cursor = 0usize;
    while cursor < piece.len() {
        let mut best_end = cursor;
        let mut candidate = [0u8; MAX_INPUT_BYTES];
        let prefix = if cursor == 0 { b"\xE2\x96\x81".as_slice() } else { b"##".as_slice() };
        let max_end = piece.len().min(cursor + MAX_INPUT_BYTES.saturating_sub(prefix.len()));
        for end in (cursor + 1..=max_end).rev() {
            let n = prefix.len() + end - cursor;
            candidate[..prefix.len()].copy_from_slice(prefix);
            candidate[prefix.len()..n].copy_from_slice(&piece[cursor..end]);
            if lookup(request.vocab, &candidate[..n]).or_else(|| lookup(request.vocab, &piece[cursor..end])).is_some() { best_end = end; break; }
        }
        if best_end == cursor {
            let unk = request.vocab.unk_id();
            if unk >= 0 { let mut tokens = request.token_ids.borrow_mut(); if (*count as usize) >= tokens.len() { return false; } tokens[*count as usize] = unk; *count += 1; }
            cursor = piece.len();
            continue;
        }
        let end = best_end;
        let n = prefix.len() + end - cursor;
        candidate[..prefix.len()].copy_from_slice(prefix);
        candidate[prefix.len()..n].copy_from_slice(&piece[cursor..end]);
        let token = lookup(request.vocab, &candidate[..n]).or_else(|| lookup(request.vocab, &piece[cursor..end])).unwrap_or(-1);
        let mut tokens = request.token_ids.borrow_mut();
        if token < 0 || (*count as usize) >= tokens.len() { return false; }
        tokens[*count as usize] = token;
        *count += 1;
        cursor = end;
    }
    true
}

/// Synchronous bounded owner of the generated WPM machine.
pub struct TextEncodersWpmActor<'event> {
    machine: TextEncodersWpmStateMachine<'event, EncodeContext>,
}
impl<'event> Default for TextEncodersWpmActor<'event> { fn default() -> Self { Self::new() } }
impl<'event> TextEncodersWpmActor<'event> {
    /// Creates an actor in the generated `initialized` state.
    pub fn new() -> Self { Self { machine: TextEncodersWpmStateMachine::new(EncodeContext::default()) } }
    /// Processes one request synchronously and returns whether it succeeded.
    pub fn process_event(&mut self, event: EventEncodeRuntime<'event>) -> bool {
        self.machine.process_event(TextEncodersWpmEvents::EventEncodeRuntime(event)).is_ok()
    }
    /// Moves the actor into the explicit unexpected state and reports failure.
    pub fn process_unexpected(&mut self) -> bool { self.machine.set_state(TextEncodersWpmStates::Unexpected); self.machine.context_mut().unexpected = true; false }
    /// Returns generated state inspection data.
    pub fn state(&self) -> &TextEncodersWpmStates { self.machine.state() }
    /// Reports whether the actor is in `state`.
    pub fn is(&self, state: &TextEncodersWpmStates) -> bool { self.machine.is(state) }
    /// Returns the actor's persistent context.
    pub fn context(&self) -> &EncodeContext { self.machine.context() }
    /// Returns mutable persistent context for caller-owned inspection/reset.
    pub fn context_mut(&mut self) -> &mut EncodeContext { self.machine.context_mut() }
}
/// Short alias matching the pinned C++ `Wpm` name.
pub type Wpm<'event> = TextEncodersWpmActor<'event>;
