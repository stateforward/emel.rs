//! Source-aligned, bounded PLaMo2 text encoder actor.
//!
//! The actor keeps request storage caller-owned, runs to completion on one
//! writer, and exposes the generated state machine for inspection.  The
//! vocabulary view deliberately contains only the data needed by this actor;
//! model loaders can adapt their private vocabulary without leaking ownership.

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

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

/// Maximum number of Unicode scalar values accepted by one request.
pub const MAX_ENCODE_CODEPOINTS: usize = 16_384;

/// A borrowed vocabulary entry used by the encoder.
#[derive(Clone, Copy, Debug)]
pub struct TokenEntry<'a> {
    /// UTF-8 token spelling.
    pub text: &'a str,
    /// Model score; larger scores are preferred.
    pub score: f32,
    /// Source token type.  Type `6` is the PLaMo2 byte-token type.
    pub kind: i32,
}

/// Minimal borrowed vocabulary contract required by PLaMo2.
pub trait VocabView {
    /// Number of entries available for lookup.
    fn token_count(&self) -> usize;
    /// Returns an entry without transferring ownership.
    fn token(&self, index: usize) -> Option<TokenEntry<'_>>;
}

/// Stable encoder error values matching the maintained C++ error bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum EncodeError {
    /// No error.
    None = 0,
    /// Request or output storage was invalid or too small.
    InvalidArgument = 1,
    /// A phase could not complete in the supplied backend/model data.
    Backend = 2,
    /// Vocabulary data cannot build a usable PLaMo2 table.
    ModelInvalid = 4,
}

impl Default for EncodeError {
    fn default() -> Self { Self::None }
}

impl EncodeError {
    const fn code(self) -> i32 { self as i32 }
    fn from_code(code: i32) -> Self {
        match code {
            0 => Self::None,
            1 => Self::InvalidArgument,
            4 => Self::ModelInvalid,
            _ => Self::Backend,
        }
    }
}

/// Successful bounded encoding callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingDone {
    /// Number of token IDs written to the caller's buffer.
    pub token_count: usize,
}

/// Failed bounded encoding callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventsEncodingError {
    /// Stable error value for the failed request.
    pub error: EncodeError,
}

/// Synchronous completion callbacks.  Their boolean return is intentionally
/// ignored, matching the source callback contract.
pub type DoneCallback = fn(EventsEncodingDone) -> bool;
/// Synchronous failure callback.
pub type ErrorCallback = fn(EventsEncodingError) -> bool;

/// Caller-owned encode request.  No request storage is allocated by the actor.
pub struct EncodeRequest<'a> {
    /// Borrowed vocabulary.
    pub vocab: &'a dyn VocabView,
    /// UTF-8 input text.
    pub text: &'a str,
    /// Caller-owned token output storage.
    pub token_ids: &'a RefCell<&'a mut [i32]>,
    /// Optional output callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional failure callback.
    pub dispatch_error: Option<ErrorCallback>,
}

impl fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeRequest")
            .field("text", &self.text)
            .field("token_capacity", &self.token_ids.borrow().len())
            .finish_non_exhaustive()
    }
}

impl<'a> EncodeRequest<'a> {
    /// Constructs a request with both callbacks installed.
    #[must_use]
    pub const fn new(
        vocab: &'a dyn VocabView,
        text: &'a str,
        token_ids: &'a RefCell<&'a mut [i32]>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self { vocab, text, token_ids, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }

    /// Constructs a request with independently optional callbacks.
    #[must_use]
    pub const fn with_callbacks(
        vocab: &'a dyn VocabView,
        text: &'a str,
        token_ids: &'a RefCell<&'a mut [i32]>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self { vocab, text, token_ids, dispatch_done, dispatch_error }
    }
}

/// Mutable result fields corresponding to `event::encode_ctx`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeContext {
    /// Last phase error.
    pub error: EncodeError,
    /// Number of IDs committed by the successful request.
    pub token_count: usize,
}

impl Default for EncodeContext {
    fn default() -> Self { Self { error: EncodeError::None, token_count: 0 } }
}

/// Runtime event carrying borrowed request and result context.
pub struct RuntimeEncodeRuntime<'a> {
    /// Borrowed request.
    pub request: EncodeRequest<'a>,
    /// Caller-owned result context.
    pub context: &'a RefCell<EncodeContext>,
    pub data_len: Cell<usize>,
    /// Error produced by the emission phase.
    pub emit_result_error: Cell<EncodeError>,
    /// Number of IDs produced by the emission phase.
    pub emit_result_token_count: Cell<usize>,
}

impl fmt::Debug for RuntimeEncodeRuntime<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeEncodeRuntime")
            .field("request", &self.request)
            .field("data_len", &self.data_len)
            .finish_non_exhaustive()
    }
}

impl<'a> RuntimeEncodeRuntime<'a> {
    /// Creates a fresh runtime event.
    #[must_use]
    pub fn new(request: EncodeRequest<'a>, context: &'a RefCell<EncodeContext>) -> Self {
        Self { request, context, data_len: Cell::new(0), emit_result_error: Cell::new(EncodeError::None), emit_result_token_count: Cell::new(0) }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PathEntry {
    token_len: usize,
    token_id: i32,
}

/// State owned by the generated PLaMo2 machine.
pub struct TextEncodersPlamo2Context<'a> {
    /// Whether vocabulary-derived lookup is usable.
    pub tables_ready: bool,
    /// Last request vocabulary, used for source-aligned sync guards.
    vocab: Option<&'a dyn VocabView>,
    /// Decoded input codepoints.
    codepoints: [u32; MAX_ENCODE_CODEPOINTS],
    /// DP scores, indexed from the current codepoint.
    scores: [i64; MAX_ENCODE_CODEPOINTS + 1],
    /// Best path at each codepoint.
    paths: [PathEntry; MAX_ENCODE_CODEPOINTS + 1],
    /// Whether the actor encountered a sequencing violation.
    pub unexpected: bool,
}

impl<'a> fmt::Debug for TextEncodersPlamo2Context<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextEncodersPlamo2Context")
            .field("tables_ready", &self.tables_ready)
            .field("unexpected", &self.unexpected)
            .finish_non_exhaustive()
    }
}

impl<'a> Default for TextEncodersPlamo2Context<'a> {
    fn default() -> Self {
        Self {
            tables_ready: false,
            vocab: None,
            codepoints: [0; MAX_ENCODE_CODEPOINTS],
            scores: [0; MAX_ENCODE_CODEPOINTS + 1],
            paths: [PathEntry::default(); MAX_ENCODE_CODEPOINTS + 1],
            unexpected: false,
        }
    }
}

sml! {
    TextEncodersPlamo2<'a> {
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
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [tables_missing],
        "decode_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / sync_tables,
        "decode_exec"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "decode_result_decision"_s <= "decode_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / decode_input,
        "done"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_empty_ok] / mark_done_from_decode_result_decision,
        "dp_prepare_exec"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_non_empty_ok],
        "errored"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_invalid_argument_error] / ensure_last_error_from_decode_result_decision,
        "errored"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_backend_error] / ensure_last_error_from_decode_result_decision,
        "errored"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_model_invalid_error] / ensure_last_error_from_decode_result_decision,
        "errored"_s <= "decode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [decode_result_unclassified_error_code] / ensure_last_error_from_decode_result_decision,
        "dp_exec"_s <= "dp_prepare_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / prepare_dp,
        "backtrace_exec"_s <= "dp_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / run_dp,
        "backtrace_result_decision"_s <= "backtrace_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / validate_backtrace,
        "emit_exec"_s <= "backtrace_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [backtrace_ok],
        "errored"_s <= "backtrace_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [backtrace_failed] / mark_backtrace_failed,
        "errored"_s <= "backtrace_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_backtrace_result_decision,
        "emit_result_decision"_s <= "emit_exec"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / emit_tokens,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [emit_result_ok] / apply_emit_result_ok,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [emit_result_failed] / apply_emit_result_failed,
        "errored"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_emit_result_decision,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime>(RuntimeEncodeRuntime<'a>) / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_policy_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "decode_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "decode_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_prepare_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "backtrace_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "backtrace_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_exec"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime<'a>> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

impl<'a> TextEncodersPlamo2StateMachineContext for TextEncodersPlamo2Context<'a> {
    fn apply_emit_result_failed(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let mut context = event.context.borrow_mut();
        context.token_count = 0;
        context.error = event.emit_result_error.get();
        Ok(())
    }
    fn apply_emit_result_ok(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let mut context = event.context.borrow_mut();
        context.token_count = event.emit_result_token_count.get();
        context.error = EncodeError::None;
        Ok(())
    }
    fn begin_encode(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let mut context = event.context.borrow_mut();
        context.token_count = 0;
        context.error = EncodeError::None;
        event.data_len.set(0);
        event.emit_result_error.set(EncodeError::None);
        event.emit_result_token_count.set(0);
        self.unexpected = false;
        Ok(())
    }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        self.begin_encode(event)?;
        self.vocab = Some(event.request.vocab);
        self.tables_ready = false;
        Ok(())
    }
    fn decode_input(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let text = event.request.text;
        let mut index = 0usize;
        for codepoint in text.chars() {
            if codepoint == '\u{feff}' && index == 0 { continue; }
            if index == MAX_ENCODE_CODEPOINTS {
                event.context.borrow_mut().error = EncodeError::InvalidArgument;
                event.data_len.set(0);
                return Ok(());
            }
            self.codepoints[index] = codepoint as u32;
            index += 1;
        }
        event.data_len.set(index);
        Ok(())
    }
    fn decode_result_backend_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::Backend) }
    fn decode_result_empty_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None && event.data_len.get() == 0) }
    fn decode_result_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::InvalidArgument) }
    fn decode_result_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::ModelInvalid) }
    fn decode_result_non_empty_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None && event.data_len.get() > 0) }
    fn decode_result_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error != EncodeError::None && !matches!(event.context.borrow().error, EncodeError::InvalidArgument | EncodeError::Backend | EncodeError::ModelInvalid)) }
    fn emit_result_failed(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.emit_result_error.get() != EncodeError::None) }
    fn emit_result_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.emit_result_error.get() == EncodeError::None) }
    fn emit_tokens(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let mut output = event.request.token_ids.borrow_mut();
        let capacity = output.len();
        let mut count = 0usize;
        let mut pos = 0usize;
        let mut error = EncodeError::None;
        while pos < event.data_len.get() {
            let path = self.paths[pos];
            if path.token_len == 0 || pos.saturating_add(path.token_len) > event.data_len.get() { error = EncodeError::Backend; break; }
            if path.token_id >= 0 {
                if count == capacity { error = EncodeError::InvalidArgument; break; }
                output[count] = path.token_id;
                count += 1;
            } else {
                let cp = self.codepoints[pos];
                let mut bytes = [0u8; 4];
                let encoded = char::from_u32(cp).map_or(0, |ch| ch.encode_utf8(&mut bytes).len());
                for byte in &bytes[..encoded] {
                    let token = find_byte_token(event.request.vocab, *byte);
                    if token < 0 { error = EncodeError::Backend; break; }
                    if count == capacity { error = EncodeError::InvalidArgument; break; }
                    output[count] = token;
                    count += 1;
                }
                if error != EncodeError::None { break; }
            }
            pos += path.token_len;
        }
        event.emit_result_token_count.set(if error == EncodeError::None { count } else { 0 });
        event.emit_result_error.set(error);
        Ok(())
    }
    fn encode_result_backend_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::Backend) }
    fn encode_result_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::InvalidArgument) }
    fn encode_result_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::ModelInvalid) }
    fn encode_result_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None) }
    fn encode_result_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error != EncodeError::None && !matches!(event.context.borrow().error, EncodeError::InvalidArgument | EncodeError::Backend | EncodeError::ModelInvalid)) }
    fn ensure_last_error_from_decode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_emit_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_encode_precheck_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_table_policy_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_table_sync_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn ensure_last_error_from_backtrace_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { ensure_last_error(event) }
    fn invalid_encode(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.request.token_ids.borrow().is_empty()) }
    fn mark_done_from_decode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        event.context.borrow_mut().error = EncodeError::None;
        dispatch_done(event, 0);
        Ok(())
    }
    fn mark_done_from_encode_precheck_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        event.context.borrow_mut().error = EncodeError::None;
        dispatch_done(event, 0);
        Ok(())
    }
    fn mark_done_from_encode_result_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        event.context.borrow_mut().error = EncodeError::None;
        dispatch_done(event, event.emit_result_token_count.get());
        Ok(())
    }
    fn on_unexpected_events_encoding_done(&mut self, _event: &EventsEncodingDone) -> Result<(), ()> { self.unexpected = true; Ok(()) }
    fn on_unexpected_events_encoding_error(&mut self, _event: &EventsEncodingError) -> Result<(), ()> { self.unexpected = true; Ok(()) }
    fn on_unexpected_runtime_encode_runtime(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { self.unexpected = true; let mut context = event.context.borrow_mut(); context.token_count = 0; context.error = EncodeError::InvalidArgument; Ok(()) }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> { self.unexpected = true; Ok(()) }
    fn prepare_dp(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let len = event.data_len.get();
        for index in 0..=len { self.scores[index] = i64::MAX / 4; self.paths[index] = PathEntry::default(); }
        self.scores[len] = 0;
        Ok(())
    }
    fn reject_invalid_encode_from_encode_validity_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { reject_invalid(event) }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { reject_invalid(event) }
    fn run_dp(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        let Some(vocab) = self.vocab.or(Some(event.request.vocab)) else { event.context.borrow_mut().error = EncodeError::ModelInvalid; return Ok(()); };
        for pos in (0..event.data_len.get()).rev() {
            let mut best_score = i64::MAX / 4;
            let mut best = PathEntry::default();
            for index in 0..vocab.token_count() {
                let Some(entry) = vocab.token(index) else { continue; };
                if entry.text.is_empty() || entry.kind == 6 { continue; }
                let mut end = pos;
                let mut matched = true;
                for token_cp in entry.text.chars() {
                    if end >= event.data_len.get() || self.codepoints[end] != token_cp as u32 { matched = false; break; }
                    end += 1;
                }
                if matched && end > pos {
                    let score = self.scores[end].saturating_sub((entry.score as f64 * 10_000.0) as i64);
                    if score < best_score { best_score = score; best = PathEntry { token_len: end - pos, token_id: index as i32 }; }
                }
            }
            if best.token_len == 0 {
                let cp = self.codepoints[pos];
                let mut bytes = [0u8; 4];
                let Some(ch) = char::from_u32(cp) else { event.context.borrow_mut().error = EncodeError::Backend; return Ok(()); };
                let encoded = ch.encode_utf8(&mut bytes).len();
                let mut fallback_ok = true;
                for byte in &bytes[..encoded] { if find_byte_token(vocab, *byte) < 0 { fallback_ok = false; break; } }
                if fallback_ok { best = PathEntry { token_len: 1, token_id: -1 }; best_score = self.scores[pos + 1].saturating_add(10_000_000); }
            }
            if best.token_len == 0 { event.context.borrow_mut().error = EncodeError::Backend; return Ok(()); }
            self.scores[pos] = best_score;
            self.paths[pos] = best;
        }
        Ok(())
    }
    fn sync_tables(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> {
        self.vocab = Some(event.request.vocab);
        self.tables_ready = event.request.vocab.token_count() > 0;
        event.context.borrow_mut().error = if self.tables_ready { EncodeError::None } else { EncodeError::ModelInvalid };
        Ok(())
    }
    fn table_sync_backend_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::Backend) }
    fn table_sync_invalid_argument_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::InvalidArgument) }
    fn table_sync_model_invalid_error(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::ModelInvalid) }
    fn table_sync_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None) }
    fn table_sync_unclassified_error_code(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error != EncodeError::None && !matches!(event.context.borrow().error, EncodeError::InvalidArgument | EncodeError::Backend | EncodeError::ModelInvalid)) }
    fn tables_missing(&self, _event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!self.tables_ready) }
    fn tables_ready(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(self.tables_ready && self.vocab.is_some_and(|vocab| core::ptr::eq(vocab, event.request.vocab))) }
    fn text_empty(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.request.text.is_empty()) }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!event.request.text.is_empty()) }
    fn valid_encode(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!event.request.token_ids.borrow().is_empty()) }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!self.vocab.is_some_and(|vocab| core::ptr::eq(vocab, event.request.vocab))) }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!self.vocab_changed(event)?) }
    fn backtrace_ok(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(event.context.borrow().error == EncodeError::None && (event.data_len.get() == 0 || self.paths[0].token_len > 0)) }
    fn backtrace_failed(&self, event: &RuntimeEncodeRuntime<'a>) -> Result<bool, ()> { Ok(!self.backtrace_ok(event)?) }
    fn validate_backtrace(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { if event.data_len.get() > 0 && self.paths[0].token_len == 0 { event.context.borrow_mut().error = EncodeError::Backend; } Ok(()) }
    fn mark_backtrace_failed(&mut self, event: &RuntimeEncodeRuntime<'a>) -> Result<(), ()> { event.context.borrow_mut().error = EncodeError::Backend; Ok(()) }
}

fn reject_invalid(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> {
    let mut context = event.context.borrow_mut();
    context.token_count = 0;
    context.error = EncodeError::InvalidArgument;
    Ok(())
}

fn ensure_last_error(event: &RuntimeEncodeRuntime<'_>) -> Result<(), ()> {
    let mut context = event.context.borrow_mut();
    if context.error == EncodeError::None { context.error = EncodeError::Backend; }
    context.token_count = 0;
    Ok(())
}

fn find_byte_token(vocab: &dyn VocabView, byte: u8) -> i32 {
    let digits = *b"0123456789ABCDEF";
    for index in 0..vocab.token_count() {
        let Some(entry) = vocab.token(index) else { continue; };
        let bytes = entry.text.as_bytes();
        if entry.kind == 6 && bytes.len() == 6 && bytes[0] == b'<' && bytes[1] == b'0' && (bytes[2] == b'x' || bytes[2] == b'X') && bytes[5] == b'>' && bytes[3].to_ascii_uppercase() == digits[(byte >> 4) as usize] && bytes[4].to_ascii_uppercase() == digits[(byte & 0x0f) as usize] { return index as i32; }
    }
    -1
}

fn dispatch_done(event: &RuntimeEncodeRuntime<'_>, token_count: usize) {
    if let Some(callback) = event.request.dispatch_done {
        let _ = callback(EventsEncodingDone { token_count });
    }
}

/// Synchronous bounded PLaMo2 actor around the generated machine.
pub struct TextEncodersPlamo2Actor<'a> {
    machine: TextEncodersPlamo2StateMachine<'a, TextEncodersPlamo2Context<'a>>,
}

impl<'a> Default for TextEncodersPlamo2Actor<'a> {
    fn default() -> Self { Self::new() }
}

impl<'a> TextEncodersPlamo2Actor<'a> {
    /// Creates an actor in the generated `initialized` state.
    #[must_use]
    pub fn new() -> Self { Self { machine: TextEncodersPlamo2StateMachine::new(TextEncodersPlamo2Context::default()) } }
    pub fn process_event(&mut self, event: RuntimeEncodeRuntime<'a>) -> bool {
        let context = event.context;
        let error_callback = event.request.dispatch_error;
        let accepted = self
            .machine
            .process_event(TextEncodersPlamo2Events::EventRuntimeEncodeRuntime(event))
            .is_ok();
        let result = *context.borrow();
        if result.error != EncodeError::None {
            if let Some(callback) = error_callback {
                let _ = callback(EventsEncodingError { error: result.error });
            }
        }
        accepted && result.error == EncodeError::None
    }
    /// Dispatches an explicit sequencing violation.
    pub fn process_unexpected(&mut self) -> bool { self.machine.context_mut().unexpected = true; self.machine.set_state(TextEncodersPlamo2States::Unexpected); false }
    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextEncodersPlamo2States { self.machine.state() }
    /// Reports whether the actor is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextEncodersPlamo2States) -> bool { self.machine.is(state) }
    /// Returns generated context inspection data.
    #[must_use]
    pub fn context(&self) -> &TextEncodersPlamo2Context<'a> { self.machine.context() }
}

/// Short actor alias matching the pinned PLaMo2 encoder name.
pub type Plamo2<'a> = TextEncodersPlamo2Actor<'a>;
