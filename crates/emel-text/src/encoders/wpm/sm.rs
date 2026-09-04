//! Source-aligned bounded WPM text encoder actor.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::manual_let_else,
    clippy::too_many_arguments,
    clippy::needless_lifetimes,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::RefCell;
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
    fn unk_id(&self) -> i32 {
        -1
    }
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
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Caller-owned bounded encode request.
#[derive(Clone, Copy)]
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
    pub fn new(
        vocab: &'a dyn WpmVocabularyView,
        text: &'a str,
        token_ids: &'a RefCell<&'a mut [i32]>,
        token_count_out: &'a RefCell<i32>,
        error_out: &'a RefCell<i32>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self {
            vocab,
            text,
            token_ids,
            token_count_out: Some(token_count_out),
            error_out: Some(error_out),
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }
    pub const fn with_callbacks(
        vocab: &'a dyn WpmVocabularyView,
        text: &'a str,
        token_ids: &'a RefCell<&'a mut [i32]>,
        token_count_out: Option<&'a RefCell<i32>>,
        error_out: Option<&'a RefCell<i32>>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self {
            vocab,
            text,
            token_ids,
            token_count_out,
            error_out,
            dispatch_done,
            dispatch_error,
        }
    }
}
impl core::fmt::Debug for EncodeRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeRequest")
            .field("text", &self.text)
            .field("token_capacity", &self.token_ids.borrow().len())
            .finish_non_exhaustive()
    }
}

/// Mutable result context corresponding to the source `encode_ctx`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EncodeContext {
    pub token_count: i32,
    pub error: EncodeError,
    pub tables_ready: bool,
    pub vocab_identity: usize,
    pub max_token_len: usize,
    pub unexpected: bool,
}

/// Runtime event carrying an owned request and caller-owned result context.
#[derive(Clone)]
pub struct EventEncodeRuntime<'a> {
    pub request: EncodeRequest<'a>,
    pub context: &'a RefCell<EncodeContext>,
}
impl<'a> EventEncodeRuntime<'a> {
    pub const fn new(request: EncodeRequest<'a>, context: &'a RefCell<EncodeContext>) -> Self {
        Self { request, context }
    }
}
impl core::fmt::Debug for EventEncodeRuntime<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventEncodeRuntime")
            .field("request", &self.request)
            .field("context", &self.context.borrow())
            .finish_non_exhaustive()
    }
}

sml! {
    TextEncodersWpm<'event> {
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
        "encode_input_capacity_decision"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / sync_tables,
        "encode_input_capacity_decision"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "encode_exec"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prefix_buffer_capacity_within_limit],
        "errored"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [prefix_buffer_capacity_exceeded] / reject_invalid_encode_from_encode_input_capacity_decision,
        "errored"_s <= "encode_input_capacity_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / reject_invalid_encode_from_encode_input_capacity_decision,
        "encode_result_decision"_s <= "encode_exec"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) / run_encode,
        "done"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime>(EventEncodeRuntime<'event>) [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
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
    fn begin_encode(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        reset_runtime(event);
        Ok(())
    }
    fn begin_encode_sync_vocab(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        reset_runtime(event);
        self.vocab_identity = event.request.vocab.identity();
        self.tables_ready = false;
        Ok(())
    }
    fn encode_result_backend_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::Backend)
    }
    fn encode_result_invalid_argument_error(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::InvalidArgument)
    }
    fn encode_result_model_invalid_error(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::ModelInvalid)
    }
    fn encode_result_ok(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::Ok)
    }
    fn encode_result_unclassified_error_code(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!matches!(
            event.context.borrow().error,
            EncodeError::Ok
                | EncodeError::InvalidArgument
                | EncodeError::Backend
                | EncodeError::ModelInvalid
        ))
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        ensure_error(event);
        Ok(())
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        ensure_error(event);
        Ok(())
    }
    fn ensure_last_error_from_table_policy_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        ensure_error(event);
        Ok(())
    }
    fn ensure_last_error_from_table_sync_result_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        ensure_error(event);
        Ok(())
    }
    fn invalid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_encode(event))
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        finish_done(event);
        Ok(())
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        finish_done(event);
        Ok(())
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.error = EncodeError::InvalidArgument;
        Ok(())
    }
    fn prefix_buffer_capacity_exceeded(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!prefix_capacity(event))
    }
    fn prefix_buffer_capacity_within_limit(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(prefix_capacity(event))
    }
    fn reject_invalid_encode_from_encode_input_capacity_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        reject(event);
        Ok(())
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        reject(event);
        Ok(())
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<(), ()> {
        reject(event);
        Ok(())
    }
    fn run_encode(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        encode_wpm(event);
        Ok(())
    }
    fn sync_tables(&mut self, event: &EventEncodeRuntime<'_>) -> Result<(), ()> {
        sync_tables(event);
        Ok(())
    }
    fn table_sync_backend_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::Backend)
    }
    fn table_sync_invalid_argument_error(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::InvalidArgument)
    }
    fn table_sync_model_invalid_error(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::ModelInvalid)
    }
    fn table_sync_ok(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().error == EncodeError::Ok)
    }
    fn table_sync_unclassified_error_code(
        &self,
        event: &EventEncodeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!matches!(
            event.context.borrow().error,
            EncodeError::Ok
                | EncodeError::InvalidArgument
                | EncodeError::Backend
                | EncodeError::ModelInvalid
        ))
    }
    fn tables_missing(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!tables_ready(event))
    }
    fn tables_ready(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(tables_ready(event))
    }
    fn text_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.text.is_empty())
    }
    fn text_non_empty(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.text.is_empty())
    }
    fn valid_encode(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_encode(event))
    }
    fn vocab_changed(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.vocab_identity != event.request.vocab.identity())
    }
    fn vocab_unchanged(&self, event: &EventEncodeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.vocab_identity == event.request.vocab.identity())
    }
}

fn reset_runtime(event: &EventEncodeRuntime<'_>) {
    let mut context = event.context.borrow_mut();
    context.token_count = 0;
    context.error = EncodeError::Ok;
    context.unexpected = false;
}

fn valid_encode(event: &EventEncodeRuntime<'_>) -> bool {
    !event.request.token_ids.borrow().is_empty() && event.request.text.len() <= MAX_INPUT_BYTES
}

fn prefix_capacity(event: &EventEncodeRuntime<'_>) -> bool {
    event.request.text.len() <= MAX_INPUT_BYTES.saturating_sub(3)
}

fn tables_ready(event: &EventEncodeRuntime<'_>) -> bool {
    let context = event.context.borrow();
    context.tables_ready && context.vocab_identity == event.request.vocab.identity()
}

fn reject(event: &EventEncodeRuntime<'_>) {
    let mut context = event.context.borrow_mut();
    context.token_count = 0;
    context.error = EncodeError::InvalidArgument;
}

fn ensure_error(event: &EventEncodeRuntime<'_>) {
    let mut context = event.context.borrow_mut();
    if context.error == EncodeError::Ok {
        context.error = EncodeError::Backend;
    }
}

fn publish_error(event: &EventEncodeRuntime<'_>) {
    if let Some(out) = event.request.token_count_out {
        *out.borrow_mut() = 0;
    }
    if let Some(out) = event.request.error_out {
        *out.borrow_mut() = event.context.borrow().error.code();
    }
}

fn finish_done(event: &EventEncodeRuntime<'_>) {
    let count = event.context.borrow().token_count;
    if let Some(out) = event.request.token_count_out {
        *out.borrow_mut() = count;
    }
    if let Some(out) = event.request.error_out {
        *out.borrow_mut() = EncodeError::Ok.code();
    }
}

fn sync_tables(event: &EventEncodeRuntime<'_>) {
    let declared = event.request.vocab.token_count() as usize;
    if declared > MAX_VOCAB_ENTRIES {
        let mut context = event.context.borrow_mut();
        context.error = EncodeError::ModelInvalid;
        context.tables_ready = false;
        return;
    }
    let mut max_len = 0usize;
    for index in 0..declared {
        let Some(token) = event.request.vocab.token(index as u32) else {
            let mut context = event.context.borrow_mut();
            context.error = EncodeError::ModelInvalid;
            context.tables_ready = false;
            return;
        };
        max_len = max_len.max(token.len());
    }
    let unk = event.request.vocab.unk_id();
    if unk >= 0 && (unk as usize) >= declared {
        let mut context = event.context.borrow_mut();
        context.error = EncodeError::ModelInvalid;
        context.tables_ready = false;
        return;
    }
    let mut context = event.context.borrow_mut();
    context.max_token_len = max_len;
    context.tables_ready = true;
    context.vocab_identity = event.request.vocab.identity();
    context.error = EncodeError::Ok;
}

fn lookup(vocab: &dyn WpmVocabularyView, candidate: &[u8]) -> Option<i32> {
    if candidate.is_empty() {
        return None;
    }
    let count = (vocab.token_count() as usize).min(MAX_VOCAB_ENTRIES);
    for index in 0..count {
        if vocab
            .token(index as u32)
            .is_some_and(|token| token == candidate)
        {
            return i32::try_from(index).ok();
        }
    }
    None
}

fn push_token(request: &EncodeRequest<'_>, token: i32, count: &mut i32) -> bool {
    if token < 0 {
        return true;
    }
    let mut output = request.token_ids.borrow_mut();
    let index = match usize::try_from(*count) {
        Ok(index) => index,
        Err(_) => return false,
    };
    if index >= output.len() {
        return false;
    }
    output[index] = token;
    *count += 1;
    true
}

fn encode_piece(
    request: &EncodeRequest<'_>,
    max_token_len: usize,
    piece: &[u8],
    count: &mut i32,
) -> bool {
    // Match the pinned kernel's whole-word rollback: a failed piece causes
    // the partial word output to be discarded before resolving <unk>.
    let word_token_start = *count;
    let mut cursor = 0usize;
    let mut matched_all = true;
    while cursor < piece.len() {
        let continuation = cursor != 0;
        const WORD_START_PREFIX: &[u8] = &[0xE2, 0x96, 0x81];
        let prefix: &[u8] = if continuation {
            b"##"
        } else {
            WORD_START_PREFIX
        };
        let max_piece_len = max_token_len.saturating_sub(prefix.len()).max(1);
        let max_end = piece.len().min(cursor.saturating_add(max_piece_len));
        let mut matched_end = cursor;
        let mut matched_token = -1i32;
        for end in (cursor + 1..=max_end).rev() {
            let raw = &piece[cursor..end];
            let candidate_len = prefix.len().saturating_add(raw.len());
            if candidate_len <= MAX_INPUT_BYTES {
                let mut candidate = [0u8; MAX_INPUT_BYTES];
                candidate[..prefix.len()].copy_from_slice(prefix);
                candidate[prefix.len()..candidate_len].copy_from_slice(raw);
                if let Some(token) = lookup(request.vocab, &candidate[..candidate_len]) {
                    matched_end = end;
                    matched_token = token;
                    break;
                }
            }
            if let Some(token) = lookup(request.vocab, raw) {
                matched_end = end;
                matched_token = token;
                break;
            }
        }
        if matched_end == cursor {
            matched_all = false;
            break;
        }
        if !push_token(request, matched_token, count) {
            return false;
        }
        cursor = matched_end;
    }
    if !matched_all {
        *count = word_token_start;
        let mut unk = request.vocab.unk_id();
        if unk < 0 {
            unk = lookup(request.vocab, b"<unk>").unwrap_or(-1);
        }
        // The C++ path succeeds without output when the vocabulary has no
        // unknown token; do not synthesize an id or report a spurious error.
        if unk >= 0 && !push_token(request, unk, count) {
            return false;
        }
    }
    true
}

fn is_chinese(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4dbf | 0x4e00..=0x9fff | 0x20000..=0x2a6df |
        0x2a700..=0x2b73f | 0x2b740..=0x2b81f | 0x2b820..=0x2ceaf |
        0x2ceb0..=0x2ebe0 | 0x2f00..=0x2fd5 | 0x30000..=0x3134f)
}

fn is_punctuation(c: char) -> bool {
    matches!(c as u32,
        0x21..=0x23 | 0x25..=0x2a | 0x2c..=0x2f | 0x3a..=0x3b |
        0x3f | 0x40 | 0x5b..=0x5d | 0x5f | 0x7b..=0x7e |
        0x2000..=0x206f | 0x3000..=0x303f | 0xfe10..=0xfe6f)
}

fn is_symbol(c: char) -> bool {
    c.is_ascii() && matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~')
}

fn normalize_char(c: char) -> char {
    // Match the pinned Unicode NFD table's mappings to ASCII bases.  Entries
    // whose canonical base is non-ASCII remain unchanged, as they do in the
    // source's single-codepoint preprocessing representation.
    match c as u32 {
        0x00c0..=0x00c5
        | 0x0100
        | 0x0102
        | 0x0104
        | 0x01cd
        | 0x01de
        | 0x01e0
        | 0x01fa
        | 0x0200
        | 0x0202
        | 0x0226
        | 0x1e00
        | 0x1ea0
        | 0x1ea2
        | 0x1ea4
        | 0x1ea6
        | 0x1ea8
        | 0x1eaa
        | 0x1eac
        | 0x1eae
        | 0x1eb0
        | 0x1eb2
        | 0x1eb4
        | 0x1eb6
        | 0x212b => 'A',
        0x1e02 | 0x1e04 | 0x1e06 => 'B',
        0x00c7 | 0x0106 | 0x0108 | 0x010a | 0x010c | 0x1e08 => 'C',
        0x010e | 0x1e0a | 0x1e0c | 0x1e0e | 0x1e10 | 0x1e12 => 'D',
        0x00c8..=0x00cb
        | 0x0112
        | 0x0114
        | 0x0116
        | 0x0118
        | 0x011a
        | 0x0204
        | 0x0206
        | 0x0228
        | 0x1e14
        | 0x1e16
        | 0x1e18
        | 0x1e1a
        | 0x1e1c
        | 0x1eb8
        | 0x1eba
        | 0x1ebc
        | 0x1ebe
        | 0x1ec0
        | 0x1ec2
        | 0x1ec4
        | 0x1ec6 => 'E',
        0x1e1e => 'F',
        0x011c | 0x011e | 0x0120 | 0x0122 | 0x01e6 | 0x01f4 | 0x1e20 => 'G',
        0x0124 | 0x021e | 0x1e22 | 0x1e24 | 0x1e26 | 0x1e28 | 0x1e2a => 'H',
        0x00cc..=0x00cf
        | 0x0128
        | 0x012a
        | 0x012c
        | 0x012e
        | 0x0130
        | 0x01cf
        | 0x0208
        | 0x020a
        | 0x1e2c
        | 0x1e2e
        | 0x1ec8
        | 0x1eca => 'I',
        0x0134 => 'J',
        0x0136 | 0x01e8 | 0x1e30 | 0x1e32 | 0x1e34 | 0x212a => 'K',
        0x0139 | 0x013b | 0x013d | 0x1e36 | 0x1e38 | 0x1e3a | 0x1e3c => 'L',
        0x1e3e | 0x1e40 | 0x1e42 => 'M',
        0x00d1 | 0x0143 | 0x0145 | 0x0147 | 0x01f8 | 0x1e44 | 0x1e46 | 0x1e48 | 0x1e4a => 'N',
        0x00d2..=0x00d6
        | 0x014c
        | 0x014e
        | 0x0150
        | 0x01a0
        | 0x01d1
        | 0x01ea
        | 0x01ec
        | 0x020c
        | 0x020e
        | 0x022a
        | 0x022c
        | 0x022e
        | 0x0230
        | 0x1e4c
        | 0x1e4e
        | 0x1e50
        | 0x1e52
        | 0x1ecc
        | 0x1ece
        | 0x1ed0
        | 0x1ed2
        | 0x1ed4
        | 0x1ed6
        | 0x1ed8
        | 0x1eda
        | 0x1edc
        | 0x1ede
        | 0x1ee0
        | 0x1ee2 => 'O',
        0x1e54 | 0x1e56 => 'P',
        0x0154 | 0x0156 | 0x0158 | 0x0210 | 0x0212 | 0x1e58 | 0x1e5a | 0x1e5c | 0x1e5e => 'R',
        0x015a | 0x015c | 0x015e | 0x0160 | 0x0218 | 0x1e60 | 0x1e62 | 0x1e64 | 0x1e66 | 0x1e68 => {
            'S'
        }
        0x0162 | 0x0164 | 0x021a | 0x1e6a | 0x1e6c | 0x1e6e | 0x1e70 => 'T',
        0x00d9..=0x00dc
        | 0x0168
        | 0x016a
        | 0x016c
        | 0x016e
        | 0x0170
        | 0x0172
        | 0x01af
        | 0x01d3
        | 0x01d5
        | 0x01d7
        | 0x01d9
        | 0x01db
        | 0x0214
        | 0x0216
        | 0x1e72
        | 0x1e74
        | 0x1e76
        | 0x1e78
        | 0x1e7a
        | 0x1ee4
        | 0x1ee6
        | 0x1ee8
        | 0x1eea
        | 0x1eec
        | 0x1eee
        | 0x1ef0 => 'U',
        0x1e7c | 0x1e7e => 'V',
        0x0174 | 0x1e80 | 0x1e82 | 0x1e84 | 0x1e86 | 0x1e88 => 'W',
        0x1e8a | 0x1e8c => 'X',
        0x00dd | 0x0176 | 0x0178 | 0x0232 | 0x1e8e | 0x1ef2 | 0x1ef4 | 0x1ef6 | 0x1ef8 => 'Y',
        0x0179 | 0x017b | 0x017d | 0x1e90 | 0x1e92 | 0x1e94 => 'Z',
        0x1fef => '`',
        0x00e0..=0x00e5
        | 0x0101
        | 0x0103
        | 0x0105
        | 0x01ce
        | 0x01df
        | 0x01e1
        | 0x01fb
        | 0x0201
        | 0x0203
        | 0x0227
        | 0x1e01
        | 0x1ea1
        | 0x1ea3
        | 0x1ea5
        | 0x1ea7
        | 0x1ea9
        | 0x1eab
        | 0x1ead
        | 0x1eaf
        | 0x1eb1
        | 0x1eb3
        | 0x1eb5
        | 0x1eb7 => 'a',
        0x1e03 | 0x1e05 | 0x1e07 => 'b',
        0x00e7 | 0x0107 | 0x0109 | 0x010b | 0x010d | 0x1e09 => 'c',
        0x010f | 0x1e0b | 0x1e0d | 0x1e0f | 0x1e11 | 0x1e13 => 'd',
        0x00e8..=0x00eb
        | 0x0113
        | 0x0115
        | 0x0117
        | 0x0119
        | 0x011b
        | 0x0205
        | 0x0207
        | 0x0229
        | 0x1e15
        | 0x1e17
        | 0x1e19
        | 0x1e1b
        | 0x1e1d
        | 0x1eb9
        | 0x1ebb
        | 0x1ebd
        | 0x1ebf
        | 0x1ec1
        | 0x1ec3
        | 0x1ec5
        | 0x1ec7 => 'e',
        0x1e1f => 'f',
        0x011d | 0x011f | 0x0121 | 0x0123 | 0x01e7 | 0x01f5 | 0x1e21 => 'g',
        0x0125 | 0x021f | 0x1e23 | 0x1e25 | 0x1e27 | 0x1e29 | 0x1e2b | 0x1e96 => 'h',
        0x00ec..=0x00ef
        | 0x0129
        | 0x012b
        | 0x012d
        | 0x012f
        | 0x01d0
        | 0x0209
        | 0x020b
        | 0x1e2d
        | 0x1e2f
        | 0x1ec9
        | 0x1ecb => 'i',
        0x0135 | 0x01f0 => 'j',
        0x0137 | 0x01e9 | 0x1e31 | 0x1e33 | 0x1e35 => 'k',
        0x013a | 0x013c | 0x013e | 0x1e37 | 0x1e39 | 0x1e3b | 0x1e3d => 'l',
        0x1e3f | 0x1e41 | 0x1e43 => 'm',
        0x00f1 | 0x0144 | 0x0146 | 0x0148 | 0x01f9 | 0x1e45 | 0x1e47 | 0x1e49 | 0x1e4b => 'n',
        0x00f2..=0x00f6
        | 0x014d
        | 0x014f
        | 0x0151
        | 0x01a1
        | 0x01d2
        | 0x01eb
        | 0x01ed
        | 0x020d
        | 0x020f
        | 0x022b
        | 0x022d
        | 0x022f
        | 0x0231
        | 0x1e4d
        | 0x1e4f
        | 0x1e51
        | 0x1e53
        | 0x1ecd
        | 0x1ecf
        | 0x1ed1
        | 0x1ed3
        | 0x1ed5
        | 0x1ed7
        | 0x1ed9
        | 0x1edb
        | 0x1edd
        | 0x1edf
        | 0x1ee1
        | 0x1ee3 => 'o',
        0x1e55 | 0x1e57 => 'p',
        0x0155 | 0x0157 | 0x0159 | 0x0211 | 0x0213 | 0x1e59 | 0x1e5b | 0x1e5d | 0x1e5f => 'r',
        0x015b | 0x015d | 0x015f | 0x0161 | 0x0219 | 0x1e61 | 0x1e63 | 0x1e65 | 0x1e67 | 0x1e69 => {
            's'
        }
        0x0163 | 0x0165 | 0x021b | 0x1e6b | 0x1e6d | 0x1e6f | 0x1e71 | 0x1e97 => 't',
        0x00f9..=0x00fc
        | 0x0169
        | 0x016b
        | 0x016d
        | 0x016f
        | 0x0171
        | 0x0173
        | 0x01b0
        | 0x01d4
        | 0x01d6
        | 0x01d8
        | 0x01da
        | 0x01dc
        | 0x0215
        | 0x0217
        | 0x1e73
        | 0x1e75
        | 0x1e77
        | 0x1e79
        | 0x1e7b
        | 0x1ee5
        | 0x1ee7
        | 0x1ee9
        | 0x1eeb
        | 0x1eed
        | 0x1eef
        | 0x1ef1 => 'u',
        0x1e7d | 0x1e7f => 'v',
        0x0175 | 0x1e81 | 0x1e83 | 0x1e85 | 0x1e87 | 0x1e89 | 0x1e98 => 'w',
        0x1e8b | 0x1e8d => 'x',
        0x00fd | 0x00ff | 0x0177 | 0x0233 | 0x1e8f | 0x1e99 | 0x1ef3 | 0x1ef5 | 0x1ef7 | 0x1ef9 => {
            'y'
        }
        0x017a | 0x017c | 0x017e | 0x1e91 | 0x1e93 | 0x1e95 => 'z',
        _ => c,
    }
}

fn lower_char(c: char) -> char {
    normalize_char(c).to_lowercase().next().unwrap_or(c)
}

fn is_discarded(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || c.is_control()
}

fn flush_word(
    request: &EncodeRequest<'_>,
    max_token_len: usize,
    word: &[u8],
    count: &mut i32,
) -> bool {
    word.is_empty() || encode_piece(request, max_token_len, word, count)
}

fn encode_wpm(event: &EventEncodeRuntime<'_>) {
    let mut word = [0u8; MAX_INPUT_BYTES];
    let mut word_len = 0usize;
    let max_token_len = event.context.borrow().max_token_len;
    let mut count = 0i32;
    for character in event.request.text.chars() {
        // Invalid UTF-8 is represented by U+FFFD by the pinned decoder, and
        // controls/NUL are discarded during WPM preprocessing.
        if is_discarded(character) {
            continue;
        }
        let whitespace = character.is_whitespace();
        let split = is_punctuation(character) || is_symbol(character) || is_chinese(character);
        if whitespace || split {
            if !flush_word(&event.request, max_token_len, &word[..word_len], &mut count) {
                event.context.borrow_mut().error = EncodeError::InvalidArgument;
                return;
            }
            word_len = 0;
            if !whitespace {
                let mut bytes = [0u8; 4];
                let encoded = lower_char(character).encode_utf8(&mut bytes).as_bytes();
                if !flush_word(&event.request, max_token_len, encoded, &mut count) {
                    event.context.borrow_mut().error = EncodeError::InvalidArgument;
                    return;
                }
            }
            continue;
        }
        let mut bytes = [0u8; 4];
        let encoded = lower_char(character).encode_utf8(&mut bytes).as_bytes();
        if word_len + encoded.len() > word.len() {
            event.context.borrow_mut().error = EncodeError::InvalidArgument;
            return;
        }
        word[word_len..word_len + encoded.len()].copy_from_slice(encoded);
        word_len += encoded.len();
    }
    if !flush_word(&event.request, max_token_len, &word[..word_len], &mut count) {
        event.context.borrow_mut().error = EncodeError::InvalidArgument;
        return;
    }
    let mut context = event.context.borrow_mut();
    context.token_count = count;
    context.error = EncodeError::Ok;
}
/// Synchronous bounded owner of the generated WPM machine.
pub struct TextEncodersWpmActor {
    machine: TextEncodersWpmStateMachine<EncodeContext>,
    result: RefCell<EncodeContext>,
}
impl Default for TextEncodersWpmActor {
    fn default() -> Self {
        Self::new()
    }
}
impl TextEncodersWpmActor {
    /// Creates an actor in the generated `initialized` state.
    pub fn new() -> Self {
        Self {
            machine: TextEncodersWpmStateMachine::new(EncodeContext::default()),
            result: RefCell::new(EncodeContext::default()),
        }
    }
    /// Processes one request synchronously and returns whether it succeeded.
    pub fn process_event<'event>(&'event mut self, request: EncodeRequest<'event>) -> bool {
        *self.result.borrow_mut() = EncodeContext::default();
        let runtime = EventEncodeRuntime::new(request, &self.result);
        let accepted = self
            .machine
            .process_event(TextEncodersWpmEvents::EventEncodeRuntime(runtime))
            .is_ok();
        let mut result = *self.result.borrow();
        if !accepted && result.error == EncodeError::Ok {
            result.error = EncodeError::Backend;
        }
        if result.unexpected || self.machine.is(&TextEncodersWpmStates::Unexpected) {
            result.token_count = 0;
            result.error = EncodeError::InvalidArgument;
            self.result.borrow_mut().error = result.error;
        }
        let success = accepted
            && self.machine.is(&TextEncodersWpmStates::Done)
            && result.error == EncodeError::Ok;
        if success {
            if let Some(out) = request.token_count_out {
                *out.borrow_mut() = result.token_count;
            }
            if let Some(out) = request.error_out {
                *out.borrow_mut() = EncodeError::Ok.code();
            }
            if let Some(callback) = request.dispatch_done {
                let _ = callback(EventsEncodingDone {
                    token_count: result.token_count,
                });
            }
        } else {
            if let Some(out) = request.token_count_out {
                *out.borrow_mut() = 0;
            }
            if let Some(out) = request.error_out {
                *out.borrow_mut() = result.error.code();
            }
            if let Some(callback) = request.dispatch_error {
                let _ = callback(EventsEncodingError {
                    error: result.error,
                });
            }
        }
        success
    }
    /// Returns the last source-compatible phase error.
    #[must_use]
    pub fn last_error(&self) -> EncodeError {
        self.machine.context().error
    }
    /// Moves the actor into the explicit unexpected state and records invalid input.
    pub fn process_unexpected(&mut self) -> bool {
        self.machine.set_state(TextEncodersWpmStates::Unexpected);
        let context = self.machine.context_mut();
        context.unexpected = true;
        context.token_count = 0;
        context.error = EncodeError::InvalidArgument;
        false
    }
    /// Returns generated state inspection data.
    pub fn is(&self, state: &TextEncodersWpmStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the actor's persistent context.
    pub fn context(&self) -> &EncodeContext {
        self.machine.context()
    }
}
/// Short alias matching the pinned C++ `Wpm` name.
pub type Wpm = TextEncodersWpmActor;
