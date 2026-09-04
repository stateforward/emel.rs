//! Bounded UGM encoder state machine.
//!
//! The actor keeps request storage caller-owned, copies only bounded input and
//! vocabulary records, and executes normalization, dynamic programming,
//! backtrace, and emission synchronously through the SML state graph.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::unused_self,
    clippy::derivable_impls,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::manual_range_contains,
    clippy::range_plus_one,
    clippy::needless_range_loop,
    dead_code,
    missing_docs
)]

use core::cell::{Cell, RefCell};
use sml::sml;

/// Maximum normalized input accepted by one request.
pub const MAX_ENCODE_BYTES: usize = 4096;
/// Maximum vocabulary records retained by the bounded actor.
pub const MAX_VOCAB_TOKENS: usize = 4096;
/// Maximum bytes retained for one vocabulary token.
pub const MAX_TOKEN_BYTES: usize = 128;
/// Maximum output IDs retained by one request.
pub const MAX_TOKEN_IDS: usize = 4096;

/// UGM vocabulary token record.
#[derive(Clone, Copy, Debug)]
pub struct UgmToken {
    pub bytes: [u8; MAX_TOKEN_BYTES],
    pub len: u16,
    pub score: f32,
    pub kind: u8,
}

impl Default for UgmToken {
    fn default() -> Self {
        Self {
            bytes: [0; MAX_TOKEN_BYTES],
            len: 0,
            score: 0.0,
            kind: 0,
        }
    }
}

impl UgmToken {
    /// Creates a bounded token record. Oversized token text is rejected by returning `None`.
    pub fn new(bytes: &[u8], score: f32, kind: u8) -> Option<Self> {
        if bytes.len() > MAX_TOKEN_BYTES {
            return None;
        }
        let mut token = Self {
            score,
            kind,
            ..Self::default()
        };
        token.bytes[..bytes.len()].copy_from_slice(bytes);
        token.len = bytes.len() as u16;
        Some(token)
    }
    fn text(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

/// Caller-supplied bounded vocabulary for UGM encoding.
#[derive(Clone, Debug)]
pub struct UgmVocabulary {
    pub tokens: [UgmToken; MAX_VOCAB_TOKENS],
    pub token_count: usize,
    pub unk_id: i32,
    pub add_space_prefix: bool,
    pub treat_whitespace_as_suffix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    /// Changes whenever the caller changes the vocabulary contents.
    pub identity: u64,
}

impl Default for UgmVocabulary {
    fn default() -> Self {
        Self {
            tokens: [UgmToken::default(); MAX_VOCAB_TOKENS],
            token_count: 0,
            unk_id: -1,
            add_space_prefix: false,
            treat_whitespace_as_suffix: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: false,
            identity: 0,
        }
    }
}

/// Request passed to the synchronous actor.
pub struct UgmEncodeRequest<'a> {
    pub vocabulary: &'a UgmVocabulary,
    pub text: &'a [u8],
    pub token_ids: &'a mut [i32],
    pub preprocessed: bool,
    pub on_done: Option<fn(EncodingDone) -> bool>,
    pub on_error: Option<fn(EncodingError) -> bool>,
}

/// Successful bounded encoding callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingDone {
    pub token_count: usize,
}
/// Failed bounded encoding callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodingError {
    pub error: UgmError,
}

/// Stable encoder error classes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum UgmError {
    #[default]
    None = 0,
    InvalidArgument = 1,
    Backend = 2,
    ModelInvalid = 3,
    Unexpected = 4,
}

impl UgmError {
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Runtime event consumed by the generated state machine.
#[derive(Clone, Debug)]
pub struct RuntimeEncodeRuntime {
    text: [u8; MAX_ENCODE_BYTES],
    text_len: usize,
    token_ids: [i32; MAX_TOKEN_IDS],
    token_capacity: usize,
    preprocessed: bool,
    vocab_identity: u64,
    pub on_done: Option<fn(EncodingDone) -> bool>,
    pub on_error: Option<fn(EncodingError) -> bool>,
    err: Cell<UgmError>,
    token_count: Cell<usize>,
    normalized: RefCell<[u8; MAX_ENCODE_BYTES]>,
    normalized_len: Cell<usize>,
    unk_id: Cell<i32>,
    traced_count: Cell<usize>,
    backtrace_failed: Cell<bool>,
    emit_failed: Cell<bool>,
}

impl RuntimeEncodeRuntime {
    fn from_request(request: &UgmEncodeRequest<'_>) -> Result<Self, UgmError> {
        if request.text.len() > MAX_ENCODE_BYTES || request.token_ids.len() > MAX_TOKEN_IDS {
            return Err(UgmError::InvalidArgument);
        }
        let mut event = Self {
            text: [0; MAX_ENCODE_BYTES],
            text_len: request.text.len(),
            token_ids: [0; MAX_TOKEN_IDS],
            token_capacity: request.token_ids.len(),
            preprocessed: request.preprocessed,
            vocab_identity: request.vocabulary.identity,
            on_done: request.on_done,
            on_error: request.on_error,
            err: Cell::new(UgmError::None),
            token_count: Cell::new(0),
            normalized: RefCell::new([0; MAX_ENCODE_BYTES]),
            normalized_len: Cell::new(0),
            unk_id: Cell::new(-1),
            traced_count: Cell::new(0),
            backtrace_failed: Cell::new(false),
            emit_failed: Cell::new(false),
        };
        event.text[..request.text.len()].copy_from_slice(request.text);
        Ok(event)
    }
    fn text(&self) -> &[u8] {
        &self.text[..self.text_len]
    }
    fn normalized(&self) -> core::cell::Ref<'_, [u8; MAX_ENCODE_BYTES]> {
        self.normalized.borrow()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BestTokenization {
    token_id: i32,
    input_offset: usize,
    score_sum: f64,
}

#[derive(Clone, Copy, Debug)]
struct TokenBuffer {
    ids: [i32; MAX_TOKEN_IDS],
    count: usize,
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self {
            ids: [0; MAX_TOKEN_IDS],
            count: 0,
        }
    }
}

#[derive(Debug)]
pub struct TextEncodersUgmContext {
    tokens: [UgmToken; MAX_VOCAB_TOKENS],
    token_count: usize,
    output_count: usize,
    vocab_identity: u64,
    tables_ready: bool,
    unk_id: i32,
    add_space_prefix: bool,
    treat_whitespace_as_suffix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    unknown_token_score: f64,
    best: [BestTokenization; MAX_ENCODE_BYTES + 1],
    token_buffer: TokenBuffer,
    error: UgmError,
    output: [i32; MAX_TOKEN_IDS],
}

impl Default for TextEncodersUgmContext {
    fn default() -> Self {
        Self {
            tokens: [UgmToken::default(); MAX_VOCAB_TOKENS],
            token_count: 0,
            output_count: 0,
            vocab_identity: 0,
            tables_ready: false,
            unk_id: -1,
            add_space_prefix: false,
            treat_whitespace_as_suffix: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: false,
            unknown_token_score: -10.0,
            best: [BestTokenization::default(); MAX_ENCODE_BYTES + 1],
            token_buffer: TokenBuffer::default(),
            error: UgmError::None,
            output: [0; MAX_TOKEN_IDS],
        }
    }
}

impl TextEncodersUgmContext {
    fn sync_vocabulary(&mut self, vocabulary: &UgmVocabulary) {
        self.tokens = vocabulary.tokens;
        self.token_count = vocabulary.token_count;
        self.output_count = 0;
        self.vocab_identity = vocabulary.identity;
        self.unk_id = vocabulary.unk_id;
        self.add_space_prefix = vocabulary.add_space_prefix;
        self.treat_whitespace_as_suffix = vocabulary.treat_whitespace_as_suffix;
        self.remove_extra_whitespaces = vocabulary.remove_extra_whitespaces;
        self.escape_whitespaces = vocabulary.escape_whitespaces;
        let min_score = self
            .tokens
            .iter()
            .take(self.token_count)
            .filter(|token| token.kind == 1)
            .map(|token| token.score)
            .reduce(f32::min)
            .unwrap_or(0.0);
        self.unknown_token_score = f64::from(min_score) - 10.0;
        self.tables_ready = true;
    }
    fn user_defined_prefix_len(&self, input: &[u8], offset: usize) -> usize {
        self.tokens
            .iter()
            .take(self.token_count)
            .filter(|token| token.kind == 4)
            .map(UgmToken::text)
            .filter(|piece| !piece.is_empty() && input[offset..].starts_with(piece))
            .map(<[u8]>::len)
            .max()
            .unwrap_or(0)
    }
    fn reset_runtime(&mut self) {
        self.output_count = 0;
        self.error = UgmError::None;
        self.output.fill(0);
        self.token_buffer.count = 0;
    }
    fn set_error(&mut self, event: &RuntimeEncodeRuntime, error: UgmError) {
        event.err.set(error);
        self.error = error;
        self.output_count = 0;
    }
    fn phase_ok(&self, event: &RuntimeEncodeRuntime) -> bool {
        event.err.get() == UgmError::None
    }
}

sml! {
    TextEncodersUgm {
        "encode_validity_decision"_s <= *"initialized"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "done"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "errored"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "unexpected"_s + event<RuntimeEncodeRuntime>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime> / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> [text_empty] / mark_done_from_encode_precheck_decision,
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> [tables_missing],
        "unk_resolution_decision"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime> / sync_tables,
        "unk_resolution_decision"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_table_sync_result_decision,
        "normalize_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unk_present] / resolve_vocab_unk,
        "unk_lookup_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unk_missing],
        "normalize_exec"_s <= "unk_lookup_exec"_s + completion<RuntimeEncodeRuntime> / lookup_unk_id,
        "normalize_result_decision"_s <= "normalize_exec"_s + completion<RuntimeEncodeRuntime> / normalize_input,
        "input_prepare_exec"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_ok],
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_invalid_argument_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_backend_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_model_invalid_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_unclassified_error_code] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_normalize_result_decision,
        "input_prepare_result_decision"_s <= "input_prepare_exec"_s + completion<RuntimeEncodeRuntime> / prepare_dp_input,
        "dp_forward_exec"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_non_empty_ok],
        "done"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_empty_ok] / mark_done_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_invalid_argument_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_backend_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_model_invalid_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_unclassified_error_code] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_input_prepare_result_decision,
        "dp_forward_result_decision"_s <= "dp_forward_exec"_s + completion<RuntimeEncodeRuntime> / run_dp_forward,
        "dp_backtrace_exec"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_ok],
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_invalid_argument_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_backend_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_model_invalid_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_unclassified_error_code] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_dp_forward_result_decision,
        "dp_backtrace_result_decision"_s <= "dp_backtrace_exec"_s + completion<RuntimeEncodeRuntime> / run_dp_backtrace,
        "emit_exec"_s <= "dp_backtrace_result_decision"_s + completion<RuntimeEncodeRuntime> [backtrace_ok],
        "errored"_s <= "dp_backtrace_result_decision"_s + completion<RuntimeEncodeRuntime> [backtrace_failed] / mark_backtrace_failed,
        "encode_result_decision"_s <= "emit_exec"_s + completion<RuntimeEncodeRuntime> / emit_tokens,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_failed] / mark_emit_failed,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_from_encode_validity_decision,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_from_encode_vocab_sync_decision,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_from_encode_precheck_decision,
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_from_table_policy_decision,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_from_table_sync_exec,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_from_table_sync_result_decision,
        "unexpected"_s <= "unk_resolution_decision"_s + unexpected_event<_> / on_unexpected_from_unk_resolution_decision,
        "unexpected"_s <= "unk_lookup_exec"_s + unexpected_event<_> / on_unexpected_from_unk_lookup_exec,
        "unexpected"_s <= "normalize_exec"_s + unexpected_event<_> / on_unexpected_from_normalize_exec,
        "unexpected"_s <= "normalize_result_decision"_s + unexpected_event<_> / on_unexpected_from_normalize_result_decision,
        "unexpected"_s <= "input_prepare_exec"_s + unexpected_event<_> / on_unexpected_from_input_prepare_exec,
        "unexpected"_s <= "input_prepare_result_decision"_s + unexpected_event<_> / on_unexpected_from_input_prepare_result_decision,
        "unexpected"_s <= "dp_forward_exec"_s + unexpected_event<_> / on_unexpected_from_dp_forward_exec,
        "unexpected"_s <= "dp_forward_result_decision"_s + unexpected_event<_> / on_unexpected_from_dp_forward_result_decision,
        "unexpected"_s <= "dp_backtrace_exec"_s + unexpected_event<_> / on_unexpected_from_dp_backtrace_exec,
        "unexpected"_s <= "dp_backtrace_result_decision"_s + unexpected_event<_> / on_unexpected_from_dp_backtrace_result_decision,
        "unexpected"_s <= "emit_exec"_s + unexpected_event<_> / on_unexpected_from_emit_exec,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_from_encode_result_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

impl TextEncodersUgmStateMachineContext for TextEncodersUgmContext {
    fn valid_encode(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.token_capacity > 0
            && event.token_capacity <= MAX_TOKEN_IDS
            && event.text().len() <= MAX_ENCODE_BYTES)
    }
    fn vocab_changed(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!self.tables_ready || self.vocab_identity != event.vocab_identity)
    }
    fn vocab_unchanged(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.tables_ready && self.vocab_identity == event.vocab_identity)
    }
    fn text_empty(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.text().is_empty())
    }
    fn text_non_empty(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!event.text().is_empty())
    }
    fn tables_ready(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.tables_ready && self.vocab_identity == event.vocab_identity)
    }
    fn tables_missing(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!self.tables_ready(event)?)
    }
    fn table_sync_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::None)
    }
    fn vocab_unk_present(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.unk_id >= 0)
    }
    fn vocab_unk_missing(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.unk_id < 0)
    }
    fn normalize_result_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.phase_ok(event))
    }
    fn input_prepare_result_non_empty_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.phase_ok(event) && event.normalized_len.get() > 0)
    }
    fn input_prepare_result_empty_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.phase_ok(event) && event.normalized_len.get() == 0)
    }
    fn dp_forward_result_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(self.phase_ok(event))
    }
    fn table_sync_invalid_argument_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::InvalidArgument)
    }
    fn table_sync_backend_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::Backend)
    }
    fn table_sync_model_invalid_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::ModelInvalid)
    }
    fn table_sync_unclassified_error_code(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!matches!(
            event.err.get(),
            UgmError::None | UgmError::InvalidArgument | UgmError::Backend | UgmError::ModelInvalid
        ))
    }
    fn normalize_result_invalid_argument_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::InvalidArgument)
    }
    fn normalize_result_backend_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::Backend)
    }
    fn normalize_result_model_invalid_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::ModelInvalid)
    }
    fn normalize_result_unclassified_error_code(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(!matches!(
            event.err.get(),
            UgmError::None | UgmError::InvalidArgument | UgmError::Backend | UgmError::ModelInvalid
        ))
    }
    fn input_prepare_result_invalid_argument_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::InvalidArgument)
    }
    fn input_prepare_result_backend_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::Backend)
    }
    fn input_prepare_result_model_invalid_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::ModelInvalid)
    }
    fn input_prepare_result_unclassified_error_code(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(!matches!(
            event.err.get(),
            UgmError::None | UgmError::InvalidArgument | UgmError::Backend | UgmError::ModelInvalid
        ))
    }
    fn dp_forward_result_invalid_argument_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::InvalidArgument)
    }
    fn dp_forward_result_backend_error(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::Backend)
    }
    fn dp_forward_result_model_invalid_error(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(event.err.get() == UgmError::ModelInvalid)
    }
    fn dp_forward_result_unclassified_error_code(
        &self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        Ok(!matches!(
            event.err.get(),
            UgmError::None | UgmError::InvalidArgument | UgmError::Backend | UgmError::ModelInvalid
        ))
    }
    fn ensure_last_error_from_table_sync_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.ensure_last_error_from_encode_precheck_decision(event)
    }
    fn ensure_last_error_from_normalize_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.ensure_last_error_from_encode_precheck_decision(event)
    }
    fn ensure_last_error_from_input_prepare_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.ensure_last_error_from_encode_precheck_decision(event)
    }
    fn ensure_last_error_from_dp_forward_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.ensure_last_error_from_encode_precheck_decision(event)
    }
    fn backtrace_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!event.backtrace_failed.get())
    }
    fn backtrace_failed(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.backtrace_failed.get())
    }
    fn emit_ok(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(!event.emit_failed.get())
    }
    fn emit_failed(&self, event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        Ok(event.emit_failed.get())
    }

    fn begin_encode(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        self.reset_runtime();
        event.err.set(UgmError::None);
        event.token_count.set(0);
        Ok(())
    }
    fn begin_encode_sync_vocab(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        self.begin_encode(event)
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.set_error(event, UgmError::InvalidArgument);
        Ok(())
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.set_error(event, UgmError::InvalidArgument);
        Ok(())
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.error = UgmError::None;
        event.err.set(UgmError::None);
        Ok(())
    }
    fn mark_done_from_input_prepare_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.error = UgmError::None;
        event.err.set(UgmError::None);
        Ok(())
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.error = UgmError::None;
        event.err.set(UgmError::None);
        Ok(())
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        if self.phase_ok(event) {
            self.set_error(event, UgmError::Backend);
        }
        Ok(())
    }
    fn ensure_last_error_from_table_policy_decision(
        &mut self,
        event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        self.ensure_last_error_from_encode_precheck_decision(event)
    }
    fn sync_tables(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        if self.token_count > MAX_VOCAB_TOKENS {
            self.tables_ready = false;
            event.err.set(UgmError::ModelInvalid);
            return Ok(());
        }
        self.tables_ready = true;
        self.vocab_identity = event.vocab_identity;
        event.err.set(UgmError::None);
        Ok(())
    }
    fn lookup_unk_id(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let resolved = (0..self.token_count)
            .find(|&id| self.tokens[id].text() == b"<unk>")
            .and_then(|id| i32::try_from(id).ok())
            .unwrap_or(-1);
        event.unk_id.set(resolved);
        Ok(())
    }
    fn resolve_vocab_unk(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        event.unk_id.set(self.unk_id);
        Ok(())
    }
    fn prepare_dp_input(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let input_len = event.normalized_len.get();
        if input_len >= self.best.len() {
            self.set_error(event, UgmError::InvalidArgument);
            return Ok(());
        }
        for item in self.best.iter_mut().take(input_len + 1) {
            *item = BestTokenization {
                token_id: event.unk_id.get(),
                input_offset: 0,
                score_sum: f64::NEG_INFINITY,
            };
        }
        self.best[0] = BestTokenization {
            token_id: event.unk_id.get(),
            input_offset: 0,
            score_sum: 0.0,
        };
        event.traced_count.set(0);
        Ok(())
    }
    fn mark_backtrace_failed(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        self.set_error(event, UgmError::InvalidArgument);
        Ok(())
    }
    fn normalize_input(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let mut normalized = event.normalized.borrow_mut();
        let mut out = 0usize;
        let mut processing_non_ws = false;
        let mut is_space_prepended = false;
        let space = if self.escape_whitespaces {
            &[0xE2, 0x96, 0x81][..]
        } else {
            b" "
        };
        let shall_prepend_space = self.add_space_prefix && !self.treat_whitespace_as_suffix;
        let shall_append_space = self.add_space_prefix && self.treat_whitespace_as_suffix;
        let shall_merge_spaces = self.remove_extra_whitespaces;
        let input = event.text();
        let mut input_offset = 0usize;
        while input_offset < input.len() {
            let user_defined_len = self.user_defined_prefix_len(input, input_offset);
            let segment_len = if user_defined_len > 0 {
                user_defined_len
            } else {
                let first = input[input_offset];
                let width = utf8_width(first);
                if (first & 0xC0) == 0x80 || width > input.len() - input_offset {
                    1
                } else {
                    width
                }
            };
            let segment_end = input_offset + segment_len;
            let user_defined = user_defined_len > 0;
            let valid_segment =
                user_defined || core::str::from_utf8(&input[input_offset..segment_end]).is_ok();
            let mut segment_offset = input_offset;
            while segment_offset < segment_end {
                let byte = input[segment_offset];
                let is_space = byte == b' ';
                if is_space {
                    processing_non_ws = false;
                    if !shall_merge_spaces {
                        if out + space.len() > MAX_ENCODE_BYTES {
                            self.set_error(event, UgmError::InvalidArgument);
                            break;
                        }
                        normalized[out..out + space.len()].copy_from_slice(space);
                        out += space.len();
                    }
                    segment_offset += 1;
                    continue;
                }
                let (source, source_len) = if valid_segment {
                    (&input[segment_offset..segment_offset + 1], 1)
                } else {
                    (&[0xEF, 0xBF, 0xBD][..], 3)
                };
                let begin_non_ws = !processing_non_ws;
                processing_non_ws = true;
                let emit_prefix = begin_non_ws
                    && ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces);
                if emit_prefix {
                    if out + space.len() > MAX_ENCODE_BYTES {
                        self.set_error(event, UgmError::InvalidArgument);
                        break;
                    }
                    normalized[out..out + space.len()].copy_from_slice(space);
                    out += space.len();
                    is_space_prepended = true;
                }
                if out + source_len > MAX_ENCODE_BYTES {
                    self.set_error(event, UgmError::InvalidArgument);
                    break;
                }
                normalized[out..out + source_len].copy_from_slice(source);
                out += source_len;
                segment_offset += if valid_segment { 1 } else { segment_len };
            }
            if event.err.get() != UgmError::None {
                break;
            }
            input_offset = segment_end;
        }
        if shall_append_space {
            if out + space.len() > MAX_ENCODE_BYTES {
                self.set_error(event, UgmError::InvalidArgument);
            } else {
                normalized[out..out + space.len()].copy_from_slice(space);
                out += space.len();
            }
        }
        event.normalized_len.set(out);
        Ok(())
    }

    fn run_dp_forward(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let normalized = event.normalized();
        let input = &normalized[..event.normalized_len.get()];
        let mut offset = 0usize;
        while offset < input.len() {
            if self.best[offset].score_sum.is_finite() {
                let mut single_codepoint_token_found = false;
                for id in 0..self.token_count {
                    let token = self.tokens[id];
                    let text = token.text();
                    let usable_kind = matches!(token.kind, 1 | 4 | 5);
                    if text.is_empty()
                        || !usable_kind
                        || offset + text.len() > input.len()
                        || &input[offset..offset + text.len()] != text
                    {
                        continue;
                    }
                    let end = offset + text.len();
                    let width = utf8_width(input[offset]).min(input.len() - offset);
                    let score = self.best[offset].score_sum
                        + if token.kind == 4 {
                            0.0
                        } else {
                            f64::from(token.score)
                        };
                    if end - offset == width {
                        single_codepoint_token_found = true;
                    }
                    if let Ok(token_id) = i32::try_from(id) {
                        self.best[end] = BestTokenization {
                            token_id,
                            input_offset: offset,
                            score_sum: score,
                        };
                    }
                }
                let width = utf8_width(input[offset]).min(input.len() - offset);
                let end = offset + width;
                let score = self.best[offset].score_sum + self.unknown_token_score;
                if !single_codepoint_token_found
                    && event.unk_id.get() >= 0
                    && score > self.best[end].score_sum
                {
                    self.best[end] = BestTokenization {
                        token_id: event.unk_id.get(),
                        input_offset: offset,
                        score_sum: score,
                    };
                }
            }
            offset += utf8_width(input[offset]).min(input.len() - offset);
        }
        Ok(())
    }
    fn run_dp_backtrace(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let len = event.normalized_len.get();
        let mut cursor = len;
        let mut count = 0usize;
        let mut previous_unknown = false;
        while cursor != 0 {
            if cursor > len || count >= MAX_TOKEN_IDS {
                event.backtrace_failed.set(true);
                break;
            }
            let item = self.best[cursor];
            if item.input_offset >= cursor || item.token_id < 0 {
                event.backtrace_failed.set(true);
                break;
            }
            let is_unknown = item.token_id == event.unk_id.get();
            if !(previous_unknown && is_unknown) {
                self.token_buffer.ids[count] = item.token_id;
                count += 1;
            }
            previous_unknown = is_unknown;
            cursor = item.input_offset;
        }
        self.token_buffer.count = count;
        event.traced_count.set(count);
        Ok(())
    }
    fn emit_tokens(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        let count = event.traced_count.get();
        if count > event.token_capacity || count > MAX_TOKEN_IDS {
            event.emit_failed.set(true);
            event.token_count.set(0);
            self.set_error(event, UgmError::InvalidArgument);
            return Ok(());
        }
        let mut emitted = 0usize;
        for index in 0..count {
            let token = self.token_buffer.ids[count - index - 1];
            if token < 0 {
                continue;
            }
            if emitted >= event.token_capacity {
                event.emit_failed.set(true);
                event.token_count.set(0);
                self.set_error(event, UgmError::InvalidArgument);
                return Ok(());
            }
            self.output[emitted] = token;
            emitted += 1;
        }
        self.output_count = emitted;
        event.token_count.set(emitted);
        Ok(())
    }
    fn mark_emit_failed(&mut self, event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        if event.err.get() == UgmError::None {
            self.set_error(event, UgmError::InvalidArgument);
        }
        event.emit_failed.set(true);
        Ok(())
    }

    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_encode_validity_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_encode_vocab_sync_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_encode_precheck_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_table_policy_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_table_sync_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_table_sync_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_unk_resolution_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_unk_lookup_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_normalize_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_normalize_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_input_prepare_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_input_prepare_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_dp_forward_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_dp_forward_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_dp_backtrace_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_dp_backtrace_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_emit_exec(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_encode_result_decision(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.error = UgmError::Unexpected;
        Ok(())
    }
}

fn utf8_width(byte: u8) -> usize {
    // The pinned C++ lookup indexes the high nibble: 0xc/0xd => 2,
    // 0xe => 3, and every 0xf lead byte => 4 (even malformed UTF-8).
    match byte {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xFF => 4,
        _ => 1,
    }
}

/// Synchronous single-writer actor around the generated UGM machine.
pub struct TextEncodersUgmActor {
    machine: TextEncodersUgmStateMachine<TextEncodersUgmContext>,
}

impl Default for TextEncodersUgmActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextEncodersUgmActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextEncodersUgmStateMachine::new(TextEncodersUgmContext::default()),
        }
    }

    /// Processes one bounded request to completion and copies IDs to caller storage.
    pub fn process_event(&mut self, request: UgmEncodeRequest<'_>) -> Result<usize, UgmError> {
        if request.token_ids.is_empty()
            || request.token_ids.len() > MAX_TOKEN_IDS
            || request.text.len() > MAX_ENCODE_BYTES
        {
            if let Some(callback) = request.on_error {
                let _ = callback(EncodingError {
                    error: UgmError::InvalidArgument,
                });
            }
            return Err(UgmError::InvalidArgument);
        }
        if !self.machine.context().tables_ready
            || self.machine.context().vocab_identity != request.vocabulary.identity
        {
            self.machine
                .context_mut()
                .sync_vocabulary(request.vocabulary);
        }
        let event = match RuntimeEncodeRuntime::from_request(&request) {
            Ok(event) => event,
            Err(error) => {
                if let Some(callback) = request.on_error {
                    let _ = callback(EncodingError { error });
                }
                return Err(error);
            }
        };
        let accepted = self
            .machine
            .process_event(TextEncodersUgmEvents::RuntimeEncodeRuntime(event));
        let error = if accepted.is_err() {
            UgmError::Unexpected
        } else {
            self.machine.context().error
        };
        let count = if error == UgmError::None {
            self.machine
                .context()
                .output_count
                .min(request.token_ids.len())
        } else {
            0
        };
        if error == UgmError::None && count == self.machine.context().output_count {
            request.token_ids[..count].copy_from_slice(&self.machine.context().output[..count]);
            if let Some(callback) = request.on_done {
                let _ = callback(EncodingDone { token_count: count });
            }
            Ok(count)
        } else {
            if let Some(callback) = request.on_error {
                let _ = callback(EncodingError {
                    error: if error == UgmError::None {
                        UgmError::InvalidArgument
                    } else {
                        error
                    },
                });
            }
            Err(if error == UgmError::None {
                UgmError::InvalidArgument
            } else {
                error
            })
        }
    }
    #[must_use]
    pub fn state(&self) -> &TextEncodersUgmStates {
        self.machine.state()
    }
    /// Dispatches an explicit sequencing violation into the terminal route.
    pub fn process_unexpected(&mut self) -> Result<usize, UgmError> {
        self.machine.set_state(TextEncodersUgmStates::Unexpected);
        self.machine.context_mut().error = UgmError::Unexpected;
        self.machine.context_mut().output_count = 0;
        Err(UgmError::Unexpected)
    }
    #[must_use]
    pub fn last_error(&self) -> UgmError {
        self.machine.context().error
    }
    #[must_use]
    pub fn context(&self) -> &TextEncodersUgmContext {
        self.machine.context()
    }
}

/// Source-compatible actor alias.
pub type Ugm = TextEncodersUgmActor;
