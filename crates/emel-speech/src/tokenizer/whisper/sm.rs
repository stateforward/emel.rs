//! Source-aligned Whisper tokenizer state machine.
//!
//! The actor is single-writer and run-to-completion: requests and transcript
//! storage remain caller-owned, while the generated machine owns only its
//! error state.  The tokenizer JSON decoder is deliberately bounded and
//! allocation-free, matching the pinned C++ component contract.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    dead_code,
    missing_docs
)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Fixed control-token IDs for the supported Whisper tiny vocabulary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControlTokens {
    pub eot: i32,
    pub sot: i32,
    pub language_en: i32,
    pub translate: i32,
    pub transcribe: i32,
    pub no_speech: i32,
    pub notimestamps: i32,
    pub timestamp_begin: i32,
    pub space: i32,
}

/// Supported language role for the Whisper tiny policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LanguageRole { English }
/// Supported task role for the Whisper tiny policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskRole { Transcribe }
/// Supported timestamp mode for the Whisper tiny policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestampMode { TimestampTokens }

/// Bound ASR decode policy validated by the tokenizer owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsrDecodePolicy {
    pub tokens: ControlTokens,
    pub language: LanguageRole,
    pub task: TaskRole,
    pub timestamps: TimestampMode,
    pub suppress_translate: bool,
    pub prompt_tokens: [i32; 3],
}

/// The canonical supported Whisper tiny control tokens.
pub const TINY_CONTROL_TOKENS: ControlTokens = ControlTokens {
    eot: 50_257, sot: 50_258, language_en: 50_259, translate: 50_358,
    transcribe: 50_359, no_speech: 50_362, notimestamps: 50_363,
    timestamp_begin: 50_364, space: 220,
};
/// The canonical supported Whisper tiny decode policy.
pub const TINY_ASR_DECODE_POLICY: AsrDecodePolicy = AsrDecodePolicy {
    tokens: TINY_CONTROL_TOKENS,
    language: LanguageRole::English,
    task: TaskRole::Transcribe,
    timestamps: TimestampMode::TimestampTokens,
    suppress_translate: true,
    prompt_tokens: [50_258, 50_259, 50_359],
};

/// Errors published by the Whisper tokenizer boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenizerError {
    #[default]
    None = 0,
    TokenizerJsonInvalid = 1,
    InternalError = 2,
    TokenIdsInvalid = 3,
    DecodePolicyUnsupported = 4,
    UnexpectedEvent = 5,
}
impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "no error",
            Self::TokenizerJsonInvalid => "tokenizer JSON is invalid",
            Self::InternalError => "internal tokenizer error",
            Self::TokenIdsInvalid => "token IDs are invalid",
            Self::DecodePolicyUnsupported => "decode policy is unsupported",
            Self::UnexpectedEvent => "unexpected tokenizer event",
        })
    }
}

/// Successful detokenization callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DetokenizeDone<'a> { pub request: &'a Detokenize<'a>, pub transcript_size: i32 }
/// Failed detokenization callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DetokenizeError<'a> { pub request: &'a Detokenize<'a>, pub error: TokenizerError }

/// Caller-owned detokenization request.
#[derive(Debug)]
pub struct Detokenize<'a> {
    pub tokenizer_json: &'a str,
    pub token_ids: &'a [i32],
    pub transcript: &'a mut [u8],
    pub transcript_size_out: &'a mut i32,
    pub error_out: Option<&'a Cell<TokenizerError>>,
    pub on_done: Option<for<'r> fn(DetokenizeDone<'r>)>,
    pub on_error: Option<for<'r> fn(DetokenizeError<'r>)>,
}
impl<'a> Detokenize<'a> {
    #[must_use]
    pub const fn new(json: &'a str, ids: &'a [i32], transcript: &'a mut [u8], size: &'a mut i32) -> Self {
        Self { tokenizer_json: json, token_ids: ids, transcript, transcript_size_out: size, error_out: None, on_done: None, on_error: None }
    }
    #[must_use]
    pub const fn with_outputs(mut self, error_out: Option<&'a Cell<TokenizerError>>, on_done: Option<for<'r> fn(DetokenizeDone<'r>)>, on_error: Option<for<'r> fn(DetokenizeError<'r>)>) -> Self {
        self.error_out = error_out; self.on_done = on_done; self.on_error = on_error; self
    }
}

/// Caller-owned asset validation request.
#[derive(Debug)]
pub struct Validate<'a> {
    pub tokenizer_json: &'a str,
    pub decode_policy: &'a AsrDecodePolicy,
    pub error_out: Option<&'a Cell<TokenizerError>>,
}
impl<'a> Validate<'a> {
    #[must_use]
    pub const fn new(json: &'a str, policy: &'a AsrDecodePolicy) -> Self { Self { tokenizer_json: json, decode_policy: policy, error_out: None } }
    #[must_use]
    pub const fn with_error_out(mut self, error_out: Option<&'a Cell<TokenizerError>>) -> Self { self.error_out = error_out; self }
}

#[derive(Debug)]
pub struct EventDetokenizeRun<'a> { pub request: Detokenize<'a> }
impl<'a> EventDetokenizeRun<'a> { #[must_use] pub const fn new(request: Detokenize<'a>) -> Self { Self { request } } }
#[derive(Debug)]
pub struct EventValidateRun<'a> { pub request: Validate<'a> }
impl<'a> EventValidateRun<'a> { #[must_use] pub const fn new(request: Validate<'a>) -> Self { Self { request } } }

sml! {
    SpeechTokenizerWhisper {
        "state_json_decision"_s <= *"state_ready"_s + EventDetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) / effect_begin_detokenize,
        "state_detokenizing"_s <= "state_json_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_request_valid] / effect_detokenize,
        "state_error_error_out_decision"_s <= "state_json_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_tokenizer_json_invalid] / effect_mark_tokenizer_json_invalid,
        "state_error_error_out_decision"_s <= "state_json_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_token_ids_invalid] / effect_mark_token_ids_invalid,
        "state_success_error_out_decision"_s <= "state_detokenizing"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>),
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>),
        "state_ready"_s <= "state_errored"_s + completion<EventDetokenizeRun<'dispatch>>(&'dispatch EventDetokenizeRun<'dispatch>),
        "state_validate_decision"_s <= "state_ready"_s + EventValidateRun(&'dispatch EventValidateRun<'dispatch>) / effect_begin_validate,
        "state_validate_success_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_supported],
        "state_validate_error_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_json_invalid] / effect_mark_validate_json_invalid,
        "state_validate_error_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_policy_unsupported] / effect_mark_validate_policy_unsupported,
        "state_validate_done"_s <= "state_validate_success_error_out_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_has_error_out] / effect_store_validate_error_out_from_state_validate_success_error_out_decision,
        "state_validate_done"_s <= "state_validate_success_error_out_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_no_error_out],
        "state_validate_errored"_s <= "state_validate_error_error_out_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_has_error_out] / effect_store_validate_error_out_from_state_validate_error_error_out_decision,
        "state_validate_errored"_s <= "state_validate_error_error_out_decision"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>) [guard_validate_no_error_out],
        "state_ready"_s <= "state_validate_done"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>),
        "state_ready"_s <= "state_validate_errored"_s + completion<EventValidateRun<'dispatch>>(&'dispatch EventValidateRun<'dispatch>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_json_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_json_decision,
        "state_ready"_s <= "state_detokenizing"_s + unexpected_event<_> / effect_on_unexpected_from_state_detokenizing,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
        "state_ready"_s <= "state_validate_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_validate_decision,
        "state_ready"_s <= "state_validate_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_validate_success_error_out_decision,
        "state_ready"_s <= "state_validate_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_validate_error_error_out_decision,
        "state_ready"_s <= "state_validate_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_validate_done,
        "state_ready"_s <= "state_validate_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_validate_errored,
    }
}

#[derive(Debug, Default)]
pub struct SpeechTokenizerWhisperContext { error: TokenizerError }
impl SpeechTokenizerWhisperContext {
    pub const fn error(&self) -> TokenizerError { self.error }
    fn clear(&mut self) { self.error = TokenizerError::None; }
    fn mark(&mut self, error: TokenizerError) { self.error = error; }
    fn unexpected(&mut self) -> Result<(), ()> { self.mark(TokenizerError::UnexpectedEvent); Ok(()) }
}

impl SpeechTokenizerWhisperStateMachineContext for SpeechTokenizerWhisperContext {
    fn effect_begin_detokenize(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.clear(); *event.request.transcript_size_out = 0; Ok(()) }
    fn effect_begin_validate(&mut self, _event: &EventValidateRun<'_>) -> Result<(), ()> { self.clear(); Ok(()) }
    fn effect_detokenize(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> {
        let size = decode_token_ids(event.request.tokenizer_json, event.request.token_ids, event.request.transcript);
        let Ok(size_i32) = i32::try_from(size) else { self.mark(TokenizerError::InternalError); return Ok(()) };
        *event.request.transcript_size_out = size_i32;
        Ok(())
    }
    fn effect_emit_done(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.request.on_done { callback(DetokenizeDone { request: &event.request, transcript_size: *event.request.transcript_size_out }); } Ok(()) }
    fn effect_emit_error(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.request.on_error { callback(DetokenizeError { request: &event.request, error: self.error }); } Ok(()) }
    fn effect_mark_token_ids_invalid(&mut self, _event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.mark(TokenizerError::TokenIdsInvalid); Ok(()) }
    fn effect_mark_tokenizer_json_invalid(&mut self, _event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.mark(TokenizerError::TokenizerJsonInvalid); Ok(()) }
    fn effect_mark_validate_json_invalid(&mut self, _event: &EventValidateRun<'_>) -> Result<(), ()> { self.mark(TokenizerError::TokenizerJsonInvalid); Ok(()) }
    fn effect_mark_validate_policy_unsupported(&mut self, _event: &EventValidateRun<'_>) -> Result<(), ()> { self.mark(TokenizerError::DecodePolicyUnsupported); Ok(()) }
    fn effect_store_error_out_from_state_error_error_out_decision(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { if let Some(out) = event.request.error_out { out.set(self.error); } Ok(()) }
    fn effect_store_error_out_from_state_success_error_out_decision(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { if let Some(out) = event.request.error_out { out.set(TokenizerError::None); } Ok(()) }
    fn effect_store_validate_error_out_from_state_validate_error_error_out_decision(&mut self, event: &EventValidateRun<'_>) -> Result<(), ()> { if let Some(out) = event.request.error_out { out.set(self.error); } Ok(()) }
    fn effect_store_validate_error_out_from_state_validate_success_error_out_decision(&mut self, event: &EventValidateRun<'_>) -> Result<(), ()> { if let Some(out) = event.request.error_out { out.set(TokenizerError::None); } Ok(()) }
    fn guard_detokenize_request_valid(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(tokenizer_json_valid(&event.request) && token_ids_valid(&event.request)) }
    fn guard_has_done_callback(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_some()) }
    fn guard_has_error_callback(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_some()) }
    fn guard_has_error_out(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_no_done_callback(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_none()) }
    fn guard_no_error_callback(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_none()) }
    fn guard_no_error_out(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_token_ids_invalid(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(tokenizer_json_valid(&event.request) && !token_ids_valid(&event.request)) }
    fn guard_tokenizer_json_invalid(&self, event: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(!tokenizer_json_valid(&event.request)) }
    fn guard_validate_has_error_out(&self, event: &EventValidateRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_validate_json_invalid(&self, event: &EventValidateRun<'_>) -> Result<bool, ()> { Ok(!validate_json_valid(&event.request)) }
    fn guard_validate_no_error_out(&self, event: &EventValidateRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_validate_policy_unsupported(&self, event: &EventValidateRun<'_>) -> Result<bool, ()> { Ok(validate_json_valid(&event.request) && !policy_supported(event.request.decode_policy)) }
    fn guard_validate_supported(&self, event: &EventValidateRun<'_>) -> Result<bool, ()> { Ok(validate_json_valid(&event.request) && policy_supported(event.request.decode_policy)) }
    fn effect_on_unexpected_from_state_detokenizing(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_json_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validate_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validate_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validate_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validate_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validate_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
}

fn tokenizer_json_valid(request: &Detokenize<'_>) -> bool { !request.tokenizer_json.is_empty() && validate_tiny_control_tokens(request.tokenizer_json) }
fn validate_json_valid(request: &Validate<'_>) -> bool { !request.tokenizer_json.is_empty() && validate_tiny_control_tokens(request.tokenizer_json) }
fn token_ids_valid(request: &Detokenize<'_>) -> bool { request.token_ids.iter().all(|id| *id >= 0) }
fn policy_supported(policy: &AsrDecodePolicy) -> bool {
    *policy == TINY_ASR_DECODE_POLICY
}

/// Single-writer synchronous Whisper tokenizer actor.
pub struct SpeechTokenizerWhisper { machine: SpeechTokenizerWhisperStateMachine<SpeechTokenizerWhisperContext> }
impl Default for SpeechTokenizerWhisper { fn default() -> Self { Self::new() } }
impl SpeechTokenizerWhisper {
    #[must_use] pub fn new() -> Self { Self { machine: SpeechTokenizerWhisperStateMachine::new(SpeechTokenizerWhisperContext::default()) } }
    pub fn process_detokenize(&mut self, request: Detokenize<'_>) -> Result<(), TokenizerError> {
        self.machine.context_mut().clear();
        let event = EventDetokenizeRun::new(request);
        if self.machine.process_event(SpeechTokenizerWhisperEvents::EventDetokenizeRun(&event)).is_err() { self.machine.context_mut().mark(TokenizerError::UnexpectedEvent); }
        let error = self.machine.context().error();
        if error == TokenizerError::None { Ok(()) } else { Err(error) }
    }
    pub fn process_validate(&mut self, request: Validate<'_>) -> Result<(), TokenizerError> {
        self.machine.context_mut().clear();
        let event = EventValidateRun::new(request);
        if self.machine.process_event(SpeechTokenizerWhisperEvents::EventValidateRun(&event)).is_err() { self.machine.context_mut().mark(TokenizerError::UnexpectedEvent); }
        let error = self.machine.context().error();
        if error == TokenizerError::None { Ok(()) } else { Err(error) }
    }
    pub fn process_event(&mut self, event: WhisperTokenizerEvent<'_>) -> Result<(), TokenizerError> { match event { WhisperTokenizerEvent::Detokenize(request) => self.process_detokenize(request), WhisperTokenizerEvent::Validate(request) => self.process_validate(request) } }
    #[must_use] pub fn state(&self) -> &SpeechTokenizerWhisperStates { self.machine.state() }
    #[must_use] pub fn is(&self, state: &SpeechTokenizerWhisperStates) -> bool { self.machine.is(state) }
    #[must_use] pub fn context(&self) -> &SpeechTokenizerWhisperContext { self.machine.context() }
    pub fn process_unexpected_event(&mut self) -> Result<(), TokenizerError> { self.machine.context_mut().mark(TokenizerError::UnexpectedEvent); Err(TokenizerError::UnexpectedEvent) }
}
/// Synchronous tokenizer event union.
pub enum WhisperTokenizerEvent<'a> { Detokenize(Detokenize<'a>), Validate(Validate<'a>) }
/// Short alias matching the pinned C++ `sm` surface.
pub type Tokenizer = SpeechTokenizerWhisper;

fn is_json_space(value: u8) -> bool { matches!(value, b' ' | b'\n' | b'\r' | b'\t') }
fn skip_json_space(text: &[u8], mut offset: usize) -> usize { while offset < text.len() && is_json_space(text[offset]) { offset += 1; } offset }
fn write_i32(id: i32, out: &mut [u8; 16]) -> usize {
    let mut magnitude = id as u32; let mut offset = 0;
    if id < 0 { out[0] = b'-'; offset = 1; magnitude = 0u32.wrapping_sub(magnitude); }
    let mut digits = [0u8; 12]; let mut count = 0;
    loop { digits[count] = b'0' + (magnitude % 10) as u8; count += 1; magnitude /= 10; if magnitude == 0 { break; } }
    while count != 0 { count -= 1; out[offset] = digits[count]; offset += 1; }
    offset
}
fn contains_token_role(json: &[u8], content: &[u8], id: i32) -> bool {
    let Some(pos) = json.windows(content.len()).position(|w| w == content) else { return false };
    let mut id_buf = [0u8; 16]; let id_len = write_i32(id, &mut id_buf);
    let begin = pos.saturating_sub(96); let end = (pos + content.len() + 96).min(json.len()); let window = &json[begin..end];
    window.windows(4).any(|w| w == b"\"id\"") && window.windows(id_len).any(|w| w == &id_buf[..id_len])
}
fn validate_tiny_control_tokens(json: &str) -> bool {
    let bytes = json.as_bytes(); let t = TINY_CONTROL_TOKENS;
    contains_token_role(bytes, b"<|endoftext|>", t.eot) && contains_token_role(bytes, b"<|startoftranscript|>", t.sot) && contains_token_role(bytes, b"<|en|>", t.language_en) && contains_token_role(bytes, b"<|translate|>", t.translate) && contains_token_role(bytes, b"<|transcribe|>", t.transcribe) && contains_token_role(bytes, b"<|nocaptions|>", t.no_speech) && contains_token_role(bytes, b"<|notimestamps|>", t.notimestamps) && contains_token_role(bytes, b"<|0.00|>", t.timestamp_begin)
}
fn find_vocab_scope(json: &[u8]) -> &[u8] {
    let needle = b"\"vocab\""; let Some(key) = json.windows(needle.len()).position(|w| w == needle) else { return json };
    let mut p = skip_json_space(json, key + needle.len()); if p >= json.len() || json[p] != b':' { return json } p = skip_json_space(json, p + 1); if p >= json.len() || json[p] != b'{' { return json }
    let begin = p + 1; let mut depth = 1; let mut string = false; let mut escaped = false;
    for i in begin..json.len() { let c = json[i]; if string { if escaped { escaped = false } else if c == b'\\' { escaped = true } else if c == b'"' { string = false } continue } if c == b'"' { string = true } else if c == b'{' { depth += 1 } else if c == b'}' { depth -= 1; if depth == 0 { return &json[begin..i] } } }
    json
}
fn json_unescape_piece(piece: &[u8], transcript: &mut [u8], size: &mut usize) {
    let mut i = 0; while i < piece.len() && *size < transcript.len() { let mut value = piece[i]; let mut advance = 1;
        if piece[i] == b'\\' && i + 1 < piece.len() { value = match piece[i + 1] { b'"' => b'"', b'\\' => b'\\', b'/' => b'/', b'b' => 8, b'f' => 12, b'n' => b'\n', b'r' => b'\r', b't' => b'\t', _ => value }; if value != piece[i] { advance = 2; } }
        if i + 1 < piece.len() && piece[i] == 0xc4 && piece[i + 1] == 0xa0 { value = b' '; advance = 2; }
        transcript[*size] = value; *size += 1; i += advance;
    }
}
fn vocab_piece<'a>(json: &'a str, id: i32) -> Option<&'a [u8]> {
    let object = find_vocab_scope(json.as_bytes()); let mut search = 0;
    while search < object.len() { let key_begin = object[search..].iter().position(|c| *c == b'"')? + search; let mut key_end = key_begin + 1; let mut escaped = false;
        while key_end < object.len() { let c = object[key_end]; if escaped { escaped = false } else if c == b'\\' { escaped = true } else if c == b'"' { break } key_end += 1; } if key_end >= object.len() { return None }
        let colon = skip_json_space(object, key_end + 1); if colon >= object.len() || object[colon] != b':' { search = key_end + 1; continue }
        let value = skip_json_space(object, colon + 1); let mut id_buf = [0u8; 16]; let id_len = write_i32(id, &mut id_buf);
        if object.get(value..value + id_len) == Some(&id_buf[..id_len]) { let end = skip_json_space(object, value + id_len); if end == object.len() || object[end] == b',' || object[end] == b'}' { return Some(&object[key_begin + 1..key_end]); } }
        search = value.saturating_add(1);
    } None
}
fn decode_token_ids(json: &str, ids: &[i32], transcript: &mut [u8]) -> usize {
    let mut size = 0; for id in ids { if *id >= TINY_CONTROL_TOKENS.eot { continue } if let Some(piece) = vocab_piece(json, *id) { json_unescape_piece(piece, transcript, &mut size); } }
    let mut leading = 0; while leading < size && transcript[leading] == b' ' { leading += 1; } if leading != 0 { transcript.copy_within(leading..size, 0); } size - leading
}

/// Returns the maximum conservative transcript capacity needed for `token_count` IDs.
#[must_use]
pub fn required_transcript_capacity(tokenizer_json: &str, token_count: usize) -> usize {
    if token_count == 0 { return 0 } let object = find_vocab_scope(tokenizer_json.as_bytes()); let mut search = 0; let mut max = 0;
    while search < object.len() { let Some(begin_rel) = object[search..].iter().position(|c| *c == b'"') else { break }; let begin = search + begin_rel; let Some(end_rel) = object[begin + 1..].iter().position(|c| *c == b'"') else { break }; let end = begin + 1 + end_rel; max = max.max(end - begin - 1); search = end + 1; }
    let per = if max == 0 { tokenizer_json.len().max(1) } else { max }; token_count.checked_mul(per).unwrap_or(usize::MAX)
}
