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
    missing_docs,
    // Generated SML state/event values derive PartialEq without Eq.
    clippy::derive_partial_eq_without_eq,
    unpredictable_function_pointer_comparisons
)]

use core::cell::{Cell, RefCell};
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
pub enum LanguageRole {
    English,
}
/// Supported task role for the Whisper tiny policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskRole {
    Transcribe,
}
/// Supported timestamp mode for the Whisper tiny policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestampMode {
    TimestampTokens,
}

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
    eot: 50_257,
    sot: 50_258,
    language_en: 50_259,
    translate: 50_358,
    transcribe: 50_359,
    no_speech: 50_362,
    notimestamps: 50_363,
    timestamp_begin: 50_364,
    space: 220,
};
/// SHA-256 digest of the canonical Whisper tiny tokenizer asset.
pub const TINY_TOKENIZER_SHA256: &str =
    "dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759";
/// Returns the canonical tiny tokenizer SHA-256 digest.
#[must_use]
pub const fn tiny_tokenizer_sha256() -> &'static str {
    TINY_TOKENIZER_SHA256
}
/// Returns the pinned public spelling for a language role.
#[must_use]
pub const fn language_role_name(_: LanguageRole) -> &'static str {
    "english"
}
/// Returns the pinned public spelling for a task role.
#[must_use]
pub const fn task_role_name(_: TaskRole) -> &'static str {
    "transcribe"
}
/// Returns the pinned public spelling for a timestamp mode.
#[must_use]
pub const fn timestamp_mode_name(_: TimestampMode) -> &'static str {
    "timestamp_tokens"
}
/// Returns the canonical tiny decode policy.
#[must_use]
pub const fn tiny_asr_decode_policy() -> &'static AsrDecodePolicy {
    &TINY_ASR_DECODE_POLICY
}
/// Returns whether a policy exactly matches the pinned tiny policy.
#[must_use]
pub fn is_tiny_asr_decode_policy_supported(policy: &AsrDecodePolicy) -> bool {
    *policy == TINY_ASR_DECODE_POLICY
}
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
    /// The caller-owned transcript buffer cannot hold the complete output.
    TranscriptCapacity = 6,
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
            Self::TranscriptCapacity => "transcript capacity is insufficient",
        })
    }
}

/// Successful detokenization callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DetokenizeDone<'dispatch, 'event> {
    pub request: &'dispatch Detokenize<'event>,
    pub transcript_size: i32,
}
/// Failed detokenization callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DetokenizeError<'dispatch, 'event> {
    pub request: &'dispatch Detokenize<'event>,
    pub error: TokenizerError,
}

/// Caller-owned detokenization request.
#[derive(Debug)]
pub struct Detokenize<'a> {
    pub tokenizer_json: &'a str,
    pub token_ids: &'a [i32],
    pub transcript: RefCell<&'a mut [u8]>,
    pub transcript_size_out: RefCell<&'a mut i32>,
    pub error_out: Option<&'a Cell<TokenizerError>>,
    pub on_done: Option<for<'dispatch, 'event> fn(DetokenizeDone<'dispatch, 'event>)>,
    pub on_error: Option<for<'dispatch, 'event> fn(DetokenizeError<'dispatch, 'event>)>,
}
impl<'a> Detokenize<'a> {
    #[must_use]
    pub const fn new(
        json: &'a str,
        ids: &'a [i32],
        transcript: &'a mut [u8],
        size: &'a mut i32,
    ) -> Self {
        Self {
            tokenizer_json: json,
            token_ids: ids,
            transcript: RefCell::new(transcript),
            transcript_size_out: RefCell::new(size),
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
    #[must_use]
    pub const fn with_outputs(
        mut self,
        error_out: Option<&'a Cell<TokenizerError>>,
        on_done: Option<for<'dispatch, 'event> fn(DetokenizeDone<'dispatch, 'event>)>,
        on_error: Option<for<'dispatch, 'event> fn(DetokenizeError<'dispatch, 'event>)>,
    ) -> Self {
        self.error_out = error_out;
        self.on_done = on_done;
        self.on_error = on_error;
        self
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
    pub const fn new(json: &'a str, policy: &'a AsrDecodePolicy) -> Self {
        Self {
            tokenizer_json: json,
            decode_policy: policy,
            error_out: None,
        }
    }
    #[must_use]
    pub const fn with_error_out(mut self, error_out: Option<&'a Cell<TokenizerError>>) -> Self {
        self.error_out = error_out;
        self
    }
}

#[derive(Debug)]
pub struct EventDetokenizeRun<'a> {
    pub request: Detokenize<'a>,
}
impl<'a> EventDetokenizeRun<'a> {
    #[must_use]
    pub const fn new(request: Detokenize<'a>) -> Self {
        Self { request }
    }
}
#[derive(Debug)]
pub struct EventValidateRun<'a> {
    pub request: Validate<'a>,
}
impl<'a> EventValidateRun<'a> {
    #[must_use]
    pub const fn new(request: Validate<'a>) -> Self {
        Self { request }
    }
}

sml! {
    SpeechTokenizerWhisper<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_json_decision"_s <= *"state_ready"_s + EventDetokenizeRun(&'dispatch EventDetokenizeRun<'event>) / effect_begin_detokenize,
        "state_detokenizing"_s <= "state_json_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_request_valid] / effect_detokenize,
        "state_error_error_out_decision"_s <= "state_json_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_tokenizer_json_invalid] / effect_mark_tokenizer_json_invalid,
        "state_error_error_out_decision"_s <= "state_json_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_token_ids_invalid] / effect_mark_token_ids_invalid,
        "state_error_error_out_decision"_s <= "state_json_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_transcript_capacity_invalid] / effect_mark_transcript_capacity,
        "state_success_error_out_decision"_s <= "state_detokenizing"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>),
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>),
        "state_ready"_s <= "state_errored"_s + completion<EventDetokenizeRun>(&'dispatch EventDetokenizeRun<'event>),
        "state_validate_decision"_s <= "state_ready"_s + EventValidateRun(&'dispatch EventValidateRun<'event>) / effect_begin_validate,
        "state_validate_success_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_supported],
        "state_validate_error_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_json_invalid] / effect_mark_validate_json_invalid,
        "state_validate_error_error_out_decision"_s <= "state_validate_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_policy_unsupported] / effect_mark_validate_policy_unsupported,
        "state_validate_done"_s <= "state_validate_success_error_out_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_has_error_out] / effect_store_validate_error_out_from_state_validate_success_error_out_decision,
        "state_validate_done"_s <= "state_validate_success_error_out_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_no_error_out],
        "state_validate_errored"_s <= "state_validate_error_error_out_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_has_error_out] / effect_store_validate_error_out_from_state_validate_error_error_out_decision,
        "state_validate_errored"_s <= "state_validate_error_error_out_decision"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>) [guard_validate_no_error_out],
        "state_ready"_s <= "state_validate_done"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>),
        "state_ready"_s <= "state_validate_errored"_s + completion<EventValidateRun>(&'dispatch EventValidateRun<'event>),
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
pub struct SpeechTokenizerWhisperContext {
    error: TokenizerError,
}
impl SpeechTokenizerWhisperContext {
    pub const fn error(&self) -> TokenizerError {
        self.error
    }
    pub const fn clear(&mut self) {
        self.error = TokenizerError::None;
    }
    const fn mark(&mut self, error: TokenizerError) {
        self.error = error;
    }
    const fn unexpected(&mut self) {
        self.mark(TokenizerError::UnexpectedEvent);
    }
}

impl SpeechTokenizerWhisperStateMachineContext for SpeechTokenizerWhisperContext {
    fn effect_begin_detokenize<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.clear();
        **event.request.transcript_size_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_begin_validate<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventValidateRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.clear();
        Ok(())
    }
    fn effect_detokenize<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let size = {
            let mut transcript = event.request.transcript.borrow_mut();
            decode_token_ids(
                event.request.tokenizer_json,
                event.request.token_ids,
                &mut transcript,
            )
        };
        let Ok(size_i32) = i32::try_from(size) else {
            self.mark(TokenizerError::InternalError);
            return Ok(());
        };
        **event.request.transcript_size_out.borrow_mut() = size_i32;
        Ok(())
    }
    fn effect_emit_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            let transcript_size = { **event.request.transcript_size_out.borrow() };
            callback(DetokenizeDone {
                request: &event.request,
                transcript_size,
            });
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(DetokenizeError {
                request: &event.request,
                error: self.error,
            });
        }
        Ok(())
    }
    fn effect_mark_token_ids_invalid<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(TokenizerError::TokenIdsInvalid);
        Ok(())
    }
    fn effect_mark_tokenizer_json_invalid<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(TokenizerError::TokenizerJsonInvalid);
        Ok(())
    }
    fn effect_mark_transcript_capacity<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(TokenizerError::TranscriptCapacity);
        Ok(())
    }
    fn effect_mark_validate_json_invalid<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventValidateRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(TokenizerError::TokenizerJsonInvalid);
        Ok(())
    }
    fn effect_mark_validate_policy_unsupported<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EventValidateRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(TokenizerError::DecodePolicyUnsupported);
        Ok(())
    }
    fn effect_store_error_out_from_state_error_error_out_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out {
            out.set(self.error);
        }
        Ok(())
    }
    fn effect_store_error_out_from_state_success_error_out_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out {
            out.set(TokenizerError::None);
        }
        Ok(())
    }
    fn effect_store_validate_error_out_from_state_validate_error_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out {
            out.set(self.error);
        }
        Ok(())
    }
    fn effect_store_validate_error_out_from_state_validate_success_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out {
            out.set(TokenizerError::None);
        }
        Ok(())
    }
    fn guard_detokenize_request_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(tokenizer_json_valid(&event.request)
            && token_ids_valid(&event.request)
            && !transcript_capacity_invalid(&event.request))
    }
    fn guard_has_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_has_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_has_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_no_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_no_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_no_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_token_ids_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(tokenizer_json_valid(&event.request) && !token_ids_valid(&event.request))
    }
    fn guard_transcript_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(tokenizer_json_valid(&event.request)
            && token_ids_valid(&event.request)
            && required_transcript_capacity(
                event.request.tokenizer_json,
                event.request.token_ids.len(),
            ) > event.request.transcript.borrow().len())
    }
    fn guard_tokenizer_json_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!tokenizer_json_valid(&event.request))
    }
    fn guard_validate_has_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_validate_json_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!validate_json_valid(&event.request))
    }
    fn guard_validate_no_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_validate_policy_unsupported<'dispatch, 'event>(
        &self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(validate_json_valid(&event.request) && !policy_supported(event.request.decode_policy))
    }
    fn guard_validate_supported<'dispatch, 'event>(
        &self,
        event: &'dispatch EventValidateRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(validate_json_valid(&event.request) && policy_supported(event.request.decode_policy))
    }
    fn effect_on_unexpected_from_state_detokenizing(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_json_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_validate_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_validate_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_validate_error_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_validate_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_on_unexpected_from_state_validate_success_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
}
fn tokenizer_json_valid(request: &Detokenize<'_>) -> bool {
    !request.tokenizer_json.is_empty() && validate_tiny_control_tokens(request.tokenizer_json)
}
fn validate_json_valid(request: &Validate<'_>) -> bool {
    !request.tokenizer_json.is_empty() && validate_tiny_control_tokens(request.tokenizer_json)
}
fn token_ids_valid(request: &Detokenize<'_>) -> bool {
    request.token_ids.iter().all(|id| *id >= 0)
}
fn policy_supported(policy: &AsrDecodePolicy) -> bool {
    *policy == TINY_ASR_DECODE_POLICY
}
fn transcript_capacity_invalid(request: &Detokenize<'_>) -> bool {
    required_transcript_capacity(request.tokenizer_json, request.token_ids.len())
        > request.transcript.borrow().len()
}

/// Single-writer synchronous Whisper tokenizer actor.
pub struct SpeechTokenizerWhisper {
    machine: SpeechTokenizerWhisperStateMachine<SpeechTokenizerWhisperContext>,
}
impl fmt::Debug for SpeechTokenizerWhisper {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("SpeechTokenizerWhisper").finish()
    }
}
impl Default for SpeechTokenizerWhisper {
    fn default() -> Self {
        Self::new()
    }
}
impl SpeechTokenizerWhisper {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechTokenizerWhisperStateMachine::new(
                SpeechTokenizerWhisperContext::default(),
            ),
        }
    }
    pub fn process_detokenize(&mut self, request: Detokenize<'_>) -> Result<(), TokenizerError> {
        self.machine.context_mut().clear();
        let event = EventDetokenizeRun::new(request);
        if self
            .machine
            .process_event(SpeechTokenizerWhisperEvents::EventDetokenizeRun(&event))
            .is_err()
        {
            self.machine
                .context_mut()
                .mark(TokenizerError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == TokenizerError::None {
            Ok(())
        } else {
            Err(error)
        }
    }
    pub fn process_validate(&mut self, request: Validate<'_>) -> Result<(), TokenizerError> {
        self.machine.context_mut().clear();
        let event = EventValidateRun::new(request);
        if self
            .machine
            .process_event(SpeechTokenizerWhisperEvents::EventValidateRun(&event))
            .is_err()
        {
            self.machine
                .context_mut()
                .mark(TokenizerError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == TokenizerError::None {
            Ok(())
        } else {
            Err(error)
        }
    }
    pub fn process_event(
        &mut self,
        event: WhisperTokenizerEvent<'_>,
    ) -> Result<(), TokenizerError> {
        match event {
            WhisperTokenizerEvent::Detokenize(request) => self.process_detokenize(request),
            WhisperTokenizerEvent::Validate(request) => self.process_validate(request),
        }
    }
    #[must_use]
    pub fn state(&self) -> &SpeechTokenizerWhisperStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &SpeechTokenizerWhisperStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &SpeechTokenizerWhisperContext {
        self.machine.context()
    }
    pub fn process_unexpected_event(&mut self) -> Result<(), TokenizerError> {
        self.machine
            .context_mut()
            .mark(TokenizerError::UnexpectedEvent);
        Err(TokenizerError::UnexpectedEvent)
    }
}
/// Synchronous tokenizer event union.
#[derive(Debug)]
pub enum WhisperTokenizerEvent<'a> {
    Detokenize(Detokenize<'a>),
    Validate(Validate<'a>),
}
/// Short alias matching the pinned C++ `sm` surface.
pub type Tokenizer = SpeechTokenizerWhisper;

#[must_use]
const fn is_json_space(value: u8) -> bool {
    matches!(value, b' ' | b'\n' | b'\r' | b'\t')
}
const fn skip_json_space(text: &[u8], mut offset: usize) -> usize {
    while offset < text.len() && is_json_space(text[offset]) {
        offset += 1;
    }
    offset
}
const fn write_i32(id: i32, out: &mut [u8; 16]) -> usize {
    let mut magnitude = id.unsigned_abs();
    let mut offset = if id < 0 {
        out[0] = b'-';
        1
    } else {
        0
    };
    let mut digits = [0u8; 12];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (magnitude % 10) as u8;
        count += 1;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    while count != 0 {
        count -= 1;
        out[offset] = digits[count];
        offset += 1;
    }
    offset
}

fn json_string_end(json: &[u8], mut offset: usize) -> Option<usize> {
    if json.get(offset) != Some(&b'"') {
        return None;
    }
    offset += 1;
    while offset < json.len() {
        match json[offset] {
            b'"' => return Some(offset + 1),
            b'\\' => {
                offset += 1;
                let escaped = *json.get(offset)?;
                if escaped == b'u' {
                    for _ in 0..4 {
                        offset += 1;
                        if !json.get(offset).is_some_and(u8::is_ascii_hexdigit) {
                            return None;
                        }
                    }
                } else if !matches!(
                    escaped,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                ) {
                    return None;
                }
            }
            c if c < 0x20 => return None,
            _ => {}
        }
        offset += 1;
    }
    None
}

fn json_value_end(json: &[u8], mut offset: usize) -> Option<usize> {
    offset = skip_json_space(json, offset);
    match *json.get(offset)? {
        b'"' => json_string_end(json, offset),
        b'{' => {
            offset += 1;
            offset = skip_json_space(json, offset);
            if json.get(offset) == Some(&b'}') {
                return Some(offset + 1);
            }
            loop {
                offset = json_string_end(json, skip_json_space(json, offset))?;
                offset = skip_json_space(json, offset);
                if json.get(offset) != Some(&b':') {
                    return None;
                }
                offset = json_value_end(json, offset + 1)?;
                offset = skip_json_space(json, offset);
                match json.get(offset) {
                    Some(b',') => offset += 1,
                    Some(b'}') => return Some(offset + 1),
                    _ => return None,
                }
            }
        }
        b'[' => {
            offset += 1;
            offset = skip_json_space(json, offset);
            if json.get(offset) == Some(&b']') {
                return Some(offset + 1);
            }
            loop {
                offset = json_value_end(json, offset)?;
                offset = skip_json_space(json, offset);
                match json.get(offset) {
                    Some(b',') => offset += 1,
                    Some(b']') => return Some(offset + 1),
                    _ => return None,
                }
            }
        }
        b'-' | b'0'..=b'9' => {
            if json[offset] == b'-' {
                offset += 1;
            }
            match json.get(offset) {
                Some(b'0') => offset += 1,
                Some(b'1'..=b'9') => {
                    while json.get(offset).is_some_and(u8::is_ascii_digit) {
                        offset += 1;
                    }
                }
                _ => return None,
            }
            if json.get(offset) == Some(&b'.') {
                offset += 1;
                if !json.get(offset).is_some_and(u8::is_ascii_digit) {
                    return None;
                }
                while json.get(offset).is_some_and(u8::is_ascii_digit) {
                    offset += 1;
                }
            }
            if matches!(json.get(offset), Some(b'e' | b'E')) {
                offset += 1;
                if matches!(json.get(offset), Some(b'+' | b'-')) {
                    offset += 1;
                }
                if !json.get(offset).is_some_and(u8::is_ascii_digit) {
                    return None;
                }
                while json.get(offset).is_some_and(u8::is_ascii_digit) {
                    offset += 1;
                }
            }
            Some(offset)
        }
        b't' if json.get(offset..offset + 4) == Some(b"true") => Some(offset + 4),
        b'f' if json.get(offset..offset + 5) == Some(b"false") => Some(offset + 5),
        b'n' if json.get(offset..offset + 4) == Some(b"null") => Some(offset + 4),
        _ => None,
    }
}

fn json_valid(json: &[u8]) -> bool {
    json_value_end(json, 0).is_some_and(|end| skip_json_space(json, end) == json.len())
}

fn contains_token_role(json: &[u8], content: &[u8], id: i32) -> bool {
    let mut id_buf = [0u8; 16];
    let id_len = write_i32(id, &mut id_buf);
    let quoted_len = content.len() + 2;
    let mut pos = 0;
    while let Some(relative) = json[pos..].iter().position(|c| *c == b'"') {
        let begin = pos + relative;
        let Some(end) = json_string_end(json, begin) else {
            return false;
        };
        let string = &json[begin + 1..end - 1];
        if string == content {
            let window_begin = begin.saturating_sub(96);
            let window_end = (end + 96).min(json.len());
            let window = &json[window_begin..window_end];
            let id_key = b"\"id\"";
            if window.windows(id_key.len()).any(|w| w == id_key)
                && window.windows(id_len).any(|w| w == &id_buf[..id_len])
            {
                return true;
            }
        }
        let _ = quoted_len;
        pos = end;
    }
    false
}

fn validate_tiny_control_tokens(json: &str) -> bool {
    let bytes = json.as_bytes();
    if !json_valid(bytes) {
        return false;
    }
    let t = TINY_CONTROL_TOKENS;
    contains_token_role(bytes, b"<|endoftext|>", t.eot)
        && contains_token_role(bytes, b"<|startoftranscript|>", t.sot)
        && contains_token_role(bytes, b"<|en|>", t.language_en)
        && contains_token_role(bytes, b"<|translate|>", t.translate)
        && contains_token_role(bytes, b"<|transcribe|>", t.transcribe)
        && contains_token_role(bytes, b"<|nocaptions|>", t.no_speech)
        && contains_token_role(bytes, b"<|notimestamps|>", t.notimestamps)
        && contains_token_role(bytes, b"<|0.00|>", t.timestamp_begin)
}
fn find_vocab_scope(json: &[u8]) -> &[u8] {
    let needle = b"\"vocab\"";
    let Some(key) = json.windows(needle.len()).position(|w| w == needle) else {
        return json;
    };
    let mut p = skip_json_space(json, key + needle.len());
    if p >= json.len() || json[p] != b':' {
        return json;
    }
    p = skip_json_space(json, p + 1);
    if p >= json.len() || json[p] != b'{' {
        return json;
    }
    let begin = p + 1;
    let mut depth = 1;
    let mut string = false;
    let mut escaped = false;
    for i in begin..json.len() {
        let c = json[i];
        if string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                string = false;
            }
            continue;
        }
        if c == b'"' {
            string = true;
        } else if c == b'{' {
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                return &json[begin..i];
            }
        }
    }
    json
}
const fn json_unescape_piece(piece: &[u8], transcript: &mut [u8], size: &mut usize) {
    let mut i = 0;
    while i < piece.len() && *size < transcript.len() {
        let mut value = piece[i];
        let mut advance = 1;
        if piece[i] == b'\\' && i + 1 < piece.len() {
            value = match piece[i + 1] {
                b'"' => b'"',
                b'\\' => b'\\',
                b'/' => b'/',
                b'b' => 8,
                b'f' => 12,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                _ => value,
            };
            if value != piece[i] {
                advance = 2;
            }
        }
        if i + 1 < piece.len() && piece[i] == 0xc4 && piece[i + 1] == 0xa0 {
            value = b' ';
            advance = 2;
        }
        transcript[*size] = value;
        *size += 1;
        i += advance;
    }
}
fn vocab_piece(json: &str, id: i32) -> Option<&[u8]> {
    let object = find_vocab_scope(json.as_bytes());
    let mut search = 0;
    while search < object.len() {
        let key_begin = object[search..].iter().position(|c| *c == b'"')? + search;
        let mut key_end = key_begin + 1;
        let mut escaped = false;
        while key_end < object.len() {
            let c = object[key_end];
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                break;
            }
            key_end += 1;
        }
        if key_end >= object.len() {
            return None;
        }
        let colon = skip_json_space(object, key_end + 1);
        if colon >= object.len() || object[colon] != b':' {
            search = key_end + 1;
            continue;
        }
        let value = skip_json_space(object, colon + 1);
        let mut id_buf = [0u8; 16];
        let id_len = write_i32(id, &mut id_buf);
        if object.get(value..value + id_len) == Some(&id_buf[..id_len]) {
            let end = skip_json_space(object, value + id_len);
            if end == object.len() || object[end] == b',' || object[end] == b'}' {
                return Some(&object[key_begin + 1..key_end]);
            }
        }
        search = value.saturating_add(1);
    }
    None
}
fn decode_token_ids(json: &str, ids: &[i32], transcript: &mut [u8]) -> usize {
    let mut size = 0;
    for id in ids {
        if *id >= TINY_CONTROL_TOKENS.eot {
            continue;
        }
        if let Some(piece) = vocab_piece(json, *id) {
            json_unescape_piece(piece, transcript, &mut size);
        }
    }
    let mut leading = 0;
    while leading < size && transcript[leading] == b' ' {
        leading += 1;
    }
    if leading != 0 {
        transcript.copy_within(leading..size, 0);
    }
    size - leading
}

/// Returns the maximum conservative transcript capacity needed for `token_count` IDs.
#[must_use]
pub fn required_transcript_capacity(tokenizer_json: &str, token_count: usize) -> usize {
    if token_count == 0 {
        return 0;
    }
    let object = find_vocab_scope(tokenizer_json.as_bytes());
    let mut search = 0;
    let mut max = 0;
    while search < object.len() {
        let Some(begin_rel) = object[search..].iter().position(|c| *c == b'"') else {
            break;
        };
        let begin = search + begin_rel;
        let Some(end_rel) = object[begin + 1..].iter().position(|c| *c == b'"') else {
            break;
        };
        let end = begin + 1 + end_rel;
        max = max.max(end - begin - 1);
        search = end + 1;
    }
    let per = if max == 0 {
        tokenizer_json.len().max(1)
    } else {
        max
    };
    token_count.saturating_mul(per)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_JSON: &str = r#"{"added_tokens":[{"id":50257,"content":"<|endoftext|>"},{"id":50258,"content":"<|startoftranscript|>"},{"id":50259,"content":"<|en|>"},{"id":50358,"content":"<|translate|>"},{"id":50359,"content":"<|transcribe|>"},{"id":50362,"content":"<|nocaptions|>"},{"id":50363,"content":"<|notimestamps|>"},{"id":50364,"content":"<|0.00|>"}],"model":{"vocab":{"hello":1,"world":2}}}"#;

    #[test]
    fn exposes_pinned_facade_values() {
        assert_eq!(tiny_tokenizer_sha256(), TINY_TOKENIZER_SHA256);
        assert_eq!(language_role_name(LanguageRole::English), "english");
        assert_eq!(task_role_name(TaskRole::Transcribe), "transcribe");
        assert_eq!(
            timestamp_mode_name(TimestampMode::TimestampTokens),
            "timestamp_tokens"
        );
        assert_eq!(tiny_asr_decode_policy(), &TINY_ASR_DECODE_POLICY);
        assert!(is_tiny_asr_decode_policy_supported(&TINY_ASR_DECODE_POLICY));
    }

    #[test]
    fn rejects_malformed_json_and_wrong_role_ids() {
        assert!(!validate_tiny_control_tokens("{}"));
        assert!(!validate_tiny_control_tokens(
            &VALID_JSON[..VALID_JSON.len() - 1]
        ));
        let wrong = VALID_JSON.replace("50257", "50256");
        assert!(!validate_tiny_control_tokens(&wrong));
    }

    #[test]
    fn rejects_negative_ids_and_short_transcript() {
        let mut transcript = [0xA5; 2];
        let mut size = -1;
        let mut tokenizer = SpeechTokenizerWhisper::new();
        let request = Detokenize::new(VALID_JSON, &[1, -1], &mut transcript, &mut size);
        assert_eq!(
            tokenizer.process_detokenize(request),
            Err(TokenizerError::TokenIdsInvalid)
        );
        assert_eq!(size, 0);

        let mut transcript = [0u8; 2];
        let mut size = -1;
        let mut tokenizer = SpeechTokenizerWhisper::new();
        let request = Detokenize::new(VALID_JSON, &[1, 2], &mut transcript, &mut size);
        assert_eq!(
            tokenizer.process_detokenize(request),
            Err(TokenizerError::TranscriptCapacity)
        );
        assert_eq!(size, 0);
    }

    #[test]
    fn accepts_empty_output() {
        let mut transcript = [0u8; 1];
        let mut size = -1;
        let mut tokenizer = SpeechTokenizerWhisper::new();
        let request = Detokenize::new(VALID_JSON, &[], &mut transcript, &mut size);
        assert_eq!(tokenizer.process_detokenize(request), Ok(()));
        assert_eq!(size, 0);
    }
}
