//! Source-aligned speech transcriber orchestration state machine.
//!
//! The transcriber owns request-local bookkeeping only.  Encode, decode, and
//! detokenize are synchronous callbacks supplied by the bound component
//! variants; no callback may retain a borrowed value or re-enter the actor.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_braces,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use std::cell::{Cell, RefCell};
use sml::sml;

/// Errors selected by the transcriber lifecycle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TranscriberError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ModelInvalid = 2,
    TokenizerInvalid = 3,
    UnsupportedModel = 4,
    Uninitialized = 5,
    Backend = 6,
    OutputCapacity = 7,
    UnexpectedEvent = 8,
}

/// Bounded tokenizer assets copied into a request view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizerAssets<'a> {
    pub model_json: &'a str,
    pub sha256: &'a str,
}

/// Caller-owned intermediate buffers used by the recognition pipeline.
///
/// The interior cells let generated SML actions mutate caller-owned output
/// without requiring mutable access to the event itself.  The buffers are
/// borrowed for one run-to-completion dispatch and are never retained.
pub struct RuntimeStorage<'a> {
    pub encoder_workspace: RefCell<&'a mut [f32]>,
    pub encoder_state: RefCell<&'a mut [f32]>,
    pub decoder_workspace: RefCell<&'a mut [f32]>,
    pub logits: RefCell<&'a mut [f32]>,
    pub generated_tokens: RefCell<&'a mut [i32]>,
}

impl<'a> RuntimeStorage<'a> {
    #[must_use]
    pub fn new(
        encoder_workspace: &'a mut [f32],
        encoder_state: &'a mut [f32],
        decoder_workspace: &'a mut [f32],
        logits: &'a mut [f32],
        generated_tokens: &'a mut [i32],
    ) -> Self {
        Self {
            encoder_workspace: RefCell::new(encoder_workspace),
            encoder_state: RefCell::new(encoder_state),
            decoder_workspace: RefCell::new(decoder_workspace),
            logits: RefCell::new(logits),
            generated_tokens: RefCell::new(generated_tokens),
        }
    }
}

/// Result fields published by a successful encoder callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EncoderResult {
    pub frame_count: i32,
    pub width: i32,
    pub digest: u64,
}

/// Result fields published by a successful decoder callback.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DecoderResult {
    pub generated_token_count: i32,
    pub selected_token: i32,
    pub confidence: f32,
    pub digest: u64,
}

/// Synchronous tokenizer asset validation callback.
pub type ValidateTokenizerFn = fn(&str, &str) -> bool;
/// Synchronous encoder callback.
pub type EncodeFn = fn(&[f32], i32, &mut [f32], &mut [f32], &mut EncoderResult) -> bool;
/// Synchronous decoder callback.
pub type DecodeFn = fn(&[f32], i32, &mut [i32], &mut [f32], &mut [f32], &mut DecoderResult) -> bool;
/// Synchronous detokenizer callback.
pub type DetokenizeFn = fn(&str, &[i32], &mut [u8]) -> Option<i32>;

/// Component contracts injected by the bound speech variants.
#[derive(Clone, Copy, Debug)]
pub struct Dependencies {
    pub encoder_supported: bool,
    pub decoder_supported: bool,
    pub tokenizer_supported: bool,
    pub model_id: usize,
    pub encoder_model_id: usize,
    pub decoder_model_id: usize,
    pub embedding_length: usize,
    pub tokenizer_sha256: &'static str,
    pub validate_tokenizer: Option<ValidateTokenizerFn>,
    pub encode: Option<EncodeFn>,
    pub decode: Option<DecodeFn>,
    pub detokenize: Option<DetokenizeFn>,
}

impl Default for Dependencies {
    fn default() -> Self {
        Self {
            encoder_supported: false,
            decoder_supported: false,
            tokenizer_supported: false,
            model_id: 0,
            encoder_model_id: 0,
            decoder_model_id: 0,
            embedding_length: 0,
            tokenizer_sha256: "",
            validate_tokenizer: None,
            encode: None,
            decode: None,
            detokenize: None,
        }
    }
}

/// Completion callback payload for initialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeDone;
/// Completion callback payload for initialization failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeError {
    pub error: TranscriberError,
}
/// Completion callback payload for recognition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecognitionDone {
    pub transcript_size: i32,
    pub selected_token: i32,
    pub confidence: f32,
    pub encoder_frame_count: i32,
    pub encoder_width: i32,
    pub generated_token_count: i32,
    pub encoder_digest: u64,
    pub decoder_digest: u64,
}
/// Completion callback payload for recognition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecognitionError {
    pub error: TranscriberError,
}

pub type InitializeDoneFn = fn(InitializeDone) -> bool;
pub type InitializeErrorFn = fn(InitializeError) -> bool;
pub type RecognitionDoneFn = fn(RecognitionDone) -> bool;
pub type RecognitionErrorFn = fn(RecognitionError) -> bool;

/// Initialization request.  Scalar fields and asset views are bounded; the
/// callback channels are synchronous and optional, matching the C++ event.
pub struct EventInitializeRun<'a> {
    pub model: usize,
    pub tokenizer: TokenizerAssets<'a>,
    pub error_out: Option<&'a Cell<TranscriberError>>,
    pub on_done: Option<InitializeDoneFn>,
    pub on_error: Option<InitializeErrorFn>,
}

/// Recognition request and caller-owned result channels.
pub struct EventRecognizeRun<'a> {
    pub model: usize,
    pub tokenizer: TokenizerAssets<'a>,
    pub pcm: &'a [f32],
    pub sample_rate: i32,
    pub channel_count: i32,
    pub transcript: RefCell<&'a mut [u8]>,
    pub storage: RuntimeStorage<'a>,
    pub transcript_size_out: &'a Cell<i32>,
    pub selected_token_out: &'a Cell<i32>,
    pub confidence_out: &'a Cell<f32>,
    pub encoder_frame_count_out: &'a Cell<i32>,
    pub encoder_width_out: &'a Cell<i32>,
    pub encoder_digest_out: &'a Cell<u64>,
    pub decoder_digest_out: &'a Cell<u64>,
    pub generated_token_count_out: &'a Cell<i32>,
    pub error_out: Option<&'a Cell<TranscriberError>>,
    pub on_done: Option<RecognitionDoneFn>,
    pub on_error: Option<RecognitionErrorFn>,
}

/// Explicit event used by callers that need to exercise unexpected handling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

sml! {
    SpeechTranscriber<'event> {
        "state_initializing"_s <= *"state_uninitialized"_s + event<EventInitializeRun<'event>> [guard_valid_initialize] / effect_begin_initialize_from_state_uninitialized,
        "state_initialize_error_out_decision"_s <= "state_uninitialized"_s + event<EventInitializeRun<'event>> [guard_invalid_initialize] / effect_reject_initialize_from_state_uninitialized,
        "state_initializing"_s <= "state_ready"_s + event<EventInitializeRun<'event>> [guard_valid_initialize] / effect_begin_initialize_from_state_ready,
        "state_initialize_error_out_decision"_s <= "state_ready"_s + event<EventInitializeRun<'event>> [guard_invalid_initialize] / effect_reject_initialize_from_state_ready,
        "state_initialize_error_out_decision"_s <= "state_errored"_s + event<EventInitializeRun<'event>> / effect_reject_initialize_from_state_errored,
        "state_tokenizer_decision"_s <= "state_initializing"_s + completion<EventInitializeRun>(EventInitializeRun<'event>),
        "state_tokenizer_validation_decision"_s <= "state_tokenizer_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_initialize_tokenizer_supported] / effect_validate_tokenizer_assets,
        "state_initialize_error_out_decision"_s <= "state_tokenizer_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_initialize_tokenizer_unsupported] / effect_mark_tokenizer_invalid_from_state_tokenizer_decision,
        "state_model_support_decision"_s <= "state_tokenizer_validation_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_tokenizer_validation_accepted],
        "state_initialize_error_out_decision"_s <= "state_tokenizer_validation_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_tokenizer_validation_rejected] / effect_mark_tokenizer_invalid_from_state_tokenizer_validation_decision,
        "state_initialize_success"_s <= "state_model_support_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_initialize_model_supported],
        "state_initialize_error_out_decision"_s <= "state_model_support_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_initialize_unsupported_model] / effect_mark_unsupported_model,
        "state_initialize_done_callback_decision"_s <= "state_initialize_success"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_has_initialize_error_out] / effect_store_initialize_success,
        "state_initialize_done_callback_decision"_s <= "state_initialize_success"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_no_initialize_error_out],
        "state_initialize_error_callback_decision"_s <= "state_initialize_error_out_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_has_initialize_error_out] / effect_store_initialize_error,
        "state_initialize_error_callback_decision"_s <= "state_initialize_error_out_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_no_initialize_error_out],
        "state_ready"_s <= "state_initialize_done_callback_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_has_initialize_done_callback] / effect_emit_initialize_done,
        "state_ready"_s <= "state_initialize_done_callback_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_no_initialize_done_callback],
        "state_errored"_s <= "state_initialize_error_callback_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_has_initialize_error_callback] / effect_emit_initialize_error,
        "state_errored"_s <= "state_initialize_error_callback_decision"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [guard_no_initialize_error_callback],
        "state_recognize_support_decision"_s <= "state_ready"_s + event<EventRecognizeRun<'event>> [guard_valid_recognize] / effect_begin_recognize,
        "state_recognize_error_out_decision"_s <= "state_ready"_s + event<EventRecognizeRun<'event>> [guard_invalid_recognize] / effect_reject_recognize,
        "state_recognize_uninitialized_error_out_decision"_s <= "state_uninitialized"_s + event<EventRecognizeRun<'event>> / effect_mark_uninitialized_from_state_uninitialized,
        "state_recognize_errored_error_out_decision"_s <= "state_errored"_s + event<EventRecognizeRun<'event>> / effect_mark_uninitialized_from_state_errored,
        "state_recognize_error_out_decision"_s <= "state_recognize_support_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_transcriber_unsupported] / effect_mark_uninitialized_from_state_recognize_support_decision,
        "state_encoding"_s <= "state_recognize_support_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_transcriber_ready] / effect_encode,
        "state_encoder_decision"_s <= "state_encoding"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>),
        "state_decoding"_s <= "state_encoder_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_encoder_success] / effect_decode,
        "state_recognize_error_out_decision"_s <= "state_encoder_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_encoder_failure] / effect_mark_backend_error_from_state_encoder_decision,
        "state_decoder_decision"_s <= "state_decoding"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>),
        "state_detokenizing"_s <= "state_decoder_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_decoder_success] / effect_detokenize,
        "state_recognize_error_out_decision"_s <= "state_decoder_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_decoder_failure] / effect_mark_backend_error_from_state_decoder_decision,
        "state_detokenize_decision"_s <= "state_detokenizing"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>),
        "state_recognize_success"_s <= "state_detokenize_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_detokenize_success] / effect_publish_recognition_outputs,
        "state_recognize_error_out_decision"_s <= "state_detokenize_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_detokenize_failure] / effect_mark_backend_error_from_state_detokenize_decision,
        "state_recognize_done_callback_decision"_s <= "state_recognize_success"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_out] / effect_store_recognize_success,
        "state_recognize_done_callback_decision"_s <= "state_recognize_success"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_out],
        "state_recognize_error_callback_decision"_s <= "state_recognize_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_error_out_decision,
        "state_recognize_error_callback_decision"_s <= "state_recognize_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_out],
        "state_done"_s <= "state_recognize_done_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_done_callback] / effect_emit_recognize_done,
        "state_done"_s <= "state_recognize_done_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_done_callback],
        "state_ready"_s <= "state_recognize_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_error_callback_decision,
        "state_ready"_s <= "state_recognize_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>),
        "state_recognize_uninitialized_error_callback_decision"_s <= "state_recognize_uninitialized_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_uninitialized_error_out_decision,
        "state_recognize_uninitialized_error_callback_decision"_s <= "state_recognize_uninitialized_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_out],
        "state_uninitialized"_s <= "state_recognize_uninitialized_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_uninitialized_error_callback_decision,
        "state_uninitialized"_s <= "state_recognize_uninitialized_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_callback],
        "state_recognize_errored_error_callback_decision"_s <= "state_recognize_errored_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_errored_error_out_decision,
        "state_recognize_errored_error_callback_decision"_s <= "state_recognize_errored_error_out_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_out],
        "state_errored"_s <= "state_recognize_errored_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_errored_error_callback_decision,
        "state_errored"_s <= "state_recognize_errored_error_callback_decision"_s + completion<EventRecognizeRun>(EventRecognizeRun<'event>) [guard_no_recognize_error_callback],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_on_unexpected_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Mutable request-local context retained by the generated machine.
#[derive(Debug)]
pub struct SpeechTranscriberContext {
    pub deps: Dependencies,
    pub err: TranscriberError,
    pub tokenizer_validation_accepted: bool,
    pub encoder_accepted: bool,
    pub decoder_accepted: bool,
    pub detokenize_accepted: bool,
    pub encoder_frame_count: i32,
    pub encoder_width: i32,
    pub generated_token_count: i32,
    pub selected_token: i32,
    pub confidence: f32,
    pub transcript_size: i32,
    pub encoder_digest: u64,
    pub decoder_digest: u64,
    last_encoder_state_capacity: usize,
}

impl Default for SpeechTranscriberContext {
    fn default() -> Self { Self::new(Dependencies::default()) }
}

impl SpeechTranscriberContext {
    pub fn new(deps: Dependencies) -> Self {
        Self { deps, err: TranscriberError::None, tokenizer_validation_accepted: false, encoder_accepted: false, decoder_accepted: false, detokenize_accepted: false, encoder_frame_count: 0, encoder_width: 0, generated_token_count: 0, selected_token: 0, confidence: 0.0, transcript_size: 0, encoder_digest: 0, decoder_digest: 0, last_encoder_state_capacity: 0 }
    }
    fn reset_initialize(&mut self) { self.err = TranscriberError::None; self.tokenizer_validation_accepted = false; }
    fn reset_recognize(&mut self) { self.err = TranscriberError::None; self.encoder_accepted = false; self.decoder_accepted = false; self.detokenize_accepted = false; self.encoder_frame_count = 0; self.encoder_width = 0; self.generated_token_count = 0; self.selected_token = 0; self.confidence = 0.0; self.transcript_size = 0; self.encoder_digest = 0; self.decoder_digest = 0; }
    fn store_error(&self, out: Option<&Cell<TranscriberError>>) { if let Some(out) = out { out.set(self.err); } }
    fn unexpected(&mut self) -> Result<(), ()> { self.err = TranscriberError::UnexpectedEvent; Ok(()) }
}

impl SpeechTranscriberStateMachineContext for SpeechTranscriberContext {
    fn effect_begin_initialize_from_state_ready(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.reset_initialize(); Ok(()) }
    fn effect_begin_initialize_from_state_uninitialized(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.reset_initialize(); Ok(()) }
    fn effect_begin_recognize(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.reset_recognize(); event.transcript_size_out.set(0); event.generated_token_count_out.set(0); Ok(()) }
    fn effect_decode(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> {
        let Some(callback) = self.deps.decode else { self.decoder_accepted = false; return Ok(()); };
        let state_len = self.encoder_frame_count.max(0) as usize * self.deps.embedding_length;
        let state = event.storage.encoder_state.borrow();
        let state = state.get(..state_len).unwrap_or(&[]);
        let mut tokens = event.storage.generated_tokens.borrow_mut();
        let mut workspace = event.storage.decoder_workspace.borrow_mut();
        let mut logits = event.storage.logits.borrow_mut();
        let mut result = DecoderResult::default();
        self.decoder_accepted = callback(state, self.encoder_frame_count, &mut **tokens, &mut **workspace, &mut **logits, &mut result);
        if self.decoder_accepted { self.generated_token_count = result.generated_token_count; self.selected_token = result.selected_token; self.confidence = result.confidence; self.decoder_digest = result.digest; }
        Ok(())
    }
    fn effect_detokenize(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> {
        let Some(callback) = self.deps.detokenize else { self.detokenize_accepted = false; return Ok(()); };
        let tokens = event.storage.generated_tokens.borrow();
        let count = self.generated_token_count.max(0) as usize;
        let tokens = tokens.get(..count).unwrap_or(&[]);
        let mut transcript = event.transcript.borrow_mut();
        let result = callback(event.tokenizer.model_json, tokens, &mut **transcript);
        self.detokenize_accepted = result.is_some();
        if let Some(size) = result { self.transcript_size = size; }
        Ok(())
    }
    fn effect_emit_initialize_error(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { let _ = callback(InitializeError { error: self.err }); } Ok(()) }
    fn effect_emit_initialize_done(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { let _ = callback(InitializeDone); } Ok(()) }
    fn effect_emit_recognize_done(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { let _ = callback(RecognitionDone { transcript_size: self.transcript_size, selected_token: self.selected_token, confidence: self.confidence, encoder_frame_count: self.encoder_frame_count, encoder_width: self.encoder_width, generated_token_count: self.generated_token_count, encoder_digest: self.encoder_digest, decoder_digest: self.decoder_digest }); } Ok(()) }
    fn effect_emit_recognize_error_from_state_recognize_error_callback_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { let _ = callback(RecognitionError { error: self.err }); } Ok(()) }
    fn effect_emit_recognize_error_from_state_recognize_errored_error_callback_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { let _ = callback(RecognitionError { error: self.err }); } Ok(()) }
    fn effect_emit_recognize_error_from_state_recognize_uninitialized_error_callback_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { let _ = callback(RecognitionError { error: self.err }); } Ok(()) }
    fn effect_encode(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> {
        let Some(callback) = self.deps.encode else { self.encoder_accepted = false; return Ok(()); };
        let mut workspace = event.storage.encoder_workspace.borrow_mut(); let mut state = event.storage.encoder_state.borrow_mut(); let mut result = EncoderResult::default();
        self.encoder_accepted = callback(event.pcm, event.sample_rate, &mut **workspace, &mut **state, &mut result);
        if self.encoder_accepted { self.encoder_frame_count = result.frame_count; self.encoder_width = result.width; self.encoder_digest = result.digest; self.last_encoder_state_capacity = state.len(); }
        Ok(())
    }
    fn effect_mark_backend_error_from_state_decoder_decision(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Backend; Ok(()) }
    fn effect_mark_backend_error_from_state_detokenize_decision(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Backend; Ok(()) }
    fn effect_mark_backend_error_from_state_encoder_decision(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Backend; Ok(()) }
    fn effect_mark_tokenizer_invalid_from_state_tokenizer_decision(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::TokenizerInvalid; Ok(()) }
    fn effect_mark_tokenizer_invalid_from_state_tokenizer_validation_decision(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::TokenizerInvalid; Ok(()) }
    fn effect_mark_uninitialized_from_state_errored(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Uninitialized; Ok(()) }
    fn effect_mark_uninitialized_from_state_recognize_support_decision(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Uninitialized; Ok(()) }
    fn effect_mark_uninitialized_from_state_uninitialized(&mut self, _: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::Uninitialized; Ok(()) }
    fn effect_mark_unsupported_model(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::UnsupportedModel; Ok(()) }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_publish_recognition_outputs(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { event.transcript_size_out.set(self.transcript_size); event.selected_token_out.set(self.selected_token); event.confidence_out.set(self.confidence); event.encoder_frame_count_out.set(self.encoder_frame_count); event.encoder_width_out.set(self.encoder_width); event.encoder_digest_out.set(self.encoder_digest); event.decoder_digest_out.set(self.decoder_digest); event.generated_token_count_out.set(self.generated_token_count); Ok(()) }
    fn effect_reject_initialize_from_state_errored(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::InvalidRequest; Ok(()) }
    fn effect_reject_initialize_from_state_ready(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::InvalidRequest; Ok(()) }
    fn effect_reject_initialize_from_state_uninitialized(&mut self, _: &EventInitializeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::InvalidRequest; Ok(()) }
    fn effect_reject_recognize(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.err = TranscriberError::InvalidRequest; event.transcript_size_out.set(0); event.generated_token_count_out.set(0); Ok(()) }
    fn effect_store_initialize_error(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_store_initialize_success(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_store_recognize_error_from_state_recognize_error_out_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_store_recognize_error_from_state_recognize_errored_error_out_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_store_recognize_error_from_state_recognize_uninitialized_error_out_decision(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_store_recognize_success(&mut self, event: &EventRecognizeRun<'_>) -> Result<(), ()> { self.store_error(event.error_out); Ok(()) }
    fn effect_validate_tokenizer_assets(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> { self.tokenizer_validation_accepted = self.deps.validate_tokenizer.is_some_and(|f| f(event.tokenizer.model_json, event.tokenizer.sha256)); Ok(()) }
    fn guard_decoder_failure(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_decoder_success_impl()) }
    fn guard_decoder_success(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(self.guard_decoder_success_impl()) }
    fn guard_detokenize_failure(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(!self.detokenize_accepted || self.err != TranscriberError::None) }
    fn guard_detokenize_success(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(self.detokenize_accepted && self.err == TranscriberError::None) }
    fn guard_encoder_failure(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_encoder_success_impl()) }
    fn guard_encoder_success(&self, _: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(self.guard_encoder_success_impl()) }
    fn guard_has_initialize_done_callback(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.on_done.is_some()) }
    fn guard_has_initialize_error_callback(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.on_error.is_some()) }
    fn guard_has_initialize_error_out(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.error_out.is_some()) }
    fn guard_has_recognize_done_callback(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.on_done.is_some()) }
    fn guard_has_recognize_error_callback(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.on_error.is_some()) }
    fn guard_has_recognize_error_out(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.error_out.is_some()) }
    fn guard_initialize_model_supported(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.model == self.deps.model_id && self.deps.encoder_supported && self.deps.decoder_supported && self.deps.encoder_model_id == event.model && self.deps.decoder_model_id == event.model && self.deps.embedding_length > 0) }
    fn guard_initialize_tokenizer_supported(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(self.tokenizer_assets_supported(event.tokenizer)) }
    fn guard_initialize_tokenizer_unsupported(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(!self.tokenizer_assets_supported(event.tokenizer)) }
    fn guard_initialize_unsupported_model(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_initialize_model_supported(event)?) }
    fn guard_invalid_initialize(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.tokenizer.model_json.is_empty()) }
    fn guard_invalid_recognize(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.pcm.is_empty() || event.transcript.borrow().is_empty()) }
    fn guard_no_initialize_done_callback(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.on_done.is_none()) }
    fn guard_no_initialize_error_callback(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.on_error.is_none()) }
    fn guard_no_initialize_error_out(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(event.error_out.is_none()) }
    fn guard_no_recognize_done_callback(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.on_done.is_none()) }
    fn guard_no_recognize_error_callback(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.on_error.is_none()) }
    fn guard_no_recognize_error_out(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(event.error_out.is_none()) }
    fn guard_tokenizer_validation_accepted(&self, _: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(self.tokenizer_validation_accepted) }
    fn guard_tokenizer_validation_rejected(&self, _: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(!self.tokenizer_validation_accepted) }
    fn guard_transcriber_ready(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(self.guard_initialize_model_supported_for(event.model) && self.tokenizer_assets_supported(event.tokenizer)) }
    fn guard_transcriber_unsupported(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_transcriber_ready(event)?) }
    fn guard_valid_initialize(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> { Ok(!event.tokenizer.model_json.is_empty()) }
    fn guard_valid_recognize(&self, event: &EventRecognizeRun<'_>) -> Result<bool, ()> { Ok(!event.pcm.is_empty() && !event.transcript.borrow().is_empty()) }
}

impl SpeechTranscriberContext {
    fn tokenizer_assets_supported(&self, assets: TokenizerAssets<'_>) -> bool { self.deps.tokenizer_supported && !assets.model_json.is_empty() && !self.deps.tokenizer_sha256.is_empty() && assets.sha256 == self.deps.tokenizer_sha256 }
    fn guard_initialize_model_supported_for(&self, model: usize) -> bool { model == self.deps.model_id && self.deps.encoder_supported && self.deps.decoder_supported && self.deps.encoder_model_id == model && self.deps.decoder_model_id == model && self.deps.embedding_length > 0 }
    fn guard_encoder_success_impl(&self) -> bool { self.encoder_accepted && self.err == TranscriberError::None && self.encoder_frame_count >= 0 && (self.encoder_frame_count as usize).saturating_mul(self.deps.embedding_length) <= self.last_encoder_state_capacity }
    fn guard_decoder_success_impl(&self) -> bool { self.decoder_accepted && self.err == TranscriberError::None }
}

/// Single-writer synchronous transcriber wrapper.
pub struct SpeechTranscriber {
    machine: SpeechTranscriberStateMachine<SpeechTranscriberContext>,
}

impl SpeechTranscriber {
    #[must_use]
    pub fn new(deps: Dependencies) -> Self { Self { machine: SpeechTranscriberStateMachine::new(SpeechTranscriberContext::new(deps)) } }
    #[must_use]
    pub fn context(&self) -> &SpeechTranscriberContext { self.machine.context() }
    pub fn initialize(&mut self, event: EventInitializeRun<'_>) -> Result<(), TranscriberError> { self.machine.process_event(SpeechTranscriberEvents::EventInitializeRun(event)).map_err(|_| TranscriberError::UnexpectedEvent).and_then(|_| self.result()) }
    pub fn recognize(&mut self, event: EventRecognizeRun<'_>) -> Result<(), TranscriberError> { self.machine.process_event(SpeechTranscriberEvents::EventRecognizeRun(event)).map_err(|_| TranscriberError::UnexpectedEvent).and_then(|_| self.result()) }
    /// Returns the generated state identity for parent-actor inspection.
    #[must_use]
    pub fn state(&self) -> SpeechTranscriberStates { self.machine.state() }
    /// Tests the generated state identity.
    #[must_use]
    pub fn is(&self, state: &SpeechTranscriberStates) -> bool { self.machine.is(state) }
    fn result(&self) -> Result<(), TranscriberError> { match self.context().err { TranscriberError::None => Ok(()), error => Err(error) } }
}

impl Default for SpeechTranscriber {
    fn default() -> Self { Self::new(Dependencies::default()) }
}
