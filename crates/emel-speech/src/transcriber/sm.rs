//! Source-aligned synchronous speech transcriber.
//!
//! The transcriber owns the maintained Whisper encoder, decoder/tokenizer owner,
//! and tokenizer children. Requests and all work buffers remain caller-owned;
//! dispatch runs to completion and callbacks are limited to immediate publish.

#![allow(
    clippy::enum_variant_names,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    missing_debug_implementations,
    dead_code,
    missing_docs
)]

use crate::decoder::whisper::owner::{
    Decode as WhisperDecode, EventDecodeRun as WhisperDecodeEvent, SpeechWhisperOwnerActor,
    WhisperOwnerError, WhisperOwnerEvent,
};
use crate::decoder::whisper::sm::{DecodePolicy, WhisperDecoderError, bind_execution_contract};
use crate::encoder::whisper::sm::{
    Error as WhisperEncoderError, EventEncodeRun as WhisperEncodeEvent, ExecutionContract,
    SpeechEncoderWhisperActor,
};
use crate::tokenizer::whisper::sm::{
    SpeechTokenizerWhisper, TokenizerError, Validate, tiny_asr_decode_policy,
};
use emel_model::bridge::Data;
use std::cell::{Cell, RefCell};

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
    /// A maintained child exists but its numeric implementation is unavailable.
    UnsupportedDependency = 9,
}

/// Public lifecycle states.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpeechTranscriberStates {
    #[default]
    StateUninitialized,
    StateReady,
    StateErrored,
}

/// Bounded tokenizer assets copied into a request view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizerAssets<'a> {
    pub model_json: &'a str,
    pub sha256: &'a str,
}

/// Caller-owned intermediate buffers used by recognition.
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

/// Component identity and support metadata. Numeric execution belongs to the
/// maintained child actors rather than to injected transcriber callbacks.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dependencies {
    pub encoder_supported: bool,
    pub decoder_supported: bool,
    pub tokenizer_supported: bool,
    pub model_id: usize,
    pub encoder_model_id: usize,
    pub decoder_model_id: usize,
    pub embedding_length: usize,
    pub tokenizer_sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeDone;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeError {
    pub error: TranscriberError,
}
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecognitionError {
    pub error: TranscriberError,
}
pub type InitializeDoneFn = fn(InitializeDone) -> bool;
pub type InitializeErrorFn = fn(InitializeError) -> bool;
pub type RecognitionDoneFn = fn(RecognitionDone) -> bool;
pub type RecognitionErrorFn = fn(RecognitionError) -> bool;

/// Initialization request. `model_data` is borrowed for synchronous validation.
pub struct EventInitializeRun<'a> {
    pub model: usize,
    pub model_data: Option<&'a Data>,
    pub tokenizer: TokenizerAssets<'a>,
    pub error_out: Option<&'a Cell<TranscriberError>>,
    pub on_done: Option<InitializeDoneFn>,
    pub on_error: Option<InitializeErrorFn>,
}

/// Recognition request and caller-owned result channels.
pub struct EventRecognizeRun<'a> {
    pub model: usize,
    pub model_data: Option<&'a Data>,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

/// Mutable transcriber context and owned child actors.
pub struct SpeechTranscriberContext {
    pub deps: Dependencies,
    pub err: TranscriberError,
    pub state: SpeechTranscriberStates,
    pub encoder_frame_count: i32,
    pub encoder_width: i32,
    pub generated_token_count: i32,
    pub selected_token: i32,
    pub confidence: f32,
    pub transcript_size: i32,
    pub encoder_digest: u64,
    pub decoder_digest: u64,
    encoder: SpeechEncoderWhisperActor,
    whisper: SpeechWhisperOwnerActor,
    tokenizer: SpeechTokenizerWhisper,
}

impl SpeechTranscriberContext {
    #[must_use]
    pub fn new(deps: Dependencies) -> Self {
        Self {
            deps,
            err: TranscriberError::None,
            state: SpeechTranscriberStates::StateUninitialized,
            encoder_frame_count: 0,
            encoder_width: 0,
            generated_token_count: 0,
            selected_token: 0,
            confidence: 0.0,
            transcript_size: 0,
            encoder_digest: 0,
            decoder_digest: 0,
            encoder: SpeechEncoderWhisperActor::new(),
            whisper: SpeechWhisperOwnerActor::new(),
            tokenizer: SpeechTokenizerWhisper::new(),
        }
    }
    fn clear_outputs(&mut self) {
        self.encoder_frame_count = 0;
        self.encoder_width = 0;
        self.generated_token_count = 0;
        self.selected_token = 0;
        self.confidence = 0.0;
        self.transcript_size = 0;
        self.encoder_digest = 0;
        self.decoder_digest = 0;
    }
    fn fail(&mut self, error: TranscriberError) -> Result<(), TranscriberError> {
        self.err = error;
        self.state = SpeechTranscriberStates::StateErrored;
        Err(error)
    }
    fn tokenizer_ok(&mut self, assets: TokenizerAssets<'_>) -> bool {
        self.deps.tokenizer_supported
            && assets.sha256 == self.deps.tokenizer_sha256
            && self
                .tokenizer
                .process_validate(Validate::new(assets.model_json, tiny_asr_decode_policy()))
                .is_ok()
    }
}

impl Default for SpeechTranscriberContext {
    fn default() -> Self {
        Self::new(Dependencies::default())
    }
}

/// Single-writer synchronous transcriber wrapper.
pub struct SpeechTranscriber {
    context: SpeechTranscriberContext,
}

impl SpeechTranscriber {
    #[must_use]
    pub fn new(deps: Dependencies) -> Self {
        Self {
            context: SpeechTranscriberContext::new(deps),
        }
    }
    #[must_use]
    pub fn context(&self) -> &SpeechTranscriberContext {
        &self.context
    }
    #[must_use]
    pub fn state(&self) -> SpeechTranscriberStates {
        self.context.state
    }
    #[must_use]
    pub fn is(&self, state: SpeechTranscriberStates) -> bool {
        self.context.state == state
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn initialize(&mut self, event: EventInitializeRun<'_>) -> Result<(), TranscriberError> {
        let _ = if event.tokenizer.model_json.is_empty()
            || self.context.state == SpeechTranscriberStates::StateErrored
        {
            self.context.fail(TranscriberError::InvalidRequest)
        } else if !self.context.tokenizer_ok(event.tokenizer) {
            self.context.fail(TranscriberError::TokenizerInvalid)
        } else if event.model != self.context.deps.model_id
            || event.model != self.context.deps.encoder_model_id
            || event.model != self.context.deps.decoder_model_id
            || !self.context.deps.encoder_supported
            || !self.context.deps.decoder_supported
            || event
                .model_data
                .is_none_or(|model| !ExecutionContract::bind_model(model).model_contract_valid())
        {
            self.context.fail(TranscriberError::UnsupportedModel)
        } else {
            self.context.err = TranscriberError::None;
            self.context.state = SpeechTranscriberStates::StateReady;
            if let Some(out) = event.error_out {
                out.set(TranscriberError::None);
            }
            if let Some(done) = event.on_done {
                let _ = done(InitializeDone);
            }
            Ok(())
        };
        let result = if self.context.err == TranscriberError::None {
            Ok(())
        } else {
            Err(self.context.err)
        };
        if let Err(error) = result {
            if let Some(out) = event.error_out {
                out.set(error);
            }
            if let Some(callback) = event.on_error {
                let _ = callback(InitializeError { error });
            }
            Err(error)
        } else {
            Ok(())
        }
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub fn recognize(&mut self, event: EventRecognizeRun<'_>) -> Result<(), TranscriberError> {
        if self.context.state != SpeechTranscriberStates::StateReady {
            return self.context.fail(TranscriberError::Uninitialized);
        }
        self.context.err = TranscriberError::None;
        self.context.clear_outputs();
        event.transcript_size_out.set(0);
        event.generated_token_count_out.set(0);
        let Some(model) = event.model_data else {
            return self.publish_error(&event, TranscriberError::UnsupportedDependency);
        };
        let contract = ExecutionContract::bind_model(model);
        let mut frames = 0;
        let mut width = 0;
        let mut encoder_digest = 0;
        let encoded = {
            let mut workspace = event.storage.encoder_workspace.borrow_mut();
            let mut state = event.storage.encoder_state.borrow_mut();
            self.context.encoder.process_event(
                WhisperEncodeEvent::new(
                    &contract,
                    event.pcm,
                    event.sample_rate,
                    event.channel_count,
                    &mut workspace,
                    &mut state,
                    &mut frames,
                    &mut width,
                    &mut encoder_digest,
                )
                .with_model(model),
            )
        };
        if !encoded {
            let error = match self.context.encoder.context().err {
                WhisperEncoderError::ModelInvalid => TranscriberError::ModelInvalid,
                WhisperEncoderError::OutputCapacity | WhisperEncoderError::WorkspaceCapacity => {
                    TranscriberError::OutputCapacity
                }
                WhisperEncoderError::UnsupportedVariant => TranscriberError::UnsupportedModel,
                WhisperEncoderError::InternalError => TranscriberError::UnsupportedDependency,
                _ => TranscriberError::Backend,
            };
            return self.publish_error(&event, error);
        }
        self.context.encoder_frame_count = frames;
        self.context.encoder_width = width;
        self.context.encoder_digest = encoder_digest;
        let state_len = usize::try_from(frames)
            .ok()
            .and_then(|n| {
                usize::try_from(contract.embedding_length)
                    .ok()
                    .and_then(|length| n.checked_mul(length))
            })
            .ok_or(TranscriberError::OutputCapacity)?;
        let state = event.storage.encoder_state.borrow();
        let state = state
            .get(..state_len)
            .ok_or(TranscriberError::OutputCapacity)?;
        let mut generated_count = 0;
        let mut token = 0;
        let mut confidence = 0.0;
        let mut decoder_digest = 0;
        let mut transcript_size = 0;
        let decoded = {
            let mut generated = event.storage.generated_tokens.borrow_mut();
            let mut workspace = event.storage.decoder_workspace.borrow_mut();
            let mut logits = event.storage.logits.borrow_mut();
            let mut transcript = event.transcript.borrow_mut();
            self.context
                .whisper
                .process_event(WhisperOwnerEvent::Decode(WhisperDecodeEvent::new(
                    WhisperDecode::new(
                        model,
                        bind_execution_contract(model),
                        state,
                        frames,
                        DecodePolicy::tiny_asr(),
                        &mut generated,
                        &mut generated_count,
                        &mut workspace,
                        &mut logits,
                        &mut token,
                        &mut confidence,
                        &mut decoder_digest,
                        event.tokenizer.model_json,
                        &mut transcript,
                        &mut transcript_size,
                    ),
                )))
        };
        if let Err(error) = decoded {
            let error = match error {
                WhisperOwnerError::Tokenizer(TokenizerError::TranscriptCapacity) => {
                    TranscriberError::OutputCapacity
                }
                WhisperOwnerError::Tokenizer(_) => TranscriberError::TokenizerInvalid,
                WhisperOwnerError::Decoder(WhisperDecoderError::ModelInvalid) => {
                    TranscriberError::ModelInvalid
                }
                WhisperOwnerError::Decoder(
                    WhisperDecoderError::GeneratedTokenCapacity
                    | WhisperDecoderError::LogitsCapacity
                    | WhisperDecoderError::WorkspaceCapacity,
                ) => TranscriberError::OutputCapacity,
                WhisperOwnerError::Decoder(WhisperDecoderError::InternalError) => {
                    TranscriberError::UnsupportedDependency
                }
                _ => TranscriberError::Backend,
            };
            return self.publish_error(&event, error);
        }
        self.context.generated_token_count = generated_count;
        self.context.selected_token = token;
        self.context.confidence = confidence;
        self.context.decoder_digest = decoder_digest;
        self.context.transcript_size = transcript_size;
        event.transcript_size_out.set(transcript_size);
        event.selected_token_out.set(token);
        event.confidence_out.set(confidence);
        event.encoder_frame_count_out.set(frames);
        event.encoder_width_out.set(width);
        event.encoder_digest_out.set(encoder_digest);
        event.decoder_digest_out.set(decoder_digest);
        event.generated_token_count_out.set(generated_count);
        if let Some(done) = event.on_done {
            let _ = done(RecognitionDone {
                transcript_size,
                selected_token: token,
                confidence,
                encoder_frame_count: frames,
                encoder_width: width,
                generated_token_count: generated_count,
                encoder_digest,
                decoder_digest,
            });
        }
        Ok(())
    }

    pub fn process_unexpected_event(&mut self) -> Result<(), TranscriberError> {
        self.context.state = SpeechTranscriberStates::StateErrored;
        self.context.err = TranscriberError::UnexpectedEvent;
        Err(TranscriberError::UnexpectedEvent)
    }

    fn publish_error(
        &mut self,
        event: &EventRecognizeRun<'_>,
        error: TranscriberError,
    ) -> Result<(), TranscriberError> {
        self.context.err = error;
        self.context.state = SpeechTranscriberStates::StateErrored;
        if let Some(out) = event.error_out {
            out.set(error);
        }
        if let Some(callback) = event.on_error {
            let _ = callback(RecognitionError { error });
        }
        Err(error)
    }
}

impl Default for SpeechTranscriber {
    fn default() -> Self {
        Self::new(Dependencies::default())
    }
}
