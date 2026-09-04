//! Maintained speech-owned Whisper decoder/tokenizer owner boundary.
//!
//! The owner composes the existing synchronous Whisper decoder and tokenizer
//! actors. Requests and storage are caller-owned for one dispatch. Numeric
//! decoder execution remains an explicit caller-provided callback.

use core::cell::RefCell;
use core::fmt;

use emel_model::bridge::Data;

use super::sm::{
    DecodeKernelFn, DecodePolicy, DecodeRequest, EventDecodeRun as DecoderEvent,
    SpeechDecoderWhisperActor, WhisperDecoderError, WhisperExecutionContract,
    bind_execution_contract,
};
use crate::tokenizer::whisper::sm::{
    AsrDecodePolicy, Detokenize, SpeechTokenizerWhisper, TokenizerError, Validate,
};

/// Typed failure from the Whisper owner boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WhisperOwnerError {
    /// No error occurred.
    #[default]
    None,
    /// Decoder child failure.
    Decoder(WhisperDecoderError),
    /// Tokenizer child failure.
    Tokenizer(TokenizerError),
    /// A request was delivered before reset after a failed dispatch.
    Busy,
    /// Event was invalid at this boundary.
    UnexpectedEvent,
}

/// Lifecycle state of the single-writer Whisper owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhisperOwnerState {
    /// The owner accepts a new decode request.
    Ready,
    /// The last request returned a typed failure.
    Errored,
    /// An unexpected event was observed.
    Unexpected,
}

/// Short state spelling for component-oriented callers.
pub type State = WhisperOwnerState;

/// Restores the owner to its initial lifecycle state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reset;

impl Reset {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl fmt::Display for WhisperOwnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("no error"),
            Self::Decoder(error) => write!(f, "Whisper decoder error: {error:?}"),
            Self::Tokenizer(error) => write!(f, "Whisper tokenizer error: {error}"),
            Self::Busy => f.write_str("Whisper owner is busy after an error"),
            Self::UnexpectedEvent => f.write_str("unexpected Whisper owner event"),
        }
    }
}

/// Copied successful result; no request borrow escapes dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhisperOwnerDone {
    /// Selected decoder token.
    pub token: i32,
    /// Selected confidence represented as bits.
    pub confidence_bits: u32,
    /// Caller-defined decoder digest.
    pub digest: u64,
    /// Transcript byte count.
    pub transcript_size: i32,
}

/// Copied failed result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhisperOwnerErrorEvent {
    /// Typed owner error.
    pub error: WhisperOwnerError,
}

/// Source-neutral success spelling for owner integrations.
pub type DecodeDone = WhisperOwnerDone;
/// Source-neutral failure spelling for owner integrations.
pub type DecodeError = WhisperOwnerErrorEvent;

/// Synchronous completion callback.
pub type DoneCallback = fn(WhisperOwnerDone) -> bool;
/// Synchronous failure callback.
pub type ErrorCallback = fn(WhisperOwnerErrorEvent) -> bool;

/// Caller-owned decode and detokenize request.
#[derive(Debug)]
pub struct Decode<'a> {
    /// Immutable model bridge data.
    pub model: &'a Data,
    /// Bound Whisper model dimensions.
    pub contract: WhisperExecutionContract,
    /// Caller-owned encoder output.
    pub encoder_state: &'a [f32],
    /// Encoder frame count.
    pub encoder_frame_count: i32,
    /// Explicit decoder policy.
    pub policy: DecodePolicy,
    /// Generated token destination.
    pub generated_tokens: RefCell<&'a mut [i32]>,
    /// Generated token count destination.
    pub generated_token_count_out: RefCell<&'a mut i32>,
    /// Decoder workspace.
    pub workspace: RefCell<&'a mut [f32]>,
    /// Logits destination.
    pub logits: RefCell<&'a mut [f32]>,
    /// Selected token destination.
    pub token_out: RefCell<&'a mut i32>,
    /// Confidence destination.
    pub confidence_out: RefCell<&'a mut f32>,
    /// Digest destination.
    pub digest_out: RefCell<&'a mut u64>,
    /// Whisper tokenizer JSON asset.
    pub tokenizer_json: &'a str,
    /// Transcript destination.
    pub transcript: RefCell<&'a mut [u8]>,
    /// Transcript size destination.
    pub transcript_size_out: RefCell<&'a mut i32>,
    /// Optional owner error destination.
    pub error_out: RefCell<Option<&'a mut WhisperOwnerError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<DoneCallback>,
    /// Optional synchronous failure callback.
    pub on_error: Option<ErrorCallback>,
    /// Explicit caller-owned numeric decoder callback.
    pub decode: Option<DecodeKernelFn>,
}

impl<'a> Decode<'a> {
    /// Creates a request over caller-owned state and storage.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &'a Data,
        contract: WhisperExecutionContract,
        encoder_state: &'a [f32],
        encoder_frame_count: i32,
        policy: DecodePolicy,
        generated_tokens: &'a mut [i32],
        generated_token_count_out: &'a mut i32,
        workspace: &'a mut [f32],
        logits: &'a mut [f32],
        token_out: &'a mut i32,
        confidence_out: &'a mut f32,
        digest_out: &'a mut u64,
        tokenizer_json: &'a str,
        transcript: &'a mut [u8],
        transcript_size_out: &'a mut i32,
    ) -> Self {
        Self {
            model,
            contract,
            encoder_state,
            encoder_frame_count,
            policy,
            generated_tokens: RefCell::new(generated_tokens),
            generated_token_count_out: RefCell::new(generated_token_count_out),
            workspace: RefCell::new(workspace),
            logits: RefCell::new(logits),
            token_out: RefCell::new(token_out),
            confidence_out: RefCell::new(confidence_out),
            digest_out: RefCell::new(digest_out),
            tokenizer_json,
            transcript: RefCell::new(transcript),
            transcript_size_out: RefCell::new(transcript_size_out),
            error_out: RefCell::new(None),
            on_done: None,
            on_error: None,
            decode: None,
        }
    }

    /// Sets an optional owner error destination.
    #[must_use]
    pub fn with_error_out(mut self, out: &'a mut WhisperOwnerError) -> Self {
        *self.error_out.get_mut() = Some(out);
        self
    }

    /// Sets synchronous completion callbacks.
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        done: Option<DoneCallback>,
        error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = done;
        self.on_error = error;
        self
    }

    /// Sets the explicit caller-owned numeric callback.
    #[must_use]
    pub const fn with_decoder(mut self, decode: DecodeKernelFn) -> Self {
        self.decode = Some(decode);
        self
    }
}

/// Typed decode event accepted by the owner.
#[derive(Debug)]
pub struct EventDecodeRun<'a> {
    /// Caller-owned request.
    pub request: Decode<'a>,
}

impl<'a> EventDecodeRun<'a> {
    /// Wraps one request as an event.
    #[must_use]
    pub const fn new(request: Decode<'a>) -> Self {
        Self { request }
    }
}

/// Explicit unexpected event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedEvent;

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum WhisperOwnerEvent<'a> {
    /// Decode and detokenize one request synchronously.
    Decode(EventDecodeRun<'a>),
    /// Restore the owner and both child actors to their ready states.
    Reset(Reset),
    /// Exercise unexpected-event handling.
    Unexpected(UnexpectedEvent),
}

#[allow(missing_debug_implementations)]
pub struct SpeechWhisperOwnerActor {
    decoder: SpeechDecoderWhisperActor,
    tokenizer: SpeechTokenizerWhisper,
    error: WhisperOwnerError,
    state: WhisperOwnerState,
    unexpected_count: u64,
}

impl Default for SpeechWhisperOwnerActor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechWhisperOwnerActor {
    /// Creates a ready owner actor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decoder: SpeechDecoderWhisperActor::new(),
            tokenizer: SpeechTokenizerWhisper::new(),
            error: WhisperOwnerError::None,
            state: WhisperOwnerState::Ready,
            unexpected_count: 0,
        }
    }
    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns the typed child or lifecycle error selected while processing the event.
    pub fn process_event(&mut self, event: WhisperOwnerEvent<'_>) -> Result<(), WhisperOwnerError> {
        match event {
            WhisperOwnerEvent::Decode(event) => self.process_decode(event),
            WhisperOwnerEvent::Reset(_) => {
                self.reset();
                Ok(())
            }
            WhisperOwnerEvent::Unexpected(_) => self.process_unexpected_event(),
        }
    }

    /// Dispatches one decode/tokenize request synchronously.
    ///
    /// # Errors
    ///
    /// Returns the typed decoder or tokenizer error selected while processing the request.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_decode(&mut self, event: EventDecodeRun<'_>) -> Result<(), WhisperOwnerError> {
        if self.state != WhisperOwnerState::Ready {
            self.error = WhisperOwnerError::Busy;
            return Err(WhisperOwnerError::Busy);
        }
        self.error = WhisperOwnerError::None;
        **event.request.generated_token_count_out.borrow_mut() = 0;
        **event.request.transcript_size_out.borrow_mut() = 0;
        let bound_contract = bind_execution_contract(event.request.model);
        if event.request.contract != bound_contract || !bound_contract.model_contract_valid() {
            return self.fail(
                &event.request,
                WhisperOwnerError::Decoder(WhisperDecoderError::ModelInvalid),
            );
        }
        let tokenizer_policy = as_tokenizer_policy(event.request.policy);
        if let Err(error) = self.tokenizer.process_validate(Validate::new(
            event.request.tokenizer_json,
            &tokenizer_policy,
        )) {
            return self.fail(&event.request, WhisperOwnerError::Tokenizer(error));
        }
        let result = {
            let mut generated = event.request.generated_tokens.borrow_mut();
            let mut generated_count = event.request.generated_token_count_out.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            let mut logits = event.request.logits.borrow_mut();
            let mut token = event.request.token_out.borrow_mut();
            let mut confidence = event.request.confidence_out.borrow_mut();
            let mut digest = event.request.digest_out.borrow_mut();
            let child = DecodeRequest::new(
                event.request.model,
                event.request.contract,
                event.request.encoder_state,
                event.request.encoder_frame_count,
                event.request.policy,
                &mut generated,
                &mut generated_count,
                &mut workspace,
                &mut logits,
                &mut token,
                &mut confidence,
                &mut digest,
            );
            let child = match event.request.decode {
                Some(decode) => child.with_decoder(decode),
                None => child,
            };
            self.decoder.process_event(DecoderEvent::new(child))
        };
        if let Err(error) = result {
            return self.fail(&event.request, WhisperOwnerError::Decoder(error));
        }
        let generated = event.request.generated_tokens.borrow();
        let count = **event.request.generated_token_count_out.borrow();
        let Ok(count) = usize::try_from(count) else {
            return self.fail(
                &event.request,
                WhisperOwnerError::Decoder(WhisperDecoderError::InternalError),
            );
        };
        if count > generated.len() {
            return self.fail(
                &event.request,
                WhisperOwnerError::Decoder(WhisperDecoderError::InternalError),
            );
        }
        let result = {
            let mut transcript = event.request.transcript.borrow_mut();
            let mut size = event.request.transcript_size_out.borrow_mut();
            self.tokenizer.process_detokenize(Detokenize::new(
                event.request.tokenizer_json,
                &generated[..count],
                &mut transcript,
                &mut size,
            ))
        };
        if let Err(error) = result {
            return self.fail(&event.request, WhisperOwnerError::Tokenizer(error));
        }
        self.error = WhisperOwnerError::None;
        self.state = WhisperOwnerState::Ready;
        if let Some(out) = event.request.error_out.borrow_mut().as_deref_mut() {
            *out = WhisperOwnerError::None;
        }
        if let Some(done) = event.request.on_done {
            let _ = done(WhisperOwnerDone {
                token: **event.request.token_out.borrow(),
                confidence_bits: (**event.request.confidence_out.borrow()).to_bits(),
                digest: **event.request.digest_out.borrow(),
                transcript_size: **event.request.transcript_size_out.borrow(),
            });
        }
        Ok(())
    }

    /// Reports an explicit unexpected event synchronously.
    ///
    /// # Errors
    ///
    /// Always returns [`WhisperOwnerError::UnexpectedEvent`].
    pub const fn process_unexpected_event(&mut self) -> Result<(), WhisperOwnerError> {
        self.error = WhisperOwnerError::UnexpectedEvent;
        self.state = WhisperOwnerState::Unexpected;
        self.unexpected_count = self.unexpected_count.saturating_add(1);
        Err(WhisperOwnerError::UnexpectedEvent)
    }

    /// Restores both child actors and this owner to the ready state.
    pub fn reset(&mut self) {
        self.decoder = SpeechDecoderWhisperActor::new();
        self.tokenizer = SpeechTokenizerWhisper::new();
        self.error = WhisperOwnerError::None;
        self.state = WhisperOwnerState::Ready;
    }

    /// Returns the current owner lifecycle state.
    #[must_use]
    pub const fn state(&self) -> WhisperOwnerState {
        self.state
    }
    /// Returns the most recent owner error.
    #[must_use]
    pub const fn error(&self) -> WhisperOwnerError {
        self.error
    }
    /// Returns the unexpected-event count.
    #[must_use]
    pub const fn unexpected_count(&self) -> u64 {
        self.unexpected_count
    }
    /// Returns the child decoder for focused lifecycle inspection.
    #[must_use]
    pub const fn decoder(&self) -> &SpeechDecoderWhisperActor {
        &self.decoder
    }
    /// Returns the child tokenizer for focused lifecycle inspection.
    #[must_use]
    pub const fn tokenizer(&self) -> &SpeechTokenizerWhisper {
        &self.tokenizer
    }

    fn fail(
        &mut self,
        request: &Decode<'_>,
        error: WhisperOwnerError,
    ) -> Result<(), WhisperOwnerError> {
        self.error = error;
        self.state = WhisperOwnerState::Errored;
        if let Some(out) = request.error_out.borrow_mut().as_deref_mut() {
            *out = error;
        }
        if let Some(callback) = request.on_error {
            let _ = callback(WhisperOwnerErrorEvent { error });
        }
        Err(error)
    }
}

const fn as_tokenizer_policy(policy: DecodePolicy) -> AsrDecodePolicy {
    AsrDecodePolicy {
        tokens: crate::tokenizer::whisper::sm::ControlTokens {
            eot: policy.tokens.eot,
            sot: policy.tokens.sot,
            language_en: policy.tokens.language_en,
            translate: policy.tokens.translate,
            transcribe: policy.tokens.transcribe,
            no_speech: policy.tokens.no_speech,
            notimestamps: policy.tokens.notimestamps,
            timestamp_begin: policy.tokens.timestamp_begin,
            space: policy.tokens.space,
        },
        language: crate::tokenizer::whisper::sm::LanguageRole::English,
        task: crate::tokenizer::whisper::sm::TaskRole::Transcribe,
        timestamps: crate::tokenizer::whisper::sm::TimestampMode::TimestampTokens,
        suppress_translate: policy.suppress_translate,
        prompt_tokens: policy.prompt_tokens,
    }
}

/// Short model-domain-friendly alias.
pub type WhisperOwner = SpeechWhisperOwnerActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_model_is_rejected_before_tokenizer_validation() {
        let model = Data::try_new().expect("bounded model storage");
        let mut generated = [0; 4];
        let mut generated_count = -1;
        let mut workspace = [0.0; 8];
        let mut logits = [0.0; 8];
        let mut token = -1;
        let mut confidence = -1.0;
        let mut digest = u64::MAX;
        let mut transcript = [0; 8];
        let mut transcript_size = -1;
        let request = Decode::new(
            &model,
            bind_execution_contract(&model),
            &[0.0; 384],
            1,
            DecodePolicy::tiny_asr(),
            &mut generated,
            &mut generated_count,
            &mut workspace,
            &mut logits,
            &mut token,
            &mut confidence,
            &mut digest,
            "",
            &mut transcript,
            &mut transcript_size,
        );
        let mut actor = SpeechWhisperOwnerActor::new();
        assert_eq!(
            actor.process_event(WhisperOwnerEvent::Decode(EventDecodeRun::new(request))),
            Err(WhisperOwnerError::Decoder(
                WhisperDecoderError::ModelInvalid
            ))
        );
        assert_eq!(generated_count, 0);
        assert_eq!(transcript_size, 0);
    }

    #[test]
    fn owner_rejects_contract_not_bound_to_request_model() {
        let model = Data::try_new().expect("bounded model storage");
        let mut generated = [0; 4];
        let mut generated_count = -1;
        let mut workspace = [0.0; 8];
        let mut logits = [0.0; 8];
        let mut token = -1;
        let mut confidence = -1.0;
        let mut digest = u64::MAX;
        let mut transcript = [0; 8];
        let mut transcript_size = -1;
        let mut contract = WhisperExecutionContract::bind();
        contract.vocab_size -= 1;
        let request = Decode::new(
            &model,
            contract,
            &[0.0; 384],
            1,
            DecodePolicy::tiny_asr(),
            &mut generated,
            &mut generated_count,
            &mut workspace,
            &mut logits,
            &mut token,
            &mut confidence,
            &mut digest,
            "",
            &mut transcript,
            &mut transcript_size,
        );
        let mut actor = SpeechWhisperOwnerActor::new();
        assert_eq!(
            actor.process_event(WhisperOwnerEvent::Decode(EventDecodeRun::new(request))),
            Err(WhisperOwnerError::Decoder(
                WhisperDecoderError::ModelInvalid
            ))
        );
        assert_eq!(generated_count, 0);
        assert_eq!(transcript_size, 0);
    }

    #[test]
    fn owner_failure_requires_reset_before_next_request() {
        let mut actor = SpeechWhisperOwnerActor::new();
        assert_eq!(actor.state(), WhisperOwnerState::Ready);
        actor.process_unexpected_event().unwrap_err();
        assert_eq!(actor.state(), WhisperOwnerState::Unexpected);
        assert_eq!(
            actor.process_unexpected_event(),
            Err(WhisperOwnerError::UnexpectedEvent)
        );
        assert_eq!(
            actor.process_event(WhisperOwnerEvent::Reset(Reset::new())),
            Ok(())
        );
        assert_eq!(actor.state(), WhisperOwnerState::Ready);
        assert_eq!(actor.error(), WhisperOwnerError::None);
    }
}
