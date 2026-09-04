//! Speech-model components, including encoders and codec support.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// The sampled audio format supplied to speech components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioFormat {
    /// Samples per second.
    pub sample_rate_hz: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}
pub mod codec;
pub mod decoder;
pub mod encoder;
pub mod generator;
pub mod predictor;
pub mod tokenizer;
pub mod transcriber;

/// Public Mimi model-binding and prepared-runtime contracts.
pub use codec::mimi::binding as mimi;

pub use codec::mimi::sm::InitializeDone as MimiInitializeDone;
/// Maintained top-level Mimi streaming codec facade and typed events.
pub use codec::mimi::{
    DecodeRun as MimiDecodeRun, EncodeRun as MimiEncodeRun,
    EventCaptureDiagnostics as MimiDiagnosticsEvent, EventDecodeRun as MimiDecodeEvent,
    EventEncodeRun as MimiEncodeEvent, EventInitRun as MimiInitEvent,
    EventResetStreamRun as MimiResetStreamEvent, InitRun as MimiInitRun, Mimi,
    MimiDecoderBackendStage, MimiDecoderStreamingState, MimiDecoderTransformerStage,
    MimiDiagnostics, MimiEncoderRuntime, MimiEncoderStage, MimiEncoderStreamingState, MimiError,
    MimiEvent, MimiQuantizerRuntime, SpeechCodecMimi,
};

/// Maintained speech generator actors and typed request/result contracts.
pub use generator::{
    ConditionPhase as SpeechGeneratorConditionPhase,
    ConditionResult as SpeechGeneratorConditionResult, ConditionRun as SpeechGeneratorConditionRun,
    EventReset as SpeechGeneratorResetEvent, FlushRun as SpeechGeneratorFlushRun,
    FramePhase as SpeechGeneratorFramePhase, FrameResult as SpeechGeneratorFrameResult,
    GenerateRun as SpeechGeneratorGenerateRun, InitPhase as SpeechGeneratorInitPhase,
    InitRun as SpeechGeneratorInitRun, SpeechGeneratorDuplexModelActor,
    SpeechGeneratorDuplexModelContext, SpeechGeneratorDuplexModelStates, SpeechGeneratorError,
    SpeechGeneratorSynthesisModelActor, SpeechGeneratorSynthesisModelContext,
    SpeechGeneratorSynthesisModelStates, StreamRun as SpeechGeneratorStreamRun,
};

/// Maintained Moshi predictor actor and typed request/result contracts.
pub use predictor::moshi::{
    MoshiTextEmbeddingBinding, MoshiTextEmbeddingBindingError, MoshiTextEmbeddingRow,
    SpeechPredictorMoshiActor, SpeechPredictorMoshiContext, SpeechPredictorMoshiStates,
    TEXT_EMBEDDING_TENSOR,
};

/// Maintained Whisper encoder actor and typed request/result contracts.
pub use encoder::whisper::sm::{
    EncodeDone as WhisperEncodeDone, EncodeError as WhisperEncodeError,
    EncoderError as WhisperEncoderError, EventEncodeRun as WhisperEncodeEvent,
    ExecutionContract as WhisperEncoderExecutionContract, ModelAssets as WhisperEncoderModelAssets,
    SpeechEncoderWhisperActor, SpeechEncoderWhisperStates, WeightVariant as WhisperWeightVariant,
    encoder_frame_count as whisper_encoder_frame_count, mel_frame_count as whisper_mel_frame_count,
    required_encoder_output_floats as whisper_required_encoder_output_floats,
    required_workspace_floats as whisper_encoder_required_workspace_floats,
};

/// Maintained Whisper decoder actor and typed request/result contracts.
pub use decoder::whisper::sm::{
    ControlTokens as WhisperDecoderControlTokens, DecodeDone as WhisperDecoderDone,
    DecodeError as WhisperDecoderErrorEvent, DecodeKernelRequest as WhisperDecodeKernelRequest,
    DecodeRequest as WhisperDecodeRequest, DecodeVariant as WhisperDecodeVariant,
    EventDecodeRun as WhisperDecoderEvent, LanguageRole as WhisperDecoderLanguageRole,
    SpeechDecoderWhisperActor, SpeechDecoderWhisperStates, TaskRole as WhisperDecoderTaskRole,
    TimestampMode as WhisperDecoderTimestampMode, WhisperDecoderError,
    WhisperExecutionContract as WhisperDecoderExecutionContract,
    bind_execution_contract as bind_whisper_decoder_execution_contract,
    max_generated_token_count as whisper_max_generated_token_count,
    required_workspace_floats as whisper_decoder_required_workspace_floats,
    vocab_size as whisper_vocab_size,
};

/// Maintained Whisper tokenizer actor and typed request/result contracts.
pub use tokenizer::whisper::sm::{
    AsrDecodePolicy as WhisperAsrDecodePolicy, ControlTokens as WhisperTokenizerControlTokens,
    Detokenize as WhisperDetokenize, DetokenizeDone as WhisperDetokenizeDone,
    DetokenizeError as WhisperDetokenizeError, EventDetokenizeRun as WhisperDetokenizeEvent,
    EventValidateRun as WhisperValidateEvent, LanguageRole as WhisperTokenizerLanguageRole,
    SpeechTokenizerWhisper, SpeechTokenizerWhisperStates, TINY_ASR_DECODE_POLICY,
    TINY_CONTROL_TOKENS, TINY_TOKENIZER_SHA256, TaskRole as WhisperTokenizerTaskRole,
    TimestampMode as WhisperTokenizerTimestampMode, TokenizerError as WhisperTokenizerError,
    Validate as WhisperTokenizerValidate, WhisperTokenizerEvent,
    is_tiny_asr_decode_policy_supported, language_role_name, required_transcript_capacity,
    task_role_name, timestamp_mode_name, tiny_asr_decode_policy, tiny_tokenizer_sha256,
};

/// Maintained synchronous Whisper transcriber and typed lifecycle contracts.
pub use transcriber::sm::{
    Dependencies as SpeechTranscriberDependencies, EventInitializeRun as SpeechInitializeEvent,
    EventRecognizeRun as SpeechRecognizeEvent, InitializeDone as SpeechInitializeDone,
    InitializeError as SpeechInitializeError, RecognitionDone as SpeechRecognitionDone,
    RecognitionError as SpeechRecognitionError, RuntimeStorage as SpeechRuntimeStorage,
    SpeechTranscriber, SpeechTranscriberContext, SpeechTranscriberStates, TokenizerAssets,
    TranscriberError,
};
#[cfg(test)]
mod mimi_public_fixture_tests;
