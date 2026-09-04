//! Source-aligned bounded duplex speech generator state machine.
#![allow(
    dead_code,
    missing_docs,
    unused_imports,
    clippy::type_complexity,
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::result_unit_err,
    clippy::missing_errors_doc,
    missing_debug_implementations,
    reason = "SML-generated state/event names, equality implementations, and actor wrappers preserve source-aligned public machine contracts"
)]

use sml::sml;
use std::cell::Cell;

pub const MAX_FRAME_SAMPLES: usize = 16_384;
pub const MAX_CODEBOOKS: usize = 64;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpeechGeneratorError {
    #[default]
    None = 0,
    Uninitialized = 1 << 0,
    InvalidRequest = 1 << 2,
    InternalError = 1 << 4,
    MemoryInitializeFailed = 1 << 5,
    EncoderInitializeFailed = 1 << 6,
    DecoderInitializeFailed = 1 << 7,
    PredictorInitializeFailed = 1 << 9,
    ConditioningFailed = 1 << 10,
    EncodeFailed = 1 << 11,
    PredictFailed = 1 << 12,
    DecodeFailed = 1 << 13,
    ConditionerInitializeFailed = 1 << 14,
    PrefillerInitializeFailed = 1 << 15,
    SamplerInitializeFailed = 1 << 16,
    PostprocessorInitializeFailed = 1 << 17,
    PrefillFailed = 1 << 18,
    SampleFailed = 1 << 19,
    PostprocessFailed = 1 << 20,
    UnsupportedRequest = 1 << 21,
    TokenizerInitializeFailed = 1 << 22,
    TokenizeFailed = 1 << 23,
    DetokenizeFailed = 1 << 24,
    PlanningFailed = 1 << 25,
    GraphFailed = 1 << 26,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitPhase {
    TemporalPositions,
    SecondaryPositions,
    Encoder,
    Decoder,
    Predictor,
    Conditioning,
    Tokenizer,
    SynthesisConditioner,
    SynthesisPrefiller,
    SynthesisPredictor,
    SynthesisSampler,
    SynthesisDecoder,
    SynthesisPostprocessor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SynthesisGeneratePhase {
    Conditioning,
    Prefill,
    Predict,
    Sample,
    Decode,
    Postprocess,
}

pub type SynthesisGenerateAction = fn(SynthesisGeneratePhase) -> Result<(), SpeechGeneratorError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionPhase {
    Voice,
    PromptBegin,
    Prompt,
    CaptureTokenizer,
    RestoreTokenizer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramePhase {
    Encode,
    Tokenize,
    Plan,
    Predict,
    Graph,
    Sample,
    Detokenize,
    Decode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConditionResult {
    pub complete: bool,
    pub remaining: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrameResult {
    pub produced: bool,
    pub text_token: i32,
    pub sample_count: usize,
}

/// Typed synchronous initialization collaborator. `Err` is the child error.
pub type InitAction = fn(InitPhase) -> Result<(), SpeechGeneratorError>;
/// Typed synchronous conditioning collaborator. Outputs are returned by value.
pub type ConditionAction = fn(ConditionPhase, i32) -> Result<ConditionResult, SpeechGeneratorError>;
/// Typed synchronous frame collaborator. Outputs are returned by value.
pub type FrameAction = fn(FramePhase) -> Result<FrameResult, SpeechGeneratorError>;
pub type DoneCallback = fn();
pub type ErrorCallback = fn(SpeechGeneratorError);

#[derive(Debug, Clone)]
pub struct ConditionRun {
    pub token: i32,
    pub reference_len: usize,
    pub text_len: usize,
    pub condition_voice: Option<ConditionAction>,
    pub prompt_begin: Option<ConditionAction>,
    pub condition_prompt: Option<ConditionAction>,
    pub capture_tokenizer: Option<ConditionAction>,
    pub restore_tokenizer: Option<ConditionAction>,
    pub complete_out: Cell<bool>,
    pub remaining_out: Cell<i32>,
    pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl Default for ConditionRun {
    fn default() -> Self {
        Self {
            token: -1,
            reference_len: 0,
            text_len: 0,
            condition_voice: None,
            prompt_begin: None,
            condition_prompt: None,
            capture_tokenizer: None,
            restore_tokenizer: None,
            complete_out: Cell::new(false),
            remaining_out: Cell::new(-1),
            error_out: Cell::new(SpeechGeneratorError::None),
            on_done: None,
            on_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InitRun {
    pub model_ready: bool,
    pub tokenizer_ready: bool,
    pub codec_ready: bool,
    pub frame_samples: usize,
    pub codebook_count: usize,
    pub initialize: Option<InitAction>,
    pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl Default for InitRun {
    fn default() -> Self {
        Self {
            model_ready: false,
            tokenizer_ready: false,
            codec_ready: false,
            frame_samples: 0,
            codebook_count: 0,
            initialize: None,
            error_out: Cell::new(SpeechGeneratorError::None),
            on_done: None,
            on_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamRun {
    pub pcm_len: usize,
    pub output_capacity: usize,
    pub token_capacity: usize,
    pub encode: Option<FrameAction>,
    pub tokenize: Option<FrameAction>,
    pub plan: Option<FrameAction>,
    pub predict: Option<FrameAction>,
    pub graph: Option<FrameAction>,
    pub sample: Option<FrameAction>,
    pub detokenize: Option<FrameAction>,
    pub decode: Option<FrameAction>,
    pub text_token_out: Cell<i32>,
    pub sample_count_out: Cell<usize>,
    pub produced_out: Cell<bool>,
    pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl Default for StreamRun {
    fn default() -> Self {
        Self {
            pcm_len: 0,
            output_capacity: 0,
            token_capacity: 0,
            encode: None,
            tokenize: None,
            plan: None,
            predict: None,
            graph: None,
            sample: None,
            detokenize: None,
            decode: None,
            text_token_out: Cell::new(-1),
            sample_count_out: Cell::new(0),
            produced_out: Cell::new(false),
            error_out: Cell::new(SpeechGeneratorError::None),
            on_done: None,
            on_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlushRun {
    pub output_capacity: usize,
    pub token_capacity: usize,
    pub encode: Option<FrameAction>,
    pub tokenize: Option<FrameAction>,
    pub plan: Option<FrameAction>,
    pub predict: Option<FrameAction>,
    pub graph: Option<FrameAction>,
    pub sample: Option<FrameAction>,
    pub detokenize: Option<FrameAction>,
    pub decode: Option<FrameAction>,
    pub text_token_out: Cell<i32>,
    pub sample_count_out: Cell<usize>,
    pub produced_out: Cell<bool>,
    pub complete_out: Cell<bool>,
    pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl Default for FlushRun {
    fn default() -> Self {
        Self {
            output_capacity: 0,
            token_capacity: 0,
            encode: None,
            tokenize: None,
            plan: None,
            predict: None,
            graph: None,
            sample: None,
            detokenize: None,
            decode: None,
            text_token_out: Cell::new(-1),
            sample_count_out: Cell::new(0),
            produced_out: Cell::new(false),
            complete_out: Cell::new(false),
            error_out: Cell::new(SpeechGeneratorError::None),
            on_done: None,
            on_error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRun {
    pub valid: bool,
    pub condition: Option<SynthesisGenerateAction>,
    pub prefill: Option<SynthesisGenerateAction>,
    pub predict: Option<SynthesisGenerateAction>,
    pub sample: Option<SynthesisGenerateAction>,
    pub decode: Option<SynthesisGenerateAction>,
    pub postprocess: Option<SynthesisGenerateAction>,
    pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl Default for GenerateRun {
    fn default() -> Self {
        Self {
            valid: false,
            condition: None,
            prefill: None,
            predict: None,
            sample: None,
            decode: None,
            postprocess: None,
            error_out: Cell::new(SpeechGeneratorError::None),
            on_done: None,
            on_error: None,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct EventReset {
    pub error_out: Cell<SpeechGeneratorError>,
}

// --- machine SpeechGeneratorDuplexModel from emel.cpp/src/emel/speech/generator/sm.hpp ---
sml! {
    SpeechGeneratorDuplexModel {
        "state_initialize_temporal_result"_s <= *"state_uninitialized"_s + event<InitRun> [guard_initialize_request_valid_dependencies_type] / effect_initialize_temporal_positions_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_uninitialized"_s + event<InitRun> [guard_initialize_request_invalid_dependencies_type] / effect_fail_initialize_dependencies_type_error_invalid_request,
        "state_initialize_secondary_result"_s <= "state_initialize_temporal_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_secondary_positions_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_temporal_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_temporal_result,
        "state_initialize_encoder_result"_s <= "state_initialize_secondary_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_encoder_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_secondary_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_secondary_result,
        "state_initialize_decoder_result"_s <= "state_initialize_encoder_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_decoder_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_encoder_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_encoder_initialize_failed,
        "state_initialize_predictor_result"_s <= "state_initialize_decoder_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_predictor_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_decoder_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_decoder_initialize_failed,
        "state_initialize_conditioning_result"_s <= "state_initialize_predictor_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_conditioning_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_predictor_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_predictor_initialize_failed,
        "state_initialize_tokenizer_result"_s <= "state_initialize_conditioning_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_tokenizer_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_conditioning_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_conditioning_failed,
        "state_initialize_done_channel_decision"_s <= "state_initialize_tokenizer_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_publish_initialize_done_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_tokenizer_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_tokenizer_initialize_failed,
        "state_condition_voice"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_present] / effect_emit_initialize_done_dependencies_type,
        "state_condition_voice"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_absent],
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_present] / effect_emit_initialize_error_dependencies_type,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_absent],
        "state_condition_voice_result"_s <= "state_condition_voice"_s + event<ConditionRun> / effect_condition_voice_dependencies_type,
        "state_condition_voice_done_channel_decision"_s <= "state_condition_voice_result"_s + completion<ConditionRun> [guard_condition_succeeded_pending_dependencies_type] / effect_publish_condition_pending_dependencies_type_from_state_condition_voice_result,
        "state_condition_prompt_begin_result"_s <= "state_condition_voice_result"_s + completion<ConditionRun> [guard_condition_succeeded_complete_dependencies_type] / effect_begin_prompt_conditioning_dependencies_type,
        "state_condition_error_channel_decision"_s <= "state_condition_voice_result"_s + completion<ConditionRun> [guard_condition_failed_dependencies_type] / effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_voice_result,
        "state_condition_voice_done_channel_decision"_s <= "state_condition_prompt_begin_result"_s + completion<ConditionRun> [guard_child_succeeded_dependencies_type_condition_run] / effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_begin_result,
        "state_condition_error_channel_decision"_s <= "state_condition_prompt_begin_result"_s + completion<ConditionRun> [guard_child_failed_dependencies_type_condition_run] / effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_begin_result,
        "state_condition_voice"_s <= "state_condition_voice_done_channel_decision"_s + completion<ConditionRun> [guard_condition_pending_done_callback_present_dependencies_type] / effect_emit_condition_done_dependencies_type_from_state_condition_voice_done_channel_decision,
        "state_condition_voice"_s <= "state_condition_voice_done_channel_decision"_s + completion<ConditionRun> [guard_condition_pending_done_callback_absent_dependencies_type],
        "state_condition_prompt"_s <= "state_condition_voice_done_channel_decision"_s + completion<ConditionRun> [guard_condition_complete_done_callback_present_dependencies_type] / effect_emit_condition_done_dependencies_type_from_state_condition_voice_done_channel_decision,
        "state_condition_prompt"_s <= "state_condition_voice_done_channel_decision"_s + completion<ConditionRun> [guard_condition_complete_done_callback_absent_dependencies_type],
        "state_condition_prompt_result"_s <= "state_condition_prompt"_s + event<ConditionRun> / effect_condition_prompt_dependencies_type,
        "state_condition_prompt_done_channel_decision"_s <= "state_condition_prompt_result"_s + completion<ConditionRun> [guard_condition_succeeded_pending_dependencies_type] / effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_result,
        "state_condition_capture_tokenizer_result"_s <= "state_condition_prompt_result"_s + completion<ConditionRun> [guard_condition_succeeded_complete_dependencies_type] / effect_capture_tokenizer_state_dependencies_type,
        "state_condition_error_channel_decision"_s <= "state_condition_prompt_result"_s + completion<ConditionRun> [guard_condition_failed_dependencies_type] / effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_result,
        "state_condition_restore_tokenizer_result"_s <= "state_condition_capture_tokenizer_result"_s + completion<ConditionRun> [guard_child_succeeded_dependencies_type_condition_run] / effect_restore_tokenizer_state_dependencies_type,
        "state_condition_error_channel_decision"_s <= "state_condition_capture_tokenizer_result"_s + completion<ConditionRun> [guard_child_failed_dependencies_type_condition_run] / effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_capture_tokenizer_result,
        "state_condition_prompt_done_channel_decision"_s <= "state_condition_restore_tokenizer_result"_s + completion<ConditionRun> [guard_child_succeeded_dependencies_type_condition_run] / effect_publish_condition_complete_dependencies_type,
        "state_condition_error_channel_decision"_s <= "state_condition_restore_tokenizer_result"_s + completion<ConditionRun> [guard_child_failed_dependencies_type_condition_run] / effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_restore_tokenizer_result,
        "state_condition_prompt"_s <= "state_condition_prompt_done_channel_decision"_s + completion<ConditionRun> [guard_condition_pending_done_callback_present_dependencies_type] / effect_emit_condition_done_dependencies_type_from_state_condition_prompt_done_channel_decision,
        "state_condition_prompt"_s <= "state_condition_prompt_done_channel_decision"_s + completion<ConditionRun> [guard_condition_pending_done_callback_absent_dependencies_type],
        "state_ready"_s <= "state_condition_prompt_done_channel_decision"_s + completion<ConditionRun> [guard_condition_complete_done_callback_present_dependencies_type] / effect_emit_condition_done_dependencies_type_from_state_condition_prompt_done_channel_decision,
        "state_ready"_s <= "state_condition_prompt_done_channel_decision"_s + completion<ConditionRun> [guard_condition_complete_done_callback_absent_dependencies_type],
        "state_errored"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_present] / effect_emit_condition_error_dependencies_type,
        "state_errored"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_absent],
        "state_generate_error_channel_decision"_s <= "state_ready"_s + event<GenerateRun> / effect_fail_generate_dependencies_type_error_unsupported_request,
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_present] / effect_emit_generation_error_dependencies_type,
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_absent],
        "state_stream_encode_result"_s <= "state_ready"_s + event<StreamRun> [guard_stream_request_valid_dependencies_type] / effect_encode_stream_frame_dependencies_type,
        "state_stream_error_channel_decision"_s <= "state_ready"_s + event<StreamRun> [guard_stream_request_invalid_dependencies_type] / effect_fail_stream_frame_dependencies_type_error_invalid_request,
        "state_stream_tokenize_result"_s <= "state_stream_encode_result"_s + completion<StreamRun> [guard_child_succeeded_dependencies_type_stream_run] / effect_tokenize_frame_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_encode_result"_s + completion<StreamRun> [guard_child_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_encode_failed,
        "state_stream_plan_result"_s <= "state_stream_tokenize_result"_s + completion<StreamRun> [guard_child_succeeded_dependencies_type_stream_run] / effect_plan_frame_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_tokenize_result"_s + completion<StreamRun> [guard_child_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_tokenize_failed,
        "state_stream_predict_result"_s <= "state_stream_plan_result"_s + completion<StreamRun> [guard_child_succeeded_dependencies_type_stream_run] / effect_predict_frame_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_plan_result"_s + completion<StreamRun> [guard_child_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_planning_failed,
        "state_stream_graph_result"_s <= "state_stream_predict_result"_s + completion<StreamRun> [guard_prediction_succeeded_dependencies_type_stream_run] / effect_execute_prediction_graph_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_predict_result"_s + completion<StreamRun> [guard_prediction_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_predict_failed,
        "state_stream_sample_result"_s <= "state_stream_graph_result"_s + completion<StreamRun> [guard_prediction_succeeded_dependencies_type_stream_run] / effect_sample_frame_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_graph_result"_s + completion<StreamRun> [guard_prediction_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_graph_failed,
        "state_stream_detokenize_result"_s <= "state_stream_sample_result"_s + completion<StreamRun> [guard_prediction_succeeded_dependencies_type_stream_run] / effect_detokenize_frame_dependencies_type_stream_run,
        "state_stream_error_channel_decision"_s <= "state_stream_sample_result"_s + completion<StreamRun> [guard_prediction_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_sample_failed,
        "state_stream_decode_result"_s <= "state_stream_detokenize_result"_s + completion<StreamRun> [guard_frame_produced_dependencies_type_stream_run] / effect_decode_frame_dependencies_type_stream_run,
        "state_stream_done_channel_decision"_s <= "state_stream_detokenize_result"_s + completion<StreamRun> [guard_frame_pending_dependencies_type_stream_run] / effect_publish_stream_frame_pending_dependencies_type,
        "state_stream_error_channel_decision"_s <= "state_stream_detokenize_result"_s + completion<StreamRun> [guard_frame_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_detokenize_failed,
        "state_stream_done_channel_decision"_s <= "state_stream_decode_result"_s + completion<StreamRun> [guard_child_succeeded_dependencies_type_stream_run] / effect_publish_stream_frame_produced_dependencies_type,
        "state_stream_error_channel_decision"_s <= "state_stream_decode_result"_s + completion<StreamRun> [guard_child_failed_dependencies_type_stream_run] / effect_fail_stream_frame_dependencies_type_error_decode_failed,
        "state_ready"_s <= "state_stream_done_channel_decision"_s + completion<StreamRun> [stream_done_present] / effect_emit_stream_frame_done_dependencies_type,
        "state_ready"_s <= "state_stream_done_channel_decision"_s + completion<StreamRun> [stream_done_absent],
        "state_errored"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_present] / effect_emit_stream_frame_error_dependencies_type,
        "state_errored"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_absent],
        "state_flush_encode_result"_s <= "state_ready"_s + event<FlushRun> [guard_flush_request_valid_dependencies_type] / effect_encode_flush_frame_dependencies_type_from_state_ready,
        "state_flush_error_channel_decision"_s <= "state_ready"_s + event<FlushRun> [guard_flush_request_invalid_dependencies_type] / effect_fail_flush_dependencies_type_error_invalid_request_from_state_ready,
        "state_flush_encode_result"_s <= "state_flushing"_s + event<FlushRun> [guard_flush_request_valid_dependencies_type] / effect_encode_flush_frame_dependencies_type_from_state_flushing,
        "state_flush_error_channel_decision"_s <= "state_flushing"_s + event<FlushRun> [guard_flush_request_invalid_dependencies_type] / effect_fail_flush_dependencies_type_error_invalid_request_from_state_flushing,
        "state_flush_tokenize_result"_s <= "state_flush_encode_result"_s + completion<FlushRun> [guard_child_succeeded_dependencies_type_flush_run] / effect_tokenize_frame_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_encode_result"_s + completion<FlushRun> [guard_child_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_encode_failed,
        "state_flush_plan_result"_s <= "state_flush_tokenize_result"_s + completion<FlushRun> [guard_child_succeeded_dependencies_type_flush_run] / effect_plan_frame_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_tokenize_result"_s + completion<FlushRun> [guard_child_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_tokenize_failed,
        "state_flush_predict_result"_s <= "state_flush_plan_result"_s + completion<FlushRun> [guard_child_succeeded_dependencies_type_flush_run] / effect_predict_frame_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_plan_result"_s + completion<FlushRun> [guard_child_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_planning_failed,
        "state_flush_graph_result"_s <= "state_flush_predict_result"_s + completion<FlushRun> [guard_prediction_succeeded_dependencies_type_flush_run] / effect_execute_prediction_graph_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_predict_result"_s + completion<FlushRun> [guard_prediction_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_predict_failed,
        "state_flush_sample_result"_s <= "state_flush_graph_result"_s + completion<FlushRun> [guard_prediction_succeeded_dependencies_type_flush_run] / effect_sample_frame_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_graph_result"_s + completion<FlushRun> [guard_prediction_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_graph_failed,
        "state_flush_detokenize_result"_s <= "state_flush_sample_result"_s + completion<FlushRun> [guard_prediction_succeeded_dependencies_type_flush_run] / effect_detokenize_frame_dependencies_type_flush_run,
        "state_flush_error_channel_decision"_s <= "state_flush_sample_result"_s + completion<FlushRun> [guard_prediction_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_sample_failed,
        "state_flush_decode_result"_s <= "state_flush_detokenize_result"_s + completion<FlushRun> [guard_frame_produced_dependencies_type_flush_run] / effect_decode_frame_dependencies_type_flush_run,
        "state_flush_pending_result"_s <= "state_flush_detokenize_result"_s + completion<FlushRun> [guard_frame_pending_dependencies_type_flush_run] / effect_publish_flush_pending_dependencies_type,
        "state_flush_error_channel_decision"_s <= "state_flush_detokenize_result"_s + completion<FlushRun> [guard_frame_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_detokenize_failed,
        "state_flushing"_s <= "state_flush_pending_result"_s + completion<FlushRun>,
        "state_flush_produced_result"_s <= "state_flush_decode_result"_s + completion<FlushRun> [guard_child_succeeded_dependencies_type_flush_run] / effect_publish_flush_produced_dependencies_type,
        "state_flush_error_channel_decision"_s <= "state_flush_decode_result"_s + completion<FlushRun> [guard_child_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_decode_failed,
        "state_flush_done_channel_decision"_s <= "state_flush_produced_result"_s + completion<FlushRun>,
        "state_ready"_s <= "state_flush_done_channel_decision"_s + completion<FlushRun> [flush_done_present] / effect_emit_flush_done_dependencies_type,
        "state_ready"_s <= "state_flush_done_channel_decision"_s + completion<FlushRun> [flush_done_absent],
        "state_errored"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_present] / effect_emit_flush_error_dependencies_type,
        "state_errored"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_absent],
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_uninitialized,
        "state_ready"_s <= "state_ready"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_unsupported_request_from_state_ready,
        "state_flushing"_s <= "state_flushing"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_unsupported_request_from_state_flushing,
        "state_errored"_s <= "state_errored"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_internal_error,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_uninitialized,
        "state_errored"_s <= "state_condition_voice"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_condition_voice,
        "state_errored"_s <= "state_condition_prompt"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_condition_prompt,
        "state_errored"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_ready,
        "state_errored"_s <= "state_flushing"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_flushing,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_errored,
    }
}

// --- machine SpeechGeneratorSynthesisModel from emel.cpp/src/emel/speech/generator/sm.hpp ---
sml! {
    SpeechGeneratorSynthesisModel {
        "state_initialize_synthesis_conditioner_result"_s <= *"state_uninitialized"_s + event<InitRun> [guard_synthesis_initialize_request_valid] / effect_initialize_synthesis_conditioner,
        "state_initialize_synthesis_prefiller_result"_s <= "state_initialize_synthesis_conditioner_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_initialize_synthesis_prefiller,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_conditioner_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_conditioner,
        "state_initialize_synthesis_predictor_result"_s <= "state_initialize_synthesis_prefiller_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_initialize_synthesis_predictor,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_prefiller_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_prefiller,
        "state_initialize_synthesis_sampler_result"_s <= "state_initialize_synthesis_predictor_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_initialize_synthesis_sampler,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_predictor_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_predictor,
        "state_initialize_synthesis_decoder_result"_s <= "state_initialize_synthesis_sampler_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_initialize_synthesis_decoder,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_sampler_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_sampler,
        "state_initialize_synthesis_postprocessor_result"_s <= "state_initialize_synthesis_decoder_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_initialize_synthesis_postprocessor,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_decoder_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_decoder,
        "state_initialize_done_channel_decision"_s <= "state_initialize_synthesis_postprocessor_result"_s + completion<InitRun> [guard_child_succeeded_init] / effect_publish_synthesis_initialize_done,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_postprocessor_result"_s + completion<InitRun> [guard_child_failed_init] / effect_fail_initialize_postprocessor,
        "state_ready"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_present_synthesis] / effect_emit_synthesis_initialize_done,
        "state_ready"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_absent_synthesis],
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_present_synthesis] / effect_emit_synthesis_initialize_error,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_absent_synthesis],

        "state_generate_conditioning"_s <= "state_ready"_s + event<GenerateRun> [guard_generate_request_valid] / effect_condition_generate,
        "state_generate_error_channel_decision"_s <= "state_ready"_s + event<GenerateRun> [guard_generate_request_invalid] / effect_fail_generate_invalid,
        "state_generate_prefill"_s <= "state_generate_conditioning"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_prefill_generate,
        "state_generate_error_channel_decision"_s <= "state_generate_conditioning"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_conditioning,
        "state_generate_predict"_s <= "state_generate_prefill"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_predict_generate,
        "state_generate_error_channel_decision"_s <= "state_generate_prefill"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_prefill,
        "state_generate_sample"_s <= "state_generate_predict"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_sample_generate,
        "state_generate_error_channel_decision"_s <= "state_generate_predict"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_predict,
        "state_generate_decode"_s <= "state_generate_sample"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_decode_generate,
        "state_generate_error_channel_decision"_s <= "state_generate_sample"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_sample,
        "state_generate_postprocess"_s <= "state_generate_decode"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_postprocess_generate,
        "state_generate_error_channel_decision"_s <= "state_generate_decode"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_decode,
        "state_generate_done_channel_decision"_s <= "state_generate_postprocess"_s + completion<GenerateRun> [guard_child_succeeded_generate] / effect_publish_generation_done,
        "state_generate_error_channel_decision"_s <= "state_generate_postprocess"_s + completion<GenerateRun> [guard_child_failed_generate] / effect_fail_generate_postprocess,
        "state_ready"_s <= "state_generate_done_channel_decision"_s + completion<GenerateRun> [generate_done_present_synthesis] / effect_emit_generation_done_synthesis,
        "state_ready"_s <= "state_generate_done_channel_decision"_s + completion<GenerateRun> [generate_done_absent_synthesis],
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_present_synthesis] / effect_emit_generation_error_synthesis,
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_absent_synthesis],

        "state_condition_error_channel_decision"_s <= "state_ready"_s + event<ConditionRun> / effect_fail_condition_unsupported,
        "state_ready"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_present_synthesis] / effect_emit_condition_error_synthesis,
        "state_ready"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_absent_synthesis],
        "state_stream_error_channel_decision"_s <= "state_ready"_s + event<StreamRun> / effect_fail_stream_unsupported,
        "state_ready"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_present_synthesis] / effect_emit_stream_error_synthesis,
        "state_ready"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_absent_synthesis],
        "state_flush_error_channel_decision"_s <= "state_ready"_s + event<FlushRun> / effect_fail_flush_unsupported,
        "state_ready"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_present_synthesis] / effect_emit_flush_error_synthesis,
        "state_ready"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_absent_synthesis],
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset> / effect_reject_reset_uninitialized_synthesis,
        "state_ready"_s <= "state_ready"_s + event<EventReset> / effect_reject_reset_unsupported_synthesis,
        "state_errored"_s <= "state_errored"_s + event<EventReset> / effect_reject_reset_internal_synthesis,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_synthesis,
        "state_errored"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected_synthesis,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_synthesis,
    }
}

#[derive(Debug, Default)]
pub struct SpeechGeneratorSynthesisModelContext {
    pub initialized: bool,
    pub child_accepted: bool,
    pub child_err: SpeechGeneratorError,
    pub error: SpeechGeneratorError,
}

impl SpeechGeneratorSynthesisModelContext {
    fn set_init(&mut self, action: Option<InitAction>, phase: InitPhase) {
        self.child_accepted = false;
        self.child_err = match action {
            Some(callback) => match callback(phase) {
                Ok(()) => {
                    self.child_accepted = true;
                    SpeechGeneratorError::None
                }
                Err(error) => error,
            },
            None => SpeechGeneratorError::UnsupportedRequest,
        };
    }

    fn set_generate(
        &mut self,
        action: Option<SynthesisGenerateAction>,
        phase: SynthesisGeneratePhase,
    ) {
        self.child_accepted = false;
        self.child_err = match action {
            Some(callback) => match callback(phase) {
                Ok(()) => {
                    self.child_accepted = true;
                    SpeechGeneratorError::None
                }
                Err(error) => error,
            },
            None => SpeechGeneratorError::UnsupportedRequest,
        };
    }

    fn fail(&mut self, error: SpeechGeneratorError) {
        self.error = if self.child_err == SpeechGeneratorError::None {
            error
        } else {
            self.child_err
        };
    }
}

impl SpeechGeneratorSynthesisModelStateMachineContext for SpeechGeneratorSynthesisModelContext {
    fn guard_synthesis_initialize_request_valid(&self, e: &InitRun) -> Result<bool, ()> {
        Ok(e.model_ready
            && e.tokenizer_ready
            && e.codec_ready
            && e.frame_samples > 0
            && e.frame_samples <= MAX_FRAME_SAMPLES
            && e.codebook_count > 0
            && e.codebook_count <= MAX_CODEBOOKS
            && e.initialize.is_some())
    }
    fn guard_child_succeeded_init(&self, _e: &InitRun) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_child_failed_init(&self, _e: &InitRun) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_child_succeeded_generate(&self, _e: &GenerateRun) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_child_failed_generate(&self, _e: &GenerateRun) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_generate_request_valid(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(self.initialized
            && e.valid
            && e.condition.is_some()
            && e.prefill.is_some()
            && e.predict.is_some()
            && e.sample.is_some()
            && e.decode.is_some()
            && e.postprocess.is_some())
    }
    fn guard_generate_request_invalid(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(!self.guard_generate_request_valid(e)?)
    }
    fn init_done_present_synthesis(&self, e: &InitRun) -> Result<bool, ()> {
        Ok(e.on_done.is_some())
    }
    fn init_done_absent_synthesis(&self, e: &InitRun) -> Result<bool, ()> {
        Ok(e.on_done.is_none())
    }
    fn init_error_present_synthesis(&self, e: &InitRun) -> Result<bool, ()> {
        Ok(e.on_error.is_some())
    }
    fn init_error_absent_synthesis(&self, e: &InitRun) -> Result<bool, ()> {
        Ok(e.on_error.is_none())
    }
    fn generate_done_present_synthesis(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(e.on_done.is_some())
    }
    fn generate_done_absent_synthesis(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(e.on_done.is_none())
    }
    fn generate_error_present_synthesis(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(e.on_error.is_some())
    }
    fn generate_error_absent_synthesis(&self, e: &GenerateRun) -> Result<bool, ()> {
        Ok(e.on_error.is_none())
    }
    fn condition_error_present_synthesis(&self, e: &ConditionRun) -> Result<bool, ()> {
        Ok(e.on_error.is_some())
    }
    fn condition_error_absent_synthesis(&self, e: &ConditionRun) -> Result<bool, ()> {
        Ok(e.on_error.is_none())
    }
    fn stream_error_present_synthesis(&self, e: &StreamRun) -> Result<bool, ()> {
        Ok(e.on_error.is_some())
    }
    fn stream_error_absent_synthesis(&self, e: &StreamRun) -> Result<bool, ()> {
        Ok(e.on_error.is_none())
    }
    fn flush_error_present_synthesis(&self, e: &FlushRun) -> Result<bool, ()> {
        Ok(e.on_error.is_some())
    }
    fn flush_error_absent_synthesis(&self, e: &FlushRun) -> Result<bool, ()> {
        Ok(e.on_error.is_none())
    }

    fn effect_initialize_synthesis_conditioner(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisConditioner);
        Ok(())
    }
    fn effect_initialize_synthesis_prefiller(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisPrefiller);
        Ok(())
    }
    fn effect_initialize_synthesis_predictor(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisPredictor);
        Ok(())
    }
    fn effect_initialize_synthesis_sampler(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisSampler);
        Ok(())
    }
    fn effect_initialize_synthesis_decoder(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisDecoder);
        Ok(())
    }
    fn effect_initialize_synthesis_postprocessor(&mut self, e: &InitRun) -> Result<(), ()> {
        self.set_init(e.initialize, InitPhase::SynthesisPostprocessor);
        Ok(())
    }
    fn effect_publish_synthesis_initialize_done(&mut self, e: &InitRun) -> Result<(), ()> {
        self.initialized = true;
        e.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_emit_synthesis_initialize_done(&mut self, e: &InitRun) -> Result<(), ()> {
        if let Some(f) = e.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_synthesis_initialize_error(&mut self, e: &InitRun) -> Result<(), ()> {
        if let Some(f) = e.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_fail_initialize_conditioner(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditionerInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_prefiller(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PrefillerInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_predictor(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PredictorInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_sampler(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::SamplerInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_decoder(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::DecoderInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_postprocessor(&mut self, e: &InitRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PostprocessorInitializeFailed;
        e.error_out.set(self.error);
        Ok(())
    }

    fn effect_condition_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.condition, SynthesisGeneratePhase::Conditioning);
        Ok(())
    }
    fn effect_prefill_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.prefill, SynthesisGeneratePhase::Prefill);
        Ok(())
    }
    fn effect_predict_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.predict, SynthesisGeneratePhase::Predict);
        Ok(())
    }
    fn effect_sample_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.sample, SynthesisGeneratePhase::Sample);
        Ok(())
    }
    fn effect_decode_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.decode, SynthesisGeneratePhase::Decode);
        Ok(())
    }
    fn effect_postprocess_generate(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.set_generate(e.postprocess, SynthesisGeneratePhase::Postprocess);
        Ok(())
    }
    fn effect_publish_generation_done(&mut self, e: &GenerateRun) -> Result<(), ()> {
        e.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_emit_generation_done_synthesis(&mut self, e: &GenerateRun) -> Result<(), ()> {
        if let Some(f) = e.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_generation_error_synthesis(&mut self, e: &GenerateRun) -> Result<(), ()> {
        if let Some(f) = e.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_fail_generate_invalid(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_conditioning(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_prefill(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PrefillFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_predict(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PredictFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_sample(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::SampleFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_decode(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::DecodeFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_postprocess(&mut self, e: &GenerateRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PostprocessFailed;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_condition_unsupported(&mut self, e: &ConditionRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::UnsupportedRequest;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_emit_condition_error_synthesis(&mut self, e: &ConditionRun) -> Result<(), ()> {
        if let Some(f) = e.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_fail_stream_unsupported(&mut self, e: &StreamRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::UnsupportedRequest;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_emit_stream_error_synthesis(&mut self, e: &StreamRun) -> Result<(), ()> {
        if let Some(f) = e.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_fail_flush_unsupported(&mut self, e: &FlushRun) -> Result<(), ()> {
        self.error = SpeechGeneratorError::UnsupportedRequest;
        e.error_out.set(self.error);
        Ok(())
    }
    fn effect_emit_flush_error_synthesis(&mut self, e: &FlushRun) -> Result<(), ()> {
        if let Some(f) = e.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_reject_reset_uninitialized_synthesis(&mut self, e: &EventReset) -> Result<(), ()> {
        e.error_out.set(SpeechGeneratorError::Uninitialized);
        Ok(())
    }
    fn effect_reject_reset_unsupported_synthesis(&mut self, e: &EventReset) -> Result<(), ()> {
        e.error_out.set(SpeechGeneratorError::UnsupportedRequest);
        Ok(())
    }
    fn effect_reject_reset_internal_synthesis(&mut self, e: &EventReset) -> Result<(), ()> {
        e.error_out.set(SpeechGeneratorError::InternalError);
        Ok(())
    }
    fn effect_unexpected_synthesis(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
}

pub struct SpeechGeneratorSynthesisModelActor {
    machine: SpeechGeneratorSynthesisModelStateMachine<SpeechGeneratorSynthesisModelContext>,
}
impl Default for SpeechGeneratorSynthesisModelActor {
    fn default() -> Self {
        Self::new()
    }
}
impl SpeechGeneratorSynthesisModelActor {
    pub fn new() -> Self {
        Self {
            machine: SpeechGeneratorSynthesisModelStateMachine::new(
                SpeechGeneratorSynthesisModelContext::default(),
            ),
        }
    }
    pub fn initialize(&mut self, e: InitRun) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn generate(&mut self, e: GenerateRun) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn condition(&mut self, e: ConditionRun) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn stream(&mut self, e: StreamRun) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn flush(&mut self, e: FlushRun) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn reset(&mut self, e: EventReset) -> Result<(), ()> {
        self.machine.process_event(e).map(|_| ()).map_err(|_| ())
    }
    pub fn context(&self) -> &SpeechGeneratorSynthesisModelContext {
        self.machine.context()
    }
    pub fn state(&self) -> &SpeechGeneratorSynthesisModelStates {
        self.machine.state()
    }
    pub fn is(&self, state: &SpeechGeneratorSynthesisModelStates) -> bool {
        self.machine.is(state)
    }
}

#[derive(Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct SpeechGeneratorDuplexModelContext {
    pub initialized: bool,
    pub frame_samples: usize,
    pub codebook_count: usize,
    pub child_accepted: bool,
    pub child_err: SpeechGeneratorError,
    pub complete: bool,
    pub remaining: i32,
    pub produced: bool,
    pub text_token: i32,
    pub sample_count: usize,
    pub error: SpeechGeneratorError,
}

impl SpeechGeneratorDuplexModelContext {
    fn set_init(&mut self, action: Option<InitAction>, phase: InitPhase) {
        self.child_accepted = false;
        self.child_err = match action {
            Some(callback) => match callback(phase) {
                Ok(()) => {
                    self.child_accepted = true;
                    SpeechGeneratorError::None
                }
                Err(error) => error,
            },
            None => SpeechGeneratorError::UnsupportedRequest,
        };
    }
    fn set_condition(
        &mut self,
        action: Option<ConditionAction>,
        phase: ConditionPhase,
        token: i32,
    ) {
        self.child_accepted = false;
        self.complete = false;
        self.remaining = -1;
        self.child_err = match action {
            Some(callback) => match callback(phase, token) {
                Ok(result) => {
                    self.child_accepted = true;
                    self.complete = result.complete;
                    self.remaining = result.remaining;
                    SpeechGeneratorError::None
                }
                Err(error) => error,
            },
            None => SpeechGeneratorError::UnsupportedRequest,
        };
    }
    fn set_frame(&mut self, action: Option<FrameAction>, phase: FramePhase) {
        self.child_accepted = false;
        self.produced = false;
        self.sample_count = 0;
        self.text_token = -1;
        self.child_err = match action {
            Some(callback) => match callback(phase) {
                Ok(result) => {
                    self.child_accepted = true;
                    self.produced = result.produced;
                    self.text_token = result.text_token;
                    self.sample_count = result.sample_count;
                    SpeechGeneratorError::None
                }
                Err(error) => error,
            },
            None => SpeechGeneratorError::UnsupportedRequest,
        };
    }
    fn fail(&mut self, fallback: SpeechGeneratorError) {
        self.error = if self.child_err == SpeechGeneratorError::None {
            fallback
        } else {
            self.child_err
        };
    }
}

impl SpeechGeneratorDuplexModelStateMachineContext for SpeechGeneratorDuplexModelContext {
    fn condition_error_absent(&self, event: &ConditionRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn condition_error_present(&self, event: &ConditionRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn effect_begin_prompt_conditioning_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.set_condition(event.prompt_begin, ConditionPhase::PromptBegin, event.token);
        Ok(())
    }
    fn effect_capture_tokenizer_state_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.set_condition(
            event.capture_tokenizer,
            ConditionPhase::CaptureTokenizer,
            event.token,
        );
        Ok(())
    }
    fn effect_condition_prompt_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.set_condition(event.condition_prompt, ConditionPhase::Prompt, event.token);
        Ok(())
    }
    fn effect_condition_voice_dependencies_type(&mut self, event: &ConditionRun) -> Result<(), ()> {
        self.set_condition(event.condition_voice, ConditionPhase::Voice, event.token);
        Ok(())
    }
    fn effect_decode_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.decode, FramePhase::Decode);
        Ok(())
    }
    fn effect_decode_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.decode, FramePhase::Decode);
        Ok(())
    }
    fn effect_detokenize_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.detokenize, FramePhase::Detokenize);
        Ok(())
    }
    fn effect_detokenize_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.detokenize, FramePhase::Detokenize);
        Ok(())
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_prompt_done_channel_decision(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_voice_done_channel_decision(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_condition_error_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_flush_error_dependencies_type(&mut self, event: &FlushRun) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_flush_done_dependencies_type(&mut self, event: &FlushRun) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }

    fn effect_emit_generation_error_dependencies_type(
        &mut self,
        event: &GenerateRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_initialize_done_dependencies_type(&mut self, event: &InitRun) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_initialize_error_dependencies_type(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_stream_frame_done_dependencies_type(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_stream_frame_error_dependencies_type(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        Ok(())
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_flushing(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.encode, FramePhase::Encode);
        Ok(())
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_ready(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.encode, FramePhase::Encode);
        Ok(())
    }
    fn effect_encode_stream_frame_dependencies_type(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.encode, FramePhase::Encode);
        Ok(())
    }
    fn effect_execute_prediction_graph_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.graph, FramePhase::Graph);
        Ok(())
    }
    fn effect_execute_prediction_graph_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.graph, FramePhase::Graph);
        Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_capture_tokenizer_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_begin_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_restore_tokenizer_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_voice_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_decode_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::DecodeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_detokenize_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::DetokenizeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_encode_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::EncodeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_graph_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::GraphFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_flushing(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_ready(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_planning_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::PlanningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_predict_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::PredictFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_sample_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::SampleFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_tokenize_failed(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::TokenizeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_generate_dependencies_type_error_unsupported_request(
        &mut self,
        event: &GenerateRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::UnsupportedRequest;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_conditioning_failed(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::ConditioningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_decoder_initialize_failed(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::DecoderInitializeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_encoder_initialize_failed(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::EncoderInitializeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_invalid_request(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_secondary_result(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::MemoryInitializeFailed;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_temporal_result(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::MemoryInitializeFailed;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_predictor_initialize_failed(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::PredictorInitializeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_tokenizer_initialize_failed(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::TokenizerInitializeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_decode_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::DecodeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_detokenize_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::DetokenizeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_encode_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::EncodeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_graph_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::GraphFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_invalid_request(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest;
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_planning_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::PlanningFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_predict_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::PredictFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_sample_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::SampleFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_tokenize_failed(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.fail(SpeechGeneratorError::TokenizeFailed);
        event.error_out.set(self.error);
        Ok(())
    }
    fn effect_initialize_conditioning_dependencies_type(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::Conditioning);
        Ok(())
    }
    fn effect_initialize_decoder_dependencies_type(&mut self, event: &InitRun) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::Decoder);
        Ok(())
    }
    fn effect_initialize_encoder_dependencies_type(&mut self, event: &InitRun) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::Encoder);
        Ok(())
    }
    fn effect_initialize_predictor_dependencies_type(&mut self, event: &InitRun) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::Predictor);
        Ok(())
    }
    fn effect_initialize_secondary_positions_dependencies_type(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::SecondaryPositions);
        Ok(())
    }
    fn effect_initialize_temporal_positions_dependencies_type(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::TemporalPositions);
        Ok(())
    }
    fn effect_initialize_tokenizer_dependencies_type(&mut self, event: &InitRun) -> Result<(), ()> {
        self.set_init(event.initialize, InitPhase::Tokenizer);
        Ok(())
    }
    fn effect_plan_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.plan, FramePhase::Plan);
        Ok(())
    }
    fn effect_plan_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.plan, FramePhase::Plan);
        Ok(())
    }
    fn effect_predict_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.predict, FramePhase::Predict);
        Ok(())
    }
    fn effect_predict_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.predict, FramePhase::Predict);
        Ok(())
    }
    fn effect_publish_condition_complete_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        event.complete_out.set(true);
        event.remaining_out.set(0);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_begin_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        event.complete_out.set(false);
        event.remaining_out.set(self.remaining);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        event.complete_out.set(false);
        event.remaining_out.set(self.remaining);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_voice_result(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        event.complete_out.set(false);
        event.remaining_out.set(self.remaining);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_flush_pending_dependencies_type(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        event.produced_out.set(false);
        event.complete_out.set(false);
        event.sample_count_out.set(0);
        event.text_token_out.set(self.text_token);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_flush_produced_dependencies_type(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        event.produced_out.set(true);
        event.complete_out.set(false);
        event.sample_count_out.set(self.frame_samples);
        event.text_token_out.set(self.text_token);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }

    fn effect_publish_initialize_done_dependencies_type(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        self.initialized = true;
        self.frame_samples = event.frame_samples;
        self.codebook_count = event.codebook_count;
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_stream_frame_pending_dependencies_type(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        event.produced_out.set(false);
        event.sample_count_out.set(0);
        event.text_token_out.set(self.text_token);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_publish_stream_frame_produced_dependencies_type(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        event.produced_out.set(true);
        event.sample_count_out.set(self.frame_samples);
        event.text_token_out.set(self.text_token);
        event.error_out.set(SpeechGeneratorError::None);
        Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_internal_error(
        &mut self,
        event: &EventReset,
    ) -> Result<(), ()> {
        event.error_out.set(SpeechGeneratorError::InternalError);
        Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_uninitialized(
        &mut self,
        event: &EventReset,
    ) -> Result<(), ()> {
        event.error_out.set(SpeechGeneratorError::Uninitialized);
        Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_flushing(
        &mut self,
        event: &EventReset,
    ) -> Result<(), ()> {
        event
            .error_out
            .set(SpeechGeneratorError::UnsupportedRequest);
        Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_ready(
        &mut self,
        event: &EventReset,
    ) -> Result<(), ()> {
        event
            .error_out
            .set(SpeechGeneratorError::UnsupportedRequest);
        Ok(())
    }
    fn effect_restore_tokenizer_state_dependencies_type(
        &mut self,
        event: &ConditionRun,
    ) -> Result<(), ()> {
        self.set_condition(
            event.restore_tokenizer,
            ConditionPhase::RestoreTokenizer,
            event.token,
        );
        Ok(())
    }
    fn effect_sample_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.sample, FramePhase::Sample);
        Ok(())
    }
    fn effect_sample_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.sample, FramePhase::Sample);
        Ok(())
    }
    fn effect_tokenize_frame_dependencies_type_flush_run(
        &mut self,
        event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_frame(event.tokenize, FramePhase::Tokenize);
        Ok(())
    }
    fn effect_tokenize_frame_dependencies_type_stream_run(
        &mut self,
        event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_frame(event.tokenize, FramePhase::Tokenize);
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_condition_prompt(
        &mut self,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_condition_voice(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_errored(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_flushing(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_ready(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError;
        Ok(())
    }
    fn flush_error_absent(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn flush_error_present(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn generate_error_absent(&self, event: &GenerateRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn generate_error_present(&self, event: &GenerateRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_child_failed_dependencies_type_condition_run(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_child_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_child_failed_dependencies_type_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_child_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_child_succeeded_dependencies_type_condition_run(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_child_succeeded_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_child_succeeded_dependencies_type_init_run(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_child_succeeded_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_condition_complete_done_callback_absent_dependencies_type(
        &self,
        event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted
            && self.child_err == SpeechGeneratorError::None
            && self.complete
            && event.on_done.is_none())
    }
    fn guard_condition_complete_done_callback_present_dependencies_type(
        &self,
        event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted
            && self.child_err == SpeechGeneratorError::None
            && self.complete
            && event.on_done.is_some())
    }
    fn guard_condition_failed_dependencies_type(&self, _event: &ConditionRun) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_condition_pending_done_callback_absent_dependencies_type(
        &self,
        event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted
            && self.child_err == SpeechGeneratorError::None
            && !self.complete
            && event.on_done.is_none())
    }
    fn guard_condition_pending_done_callback_present_dependencies_type(
        &self,
        event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted
            && self.child_err == SpeechGeneratorError::None
            && !self.complete
            && event.on_done.is_some())
    }
    fn guard_condition_succeeded_complete_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.complete)
    }
    fn guard_condition_succeeded_pending_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && !self.complete)
    }
    fn guard_flush_request_invalid_dependencies_type(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(!self.guard_flush_request_valid_dependencies_type(event)?)
    }
    fn guard_flush_request_valid_dependencies_type(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(self.initialized
            && event.output_capacity >= self.frame_samples
            && event.token_capacity >= self.codebook_count)
    }
    fn guard_frame_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_frame_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_frame_pending_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && !self.produced)
    }
    fn guard_frame_pending_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && !self.produced)
    }
    fn guard_frame_produced_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.produced)
    }
    fn flush_done_present(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn flush_done_absent(&self, event: &FlushRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_frame_produced_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.produced)
    }
    fn guard_initialize_request_invalid_dependencies_type(
        &self,
        event: &InitRun,
    ) -> Result<bool, ()> {
        Ok(!self.guard_initialize_request_valid_dependencies_type(event)?)
    }
    fn guard_initialize_request_valid_dependencies_type(
        &self,
        event: &InitRun,
    ) -> Result<bool, ()> {
        Ok(event.model_ready
            && event.tokenizer_ready
            && event.codec_ready
            && event.frame_samples > 0
            && event.frame_samples <= MAX_FRAME_SAMPLES
            && event.codebook_count > 0
            && event.codebook_count <= MAX_CODEBOOKS
            && event.initialize.is_some())
    }
    fn guard_prediction_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_prediction_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_prediction_succeeded_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_prediction_succeeded_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None)
    }
    fn guard_stream_request_invalid_dependencies_type(
        &self,
        event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(!self.guard_stream_request_valid_dependencies_type(event)?)
    }
    fn guard_stream_request_valid_dependencies_type(&self, event: &StreamRun) -> Result<bool, ()> {
        Ok(self.initialized
            && event.pcm_len == self.frame_samples
            && event.output_capacity >= self.frame_samples
            && event.token_capacity >= self.codebook_count)
    }
    fn init_done_absent(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn init_done_present(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn init_error_absent(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn init_error_present(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn stream_done_absent(&self, event: &StreamRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn stream_done_present(&self, event: &StreamRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn stream_error_absent(&self, event: &StreamRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn stream_error_present(&self, event: &StreamRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
}

pub struct SpeechGeneratorDuplexModelActor {
    machine: SpeechGeneratorDuplexModelStateMachine<SpeechGeneratorDuplexModelContext>,
}
impl Default for SpeechGeneratorDuplexModelActor {
    fn default() -> Self {
        Self::new()
    }
}
impl SpeechGeneratorDuplexModelActor {
    pub fn new() -> Self {
        Self {
            machine: SpeechGeneratorDuplexModelStateMachine::new(
                SpeechGeneratorDuplexModelContext::default(),
            ),
        }
    }
    pub fn initialize(&mut self, e: InitRun) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn condition(&mut self, e: ConditionRun) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn generate(&mut self, e: GenerateRun) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn stream(&mut self, e: StreamRun) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn flush(&mut self, e: FlushRun) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn reset(&mut self, e: EventReset) -> Result<(), ()> {
        let result = self.machine.process_event(e).map(|_| ()).map_err(|_| ());
        let error = self.machine.context().error;
        result?;
        if error == SpeechGeneratorError::None {
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn context(&self) -> &SpeechGeneratorDuplexModelContext {
        self.machine.context()
    }
    pub fn state(&self) -> &SpeechGeneratorDuplexModelStates {
        self.machine.state()
    }
    pub fn is(&self, state: &SpeechGeneratorDuplexModelStates) -> bool {
        self.machine.is(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::unnecessary_wraps)]
    fn initialize_ok(_: InitPhase) -> Result<(), SpeechGeneratorError> {
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn condition_complete(
        _: ConditionPhase,
        _: i32,
    ) -> Result<ConditionResult, SpeechGeneratorError> {
        Ok(ConditionResult {
            complete: true,
            remaining: 0,
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn frame_produced(_: FramePhase) -> Result<FrameResult, SpeechGeneratorError> {
        Ok(FrameResult {
            produced: true,
            text_token: 23,
            sample_count: 4,
        })
    }

    #[test]
    fn duplex_initialize_done_commits_geometry_and_clears_error() {
        let mut context = SpeechGeneratorDuplexModelContext::default();
        let event = InitRun {
            frame_samples: 4,
            codebook_count: 2,
            error_out: Cell::new(SpeechGeneratorError::InternalError),
            ..InitRun::default()
        };

        context
            .effect_publish_initialize_done_dependencies_type(&event)
            .expect("initialization publication effect is synchronous");

        assert!(context.initialized);
        assert_eq!(context.frame_samples, 4);
        assert_eq!(context.codebook_count, 2);
        assert_eq!(event.error_out.get(), SpeechGeneratorError::None);
    }

    #[test]
    fn duplex_stream_after_initialize_and_condition_uses_committed_capacities() {
        let mut actor = SpeechGeneratorDuplexModelActor::new();
        actor
            .initialize(InitRun {
                model_ready: true,
                tokenizer_ready: true,
                codec_ready: true,
                frame_samples: 4,
                codebook_count: 2,
                initialize: Some(initialize_ok),
                ..InitRun::default()
            })
            .expect("valid initialization succeeds");

        actor
            .condition(ConditionRun {
                condition_voice: Some(condition_complete),
                prompt_begin: Some(condition_complete),
                ..ConditionRun::default()
            })
            .expect("voice conditioning reaches prompt phase");

        actor
            .condition(ConditionRun {
                condition_prompt: Some(condition_complete),
                capture_tokenizer: Some(condition_complete),
                restore_tokenizer: Some(condition_complete),
                ..ConditionRun::default()
            })
            .expect("prompt conditioning reaches ready");
        assert!(actor.is(&SpeechGeneratorDuplexModelStates::StateReady));

        let stream = StreamRun {
            pcm_len: 4,
            output_capacity: 4,
            token_capacity: 2,
            encode: Some(frame_produced),
            tokenize: Some(frame_produced),
            plan: Some(frame_produced),
            predict: Some(frame_produced),
            graph: Some(frame_produced),
            sample: Some(frame_produced),
            detokenize: Some(frame_produced),
            decode: Some(frame_produced),
            ..StreamRun::default()
        };

        actor
            .stream(stream)
            .expect("stream capacities matching initialized geometry are accepted");
    }

    #[test]
    fn duplex_condition_complete_publishes_terminal_result() {
        let mut context = SpeechGeneratorDuplexModelContext {
            remaining: 17,
            ..SpeechGeneratorDuplexModelContext::default()
        };
        let event = ConditionRun::default();

        context
            .effect_publish_condition_complete_dependencies_type(&event)
            .expect("publication effect is synchronous");

        assert!(event.complete_out.get());
        assert_eq!(event.remaining_out.get(), 0);
        assert_eq!(event.error_out.get(), SpeechGeneratorError::None);
    }

    #[test]
    fn duplex_pending_and_produced_frame_publications_are_bounded() {
        let mut context = SpeechGeneratorDuplexModelContext {
            frame_samples: 4,
            sample_count: 4,
            text_token: 23,
            ..SpeechGeneratorDuplexModelContext::default()
        };
        let stream = StreamRun::default();
        context
            .effect_publish_stream_frame_pending_dependencies_type(&stream)
            .expect("pending publication effect is synchronous");
        assert!(!stream.produced_out.get());
        assert_eq!(stream.sample_count_out.get(), 0);
        assert_eq!(stream.text_token_out.get(), 23);
        assert_eq!(stream.error_out.get(), SpeechGeneratorError::None);

        context
            .effect_publish_stream_frame_produced_dependencies_type(&stream)
            .expect("produced publication effect is synchronous");
        assert!(stream.produced_out.get());
        assert_eq!(stream.sample_count_out.get(), 4);
        assert_eq!(stream.text_token_out.get(), 23);
        assert_eq!(stream.error_out.get(), SpeechGeneratorError::None);
    }

    #[test]
    fn synthesis_failure_effects_publish_pinned_error_variants() {
        let mut context = SpeechGeneratorSynthesisModelContext::default();
        let event = InitRun::default();

        context
            .effect_fail_initialize_conditioner(&event)
            .expect("conditioner failure effect is synchronous");
        assert_eq!(
            event.error_out.get(),
            SpeechGeneratorError::ConditionerInitializeFailed
        );

        context
            .effect_fail_initialize_prefiller(&event)
            .expect("prefiller failure effect is synchronous");
        assert_eq!(
            event.error_out.get(),
            SpeechGeneratorError::PrefillerInitializeFailed
        );

        context
            .effect_fail_initialize_sampler(&event)
            .expect("sampler failure effect is synchronous");
        assert_eq!(
            event.error_out.get(),
            SpeechGeneratorError::SamplerInitializeFailed
        );

        context
            .effect_fail_initialize_postprocessor(&event)
            .expect("postprocessor failure effect is synchronous");
        assert_eq!(
            event.error_out.get(),
            SpeechGeneratorError::PostprocessorInitializeFailed
        );

        let event = GenerateRun::default();
        context
            .effect_fail_generate_prefill(&event)
            .expect("prefill failure effect is synchronous");
        assert_eq!(event.error_out.get(), SpeechGeneratorError::PrefillFailed);

        context
            .effect_fail_generate_postprocess(&event)
            .expect("postprocess failure effect is synchronous");
        assert_eq!(
            event.error_out.get(),
            SpeechGeneratorError::PostprocessFailed
        );
    }
}
