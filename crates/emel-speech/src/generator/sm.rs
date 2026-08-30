//! Source-aligned bounded duplex speech generator state machine.
#![allow(dead_code, missing_docs, unused_imports, clippy::type_complexity)]

use std::cell::Cell;
use sml::sml;

pub const MAX_FRAME_SAMPLES: usize = 16_384;
pub const MAX_CODEBOOKS: usize = 64;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpeechGeneratorError {
    #[default] None = 0, Uninitialized = 1 << 0, InvalidRequest = 1 << 2,
    InternalError = 1 << 4, MemoryInitializeFailed = 1 << 5,
    EncoderInitializeFailed = 1 << 6, DecoderInitializeFailed = 1 << 7,
    PredictorInitializeFailed = 1 << 9, ConditioningFailed = 1 << 10,
    EncodeFailed = 1 << 11, PredictFailed = 1 << 12, DecodeFailed = 1 << 13,
    TokenizerInitializeFailed = 1 << 22, TokenizeFailed = 1 << 23,
    DetokenizeFailed = 1 << 24, PlanningFailed = 1 << 25, GraphFailed = 1 << 26,
    SampleFailed = 1 << 19, UnsupportedRequest = 1 << 21,
}

pub type InitAction = fn() -> bool;
pub type ConditionAction = fn(i32) -> bool;
pub type FrameAction = fn() -> bool;
pub type DoneCallback = fn();
pub type ErrorCallback = fn(SpeechGeneratorError);

#[derive(Debug, Clone)]
pub struct ConditionRun {
    pub token: i32, pub reference_len: usize, pub text_len: usize,
    pub condition_voice: Option<ConditionAction>, pub prompt_begin: Option<ConditionAction>,
    pub condition_prompt: Option<ConditionAction>, pub capture_tokenizer: Option<ConditionAction>,
    pub restore_tokenizer: Option<ConditionAction>, pub complete_out: Cell<bool>,
    pub remaining_out: Cell<i32>, pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>, pub on_error: Option<ErrorCallback>,
}
impl Default for ConditionRun { fn default() -> Self { Self { token: -1, reference_len: 0, text_len: 0, condition_voice: None, prompt_begin: None, condition_prompt: None, capture_tokenizer: None, restore_tokenizer: None, complete_out: Cell::new(false), remaining_out: Cell::new(-1), error_out: Cell::new(SpeechGeneratorError::None), on_done: None, on_error: None } } }

#[derive(Debug, Clone)]
pub struct InitRun {
    pub model_ready: bool, pub tokenizer_ready: bool, pub codec_ready: bool,
    pub frame_samples: usize, pub codebook_count: usize, pub initialize: Option<InitAction>,
    pub error_out: Cell<SpeechGeneratorError>, pub on_done: Option<DoneCallback>, pub on_error: Option<ErrorCallback>,
}
impl Default for InitRun { fn default() -> Self { Self { model_ready: false, tokenizer_ready: false, codec_ready: false, frame_samples: 0, codebook_count: 0, initialize: None, error_out: Cell::new(SpeechGeneratorError::None), on_done: None, on_error: None } } }

#[derive(Debug, Clone)]
pub struct StreamRun {
    pub pcm_len: usize, pub output_capacity: usize, pub token_capacity: usize,
    pub encode: Option<FrameAction>, pub tokenize: Option<FrameAction>, pub plan: Option<FrameAction>, pub predict: Option<FrameAction>,
    pub graph: Option<FrameAction>, pub sample: Option<FrameAction>, pub detokenize: Option<FrameAction>, pub decode: Option<FrameAction>,
    pub text_token_out: Cell<i32>, pub sample_count_out: Cell<usize>, pub produced_out: Cell<bool>, pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>, pub on_error: Option<ErrorCallback>,
}
impl Default for StreamRun { fn default() -> Self { Self { pcm_len: 0, output_capacity: 0, token_capacity: 0, encode: None, tokenize: None, plan: None, predict: None, graph: None, sample: None, detokenize: None, decode: None, text_token_out: Cell::new(-1), sample_count_out: Cell::new(0), produced_out: Cell::new(false), error_out: Cell::new(SpeechGeneratorError::None), on_done: None, on_error: None } } }

#[derive(Debug, Clone)]
pub struct FlushRun {
    pub output_capacity: usize, pub token_capacity: usize,
    pub encode: Option<FrameAction>, pub tokenize: Option<FrameAction>, pub plan: Option<FrameAction>, pub predict: Option<FrameAction>,
    pub graph: Option<FrameAction>, pub sample: Option<FrameAction>, pub detokenize: Option<FrameAction>, pub decode: Option<FrameAction>,
    pub text_token_out: Cell<i32>, pub sample_count_out: Cell<usize>, pub produced_out: Cell<bool>, pub complete_out: Cell<bool>, pub error_out: Cell<SpeechGeneratorError>,
    pub on_done: Option<DoneCallback>, pub on_error: Option<ErrorCallback>,
}
impl Default for FlushRun { fn default() -> Self { Self { output_capacity: 0, token_capacity: 0, encode: None, tokenize: None, plan: None, predict: None, graph: None, sample: None, detokenize: None, decode: None, text_token_out: Cell::new(-1), sample_count_out: Cell::new(0), produced_out: Cell::new(false), complete_out: Cell::new(false), error_out: Cell::new(SpeechGeneratorError::None), on_done: None, on_error: None } } }

#[derive(Debug, Clone)]
pub struct GenerateRun { pub error_out: Cell<SpeechGeneratorError>, pub on_error: Option<ErrorCallback> }
impl Default for GenerateRun { fn default() -> Self { Self { error_out: Cell::new(SpeechGeneratorError::None), on_error: None } } }
#[derive(Debug, Clone, Default)]
pub struct EventReset { pub error_out: Cell<SpeechGeneratorError> }

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
        "state_flush_done_channel_decision"_s <= "state_flush_detokenize_result"_s + completion<FlushRun> [guard_frame_pending_dependencies_type_flush_run] / effect_publish_flush_pending_dependencies_type,
        "state_flush_error_channel_decision"_s <= "state_flush_detokenize_result"_s + completion<FlushRun> [guard_frame_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_detokenize_failed,
        "state_flush_done_channel_decision"_s <= "state_flush_decode_result"_s + completion<FlushRun> [guard_child_succeeded_dependencies_type_flush_run] / effect_publish_flush_produced_dependencies_type,
        "state_flush_error_channel_decision"_s <= "state_flush_decode_result"_s + completion<FlushRun> [guard_child_failed_dependencies_type_flush_run] / effect_fail_flush_dependencies_type_error_decode_failed,
        "state_flushing"_s <= "state_flush_done_channel_decision"_s + completion<FlushRun> [flush_done_present] / effect_emit_flush_done_dependencies_type,
        "state_flushing"_s <= "state_flush_done_channel_decision"_s + completion<FlushRun> [flush_done_absent],
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

#[derive(Debug, Default)]
pub struct SpeechGeneratorDuplexModelContext {
    pub initialized: bool, pub frame_samples: usize, pub codebook_count: usize,
    pub child_accepted: bool, pub child_err: SpeechGeneratorError, pub complete: bool,
    pub remaining: i32, pub produced: bool, pub text_token: i32, pub sample_count: usize, pub error: SpeechGeneratorError,
}

impl SpeechGeneratorDuplexModelContext {
    fn set_child(&mut self, result: Option<bool>) { self.child_accepted = result.unwrap_or(false); self.child_err = if result.is_some() { SpeechGeneratorError::None } else { SpeechGeneratorError::UnsupportedRequest }; }
    fn fail(&mut self, e: SpeechGeneratorError) { self.error = e; }
}

impl SpeechGeneratorDuplexModelStateMachineContext for SpeechGeneratorDuplexModelContext {
    fn condition_error_absent(&self, _event: &ConditionRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_none())
    }
    fn condition_error_present(&self, _event: &ConditionRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_some())
    }
    fn effect_begin_prompt_conditioning_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.complete = false; self.remaining = -1; self.set_child(_event.prompt_begin.map(|f| f(_event.token))); self.complete = self.child_accepted; self.remaining = 0; Ok(())
    }
    fn effect_capture_tokenizer_state_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.complete = false; self.remaining = -1; self.set_child(_event.capture_tokenizer.map(|f| f(_event.token))); self.complete = self.child_accepted; self.remaining = 0; Ok(())
    }
    fn effect_condition_prompt_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.complete = false; self.remaining = -1; self.set_child(_event.condition_prompt.map(|f| f(_event.token))); self.complete = self.child_accepted; self.remaining = 0; Ok(())
    }
    fn effect_condition_voice_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.complete = false; self.remaining = -1; self.set_child(_event.condition_voice.map(|f| f(_event.token))); self.complete = self.child_accepted; self.remaining = 0; Ok(())
    }
    fn effect_decode_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.decode.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_decode_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.decode.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_detokenize_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.detokenize.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_detokenize_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.detokenize.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_prompt_done_channel_decision(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_done { f(); } Ok(())
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_voice_done_channel_decision(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_done { f(); } Ok(())
    }
    fn effect_emit_condition_error_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_error { f(self.error); } Ok(())
    }
    fn effect_emit_flush_done_dependencies_type(&mut self, _event: &FlushRun) -> Result<(), ()> {
        if let Some(f) = _event.on_done { f(); } Ok(())
    }
    fn effect_emit_flush_error_dependencies_type(&mut self, _event: &FlushRun) -> Result<(), ()> {
        if let Some(f) = _event.on_error { f(self.error); } Ok(())
    }
    fn effect_emit_generation_error_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_error { f(self.error); } Ok(())
    }
    fn effect_emit_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_done { f(); } Ok(())
    }
    fn effect_emit_initialize_error_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_error { f(self.error); } Ok(())
    }
    fn effect_emit_stream_frame_done_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_done { f(); } Ok(())
    }
    fn effect_emit_stream_frame_error_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        if let Some(f) = _event.on_error { f(self.error); } Ok(())
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_flushing(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.encode.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_ready(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.encode.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_encode_stream_frame_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.encode.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_execute_prediction_graph_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.graph.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_execute_prediction_graph_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.graph.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_capture_tokenizer_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_begin_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_restore_tokenizer_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_voice_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_decode_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::DecodeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_detokenize_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::TokenizeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_encode_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::EncodeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_graph_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::GraphFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_flushing(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_ready(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_planning_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PlanningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_predict_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PredictFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_sample_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::SampleFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_flush_dependencies_type_error_tokenize_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::TokenizeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_generate_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::UnsupportedRequest; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_conditioning_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::ConditioningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_decoder_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::DecoderInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_encoder_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::EncoderInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_invalid_request(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_secondary_result(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::MemoryInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_temporal_result(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::MemoryInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_predictor_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PredictorInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_initialize_dependencies_type_error_tokenizer_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::TokenizerInitializeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_decode_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::DecodeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_detokenize_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::TokenizeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_encode_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::EncodeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_graph_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::GraphFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_invalid_request(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InvalidRequest; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_planning_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PlanningFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_predict_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::PredictFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_sample_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::SampleFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_fail_stream_frame_dependencies_type_error_tokenize_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::TokenizeFailed; _event.error_out.set(self.error); Ok(())
    }
    fn effect_initialize_conditioning_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_decoder_dependencies_type(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_encoder_dependencies_type(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_predictor_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_secondary_positions_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_temporal_positions_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_initialize_tokenizer_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.set_child(_event.initialize.map(|f| f())); Ok(())
    }
    fn effect_plan_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.plan.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_plan_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.plan.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_predict_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.predict.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_predict_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.predict.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_publish_condition_complete_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_begin_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        _event.complete_out.set(false); _event.remaining_out.set(self.remaining); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        _event.complete_out.set(false); _event.remaining_out.set(self.remaining); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_voice_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        _event.complete_out.set(false); _event.remaining_out.set(self.remaining); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_flush_pending_dependencies_type(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        _event.produced_out.set(false); _event.complete_out.set(false); _event.sample_count_out.set(if false { self.sample_count } else { 0 }); _event.text_token_out.set(if false { self.text_token } else { -1 }); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_flush_produced_dependencies_type(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        _event.produced_out.set(true); _event.complete_out.set(false); _event.sample_count_out.set(if true { self.sample_count } else { 0 }); _event.text_token_out.set(if true { self.text_token } else { -1 }); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn effect_publish_stream_frame_pending_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        _event.produced_out.set(false); _event.sample_count_out.set(if false { self.sample_count } else { 0 }); _event.text_token_out.set(if false { self.text_token } else { -1 }); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_publish_stream_frame_produced_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        _event.produced_out.set(true); _event.sample_count_out.set(if true { self.sample_count } else { 0 }); _event.text_token_out.set(if true { self.text_token } else { -1 }); _event.error_out.set(SpeechGeneratorError::None); Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_internal_error(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        _event.error_out.set(SpeechGeneratorError::InternalError); Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_uninitialized(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        _event.error_out.set(SpeechGeneratorError::Uninitialized); Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_flushing(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        _event.error_out.set(SpeechGeneratorError::UnsupportedRequest); Ok(())
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_ready(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        _event.error_out.set(SpeechGeneratorError::UnsupportedRequest); Ok(())
    }
    fn effect_restore_tokenizer_state_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        self.complete = false; self.remaining = -1; self.set_child(_event.restore_tokenizer.map(|f| f(_event.token))); self.complete = self.child_accepted; self.remaining = 0; Ok(())
    }
    fn effect_sample_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.sample.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_sample_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.sample.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_tokenize_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        self.set_child(_event.tokenize.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_tokenize_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        self.set_child(_event.tokenize.map(|f| f())); self.produced = self.child_accepted; self.sample_count = if self.produced { self.frame_samples } else { 0 }; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_condition_prompt(
        &mut self,
    ) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_condition_voice(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_errored(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_flushing(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_ready(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn effect_unexpected_dependencies_type_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.error = SpeechGeneratorError::InternalError; Ok(())
    }
    fn flush_done_absent(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_none())
    }
    fn flush_done_present(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_some())
    }
    fn flush_error_absent(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_none())
    }
    fn flush_error_present(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_some())
    }
    fn generate_error_absent(&self, _event: &GenerateRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_none())
    }
    fn generate_error_present(&self, _event: &GenerateRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_some())
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
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.complete && _event.on_done.is_none())
    }
    fn guard_condition_complete_done_callback_present_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.complete && _event.on_done.is_some())
    }
    fn guard_condition_failed_dependencies_type(&self, _event: &ConditionRun) -> Result<bool, ()> {
        Ok(!self.child_accepted || self.child_err != SpeechGeneratorError::None)
    }
    fn guard_condition_pending_done_callback_absent_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && !self.complete && _event.on_done.is_none())
    }
    fn guard_condition_pending_done_callback_present_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && !self.complete && _event.on_done.is_some())
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
    fn guard_flush_request_invalid_dependencies_type(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(!self.guard_flush_request_valid(e)?)
    }
    fn guard_flush_request_valid_dependencies_type(&self, _event: &FlushRun) -> Result<bool, ()> {
        Ok(self.initialized && _event.output_capacity >= self.frame_samples && _event.token_capacity >= self.codebook_count)
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
    fn guard_frame_produced_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(self.child_accepted && self.child_err == SpeechGeneratorError::None && self.produced)
    }
    fn guard_initialize_request_invalid_dependencies_type(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        Ok(!self.guard_initialize_request_valid(e)?)
    }
    fn guard_initialize_request_valid_dependencies_type(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        Ok(_event.model_ready && _event.tokenizer_ready && _event.codec_ready && _event.frame_samples > 0 && _event.frame_samples <= MAX_FRAME_SAMPLES && _event.codebook_count > 0 && _event.codebook_count <= MAX_CODEBOOKS && _event.initialize.is_some())
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
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        Ok(!self.guard_stream_request_valid(e)?)
    }
    fn guard_stream_request_valid_dependencies_type(&self, _event: &StreamRun) -> Result<bool, ()> {
        Ok(self.initialized && _event.pcm_len == self.frame_samples && _event.output_capacity >= self.frame_samples && _event.token_capacity >= self.codebook_count)
    }
    fn init_done_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_none())
    }
    fn init_done_present(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_some())
    }
    fn init_error_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_none())
    }
    fn init_error_present(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_some())
    }
    fn stream_done_absent(&self, _event: &StreamRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_none())
    }
    fn stream_done_present(&self, _event: &StreamRun) -> Result<bool, ()> {
        Ok(_event.on_done.is_some())
    }
    fn stream_error_absent(&self, _event: &StreamRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_none())
    }
    fn stream_error_present(&self, _event: &StreamRun) -> Result<bool, ()> {
        Ok(_event.on_error.is_some())
    }
}

pub struct SpeechGeneratorDuplexModelActor { machine: SpeechGeneratorDuplexModelStateMachine<SpeechGeneratorDuplexModelContext> }
impl Default for SpeechGeneratorDuplexModelActor { fn default() -> Self { Self::new() } }
impl SpeechGeneratorDuplexModelActor {
    pub fn new() -> Self { Self { machine: SpeechGeneratorDuplexModelStateMachine::new(SpeechGeneratorDuplexModelContext::default()) } }
    pub fn initialize(&mut self, e: InitRun) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn condition(&mut self, e: ConditionRun) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn generate(&mut self, e: GenerateRun) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn stream(&mut self, e: StreamRun) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn flush(&mut self, e: FlushRun) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn reset(&mut self, e: EventReset) -> Result<(), ()> { self.machine.process_event(e).map(|_| ()) }
    pub fn context(&self) -> &SpeechGeneratorDuplexModelContext { self.machine.context() }
    pub fn context_mut(&mut self) -> &mut SpeechGeneratorDuplexModelContext { self.machine.context_mut() }
    pub fn state(&self) -> &SpeechGeneratorDuplexModelStates { self.machine.state() }
    pub fn is(&self, state: &SpeechGeneratorDuplexModelStates) -> bool { self.machine.is(state) }
}
