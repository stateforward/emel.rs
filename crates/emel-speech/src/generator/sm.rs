//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct ConditionRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReset;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct FlushRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct GenerateRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct InitRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct StreamRun;

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

/// Context for `SpeechGeneratorDuplexModel` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechGeneratorDuplexModelContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechGeneratorDuplexModelStateMachineContext for SpeechGeneratorDuplexModelContext {
    fn condition_error_absent(&self, _event: &ConditionRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::condition_error_absent
        todo!(
            "TODO: port guard `condition_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn condition_error_present(&self, _event: &ConditionRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::condition_error_present
        todo!(
            "TODO: port guard `condition_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn effect_begin_prompt_conditioning_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_begin_prompt_conditioning
        todo!(
            "TODO: port action `effect_begin_prompt_conditioning` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_capture_tokenizer_state_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_capture_tokenizer_state` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_condition_prompt_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_condition_prompt
        todo!(
            "TODO: port action `effect_condition_prompt` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_condition_voice_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_condition_voice
        todo!(
            "TODO: port action `effect_condition_voice` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_decode_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_decode_frame
        todo!(
            "TODO: port action `effect_decode_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_decode_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_decode_frame
        todo!(
            "TODO: port action `effect_decode_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_detokenize_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_detokenize_frame
        todo!(
            "TODO: port action `effect_detokenize_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_detokenize_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_detokenize_frame
        todo!(
            "TODO: port action `effect_detokenize_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_prompt_done_channel_decision(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_condition_done
        todo!(
            "TODO: port action `effect_emit_condition_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_condition_done_dependencies_type_from_state_condition_voice_done_channel_decision(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_condition_done
        todo!(
            "TODO: port action `effect_emit_condition_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_condition_error_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_condition_error
        todo!(
            "TODO: port action `effect_emit_condition_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_flush_done_dependencies_type(&mut self, _event: &FlushRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_flush_done
        todo!(
            "TODO: port action `effect_emit_flush_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_flush_error_dependencies_type(&mut self, _event: &FlushRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_flush_error
        todo!(
            "TODO: port action `effect_emit_flush_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_generation_error_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_generation_error
        todo!(
            "TODO: port action `effect_emit_generation_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_initialize_error_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_stream_frame_done_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_stream_frame_done
        todo!(
            "TODO: port action `effect_emit_stream_frame_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_stream_frame_error_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_stream_frame_error
        todo!(
            "TODO: port action `effect_emit_stream_frame_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_flushing(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_encode_flush_frame
        todo!(
            "TODO: port action `effect_encode_flush_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_encode_flush_frame_dependencies_type_from_state_ready(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_encode_flush_frame
        todo!(
            "TODO: port action `effect_encode_flush_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_encode_stream_frame_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_encode_stream_frame
        todo!(
            "TODO: port action `effect_encode_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_execute_prediction_graph_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_execute_prediction_graph
        todo!(
            "TODO: port action `effect_execute_prediction_graph` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_execute_prediction_graph_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_execute_prediction_graph
        todo!(
            "TODO: port action `effect_execute_prediction_graph` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_capture_tokenizer_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_begin_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_prompt_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_restore_tokenizer_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_conditioning_failed_from_state_condition_voice_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_decode_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_detokenize_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_encode_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_graph_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_flushing(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_invalid_request_from_state_ready(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_planning_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_predict_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_sample_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_tokenize_failed(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_conditioning_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_decoder_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_encoder_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_invalid_request(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_secondary_result(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_memory_initialize_failed_from_state_initialize_temporal_result(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_predictor_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_tokenizer_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_decode_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_detokenize_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_encode_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_graph_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_invalid_request(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_planning_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_predict_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_sample_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_tokenize_failed(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_conditioning_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_conditioning
        todo!(
            "TODO: port action `effect_initialize_conditioning` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_decoder_dependencies_type(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_decoder
        todo!(
            "TODO: port action `effect_initialize_decoder` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_encoder_dependencies_type(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_encoder
        todo!(
            "TODO: port action `effect_initialize_encoder` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_predictor_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_predictor
        todo!(
            "TODO: port action `effect_initialize_predictor` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_secondary_positions_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_secondary_positions
        todo!(
            "TODO: port action `effect_initialize_secondary_positions` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_temporal_positions_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_temporal_positions
        todo!(
            "TODO: port action `effect_initialize_temporal_positions` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_tokenizer_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_tokenizer
        todo!(
            "TODO: port action `effect_initialize_tokenizer` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_plan_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_plan_frame
        todo!(
            "TODO: port action `effect_plan_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_plan_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_plan_frame
        todo!(
            "TODO: port action `effect_plan_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_predict_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_predict_frame
        todo!(
            "TODO: port action `effect_predict_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_predict_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_predict_frame
        todo!(
            "TODO: port action `effect_predict_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_condition_complete_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_condition_complete
        todo!(
            "TODO: port action `effect_publish_condition_complete` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_begin_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_condition_pending
        todo!(
            "TODO: port action `effect_publish_condition_pending` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_prompt_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_condition_pending
        todo!(
            "TODO: port action `effect_publish_condition_pending` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_condition_pending_dependencies_type_from_state_condition_voice_result(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_condition_pending
        todo!(
            "TODO: port action `effect_publish_condition_pending` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_flush_pending_dependencies_type(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_flush_pending
        todo!(
            "TODO: port action `effect_publish_flush_pending` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_flush_produced_dependencies_type(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_flush_produced
        todo!(
            "TODO: port action `effect_publish_flush_produced` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_initialize_done
        todo!(
            "TODO: port action `effect_publish_initialize_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_stream_frame_pending_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_stream_frame_pending
        todo!(
            "TODO: port action `effect_publish_stream_frame_pending` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_stream_frame_produced_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_stream_frame_produced
        todo!(
            "TODO: port action `effect_publish_stream_frame_produced` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_internal_error(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_uninitialized(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_flushing(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request_from_state_ready(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_restore_tokenizer_state_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_restore_tokenizer_state
        todo!(
            "TODO: port action `effect_restore_tokenizer_state` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_sample_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_sample_frame
        todo!(
            "TODO: port action `effect_sample_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_sample_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_sample_frame
        todo!(
            "TODO: port action `effect_sample_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_tokenize_frame_dependencies_type_flush_run(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_tokenize_frame
        todo!(
            "TODO: port action `effect_tokenize_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_tokenize_frame_dependencies_type_stream_run(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_tokenize_frame
        todo!(
            "TODO: port action `effect_tokenize_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_condition_prompt(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_condition_voice(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_flushing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn flush_done_absent(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_done_absent
        todo!(
            "TODO: port guard `flush_done_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn flush_done_present(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_done_present
        todo!(
            "TODO: port guard `flush_done_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn flush_error_absent(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_error_absent
        todo!(
            "TODO: port guard `flush_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn flush_error_present(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_error_present
        todo!(
            "TODO: port guard `flush_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_error_absent(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_error_absent
        todo!(
            "TODO: port guard `generate_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_error_present(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_error_present
        todo!(
            "TODO: port guard `generate_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_condition_run(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_condition_run(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_init_run(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_complete_done_callback_absent_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_complete_done_callback_absent
        todo!(
            "TODO: port guard `guard_condition_complete_done_callback_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_complete_done_callback_present_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_complete_done_callback_present
        todo!(
            "TODO: port guard `guard_condition_complete_done_callback_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_failed_dependencies_type(&self, _event: &ConditionRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_failed
        todo!(
            "TODO: port guard `guard_condition_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_pending_done_callback_absent_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_pending_done_callback_absent
        todo!(
            "TODO: port guard `guard_condition_pending_done_callback_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_pending_done_callback_present_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_pending_done_callback_present
        todo!(
            "TODO: port guard `guard_condition_pending_done_callback_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_succeeded_complete_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_succeeded_complete
        todo!(
            "TODO: port guard `guard_condition_succeeded_complete` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_condition_succeeded_pending_dependencies_type(
        &self,
        _event: &ConditionRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_condition_succeeded_pending
        todo!(
            "TODO: port guard `guard_condition_succeeded_pending` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_flush_request_invalid_dependencies_type(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_flush_request_invalid
        todo!(
            "TODO: port guard `guard_flush_request_invalid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_flush_request_valid_dependencies_type(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_flush_request_valid
        todo!(
            "TODO: port guard `guard_flush_request_valid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_failed
        todo!(
            "TODO: port guard `guard_frame_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_failed
        todo!(
            "TODO: port guard `guard_frame_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_pending_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_pending
        todo!(
            "TODO: port guard `guard_frame_pending` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_pending_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_pending
        todo!(
            "TODO: port guard `guard_frame_pending` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_produced_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_produced
        todo!(
            "TODO: port guard `guard_frame_produced` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_frame_produced_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_frame_produced
        todo!(
            "TODO: port guard `guard_frame_produced` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_initialize_request_invalid_dependencies_type(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_initialize_request_invalid
        todo!(
            "TODO: port guard `guard_initialize_request_invalid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_initialize_request_valid_dependencies_type(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_initialize_request_valid
        todo!(
            "TODO: port guard `guard_initialize_request_valid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_prediction_failed_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_prediction_failed
        todo!(
            "TODO: port guard `guard_prediction_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_prediction_failed_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_prediction_failed
        todo!(
            "TODO: port guard `guard_prediction_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_prediction_succeeded_dependencies_type_flush_run(
        &self,
        _event: &FlushRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_prediction_succeeded
        todo!(
            "TODO: port guard `guard_prediction_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_prediction_succeeded_dependencies_type_stream_run(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_prediction_succeeded
        todo!(
            "TODO: port guard `guard_prediction_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_stream_request_invalid_dependencies_type(
        &self,
        _event: &StreamRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_stream_request_invalid
        todo!(
            "TODO: port guard `guard_stream_request_invalid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_stream_request_valid_dependencies_type(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_stream_request_valid
        todo!(
            "TODO: port guard `guard_stream_request_valid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_done_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_done_absent
        todo!(
            "TODO: port guard `init_done_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_done_present(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_done_present
        todo!(
            "TODO: port guard `init_done_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_error_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_error_absent
        todo!(
            "TODO: port guard `init_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_error_present(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_error_present
        todo!(
            "TODO: port guard `init_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_done_absent(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_done_absent
        todo!(
            "TODO: port guard `stream_done_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_done_present(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_done_present
        todo!(
            "TODO: port guard `stream_done_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_error_absent(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_error_absent
        todo!(
            "TODO: port guard `stream_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_error_present(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_error_present
        todo!(
            "TODO: port guard `stream_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
}

// --- machine SpeechGeneratorSynthesisModel from emel.cpp/src/emel/speech/generator/sm.hpp ---
sml! {
    SpeechGeneratorSynthesisModel {
        "state_initialize_synthesis_conditioner_result"_s <= *"state_uninitialized"_s + event<InitRun> / effect_initialize_synthesis_conditioner_dependencies_type,
        "state_initialize_synthesis_prefiller_result"_s <= "state_initialize_synthesis_conditioner_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_synthesis_prefiller_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_conditioner_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_conditioner_initialize_failed,
        "state_initialize_synthesis_predictor_result"_s <= "state_initialize_synthesis_prefiller_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_synthesis_predictor_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_prefiller_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_prefiller_initialize_failed,
        "state_initialize_synthesis_sampler_result"_s <= "state_initialize_synthesis_predictor_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_synthesis_sampler_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_predictor_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_predictor_initialize_failed,
        "state_initialize_synthesis_decoder_result"_s <= "state_initialize_synthesis_sampler_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_synthesis_decoder_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_sampler_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_sampler_initialize_failed,
        "state_initialize_synthesis_postprocessor_result"_s <= "state_initialize_synthesis_decoder_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_initialize_synthesis_postprocessor_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_decoder_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_decoder_initialize_failed,
        "state_initialize_done_channel_decision"_s <= "state_initialize_synthesis_postprocessor_result"_s + completion<InitRun> [guard_child_succeeded_dependencies_type_init_run] / effect_publish_initialize_done_dependencies_type,
        "state_initialize_error_channel_decision"_s <= "state_initialize_synthesis_postprocessor_result"_s + completion<InitRun> [guard_child_failed_dependencies_type_init_run] / effect_fail_initialize_dependencies_type_error_postprocessor_initialize_failed,
        "state_ready"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_present] / effect_emit_synthesis_initialize_done_dependencies_type,
        "state_ready"_s <= "state_initialize_done_channel_decision"_s + completion<InitRun> [init_done_absent],
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_present] / effect_emit_initialize_error_dependencies_type,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<InitRun> [init_error_absent],
        "state_generate_conditioning"_s <= "state_ready"_s + event<GenerateRun> [guard_generate_request_valid_dependencies_type] / effect_condition_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_ready"_s + event<GenerateRun> [guard_generate_request_invalid_dependencies_type] / effect_fail_generate_dependencies_type_error_invalid_request,
        "state_generate_prefill"_s <= "state_generate_conditioning"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_prefill_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_conditioning"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_conditioning_failed,
        "state_generate_predict"_s <= "state_generate_prefill"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_predict_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_prefill"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_prefill_failed,
        "state_generate_sample"_s <= "state_generate_predict"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_sample_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_predict"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_predict_failed,
        "state_generate_decode"_s <= "state_generate_sample"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_decode_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_sample"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_sample_failed,
        "state_generate_postprocess"_s <= "state_generate_decode"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_postprocess_generate_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_decode"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_decode_failed,
        "state_generate_done_channel_decision"_s <= "state_generate_postprocess"_s + completion<GenerateRun> [guard_child_succeeded_dependencies_type_generate_run] / effect_publish_generation_done_dependencies_type,
        "state_generate_error_channel_decision"_s <= "state_generate_postprocess"_s + completion<GenerateRun> [guard_child_failed_dependencies_type_generate_run] / effect_fail_generate_dependencies_type_error_postprocess_failed,
        "state_ready"_s <= "state_generate_done_channel_decision"_s + completion<GenerateRun> [generate_done_present] / effect_emit_generation_done_dependencies_type,
        "state_ready"_s <= "state_generate_done_channel_decision"_s + completion<GenerateRun> [generate_done_absent],
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_present] / effect_emit_generation_error_dependencies_type,
        "state_ready"_s <= "state_generate_error_channel_decision"_s + completion<GenerateRun> [generate_error_absent],
        "state_condition_error_channel_decision"_s <= "state_ready"_s + event<ConditionRun> / effect_fail_condition_dependencies_type_error_unsupported_request,
        "state_ready"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_present] / effect_emit_condition_error_dependencies_type,
        "state_ready"_s <= "state_condition_error_channel_decision"_s + completion<ConditionRun> [condition_error_absent],
        "state_stream_error_channel_decision"_s <= "state_ready"_s + event<StreamRun> / effect_fail_stream_frame_dependencies_type_error_unsupported_request,
        "state_ready"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_present] / effect_emit_stream_frame_error_dependencies_type,
        "state_ready"_s <= "state_stream_error_channel_decision"_s + completion<StreamRun> [stream_error_absent],
        "state_flush_error_channel_decision"_s <= "state_ready"_s + event<FlushRun> / effect_fail_flush_dependencies_type_error_unsupported_request,
        "state_ready"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_present] / effect_emit_flush_error_dependencies_type,
        "state_ready"_s <= "state_flush_error_channel_decision"_s + completion<FlushRun> [flush_error_absent],
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_uninitialized,
        "state_ready"_s <= "state_ready"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_unsupported_request,
        "state_errored"_s <= "state_errored"_s + event<EventReset> / effect_reject_reset_dependencies_type_error_internal_error,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_uninitialized,
        "state_errored"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_ready,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_dependencies_type_from_state_errored,
    }
}

/// Context for `SpeechGeneratorSynthesisModel` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechGeneratorSynthesisModelContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechGeneratorSynthesisModelStateMachineContext for SpeechGeneratorSynthesisModelContext {
    fn condition_error_absent(&self, _event: &ConditionRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::condition_error_absent
        todo!(
            "TODO: port guard `condition_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn condition_error_present(&self, _event: &ConditionRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::condition_error_present
        todo!(
            "TODO: port guard `condition_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn effect_condition_generate_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_condition_generate
        todo!(
            "TODO: port action `effect_condition_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_decode_generate_dependencies_type(&mut self, _event: &GenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_decode_generate
        todo!(
            "TODO: port action `effect_decode_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_condition_error_dependencies_type(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_condition_error
        todo!(
            "TODO: port action `effect_emit_condition_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_flush_error_dependencies_type(&mut self, _event: &FlushRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_flush_error
        todo!(
            "TODO: port action `effect_emit_flush_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_generation_done_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_generation_done
        todo!(
            "TODO: port action `effect_emit_generation_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_generation_error_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_generation_error
        todo!(
            "TODO: port action `effect_emit_generation_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_initialize_error_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_stream_frame_error_dependencies_type(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_stream_frame_error
        todo!(
            "TODO: port action `effect_emit_stream_frame_error` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_emit_synthesis_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_emit_synthesis_initialize_done
        todo!(
            "TODO: port action `effect_emit_synthesis_initialize_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_condition_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &ConditionRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_condition
        todo!(
            "TODO: port action `effect_fail_condition` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_flush_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &FlushRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_flush
        todo!(
            "TODO: port action `effect_fail_flush` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_conditioning_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_decode_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_invalid_request(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_postprocess_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_predict_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_prefill_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_generate_dependencies_type_error_sample_failed(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_generate
        todo!(
            "TODO: port action `effect_fail_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_conditioner_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_decoder_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_postprocessor_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_predictor_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_prefiller_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_initialize_dependencies_type_error_sampler_initialize_failed(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_initialize
        todo!(
            "TODO: port action `effect_fail_initialize` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_fail_stream_frame_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &StreamRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_fail_stream_frame
        todo!(
            "TODO: port action `effect_fail_stream_frame` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_conditioner_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_conditioner
        todo!(
            "TODO: port action `effect_initialize_synthesis_conditioner` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_decoder_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_decoder
        todo!(
            "TODO: port action `effect_initialize_synthesis_decoder` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_postprocessor_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_postprocessor
        todo!(
            "TODO: port action `effect_initialize_synthesis_postprocessor` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_predictor_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_predictor
        todo!(
            "TODO: port action `effect_initialize_synthesis_predictor` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_prefiller_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_prefiller
        todo!(
            "TODO: port action `effect_initialize_synthesis_prefiller` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_initialize_synthesis_sampler_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_initialize_synthesis_sampler
        todo!(
            "TODO: port action `effect_initialize_synthesis_sampler` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_postprocess_generate_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_postprocess_generate
        todo!(
            "TODO: port action `effect_postprocess_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_predict_generate_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_predict_generate
        todo!(
            "TODO: port action `effect_predict_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_prefill_generate_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_prefill_generate
        todo!(
            "TODO: port action `effect_prefill_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_generation_done_dependencies_type(
        &mut self,
        _event: &GenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_generation_done
        todo!(
            "TODO: port action `effect_publish_generation_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_publish_initialize_done_dependencies_type(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_publish_initialize_done
        todo!(
            "TODO: port action `effect_publish_initialize_done` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_internal_error(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_uninitialized(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_reject_reset_dependencies_type_error_unsupported_request(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_reject_reset
        todo!(
            "TODO: port action `effect_reject_reset` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_sample_generate_dependencies_type(&mut self, _event: &GenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_sample_generate
        todo!(
            "TODO: port action `effect_sample_generate` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn effect_unexpected_dependencies_type_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/generator/actions.hpp"
        )
    }
    fn flush_error_absent(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_error_absent
        todo!(
            "TODO: port guard `flush_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn flush_error_present(&self, _event: &FlushRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::flush_error_present
        todo!(
            "TODO: port guard `flush_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_done_absent(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_done_absent
        todo!(
            "TODO: port guard `generate_done_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_done_present(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_done_present
        todo!(
            "TODO: port guard `generate_done_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_error_absent(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_error_absent
        todo!(
            "TODO: port guard `generate_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn generate_error_present(&self, _event: &GenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::generate_error_present
        todo!(
            "TODO: port guard `generate_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_generate_run(
        &self,
        _event: &GenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_failed_dependencies_type_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_failed
        todo!(
            "TODO: port guard `guard_child_failed` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_generate_run(
        &self,
        _event: &GenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_child_succeeded_dependencies_type_init_run(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_child_succeeded
        todo!(
            "TODO: port guard `guard_child_succeeded` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_generate_request_invalid_dependencies_type(
        &self,
        _event: &GenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_generate_request_invalid
        todo!(
            "TODO: port guard `guard_generate_request_invalid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn guard_generate_request_valid_dependencies_type(
        &self,
        _event: &GenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::guard_generate_request_valid
        todo!(
            "TODO: port guard `guard_generate_request_valid` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_done_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_done_absent
        todo!(
            "TODO: port guard `init_done_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_done_present(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_done_present
        todo!(
            "TODO: port guard `init_done_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_error_absent(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_error_absent
        todo!(
            "TODO: port guard `init_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn init_error_present(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::init_error_present
        todo!(
            "TODO: port guard `init_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_error_absent(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_error_absent
        todo!(
            "TODO: port guard `stream_error_absent` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
    fn stream_error_present(&self, _event: &StreamRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/generator/guards.hpp::stream_error_present
        todo!(
            "TODO: port guard `stream_error_present` from emel.cpp/src/emel/speech/generator/guards.hpp"
        )
    }
}
