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

// --- machine SpeechPredictorMoshi from emel.cpp/src/emel/speech/predictor/moshi/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct BeginPromptRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventCaptureTokenizerState;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct ExecuteRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct InitRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct LoadVoiceRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct PredictRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct PrefillPromptRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct PrefillVoiceRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct ResetRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct SampleRun;

sml! {
    SpeechPredictorMoshi {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + event<InitRun>,
        "state_binding"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_valid] / effect_bind_contract,
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_lmgen_mode_decision"_s <= "state_binding"_s + completion<InitRun>,
        "state_configure_lmgen"_s <= "state_lmgen_mode_decision"_s + completion<InitRun> [guard_personaplex_lmgen] / effect_configure_personaplex_lmgen,
        "state_configure_lmgen"_s <= "state_lmgen_mode_decision"_s + completion<InitRun> [guard_standard_lmgen] / effect_configure_standard_lmgen,
        "state_initialize_graph"_s <= "state_configure_lmgen"_s + completion<InitRun> / effect_initialize_graph_graph_actor_type,
        "state_initialize_graph_decision"_s <= "state_initialize_graph"_s + completion<InitRun>,
        "state_reserve_memory"_s <= "state_initialize_graph_decision"_s + completion<InitRun> [guard_graph_initialize_succeeded] / effect_reserve_memory,
        "state_init_failed_error_out_decision"_s <= "state_initialize_graph_decision"_s + completion<InitRun> [guard_graph_initialize_failed] / effect_mark_graph_runtime_error_init_run,
        "state_reserve_memory_decision"_s <= "state_reserve_memory"_s + completion<InitRun>,
        "state_allocate_sequence"_s <= "state_reserve_memory_decision"_s + completion<InitRun> [guard_memory_accepted_init_run] / effect_allocate_sequence,
        "state_init_failed_error_out_decision"_s <= "state_reserve_memory_decision"_s + completion<InitRun> [guard_memory_rejected_init_run] / effect_mark_memory_error_init_run_from_state_reserve_memory_decision,
        "state_allocate_sequence_decision"_s <= "state_allocate_sequence"_s + completion<InitRun>,
        "state_init_error_out_decision"_s <= "state_allocate_sequence_decision"_s + completion<InitRun> [guard_memory_accepted_init_run],
        "state_init_failed_error_out_decision"_s <= "state_allocate_sequence_decision"_s + completion<InitRun> [guard_memory_rejected_init_run] / effect_mark_memory_error_init_run_from_state_allocate_sequence_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_error_out_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_failed_error_out_decision,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_has_done_callback_init_run] / effect_emit_initialize_done,
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_no_done_callback_init_run],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_has_error_callback_init_run] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_no_error_callback_init_run],
        "state_voice_request_decision"_s <= "state_session_ready"_s + event<LoadVoiceRun>,
        "state_voice_error_out_decision"_s <= "state_voice_request_decision"_s + completion<LoadVoiceRun> [guard_voice_contract_valid] / effect_bind_voice_contract,
        "state_voice_failed_error_out_decision"_s <= "state_voice_request_decision"_s + completion<LoadVoiceRun> [guard_voice_contract_invalid] / effect_mark_voice_contract_error_load_voice_run,
        "state_voice_callback_decision"_s <= "state_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_voice_error_out_decision,
        "state_voice_callback_decision"_s <= "state_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_voice_failed_callback_decision"_s <= "state_voice_failed_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_voice_failed_error_out_decision,
        "state_voice_failed_callback_decision"_s <= "state_voice_failed_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_session_ready"_s <= "state_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_has_done_callback_load_voice_run] / effect_emit_load_voice_done,
        "state_session_ready"_s <= "state_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_no_done_callback_load_voice_run],
        "state_session_ready"_s <= "state_voice_failed_callback_decision"_s + completion<LoadVoiceRun> [guard_has_error_callback_load_voice_run] / effect_emit_load_voice_error_from_state_voice_failed_callback_decision,
        "state_session_ready"_s <= "state_voice_failed_callback_decision"_s + completion<LoadVoiceRun> [guard_no_error_callback_load_voice_run],
        "state_prefill_voice_request_decision"_s <= "state_session_ready"_s + event<PrefillVoiceRun>,
        "state_prefill_voice_allocate_slot"_s <= "state_prefill_voice_request_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_request_valid] / effect_allocate_voice_prefill_slot,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_request_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_request_invalid] / effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_request_decision,
        "state_prefill_voice_allocate_slot_decision"_s <= "state_prefill_voice_allocate_slot"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_capture_memory"_s <= "state_prefill_voice_allocate_slot_decision"_s + completion<PrefillVoiceRun> [guard_memory_accepted_prefill_voice_run] / effect_capture_voice_prefill_memory,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_allocate_slot_decision"_s + completion<PrefillVoiceRun> [guard_memory_rejected_prefill_voice_run] / effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_allocate_slot_decision,
        "state_prefill_voice_capture_memory_decision"_s <= "state_prefill_voice_capture_memory"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_begin"_s <= "state_prefill_voice_capture_memory_decision"_s + completion<PrefillVoiceRun> [guard_memory_accepted_prefill_voice_run] / effect_begin_voice_prefill,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_capture_memory_decision"_s + completion<PrefillVoiceRun> [guard_memory_rejected_prefill_voice_run] / effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_capture_memory_decision,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_f32] / effect_load_voice_embedding_frame_f32,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_f16] / effect_load_voice_embedding_frame_f16,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_bf16] / effect_load_voice_embedding_frame_bf16,
        "state_prefill_voice_embedding_frame_decision"_s <= "state_prefill_voice_embedding_frame"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_graph_runtime_decision"_s <= "state_prefill_voice_embedding_frame_decision"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_loaded],
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_embedding_frame_decision"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_failed] / effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_embedding_frame_decision,
        "state_prefill_voice_running_graph"_s <= "state_prefill_voice_graph_runtime_decision"_s + completion<PrefillVoiceRun> / effect_run_voice_graph_runtime_graph_actor_type,
        "state_prefill_voice_graph_error_out_decision"_s <= "state_prefill_voice_running_graph"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_graph_result_decision"_s <= "state_prefill_voice_graph_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_graph_error_out_prefill_voice_run] / effect_store_graph_error_out_prefill_voice_run,
        "state_prefill_voice_graph_result_decision"_s <= "state_prefill_voice_graph_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_graph_error_out_prefill_voice_run],
        "state_prefill_voice_advance"_s <= "state_prefill_voice_graph_result_decision"_s + completion<PrefillVoiceRun> [guard_graph_step_accepted_prefill_voice_run] / effect_advance_voice_prefill,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_graph_result_decision"_s + completion<PrefillVoiceRun> [guard_graph_step_rejected_prefill_voice_run] / effect_mark_graph_runtime_error_prefill_voice_run,
        "state_prefill_voice_complete_decision"_s <= "state_prefill_voice_advance"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_output_decision"_s <= "state_prefill_voice_complete_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_complete] / effect_copy_voice_cache,
        "state_prefill_voice_output_decision"_s <= "state_prefill_voice_complete_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_pending],
        "state_prefill_voice_remaining_out_decision"_s <= "state_prefill_voice_output_decision"_s + completion<PrefillVoiceRun> [guard_has_voice_complete_out] / effect_store_voice_complete_out,
        "state_prefill_voice_remaining_out_decision"_s <= "state_prefill_voice_output_decision"_s + completion<PrefillVoiceRun> [guard_no_voice_complete_out],
        "state_prefill_voice_error_out_decision"_s <= "state_prefill_voice_remaining_out_decision"_s + completion<PrefillVoiceRun> [guard_has_voice_remaining_out] / effect_store_voice_remaining_out,
        "state_prefill_voice_error_out_decision"_s <= "state_prefill_voice_remaining_out_decision"_s + completion<PrefillVoiceRun> [guard_no_voice_remaining_out],
        "state_prefill_voice_callback_decision"_s <= "state_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_prefill_voice_error_out_decision,
        "state_prefill_voice_callback_decision"_s <= "state_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_prefill_voice_failed_callback_decision"_s <= "state_prefill_voice_failed_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_prefill_voice_failed_error_out_decision,
        "state_prefill_voice_failed_callback_decision"_s <= "state_prefill_voice_failed_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_session_ready"_s <= "state_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_done_callback_prefill_voice_run] / effect_emit_prefill_voice_done,
        "state_session_ready"_s <= "state_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_done_callback_prefill_voice_run],
        "state_session_ready"_s <= "state_prefill_voice_failed_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_error_callback_prefill_voice_run] / effect_emit_prefill_voice_error_from_state_prefill_voice_failed_callback_decision,
        "state_session_ready"_s <= "state_prefill_voice_failed_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_error_callback_prefill_voice_run],
        "state_personaplex_prompt_begin_decision"_s <= "state_session_ready"_s + event<BeginPromptRun>,
        "state_personaplex_prompt_begin_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_nonempty_valid] / effect_bind_personaplex_prompt,
        "state_personaplex_prompt_begin_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_empty_valid] / effect_bind_empty_personaplex_prompt,
        "state_personaplex_prompt_begin_failed_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_invalid] / effect_mark_personaplex_prompt_error_begin_prompt_run,
        "state_personaplex_prompt_begin_callback_decision"_s <= "state_personaplex_prompt_begin_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_error_out_decision,
        "state_personaplex_prompt_begin_callback_decision"_s <= "state_personaplex_prompt_begin_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_personaplex_prompt_begin_failed_callback_decision"_s <= "state_personaplex_prompt_begin_failed_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_failed_error_out_decision,
        "state_personaplex_prompt_begin_failed_callback_decision"_s <= "state_personaplex_prompt_begin_failed_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_session_ready"_s <= "state_personaplex_prompt_begin_callback_decision"_s + completion<BeginPromptRun> [guard_has_done_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_done,
        "state_session_ready"_s <= "state_personaplex_prompt_begin_callback_decision"_s + completion<BeginPromptRun> [guard_no_done_callback_begin_prompt_run],
        "state_session_ready"_s <= "state_personaplex_prompt_begin_failed_callback_decision"_s + completion<BeginPromptRun> [guard_has_error_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_error_from_state_personaplex_prompt_begin_failed_callback_decision,
        "state_session_ready"_s <= "state_personaplex_prompt_begin_failed_callback_decision"_s + completion<BeginPromptRun> [guard_no_error_callback_begin_prompt_run],
        "state_prefill_personaplex_prompt_request_decision"_s <= "state_session_ready"_s + event<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_allocate_slot"_s <= "state_prefill_personaplex_prompt_request_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_prefill_request_valid] / effect_allocate_personaplex_prompt_slot,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_request_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_prefill_request_invalid] / effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_request_decision,
        "state_prefill_personaplex_prompt_allocate_slot_decision"_s <= "state_prefill_personaplex_prompt_allocate_slot"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_capture_memory"_s <= "state_prefill_personaplex_prompt_allocate_slot_decision"_s + completion<PrefillPromptRun> [guard_memory_accepted_prefill_prompt_run] / effect_capture_personaplex_prompt_memory,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_allocate_slot_decision"_s + completion<PrefillPromptRun> [guard_memory_rejected_prefill_prompt_run] / effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_allocate_slot_decision,
        "state_prefill_personaplex_prompt_capture_memory_decision"_s <= "state_prefill_personaplex_prompt_capture_memory"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_begin"_s <= "state_prefill_personaplex_prompt_capture_memory_decision"_s + completion<PrefillPromptRun> [guard_memory_accepted_prefill_prompt_run] / effect_begin_personaplex_prompt_prefill,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_capture_memory_decision"_s + completion<PrefillPromptRun> [guard_memory_rejected_prefill_prompt_run] / effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_capture_memory_decision,
        "state_prefill_personaplex_prompt_phase_decision"_s <= "state_prefill_personaplex_prompt_begin"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_pre_silence_pending] / effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_text_pending] / effect_build_personaplex_prompt_text_frame,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_post_silence_pending] / effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_phase_invalid] / effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_write_input"_s <= "state_prefill_personaplex_prompt_frame"_s + completion<PrefillPromptRun> / effect_write_and_build_personaplex_prompt_input,
        "state_prefill_personaplex_prompt_graph_runtime_decision"_s <= "state_prefill_personaplex_prompt_write_input"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_running_graph"_s <= "state_prefill_personaplex_prompt_graph_runtime_decision"_s + completion<PrefillPromptRun> / effect_run_personaplex_prompt_graph_runtime_graph_actor_type,
        "state_prefill_personaplex_prompt_graph_error_out_decision"_s <= "state_prefill_personaplex_prompt_running_graph"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_graph_result_decision"_s <= "state_prefill_personaplex_prompt_graph_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_graph_error_out_prefill_prompt_run] / effect_store_graph_error_out_prefill_prompt_run,
        "state_prefill_personaplex_prompt_graph_result_decision"_s <= "state_prefill_personaplex_prompt_graph_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_graph_error_out_prefill_prompt_run],
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_pre_silence] / effect_advance_personaplex_prompt_pre_silence,
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_text] / effect_advance_personaplex_prompt_text,
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_post_silence] / effect_advance_personaplex_prompt_post_silence,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_rejected_prefill_prompt_run] / effect_mark_graph_runtime_error_prefill_prompt_run,
        "state_prefill_personaplex_prompt_complete_decision"_s <= "state_prefill_personaplex_prompt_advance"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_output_decision"_s <= "state_prefill_personaplex_prompt_complete_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_complete] / effect_finish_personaplex_prompt,
        "state_prefill_personaplex_prompt_output_decision"_s <= "state_prefill_personaplex_prompt_complete_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_pending] / effect_publish_personaplex_prompt_pending,
        "state_prefill_personaplex_prompt_remaining_out_decision"_s <= "state_prefill_personaplex_prompt_output_decision"_s + completion<PrefillPromptRun> [guard_has_personaplex_prompt_complete_out] / effect_store_personaplex_prompt_complete_out,
        "state_prefill_personaplex_prompt_remaining_out_decision"_s <= "state_prefill_personaplex_prompt_output_decision"_s + completion<PrefillPromptRun> [guard_no_personaplex_prompt_complete_out],
        "state_prefill_personaplex_prompt_error_out_decision"_s <= "state_prefill_personaplex_prompt_remaining_out_decision"_s + completion<PrefillPromptRun> [guard_has_personaplex_prompt_remaining_out] / effect_store_personaplex_prompt_remaining_out,
        "state_prefill_personaplex_prompt_error_out_decision"_s <= "state_prefill_personaplex_prompt_remaining_out_decision"_s + completion<PrefillPromptRun> [guard_no_personaplex_prompt_remaining_out],
        "state_prefill_personaplex_prompt_callback_decision"_s <= "state_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_error_out_decision,
        "state_prefill_personaplex_prompt_callback_decision"_s <= "state_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_prefill_personaplex_prompt_failed_callback_decision"_s <= "state_prefill_personaplex_prompt_failed_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_failed_error_out_decision,
        "state_prefill_personaplex_prompt_failed_callback_decision"_s <= "state_prefill_personaplex_prompt_failed_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_has_done_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_done,
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_no_done_callback_prefill_prompt_run],
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_failed_callback_decision"_s + completion<PrefillPromptRun> [guard_has_error_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_error_from_state_prefill_personaplex_prompt_failed_callback_decision,
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_failed_callback_decision"_s + completion<PrefillPromptRun> [guard_no_error_callback_prefill_prompt_run],
        "state_predict_request_decision"_s <= "state_session_ready"_s + event<PredictRun>,
        "state_predict_allocate_slot"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_request_valid] / effect_allocate_step_slot,
        "state_predict_failed_error_out_decision"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_blocked_by_voice_prompt] / effect_mark_voice_prompt_pending,
        "state_predict_failed_error_out_decision"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_request_shape_invalid] / effect_mark_step_request_invalid_predict_run,
        "state_predict_allocate_slot_decision"_s <= "state_predict_allocate_slot"_s + completion<PredictRun>,
        "state_predict_capture_memory"_s <= "state_predict_allocate_slot_decision"_s + completion<PredictRun> [guard_memory_accepted_predict_run] / effect_capture_memory,
        "state_predict_failed_error_out_decision"_s <= "state_predict_allocate_slot_decision"_s + completion<PredictRun> [guard_memory_rejected_predict_run] / effect_mark_memory_error_predict_run_from_state_predict_allocate_slot_decision,
        "state_predict_capture_memory_decision"_s <= "state_predict_capture_memory"_s + completion<PredictRun>,
        "state_predict_begin"_s <= "state_predict_capture_memory_decision"_s + completion<PredictRun> [guard_memory_accepted_predict_run] / effect_begin_predict,
        "state_predict_failed_error_out_decision"_s <= "state_predict_capture_memory_decision"_s + completion<PredictRun> [guard_memory_rejected_predict_run] / effect_mark_memory_error_predict_run_from_state_predict_capture_memory_decision,
        "state_predict_error_out_decision"_s <= "state_predict_begin"_s + completion<PredictRun> / effect_publish_predict,
        "state_prediction_ready"_s <= "state_predict_error_out_decision"_s + completion<PredictRun> [guard_has_error_out_predict_run] / effect_store_error_out_predict_run_from_state_predict_error_out_decision,
        "state_prediction_ready"_s <= "state_predict_error_out_decision"_s + completion<PredictRun> [guard_no_error_out_predict_run],
        "state_session_ready"_s <= "state_predict_failed_error_out_decision"_s + completion<PredictRun> [guard_has_error_out_predict_run] / effect_store_error_out_predict_run_from_state_predict_failed_error_out_decision,
        "state_session_ready"_s <= "state_predict_failed_error_out_decision"_s + completion<PredictRun> [guard_no_error_out_predict_run],
        "state_execute_request_decision"_s <= "state_prediction_ready"_s + event<ExecuteRun>,
        "state_execute_begin"_s <= "state_execute_request_decision"_s + completion<ExecuteRun> [guard_execute_request_valid] / effect_begin_execute,
        "state_execute_failed_error_out_decision"_s <= "state_execute_request_decision"_s + completion<ExecuteRun> [guard_execute_request_invalid] / effect_mark_step_request_invalid_execute_run,
        "state_execute_graph_runtime_decision"_s <= "state_execute_begin"_s + completion<ExecuteRun>,
        "state_execute_running_graph"_s <= "state_execute_graph_runtime_decision"_s + completion<ExecuteRun> / effect_run_prediction_graph_graph_actor_type,
        "state_execute_graph_error_out_decision"_s <= "state_execute_running_graph"_s + completion<ExecuteRun>,
        "state_execute_graph_result_decision"_s <= "state_execute_graph_error_out_decision"_s + completion<ExecuteRun> [guard_has_graph_error_out_execute_run] / effect_store_execute_graph_error_out,
        "state_execute_graph_result_decision"_s <= "state_execute_graph_error_out_decision"_s + completion<ExecuteRun> [guard_no_graph_error_out_execute_run],
        "state_execute_error_out_decision"_s <= "state_execute_graph_result_decision"_s + completion<ExecuteRun> [guard_graph_step_accepted_execute_run] / effect_publish_execute,
        "state_execute_failed_error_out_decision"_s <= "state_execute_graph_result_decision"_s + completion<ExecuteRun> [guard_graph_step_rejected_execute_run] / effect_mark_graph_runtime_error_execute_run,
        "state_execution_ready"_s <= "state_execute_error_out_decision"_s + completion<ExecuteRun> [guard_has_error_out_execute_run] / effect_store_error_out_execute_run_from_state_execute_error_out_decision,
        "state_execution_ready"_s <= "state_execute_error_out_decision"_s + completion<ExecuteRun> [guard_no_error_out_execute_run],
        "state_session_ready"_s <= "state_execute_failed_error_out_decision"_s + completion<ExecuteRun> [guard_has_error_out_execute_run] / effect_store_error_out_execute_run_from_state_execute_failed_error_out_decision,
        "state_session_ready"_s <= "state_execute_failed_error_out_decision"_s + completion<ExecuteRun> [guard_no_error_out_execute_run],
        "state_sample_request_decision"_s <= "state_execution_ready"_s + event<SampleRun>,
        "state_sample_begin"_s <= "state_sample_request_decision"_s + completion<SampleRun> [guard_sample_request_valid] / effect_begin_sample,
        "state_sample_failed_error_out_decision"_s <= "state_sample_request_decision"_s + completion<SampleRun> [guard_sample_request_invalid] / effect_mark_step_request_invalid_sample_run,
        "state_sample_running_graph"_s <= "state_sample_begin"_s + completion<SampleRun> / effect_run_sampling_graph_graph_actor_type,
        "state_sample_graph_error_out_decision"_s <= "state_sample_running_graph"_s + completion<SampleRun>,
        "state_sample_graph_result_decision"_s <= "state_sample_graph_error_out_decision"_s + completion<SampleRun> [guard_has_graph_error_out_sample_run] / effect_store_sample_graph_error_out,
        "state_sample_graph_result_decision"_s <= "state_sample_graph_error_out_decision"_s + completion<SampleRun> [guard_no_graph_error_out_sample_run],
        "state_sample_error_out_decision"_s <= "state_sample_graph_result_decision"_s + completion<SampleRun> [guard_graph_step_accepted_sample_run] / effect_publish_sample,
        "state_sample_graph_failed_error_out_decision"_s <= "state_sample_graph_result_decision"_s + completion<SampleRun> [guard_graph_step_rejected_sample_run] / effect_mark_graph_runtime_error_sample_run,
        "state_session_ready"_s <= "state_sample_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_error_out_decision,
        "state_session_ready"_s <= "state_sample_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_execution_ready"_s <= "state_sample_failed_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_failed_error_out_decision,
        "state_execution_ready"_s <= "state_sample_failed_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_session_ready"_s <= "state_sample_graph_failed_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_graph_failed_error_out_decision,
        "state_session_ready"_s <= "state_sample_graph_failed_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_session_ready"_s <= "state_session_ready"_s + event<EventCaptureTokenizerState> [guard_capture_tokenizer_state_valid] / effect_capture_tokenizer_state,
        "state_session_ready"_s <= "state_session_ready"_s + event<EventCaptureTokenizerState> [guard_capture_tokenizer_state_invalid] / effect_reject_capture_tokenizer_state_error_request_shape_from_state_session_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_request_shape_from_state_prediction_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_request_shape_from_state_execution_ready,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<PredictRun> [guard_has_error_out_predict_run] / effect_mark_not_initialized_and_store_predict_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<PredictRun> [guard_no_error_out_predict_run] / effect_mark_not_initialized_predict_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<ExecuteRun> [guard_has_error_out_execute_run] / effect_mark_not_initialized_and_store_execute_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<ExecuteRun> [guard_no_error_out_execute_run] / effect_mark_not_initialized_execute_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_not_initialized_and_store_sample_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_not_initialized_sample_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_not_initialized,
        "state_uninit_voice_error_out_decision"_s <= "state_uninitialized"_s + event<LoadVoiceRun> / effect_mark_not_initialized_load_voice_run,
        "state_uninit_voice_callback_decision"_s <= "state_uninit_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_uninit_voice_error_out_decision,
        "state_uninit_voice_callback_decision"_s <= "state_uninit_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_uninitialized"_s <= "state_uninit_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_has_error_callback_load_voice_run] / effect_emit_load_voice_error_from_state_uninit_voice_callback_decision,
        "state_uninitialized"_s <= "state_uninit_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_no_error_callback_load_voice_run],
        "state_uninit_prefill_voice_error_out_decision"_s <= "state_uninitialized"_s + event<PrefillVoiceRun> / effect_mark_not_initialized_prefill_voice_run,
        "state_uninit_prefill_voice_callback_decision"_s <= "state_uninit_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_uninit_prefill_voice_error_out_decision,
        "state_uninit_prefill_voice_callback_decision"_s <= "state_uninit_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_uninitialized"_s <= "state_uninit_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_error_callback_prefill_voice_run] / effect_emit_prefill_voice_error_from_state_uninit_prefill_voice_callback_decision,
        "state_uninitialized"_s <= "state_uninit_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_error_callback_prefill_voice_run],
        "state_uninit_begin_personaplex_prompt_error_out_decision"_s <= "state_uninitialized"_s + event<BeginPromptRun> / effect_mark_not_initialized_begin_prompt_run,
        "state_uninit_begin_personaplex_prompt_callback_decision"_s <= "state_uninit_begin_personaplex_prompt_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_uninit_begin_personaplex_prompt_error_out_decision,
        "state_uninit_begin_personaplex_prompt_callback_decision"_s <= "state_uninit_begin_personaplex_prompt_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_uninitialized"_s <= "state_uninit_begin_personaplex_prompt_callback_decision"_s + completion<BeginPromptRun> [guard_has_error_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_error_from_state_uninit_begin_personaplex_prompt_callback_decision,
        "state_uninitialized"_s <= "state_uninit_begin_personaplex_prompt_callback_decision"_s + completion<BeginPromptRun> [guard_no_error_callback_begin_prompt_run],
        "state_uninit_prefill_personaplex_prompt_error_out_decision"_s <= "state_uninitialized"_s + event<PrefillPromptRun> / effect_mark_not_initialized_prefill_prompt_run,
        "state_uninit_prefill_personaplex_prompt_callback_decision"_s <= "state_uninit_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_uninit_prefill_personaplex_prompt_error_out_decision,
        "state_uninit_prefill_personaplex_prompt_callback_decision"_s <= "state_uninit_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_uninitialized"_s <= "state_uninit_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_has_error_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_error_from_state_uninit_prefill_personaplex_prompt_callback_decision,
        "state_uninitialized"_s <= "state_uninit_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_no_error_callback_prefill_prompt_run],
        "state_session_ready"_s <= "state_session_ready"_s + event<ExecuteRun> [guard_has_error_out_execute_run] / effect_mark_step_request_invalid_and_store_execute_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<ExecuteRun> [guard_no_error_out_execute_run] / effect_mark_step_request_invalid_execute_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_step_request_invalid_and_store_sample_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_step_request_invalid_sample_run,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_step_request_invalid_and_store_sample_run,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_step_request_invalid_sample_run,
        "state_reset_graph_result_decision"_s <= "state_session_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_session_ready,
        "state_reset_graph_result_decision"_s <= "state_execution_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_execution_ready,
        "state_reset_graph_result_decision"_s <= "state_prediction_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_prediction_ready,
        "state_reset_memory_result_decision"_s <= "state_reset_graph_result_decision"_s + completion<ResetRun> [guard_reset_graph_succeeded] / effect_release_reset_sequence,
        "state_reset_failed"_s <= "state_reset_graph_result_decision"_s + completion<ResetRun> [guard_reset_graph_failed] / effect_mark_reset_graph_failed,
        "state_uninitialized"_s <= "state_reset_memory_result_decision"_s + completion<ResetRun> [guard_reset_memory_succeeded] / effect_reset_session_from_state_reset_memory_result_decision,
        "state_reset_failed"_s <= "state_reset_memory_result_decision"_s + completion<ResetRun> [guard_reset_memory_failed] / effect_mark_reset_memory_failed,
        "state_uninitialized"_s <= "state_reset_failed"_s + completion<ResetRun> / effect_reset_session_from_state_reset_failed,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninitialized,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_session_ready,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_session_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_execution_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_execution_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_prediction_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_prediction_ready,
    }
}

/// Context for `SpeechPredictorMoshi` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechPredictorMoshiContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechPredictorMoshiStateMachineContext for SpeechPredictorMoshiContext {
    fn effect_advance_personaplex_prompt_post_silence(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_advance_personaplex_prompt_post_silence
        todo!(
            "TODO: port action `effect_advance_personaplex_prompt_post_silence` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_advance_personaplex_prompt_pre_silence(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_advance_personaplex_prompt_pre_silence
        todo!(
            "TODO: port action `effect_advance_personaplex_prompt_pre_silence` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_advance_personaplex_prompt_text(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_advance_personaplex_prompt_text
        todo!(
            "TODO: port action `effect_advance_personaplex_prompt_text` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_advance_voice_prefill(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_advance_voice_prefill
        todo!(
            "TODO: port action `effect_advance_voice_prefill` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_allocate_personaplex_prompt_slot(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_allocate_personaplex_prompt_slot
        todo!(
            "TODO: port action `effect_allocate_personaplex_prompt_slot` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_allocate_sequence(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_allocate_sequence
        todo!(
            "TODO: port action `effect_allocate_sequence` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_allocate_step_slot(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_allocate_step_slot
        todo!(
            "TODO: port action `effect_allocate_step_slot` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_allocate_voice_prefill_slot(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_allocate_voice_prefill_slot
        todo!(
            "TODO: port action `effect_allocate_voice_prefill_slot` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_begin_execute(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_begin_execute
        todo!(
            "TODO: port action `effect_begin_execute` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_begin_personaplex_prompt_prefill(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_begin_personaplex_prompt_prefill
        todo!(
            "TODO: port action `effect_begin_personaplex_prompt_prefill` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_begin_predict(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_begin_predict
        todo!(
            "TODO: port action `effect_begin_predict` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_begin_sample(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_begin_sample
        todo!(
            "TODO: port action `effect_begin_sample` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_begin_voice_prefill(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_begin_voice_prefill
        todo!(
            "TODO: port action `effect_begin_voice_prefill` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_bind_contract(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_bind_contract
        todo!(
            "TODO: port action `effect_bind_contract` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_bind_empty_personaplex_prompt(&mut self, _event: &BeginPromptRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_bind_empty_personaplex_prompt
        todo!(
            "TODO: port action `effect_bind_empty_personaplex_prompt` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_bind_personaplex_prompt(&mut self, _event: &BeginPromptRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_bind_personaplex_prompt
        todo!(
            "TODO: port action `effect_bind_personaplex_prompt` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_bind_voice_contract(&mut self, _event: &LoadVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_bind_voice_contract
        todo!(
            "TODO: port action `effect_bind_voice_contract` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_build_personaplex_prompt_silence_frame
        todo!(
            "TODO: port action `effect_build_personaplex_prompt_silence_frame` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_build_personaplex_prompt_text_frame(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_build_personaplex_prompt_text_frame
        todo!(
            "TODO: port action `effect_build_personaplex_prompt_text_frame` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_capture_memory(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_capture_memory
        todo!(
            "TODO: port action `effect_capture_memory` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_capture_personaplex_prompt_memory(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_capture_personaplex_prompt_memory
        todo!(
            "TODO: port action `effect_capture_personaplex_prompt_memory` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_capture_tokenizer_state(
        &mut self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_capture_tokenizer_state` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_capture_voice_prefill_memory(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_capture_voice_prefill_memory
        todo!(
            "TODO: port action `effect_capture_voice_prefill_memory` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_configure_personaplex_lmgen(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_configure_personaplex_lmgen
        todo!(
            "TODO: port action `effect_configure_personaplex_lmgen` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_configure_standard_lmgen(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_configure_standard_lmgen
        todo!(
            "TODO: port action `effect_configure_standard_lmgen` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_copy_voice_cache(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_copy_voice_cache
        todo!(
            "TODO: port action `effect_copy_voice_cache` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_begin_personaplex_prompt_done(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_begin_personaplex_prompt_done
        todo!(
            "TODO: port action `effect_emit_begin_personaplex_prompt_done` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_begin_personaplex_prompt_error_from_state_personaplex_prompt_begin_failed_callback_decision(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_begin_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_emit_begin_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_begin_personaplex_prompt_error_from_state_uninit_begin_personaplex_prompt_callback_decision(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_begin_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_emit_begin_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_initialize_done(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_initialize_error(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_load_voice_done(&mut self, _event: &LoadVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_load_voice_done
        todo!(
            "TODO: port action `effect_emit_load_voice_done` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_load_voice_error_from_state_uninit_voice_callback_decision(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_load_voice_error
        todo!(
            "TODO: port action `effect_emit_load_voice_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_load_voice_error_from_state_voice_failed_callback_decision(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_load_voice_error
        todo!(
            "TODO: port action `effect_emit_load_voice_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_personaplex_prompt_done(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_personaplex_prompt_done
        todo!(
            "TODO: port action `effect_emit_prefill_personaplex_prompt_done` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_personaplex_prompt_error_from_state_prefill_personaplex_prompt_failed_callback_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_emit_prefill_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_personaplex_prompt_error_from_state_uninit_prefill_personaplex_prompt_callback_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_emit_prefill_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_voice_done(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_voice_done
        todo!(
            "TODO: port action `effect_emit_prefill_voice_done` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_voice_error_from_state_prefill_voice_failed_callback_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_voice_error
        todo!(
            "TODO: port action `effect_emit_prefill_voice_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_emit_prefill_voice_error_from_state_uninit_prefill_voice_callback_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_emit_prefill_voice_error
        todo!(
            "TODO: port action `effect_emit_prefill_voice_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_finish_personaplex_prompt(&mut self, _event: &PrefillPromptRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_finish_personaplex_prompt
        todo!(
            "TODO: port action `effect_finish_personaplex_prompt` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_initialize_graph_graph_actor_type(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_initialize_graph
        todo!(
            "TODO: port action `effect_initialize_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_load_voice_embedding_frame_bf16(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_load_voice_embedding_frame_bf16
        todo!(
            "TODO: port action `effect_load_voice_embedding_frame_bf16` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_load_voice_embedding_frame_f16(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_load_voice_embedding_frame_f16
        todo!(
            "TODO: port action `effect_load_voice_embedding_frame_f16` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_load_voice_embedding_frame_f32(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_load_voice_embedding_frame_f32
        todo!(
            "TODO: port action `effect_load_voice_embedding_frame_f32` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_bind_failed(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_graph_runtime_error_execute_run(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_graph_runtime_error
        todo!(
            "TODO: port action `effect_mark_graph_runtime_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_graph_runtime_error_init_run(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_graph_runtime_error
        todo!(
            "TODO: port action `effect_mark_graph_runtime_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_graph_runtime_error_prefill_prompt_run(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_graph_runtime_error
        todo!(
            "TODO: port action `effect_mark_graph_runtime_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_graph_runtime_error_prefill_voice_run(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_graph_runtime_error
        todo!(
            "TODO: port action `effect_mark_graph_runtime_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_graph_runtime_error_sample_run(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_graph_runtime_error
        todo!(
            "TODO: port action `effect_mark_graph_runtime_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_init_run_from_state_allocate_sequence_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_init_run_from_state_reserve_memory_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_predict_run_from_state_predict_allocate_slot_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_predict_run_from_state_predict_capture_memory_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_allocate_slot_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_capture_memory_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_allocate_slot_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_capture_memory_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_memory_error
        todo!(
            "TODO: port action `effect_mark_memory_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_and_store_execute_run(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized_and_store
        todo!(
            "TODO: port action `effect_mark_not_initialized_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_and_store_predict_run(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized_and_store
        todo!(
            "TODO: port action `effect_mark_not_initialized_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_and_store_sample_run(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized_and_store
        todo!(
            "TODO: port action `effect_mark_not_initialized_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_begin_prompt_run(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_execute_run(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_load_voice_run(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_predict_run(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_prefill_prompt_run(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_prefill_voice_run(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_sample_run(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_personaplex_prompt_error_begin_prompt_run(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_mark_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_phase_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_mark_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_request_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_personaplex_prompt_error
        todo!(
            "TODO: port action `effect_mark_personaplex_prompt_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_reset_graph_failed(&mut self, _event: &ResetRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_reset_graph_failed
        todo!(
            "TODO: port action `effect_mark_reset_graph_failed` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_reset_memory_failed(&mut self, _event: &ResetRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_reset_memory_failed
        todo!(
            "TODO: port action `effect_mark_reset_memory_failed` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_step_request_invalid_and_store_execute_run(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_step_request_invalid_and_store
        todo!(
            "TODO: port action `effect_mark_step_request_invalid_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_step_request_invalid_and_store_sample_run(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_step_request_invalid_and_store
        todo!(
            "TODO: port action `effect_mark_step_request_invalid_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_step_request_invalid_execute_run(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_step_request_invalid
        todo!(
            "TODO: port action `effect_mark_step_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_step_request_invalid_predict_run(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_step_request_invalid
        todo!(
            "TODO: port action `effect_mark_step_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_step_request_invalid_sample_run(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_step_request_invalid
        todo!(
            "TODO: port action `effect_mark_step_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_execution_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_prediction_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_session_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_execution_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_prediction_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_session_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_voice_contract_error_load_voice_run(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_voice_contract_error
        todo!(
            "TODO: port action `effect_mark_voice_contract_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_embedding_frame_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_voice_contract_error
        todo!(
            "TODO: port action `effect_mark_voice_contract_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_request_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_voice_contract_error
        todo!(
            "TODO: port action `effect_mark_voice_contract_error` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_mark_voice_prompt_pending(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_mark_voice_prompt_pending
        todo!(
            "TODO: port action `effect_mark_voice_prompt_pending` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_publish_execute(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_publish_execute
        todo!(
            "TODO: port action `effect_publish_execute` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_publish_personaplex_prompt_pending(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_publish_personaplex_prompt_pending
        todo!(
            "TODO: port action `effect_publish_personaplex_prompt_pending` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_publish_predict(&mut self, _event: &PredictRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_publish_predict
        todo!(
            "TODO: port action `effect_publish_predict` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_publish_sample(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_publish_sample
        todo!(
            "TODO: port action `effect_publish_sample` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reject_capture_tokenizer_state_error_not_initialized(
        &mut self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reject_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_reject_capture_tokenizer_state` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_execution_ready(
        &mut self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reject_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_reject_capture_tokenizer_state` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_prediction_ready(
        &mut self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reject_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_reject_capture_tokenizer_state` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_session_ready(
        &mut self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reject_capture_tokenizer_state
        todo!(
            "TODO: port action `effect_reject_capture_tokenizer_state` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_release_reset_sequence(&mut self, _event: &ResetRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_release_reset_sequence
        todo!(
            "TODO: port action `effect_release_reset_sequence` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reserve_memory(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reserve_memory
        todo!(
            "TODO: port action `effect_reserve_memory` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reset_graph_graph_actor_type_from_state_execution_ready(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reset_graph
        todo!(
            "TODO: port action `effect_reset_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reset_graph_graph_actor_type_from_state_prediction_ready(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reset_graph
        todo!(
            "TODO: port action `effect_reset_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reset_graph_graph_actor_type_from_state_session_ready(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reset_graph
        todo!(
            "TODO: port action `effect_reset_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reset_session_from_state_reset_failed(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reset_session
        todo!(
            "TODO: port action `effect_reset_session` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_reset_session_from_state_reset_memory_result_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_reset_session
        todo!(
            "TODO: port action `effect_reset_session` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_run_personaplex_prompt_graph_runtime_graph_actor_type(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_run_personaplex_prompt_graph_runtime
        todo!(
            "TODO: port action `effect_run_personaplex_prompt_graph_runtime` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_run_prediction_graph_graph_actor_type(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_run_prediction_graph
        todo!(
            "TODO: port action `effect_run_prediction_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_run_sampling_graph_graph_actor_type(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_run_sampling_graph
        todo!(
            "TODO: port action `effect_run_sampling_graph` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_run_voice_graph_runtime_graph_actor_type(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_run_voice_graph_runtime
        todo!(
            "TODO: port action `effect_run_voice_graph_runtime` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_error_out_decision(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_failed_error_out_decision(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_begin_prompt_run_from_state_uninit_begin_personaplex_prompt_error_out_decision(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_execute_run_from_state_execute_error_out_decision(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_execute_run_from_state_execute_failed_error_out_decision(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_failed_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_load_voice_run_from_state_uninit_voice_error_out_decision(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_load_voice_run_from_state_voice_error_out_decision(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_load_voice_run_from_state_voice_failed_error_out_decision(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_predict_run_from_state_predict_error_out_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_predict_run_from_state_predict_failed_error_out_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_error_out_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_failed_error_out_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_prompt_run_from_state_uninit_prefill_personaplex_prompt_error_out_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_voice_run_from_state_prefill_voice_error_out_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_voice_run_from_state_prefill_voice_failed_error_out_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_prefill_voice_run_from_state_uninit_prefill_voice_error_out_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_sample_run_from_state_sample_error_out_decision(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_sample_run_from_state_sample_failed_error_out_decision(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_error_out_sample_run_from_state_sample_graph_failed_error_out_decision(
        &mut self,
        _event: &SampleRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_execute_graph_error_out(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_execute_graph_error_out
        todo!(
            "TODO: port action `effect_store_execute_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_graph_error_out_prefill_prompt_run(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_graph_error_out
        todo!(
            "TODO: port action `effect_store_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_graph_error_out_prefill_voice_run(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_graph_error_out
        todo!(
            "TODO: port action `effect_store_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_personaplex_prompt_complete_out(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_personaplex_prompt_complete_out
        todo!(
            "TODO: port action `effect_store_personaplex_prompt_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_personaplex_prompt_remaining_out(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_personaplex_prompt_remaining_out
        todo!(
            "TODO: port action `effect_store_personaplex_prompt_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_sample_graph_error_out(&mut self, _event: &SampleRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_sample_graph_error_out
        todo!(
            "TODO: port action `effect_store_sample_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_voice_complete_out(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_voice_complete_out
        todo!(
            "TODO: port action `effect_store_voice_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_store_voice_remaining_out(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_store_voice_remaining_out
        todo!(
            "TODO: port action `effect_store_voice_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn effect_write_and_build_personaplex_prompt_input(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp::effect_write_and_build_personaplex_prompt_input
        todo!(
            "TODO: port action `effect_write_and_build_personaplex_prompt_input` from emel.cpp/src/emel/speech/predictor/moshi/actions.hpp"
        )
    }
    fn guard_bind_contract_invalid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_bind_contract_invalid
        todo!(
            "TODO: port guard `guard_bind_contract_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_bind_contract_valid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_bind_contract_valid
        todo!(
            "TODO: port guard `guard_bind_contract_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_capture_tokenizer_state_invalid(
        &self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_capture_tokenizer_state_invalid
        todo!(
            "TODO: port guard `guard_capture_tokenizer_state_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_capture_tokenizer_state_valid(
        &self,
        _event: &EventCaptureTokenizerState,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_capture_tokenizer_state_valid
        todo!(
            "TODO: port guard `guard_capture_tokenizer_state_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_execute_request_invalid(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_execute_request_invalid
        todo!(
            "TODO: port guard `guard_execute_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_execute_request_valid(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_execute_request_valid
        todo!(
            "TODO: port guard `guard_execute_request_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_initialize_failed(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_initialize_failed
        todo!(
            "TODO: port guard `guard_graph_initialize_failed` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_initialize_succeeded(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_initialize_succeeded
        todo!(
            "TODO: port guard `guard_graph_initialize_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted
        todo!(
            "TODO: port guard `guard_graph_step_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_personaplex_prompt_post_silence(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted_personaplex_prompt_post_silence
        todo!(
            "TODO: port guard `guard_graph_step_accepted_personaplex_prompt_post_silence` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_personaplex_prompt_pre_silence(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted_personaplex_prompt_pre_silence
        todo!(
            "TODO: port guard `guard_graph_step_accepted_personaplex_prompt_pre_silence` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_personaplex_prompt_text(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted_personaplex_prompt_text
        todo!(
            "TODO: port guard `guard_graph_step_accepted_personaplex_prompt_text` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted
        todo!(
            "TODO: port guard `guard_graph_step_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_accepted_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_accepted
        todo!(
            "TODO: port guard `guard_graph_step_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_rejected_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_rejected
        todo!(
            "TODO: port guard `guard_graph_step_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_rejected_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_rejected
        todo!(
            "TODO: port guard `guard_graph_step_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_rejected_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_rejected
        todo!(
            "TODO: port guard `guard_graph_step_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_graph_step_rejected_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_graph_step_rejected
        todo!(
            "TODO: port guard `guard_graph_step_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_done_callback_begin_prompt_run(
        &self,
        _event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_done_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_done_callback_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_done_callback_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_done_callback_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_callback_begin_prompt_run(
        &self,
        _event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_callback_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_callback_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_callback_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_begin_prompt_run(&self, _event: &BeginPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_prefill_voice_run(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_error_out_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_graph_error_out_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_graph_error_out
        todo!(
            "TODO: port guard `guard_has_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_graph_error_out_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_graph_error_out
        todo!(
            "TODO: port guard `guard_has_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_graph_error_out_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_graph_error_out
        todo!(
            "TODO: port guard `guard_has_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_graph_error_out_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_graph_error_out
        todo!(
            "TODO: port guard `guard_has_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_personaplex_prompt_complete_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_personaplex_prompt_complete_out
        todo!(
            "TODO: port guard `guard_has_personaplex_prompt_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_personaplex_prompt_remaining_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_personaplex_prompt_remaining_out
        todo!(
            "TODO: port guard `guard_has_personaplex_prompt_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_voice_complete_out(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_voice_complete_out
        todo!(
            "TODO: port guard `guard_has_voice_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_has_voice_remaining_out(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_has_voice_remaining_out
        todo!(
            "TODO: port guard `guard_has_voice_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_accepted_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_accepted
        todo!(
            "TODO: port guard `guard_memory_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_accepted_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_accepted
        todo!(
            "TODO: port guard `guard_memory_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_accepted_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_accepted
        todo!(
            "TODO: port guard `guard_memory_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_accepted_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_accepted
        todo!(
            "TODO: port guard `guard_memory_accepted` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_rejected_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_rejected
        todo!(
            "TODO: port guard `guard_memory_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_rejected_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_rejected
        todo!(
            "TODO: port guard `guard_memory_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_rejected_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_rejected
        todo!(
            "TODO: port guard `guard_memory_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_memory_rejected_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_memory_rejected
        todo!(
            "TODO: port guard `guard_memory_rejected` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_done_callback_begin_prompt_run(&self, _event: &BeginPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_done_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_done_callback_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_done_callback_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_done_callback_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_callback_begin_prompt_run(
        &self,
        _event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_callback_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_callback_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_callback_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_begin_prompt_run(&self, _event: &BeginPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_load_voice_run(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_prefill_prompt_run(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_prefill_voice_run(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_error_out_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_graph_error_out_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_graph_error_out
        todo!(
            "TODO: port guard `guard_no_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_graph_error_out_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_graph_error_out
        todo!(
            "TODO: port guard `guard_no_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_graph_error_out_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_graph_error_out
        todo!(
            "TODO: port guard `guard_no_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_graph_error_out_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_graph_error_out
        todo!(
            "TODO: port guard `guard_no_graph_error_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_personaplex_prompt_complete_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_personaplex_prompt_complete_out
        todo!(
            "TODO: port guard `guard_no_personaplex_prompt_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_personaplex_prompt_remaining_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_personaplex_prompt_remaining_out
        todo!(
            "TODO: port guard `guard_no_personaplex_prompt_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_voice_complete_out(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_voice_complete_out
        todo!(
            "TODO: port guard `guard_no_voice_complete_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_no_voice_remaining_out(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_no_voice_remaining_out
        todo!(
            "TODO: port guard `guard_no_voice_remaining_out` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_lmgen(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_lmgen
        todo!(
            "TODO: port guard `guard_personaplex_lmgen` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_begin_empty_valid(
        &self,
        _event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_begin_empty_valid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_begin_empty_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_begin_invalid(&self, _event: &BeginPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_begin_invalid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_begin_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_begin_nonempty_valid(
        &self,
        _event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_begin_nonempty_valid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_begin_nonempty_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_complete(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_complete
        todo!(
            "TODO: port guard `guard_personaplex_prompt_complete` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_pending(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_pending
        todo!(
            "TODO: port guard `guard_personaplex_prompt_pending` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_phase_invalid(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_phase_invalid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_phase_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_post_silence_pending(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_post_silence_pending
        todo!(
            "TODO: port guard `guard_personaplex_prompt_post_silence_pending` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_pre_silence_pending(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_pre_silence_pending
        todo!(
            "TODO: port guard `guard_personaplex_prompt_pre_silence_pending` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_prefill_request_invalid(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_prefill_request_invalid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_prefill_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_prefill_request_valid(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_prefill_request_valid
        todo!(
            "TODO: port guard `guard_personaplex_prompt_prefill_request_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_personaplex_prompt_text_pending(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_personaplex_prompt_text_pending
        todo!(
            "TODO: port guard `guard_personaplex_prompt_text_pending` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_predict_blocked_by_voice_prompt(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_predict_blocked_by_voice_prompt
        todo!(
            "TODO: port guard `guard_predict_blocked_by_voice_prompt` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_predict_request_shape_invalid(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_predict_request_shape_invalid
        todo!(
            "TODO: port guard `guard_predict_request_shape_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_predict_request_valid(&self, _event: &PredictRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_predict_request_valid
        todo!(
            "TODO: port guard `guard_predict_request_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_reset_graph_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_reset_graph_failed
        todo!(
            "TODO: port guard `guard_reset_graph_failed` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_reset_graph_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_reset_graph_succeeded
        todo!(
            "TODO: port guard `guard_reset_graph_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_reset_memory_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_reset_memory_failed
        todo!(
            "TODO: port guard `guard_reset_memory_failed` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_reset_memory_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_reset_memory_succeeded
        todo!(
            "TODO: port guard `guard_reset_memory_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_sample_request_invalid(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_sample_request_invalid
        todo!(
            "TODO: port guard `guard_sample_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_sample_request_valid(&self, _event: &SampleRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_sample_request_valid
        todo!(
            "TODO: port guard `guard_sample_request_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_standard_lmgen(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_standard_lmgen
        todo!(
            "TODO: port guard `guard_standard_lmgen` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_absent(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_unexpected_error_out_absent
        todo!(
            "TODO: port guard `guard_unexpected_error_out_absent` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_present(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_unexpected_error_out_present
        todo!(
            "TODO: port guard `guard_unexpected_error_out_present` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_contract_invalid(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_contract_invalid
        todo!(
            "TODO: port guard `guard_voice_contract_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_contract_valid(&self, _event: &LoadVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_contract_valid
        todo!(
            "TODO: port guard `guard_voice_contract_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_embedding_frame_bf16(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_embedding_frame_bf16
        todo!(
            "TODO: port guard `guard_voice_embedding_frame_bf16` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_embedding_frame_f16(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_embedding_frame_f16
        todo!(
            "TODO: port guard `guard_voice_embedding_frame_f16` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_embedding_frame_f32(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_embedding_frame_f32
        todo!(
            "TODO: port guard `guard_voice_embedding_frame_f32` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_embedding_frame_failed(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_embedding_frame_failed
        todo!(
            "TODO: port guard `guard_voice_embedding_frame_failed` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_embedding_frame_loaded(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_embedding_frame_loaded
        todo!(
            "TODO: port guard `guard_voice_embedding_frame_loaded` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_prefill_complete(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_prefill_complete
        todo!(
            "TODO: port guard `guard_voice_prefill_complete` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_prefill_pending(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_prefill_pending
        todo!(
            "TODO: port guard `guard_voice_prefill_pending` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_prefill_request_invalid(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_prefill_request_invalid
        todo!(
            "TODO: port guard `guard_voice_prefill_request_invalid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
    fn guard_voice_prefill_request_valid(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp::guard_voice_prefill_request_valid
        todo!(
            "TODO: port guard `guard_voice_prefill_request_valid` from emel.cpp/src/emel/speech/predictor/moshi/guards.hpp"
        )
    }
}
