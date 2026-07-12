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

// --- machine SpeechPredictorMoshiExecutor from emel.cpp/src/emel/speech/predictor/moshi/executor/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct InitRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct ResetRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct StepRun;

sml! {
    SpeechPredictorMoshiExecutor {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + event<InitRun>,
        "state_input_embedding_contract_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_valid] / effect_bind_contract,
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_invalid] / effect_mark_bind_failed_from_state_bind_contract_decision,
        "state_temporal_projection_layout_decision"_s <= "state_input_embedding_contract_decision"_s + completion<InitRun> [guard_bound_root_operands_supported],
        "state_init_failed_error_out_decision"_s <= "state_input_embedding_contract_decision"_s + completion<InitRun> [guard_bound_root_operands_unsupported] / effect_mark_bind_failed_from_state_input_embedding_contract_decision,
        "state_depformer_projection_layout_decision"_s <= "state_temporal_projection_layout_decision"_s + completion<InitRun> [guard_temporal_split_projection_layout_supported] / effect_bind_temporal_split_projection_layout,
        "state_depformer_projection_layout_decision"_s <= "state_temporal_projection_layout_decision"_s + completion<InitRun> [guard_temporal_fused_projection_layout_supported] / effect_bind_temporal_fused_projection_layout,
        "state_init_failed_error_out_decision"_s <= "state_temporal_projection_layout_decision"_s + completion<InitRun> [guard_temporal_projection_layout_unsupported] / effect_mark_bind_failed_from_state_temporal_projection_layout_decision,
        "state_sampling_seed_decision"_s <= "state_depformer_projection_layout_decision"_s + completion<InitRun> [guard_depformer_split_projection_layout_supported] / effect_bind_depformer_split_projection_layout,
        "state_sampling_seed_decision"_s <= "state_depformer_projection_layout_decision"_s + completion<InitRun> [guard_depformer_fused_projection_layout_supported] / effect_bind_depformer_fused_projection_layout,
        "state_init_failed_error_out_decision"_s <= "state_depformer_projection_layout_decision"_s + completion<InitRun> [guard_depformer_projection_layout_unsupported] / effect_mark_bind_failed_from_state_depformer_projection_layout_decision,
        "state_text_sampling_top_k_decision"_s <= "state_sampling_seed_decision"_s + completion<InitRun> [guard_sampling_seed_nonzero] / effect_bind_nonzero_sampling_seed,
        "state_text_sampling_top_k_decision"_s <= "state_sampling_seed_decision"_s + completion<InitRun> [guard_sampling_seed_zero] / effect_bind_zero_sampling_seed,
        "state_audio_sampling_top_k_decision"_s <= "state_text_sampling_top_k_decision"_s + completion<InitRun> [guard_text_sampling_top_k_within_card] / effect_bind_requested_text_sampling_top_k,
        "state_audio_sampling_top_k_decision"_s <= "state_text_sampling_top_k_decision"_s + completion<InitRun> [guard_text_sampling_top_k_exceeds_card] / effect_bind_full_card_text_sampling_top_k,
        "state_init_error_out_decision"_s <= "state_audio_sampling_top_k_decision"_s + completion<InitRun> [guard_audio_sampling_top_k_within_card] / effect_bind_requested_audio_sampling_top_k,
        "state_init_error_out_decision"_s <= "state_audio_sampling_top_k_decision"_s + completion<InitRun> [guard_audio_sampling_top_k_exceeds_card] / effect_bind_full_card_audio_sampling_top_k,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_error_out_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_failed_error_out_decision,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_has_initialize_done_callback] / effect_emit_initialize_done,
        "state_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_no_initialize_done_callback],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_has_initialize_error_callback] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_no_initialize_error_callback],
        "state_step_model_decision"_s <= "state_ready"_s + event<StepRun>,
        "state_step_shape_decision"_s <= "state_step_model_decision"_s + completion<StepRun> [guard_step_model_matches],
        "state_step_error_out_decision"_s <= "state_step_model_decision"_s + completion<StepRun> [guard_step_model_mismatch] / effect_mark_model_mismatch,
        "state_step_execution_decision"_s <= "state_step_shape_decision"_s + completion<StepRun> [guard_step_shape_valid],
        "state_step_error_out_decision"_s <= "state_step_shape_decision"_s + completion<StepRun> [guard_step_shape_invalid] / effect_mark_request_shape_from_state_step_shape_decision,
        "state_input_embedding"_s <= "state_step_execution_decision"_s + completion<StepRun> [guard_input_embedding_supported] / effect_begin_input_embedding,
        "state_step_error_out_decision"_s <= "state_step_execution_decision"_s + completion<StepRun> [guard_input_embedding_unsupported] / effect_mark_graph_execution_unsupported_from_state_step_execution_decision,
        "state_input_embedding_result_decision"_s <= "state_input_embedding"_s + completion<StepRun> [guard_external_input_embedding_supported] / effect_apply_external_input_embedding,
        "state_input_text_embedding_bind"_s <= "state_input_embedding"_s + completion<StepRun> [guard_token_input_embedding_supported] / effect_bind_input_text_embedding,
        "state_step_error_out_decision"_s <= "state_input_embedding"_s + completion<StepRun> [guard_token_input_embedding_unsupported] / effect_mark_graph_execution_unsupported_from_state_input_embedding,
        "state_input_text_embedding_bind_result_decision"_s <= "state_input_text_embedding_bind"_s + completion<StepRun>,
        "state_input_text_embedding_row"_s <= "state_input_text_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bound] / effect_run_embedding_row_fetch_from_state_input_text_embedding_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_input_text_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_input_text_embedding_bind_result_decision,
        "state_input_text_embedding_row_result_decision"_s <= "state_input_text_embedding_row"_s + completion<StepRun>,
        "state_input_text_embedding_apply"_s <= "state_input_text_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_succeeded] / effect_apply_input_text_embedding_row,
        "state_step_error_out_decision"_s <= "state_input_text_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_failed] / effect_mark_graph_execution_unsupported_from_state_input_text_embedding_row_result_decision,
        "state_input_text_embedding_result_decision"_s <= "state_input_text_embedding_apply"_s + completion<StepRun> [guard_input_text_embedding_succeeded],
        "state_step_error_out_decision"_s <= "state_input_text_embedding_apply"_s + completion<StepRun> [guard_input_text_embedding_failed] / effect_mark_graph_execution_unsupported_from_state_input_text_embedding_apply,
        "state_input_audio_token_decision"_s <= "state_input_text_embedding_result_decision"_s + completion<StepRun>,
        "state_input_audio_embedding"_s <= "state_input_audio_token_decision"_s + completion<StepRun> [guard_current_input_audio_token_present] / effect_bind_input_audio_embedding,
        "state_input_audio_embedding_result_decision"_s <= "state_input_audio_token_decision"_s + completion<StepRun> [guard_current_input_audio_token_zero] / effect_skip_zero_input_audio_embedding,
        "state_step_error_out_decision"_s <= "state_input_audio_token_decision"_s + completion<StepRun> [guard_current_input_audio_token_unsupported] / effect_mark_graph_execution_unsupported_from_state_input_audio_token_decision,
        "state_input_audio_embedding_bind_result_decision"_s <= "state_input_audio_embedding"_s + completion<StepRun>,
        "state_input_audio_embedding_row"_s <= "state_input_audio_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bound] / effect_run_embedding_row_fetch_from_state_input_audio_embedding_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_input_audio_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_bind_result_decision,
        "state_input_audio_embedding_row_result_decision"_s <= "state_input_audio_embedding_row"_s + completion<StepRun>,
        "state_input_audio_embedding_apply"_s <= "state_input_audio_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_succeeded] / effect_apply_input_audio_embedding_row,
        "state_step_error_out_decision"_s <= "state_input_audio_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_failed] / effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_row_result_decision,
        "state_input_audio_embedding_result_decision"_s <= "state_input_audio_embedding_apply"_s + completion<StepRun> [guard_input_audio_embedding_succeeded],
        "state_step_error_out_decision"_s <= "state_input_audio_embedding_apply"_s + completion<StepRun> [guard_input_audio_embedding_failed] / effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_apply,
        "state_input_audio_codebook_advance"_s <= "state_input_audio_embedding_result_decision"_s + completion<StepRun> [guard_more_input_audio_codebooks] / effect_advance_input_audio_codebook,
        "state_input_embedding_result_decision"_s <= "state_input_audio_embedding_result_decision"_s + completion<StepRun> [guard_input_audio_codebooks_complete] / effect_finish_input_embedding,
        "state_input_audio_token_decision"_s <= "state_input_audio_codebook_advance"_s + completion<StepRun>,
        "state_step_error_out_decision"_s <= "state_input_embedding_result_decision"_s + completion<StepRun> [guard_input_embedding_failed] / effect_mark_graph_execution_unsupported_from_state_input_embedding_result_decision,
        "state_bind_temporal_kv"_s <= "state_input_embedding_result_decision"_s + completion<StepRun> [guard_temporal_kv_binding_present] / effect_bind_temporal_kv,
        "state_step_error_out_decision"_s <= "state_input_embedding_result_decision"_s + completion<StepRun> [guard_temporal_kv_binding_missing] / effect_mark_graph_execution_unsupported_from_state_input_embedding_result_decision,
        "state_temporal_kv_result_decision"_s <= "state_bind_temporal_kv"_s + completion<StepRun> [guard_temporal_kv_bound],
        "state_step_error_out_decision"_s <= "state_bind_temporal_kv"_s + completion<StepRun> [guard_temporal_kv_bind_failed] / effect_mark_graph_execution_unsupported_from_state_bind_temporal_kv,
        "state_temporal_position_result_decision"_s <= "state_temporal_kv_result_decision"_s + completion<StepRun> / effect_advance_temporal_position,
        "state_temporal_position_valid_decision"_s <= "state_temporal_position_result_decision"_s + completion<StepRun> [guard_temporal_position_advance_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_position_result_decision"_s + completion<StepRun> [guard_temporal_position_advance_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_position_result_decision,
        "state_temporal_layer_norm"_s <= "state_temporal_position_valid_decision"_s + completion<StepRun> [guard_temporal_layer_norm_supported] / effect_run_temporal_layer_norm_rms_from_state_temporal_position_valid_decision,
        "state_step_error_out_decision"_s <= "state_temporal_position_valid_decision"_s + completion<StepRun> [guard_temporal_layer_norm_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_position_valid_decision,
        "state_temporal_layer_norm_rms_result_decision"_s <= "state_temporal_layer_norm"_s + completion<StepRun>,
        "state_temporal_layer_norm_scale"_s <= "state_temporal_layer_norm_rms_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm_rms_succeeded] / effect_run_temporal_layer_norm_scale,
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm_rms_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm_rms_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_rms_result_decision,
        "state_temporal_layer_norm_result_decision"_s <= "state_temporal_layer_norm_scale"_s + completion<StepRun> [guard_temporal_layer_norm_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm_scale"_s + completion<StepRun> [guard_temporal_layer_norm_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_scale,
        "state_temporal_layer_projection"_s <= "state_temporal_layer_norm_result_decision"_s + completion<StepRun> [guard_temporal_layer_projection_supported] / effect_bind_temporal_layer_projection,
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm_result_decision"_s + completion<StepRun> [guard_temporal_layer_projection_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_result_decision,
        "state_temporal_layer_projection_bind_result_decision"_s <= "state_temporal_layer_projection"_s + completion<StepRun>,
        "state_temporal_layer_projection_run"_s <= "state_temporal_layer_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_temporal_layer_projection,
        "state_step_error_out_decision"_s <= "state_temporal_layer_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_bind_result_decision,
        "state_temporal_layer_projection_result_decision"_s <= "state_temporal_layer_projection_run"_s + completion<StepRun> [guard_temporal_layer_projection_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_projection_run"_s + completion<StepRun> [guard_temporal_layer_projection_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_run,
        "state_temporal_layer_rope"_s <= "state_temporal_layer_projection_result_decision"_s + completion<StepRun> [guard_temporal_layer_rope_supported] / effect_run_temporal_layer_query_rope,
        "state_step_error_out_decision"_s <= "state_temporal_layer_projection_result_decision"_s + completion<StepRun> [guard_temporal_layer_rope_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_result_decision,
        "state_temporal_layer_rope_result_decision"_s <= "state_temporal_layer_rope"_s + completion<StepRun> [guard_temporal_layer_rope_succeeded] / effect_copy_temporal_layer_query_rope,
        "state_step_error_out_decision"_s <= "state_temporal_layer_rope"_s + completion<StepRun> [guard_temporal_layer_rope_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_rope,
        "state_temporal_layer_key_rope"_s <= "state_temporal_layer_rope_result_decision"_s + completion<StepRun> / effect_run_temporal_layer_key_rope,
        "state_temporal_layer_key_rope_result_decision"_s <= "state_temporal_layer_key_rope"_s + completion<StepRun> [guard_temporal_layer_rope_succeeded] / effect_copy_temporal_layer_key_rope,
        "state_step_error_out_decision"_s <= "state_temporal_layer_key_rope"_s + completion<StepRun> [guard_temporal_layer_rope_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_key_rope,
        "state_temporal_layer_cache_write"_s <= "state_temporal_layer_key_rope_result_decision"_s + completion<StepRun> [guard_temporal_layer_cache_write_supported] / effect_write_temporal_layer_kv_cache,
        "state_step_error_out_decision"_s <= "state_temporal_layer_key_rope_result_decision"_s + completion<StepRun> [guard_temporal_layer_cache_write_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_key_rope_result_decision,
        "state_temporal_layer_cache_write_result_decision"_s <= "state_temporal_layer_cache_write"_s + completion<StepRun> [guard_temporal_layer_cache_write_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_cache_write"_s + completion<StepRun> [guard_temporal_layer_cache_write_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_cache_write,
        "state_temporal_layer_attention"_s <= "state_temporal_layer_cache_write_result_decision"_s + completion<StepRun> [guard_temporal_layer_attention_supported] / effect_run_temporal_layer_attention,
        "state_step_error_out_decision"_s <= "state_temporal_layer_cache_write_result_decision"_s + completion<StepRun> [guard_temporal_layer_attention_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_cache_write_result_decision,
        "state_temporal_layer_attention_result_decision"_s <= "state_temporal_layer_attention"_s + completion<StepRun> [guard_temporal_layer_attention_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_attention"_s + completion<StepRun> [guard_temporal_layer_attention_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_attention,
        "state_temporal_layer_out_projection"_s <= "state_temporal_layer_attention_result_decision"_s + completion<StepRun> [guard_temporal_layer_out_projection_supported] / effect_bind_temporal_layer_out_projection,
        "state_step_error_out_decision"_s <= "state_temporal_layer_attention_result_decision"_s + completion<StepRun> [guard_temporal_layer_out_projection_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_attention_result_decision,
        "state_temporal_layer_out_projection_bind_result_decision"_s <= "state_temporal_layer_out_projection"_s + completion<StepRun>,
        "state_temporal_layer_out_projection_run"_s <= "state_temporal_layer_out_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_temporal_layer_out_projection,
        "state_step_error_out_decision"_s <= "state_temporal_layer_out_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_out_projection_bind_result_decision,
        "state_temporal_layer_out_projection_result_decision"_s <= "state_temporal_layer_out_projection_run"_s + completion<StepRun> [guard_temporal_layer_out_projection_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_out_projection_run"_s + completion<StepRun> [guard_temporal_layer_out_projection_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_out_projection_run,
        "state_temporal_layer_residual"_s <= "state_temporal_layer_out_projection_result_decision"_s + completion<StepRun> / effect_apply_temporal_layer_attention_residual,
        "state_temporal_layer_residual_result_decision"_s <= "state_temporal_layer_residual"_s + completion<StepRun> [guard_temporal_layer_residual_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_residual"_s + completion<StepRun> [guard_temporal_layer_residual_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_residual,
        "state_temporal_layer_norm2"_s <= "state_temporal_layer_residual_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm2_supported] / effect_run_temporal_layer_norm2_rms,
        "state_step_error_out_decision"_s <= "state_temporal_layer_residual_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm2_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_residual_result_decision,
        "state_temporal_layer_norm2_rms_result_decision"_s <= "state_temporal_layer_norm2"_s + completion<StepRun>,
        "state_temporal_layer_norm2_scale"_s <= "state_temporal_layer_norm2_rms_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm2_rms_succeeded] / effect_run_temporal_layer_norm2_scale,
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm2_rms_result_decision"_s + completion<StepRun> [guard_temporal_layer_norm2_rms_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_rms_result_decision,
        "state_temporal_layer_norm2_result_decision"_s <= "state_temporal_layer_norm2_scale"_s + completion<StepRun> [guard_temporal_layer_norm2_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm2_scale"_s + completion<StepRun> [guard_temporal_layer_norm2_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_scale,
        "state_temporal_layer_gating_in"_s <= "state_temporal_layer_norm2_result_decision"_s + completion<StepRun> [guard_temporal_layer_gating_in_supported] / effect_bind_temporal_layer_gating_in,
        "state_step_error_out_decision"_s <= "state_temporal_layer_norm2_result_decision"_s + completion<StepRun> [guard_temporal_layer_gating_in_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_result_decision,
        "state_temporal_layer_gating_in_bind_result_decision"_s <= "state_temporal_layer_gating_in"_s + completion<StepRun>,
        "state_temporal_layer_gating_in_run"_s <= "state_temporal_layer_gating_in_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_temporal_layer_gating_in,
        "state_step_error_out_decision"_s <= "state_temporal_layer_gating_in_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_bind_result_decision,
        "state_temporal_layer_gating_in_result_decision"_s <= "state_temporal_layer_gating_in_run"_s + completion<StepRun> [guard_temporal_layer_gating_in_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_gating_in_run"_s + completion<StepRun> [guard_temporal_layer_gating_in_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_run,
        "state_temporal_layer_silu_gate"_s <= "state_temporal_layer_gating_in_result_decision"_s + completion<StepRun> [guard_temporal_layer_silu_gate_supported] / effect_run_temporal_layer_silu_gate_silu,
        "state_step_error_out_decision"_s <= "state_temporal_layer_gating_in_result_decision"_s + completion<StepRun> [guard_temporal_layer_silu_gate_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_result_decision,
        "state_temporal_layer_silu_gate_silu_result_decision"_s <= "state_temporal_layer_silu_gate"_s + completion<StepRun>,
        "state_temporal_layer_silu_gate_mul"_s <= "state_temporal_layer_silu_gate_silu_result_decision"_s + completion<StepRun> [guard_temporal_layer_silu_gate_silu_succeeded] / effect_run_temporal_layer_silu_gate_mul,
        "state_step_error_out_decision"_s <= "state_temporal_layer_silu_gate_silu_result_decision"_s + completion<StepRun> [guard_temporal_layer_silu_gate_silu_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_silu_result_decision,
        "state_temporal_layer_silu_gate_result_decision"_s <= "state_temporal_layer_silu_gate_mul"_s + completion<StepRun> [guard_temporal_layer_silu_gate_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_silu_gate_mul"_s + completion<StepRun> [guard_temporal_layer_silu_gate_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_mul,
        "state_temporal_layer_gating_out"_s <= "state_temporal_layer_silu_gate_result_decision"_s + completion<StepRun> [guard_temporal_layer_gating_out_supported] / effect_bind_temporal_layer_gating_out,
        "state_step_error_out_decision"_s <= "state_temporal_layer_silu_gate_result_decision"_s + completion<StepRun> [guard_temporal_layer_gating_out_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_result_decision,
        "state_temporal_layer_gating_out_bind_result_decision"_s <= "state_temporal_layer_gating_out"_s + completion<StepRun>,
        "state_temporal_layer_gating_out_run"_s <= "state_temporal_layer_gating_out_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_temporal_layer_gating_out,
        "state_step_error_out_decision"_s <= "state_temporal_layer_gating_out_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_out_bind_result_decision,
        "state_temporal_layer_gating_out_result_decision"_s <= "state_temporal_layer_gating_out_run"_s + completion<StepRun> [guard_temporal_layer_gating_out_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_gating_out_run"_s + completion<StepRun> [guard_temporal_layer_gating_out_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_out_run,
        "state_temporal_layer_ff_residual"_s <= "state_temporal_layer_gating_out_result_decision"_s + completion<StepRun> / effect_apply_temporal_layer_ff_residual,
        "state_temporal_layer_ff_residual_result_decision"_s <= "state_temporal_layer_ff_residual"_s + completion<StepRun> [guard_temporal_layer_ff_residual_succeeded],
        "state_step_error_out_decision"_s <= "state_temporal_layer_ff_residual"_s + completion<StepRun> [guard_temporal_layer_ff_residual_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_ff_residual,
        "state_temporal_layer_advance"_s <= "state_temporal_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_more_temporal_layers] / effect_advance_temporal_layer,
        "state_temporal_out_norm"_s <= "state_temporal_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_temporal_layers_complete] / effect_run_temporal_out_norm_rms,
        "state_step_error_out_decision"_s <= "state_temporal_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_temporal_out_norm_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_ff_residual_result_decision,
        "state_temporal_layer_norm"_s <= "state_temporal_layer_advance"_s + completion<StepRun> [guard_temporal_layer_norm_supported] / effect_run_temporal_layer_norm_rms_from_state_temporal_layer_advance,
        "state_step_error_out_decision"_s <= "state_temporal_layer_advance"_s + completion<StepRun> [guard_temporal_layer_norm_unsupported] / effect_mark_graph_execution_unsupported_from_state_temporal_layer_advance,
        "state_temporal_out_norm_rms_result_decision"_s <= "state_temporal_out_norm"_s + completion<StepRun>,
        "state_temporal_out_norm_scale"_s <= "state_temporal_out_norm_rms_result_decision"_s + completion<StepRun> [guard_temporal_out_norm_rms_succeeded] / effect_run_temporal_out_norm_scale,
        "state_step_error_out_decision"_s <= "state_temporal_out_norm_rms_result_decision"_s + completion<StepRun> [guard_temporal_out_norm_rms_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_out_norm_rms_result_decision,
        "state_temporal_out_norm_result_decision"_s <= "state_temporal_out_norm_scale"_s + completion<StepRun> [guard_temporal_out_norm_succeeded] / effect_publish_temporal_out_norm,
        "state_step_error_out_decision"_s <= "state_temporal_out_norm_scale"_s + completion<StepRun> [guard_temporal_out_norm_failed] / effect_mark_graph_execution_unsupported_from_state_temporal_out_norm_scale,
        "state_text_logits_phase_decision"_s <= "state_temporal_out_norm_result_decision"_s + completion<StepRun> [guard_full_graph_phase],
        "state_sampling_ready"_s <= "state_temporal_out_norm_result_decision"_s + completion<StepRun> [guard_prediction_graph_phase] / effect_publish_prediction_state,
        "state_text_logits"_s <= "state_text_logits_phase_decision"_s + completion<StepRun> [guard_text_logits_supported] / effect_bind_text_token_logits,
        "state_step_error_out_decision"_s <= "state_text_logits_phase_decision"_s + completion<StepRun> [guard_text_logits_unsupported] / effect_mark_graph_execution_unsupported_from_state_text_logits_phase_decision,
        "state_text_logits"_s <= "state_sampling_ready"_s + event<StepRun> [guard_sampling_step_valid] / effect_restore_prediction_state_and_bind_text_logits,
        "state_step_error_out_decision"_s <= "state_sampling_ready"_s + event<StepRun> [guard_sampling_step_invalid] / effect_mark_request_shape_from_state_sampling_ready,
        "state_text_logits_bind_result_decision"_s <= "state_text_logits"_s + completion<StepRun>,
        "state_text_logits_sample_projection"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_forced_text_token_valid_and_sampling_consume] / effect_compute_text_token_logits_from_state_text_logits_bind_result_decision,
        "state_text_logits_result_decision"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_forced_text_token_valid_without_sampling_consume] / effect_publish_forced_text_token_from_state_text_logits_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_forced_text_sampling_config_invalid] / effect_mark_graph_execution_unsupported_from_state_text_logits_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_forced_text_token_invalid] / effect_mark_request_shape_from_state_text_logits_bind_result_decision,
        "state_text_logits_run"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_text_logits_projection_bound_and_no_forced_token_argmax] / effect_select_text_token,
        "state_text_logits_sample_projection"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_text_logits_projection_bound_and_no_forced_token_sampling] / effect_compute_text_token_logits_from_state_text_logits_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid] / effect_mark_graph_execution_unsupported_from_state_text_logits_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_text_logits_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_text_logits_bind_result_decision,
        "state_text_logits_sample_projection_result_decision"_s <= "state_text_logits_sample_projection"_s + completion<StepRun>,
        "state_text_logits_sample_select"_s <= "state_text_logits_sample_projection_result_decision"_s + completion<StepRun> [guard_text_logits_matmul_succeeded] / effect_select_text_sampling_token,
        "state_step_error_out_decision"_s <= "state_text_logits_sample_projection_result_decision"_s + completion<StepRun> [guard_text_logits_matmul_failed] / effect_mark_graph_execution_unsupported_from_state_text_logits_sample_projection_result_decision,
        "state_text_logits_result_decision"_s <= "state_text_logits_sample_select"_s + completion<StepRun> [guard_sampled_text_token_ready] / effect_publish_sampled_text_token,
        "state_text_logits_result_decision"_s <= "state_text_logits_sample_select"_s + completion<StepRun> [guard_forced_text_sampling_consumed] / effect_publish_forced_text_token_from_state_text_logits_sample_select,
        "state_step_error_out_decision"_s <= "state_text_logits_sample_select"_s + completion<StepRun> [guard_text_sampling_failed] / effect_mark_graph_execution_unsupported_from_state_text_logits_sample_select,
        "state_text_logits_result_decision"_s <= "state_text_logits_run"_s + completion<StepRun> [guard_text_logits_succeeded],
        "state_step_error_out_decision"_s <= "state_text_logits_run"_s + completion<StepRun> [guard_text_logits_failed] / effect_mark_graph_execution_unsupported_from_state_text_logits_run,
        "state_bind_depformer_kv"_s <= "state_text_logits_result_decision"_s + completion<StepRun> [guard_depformer_kv_binding_present] / effect_bind_depformer_kv,
        "state_step_error_out_decision"_s <= "state_text_logits_result_decision"_s + completion<StepRun> [guard_depformer_kv_binding_missing] / effect_mark_graph_execution_unsupported_from_state_text_logits_result_decision,
        "state_depformer_kv_result_decision"_s <= "state_bind_depformer_kv"_s + completion<StepRun> [guard_depformer_kv_bound] / effect_reset_depformer_positions,
        "state_step_error_out_decision"_s <= "state_bind_depformer_kv"_s + completion<StepRun> [guard_depformer_kv_bind_failed] / effect_mark_graph_execution_unsupported_from_state_bind_depformer_kv,
        "state_depformer_position_reset_result_decision"_s <= "state_depformer_kv_result_decision"_s + completion<StepRun>,
        "state_depformer_position_advance_result_decision"_s <= "state_depformer_position_reset_result_decision"_s + completion<StepRun> [guard_depformer_position_reset_succeeded] / effect_advance_depformer_position_from_state_depformer_position_reset_result_decision,
        "state_step_error_out_decision"_s <= "state_depformer_position_reset_result_decision"_s + completion<StepRun> [guard_depformer_position_reset_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_position_reset_result_decision,
        "state_depformer_weight_route_decision"_s <= "state_depformer_position_advance_result_decision"_s + completion<StepRun> [guard_depformer_position_advance_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_position_advance_result_decision"_s + completion<StepRun> [guard_depformer_position_advance_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_position_advance_result_decision,
        "state_depformer_weight_decision"_s <= "state_depformer_weight_route_decision"_s + completion<StepRun> [guard_depformer_scheduled_weight_present] / effect_use_depformer_scheduled_weight,
        "state_depformer_weight_decision"_s <= "state_depformer_weight_route_decision"_s + completion<StepRun> [guard_depformer_scheduled_weight_absent] / effect_use_depformer_codebook_weight,
        "state_step_error_out_decision"_s <= "state_depformer_weight_route_decision"_s + completion<StepRun> [guard_depformer_scheduled_weight_invalid] / effect_mark_graph_execution_unsupported_from_state_depformer_weight_route_decision,
        "state_depformer_input"_s <= "state_depformer_weight_decision"_s + completion<StepRun> [guard_depformer_text_input_supported] / effect_bind_depformer_text_input_projection,
        "state_depformer_input"_s <= "state_depformer_weight_decision"_s + completion<StepRun> [guard_depformer_audio_input_supported] / effect_bind_depformer_audio_input_projection,
        "state_step_error_out_decision"_s <= "state_depformer_weight_decision"_s + completion<StepRun> [guard_depformer_input_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_weight_decision,
        "state_depformer_input_projection_bind_result_decision"_s <= "state_depformer_input"_s + completion<StepRun>,
        "state_depformer_input_projection"_s <= "state_depformer_input_projection_bind_result_decision"_s + completion<StepRun> [guard_depformer_input_projection_bound] / effect_run_depformer_input_projection,
        "state_step_error_out_decision"_s <= "state_depformer_input_projection_bind_result_decision"_s + completion<StepRun> [guard_depformer_input_projection_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_input_projection_bind_result_decision,
        "state_depformer_input_projection_result_decision"_s <= "state_depformer_input_projection"_s + completion<StepRun>,
        "state_depformer_input_embedding_bind"_s <= "state_depformer_input_projection_result_decision"_s + completion<StepRun> [guard_depformer_text_input_projection_succeeded] / effect_bind_depformer_text_input_embedding,
        "state_depformer_input_embedding_bind"_s <= "state_depformer_input_projection_result_decision"_s + completion<StepRun> [guard_depformer_audio_input_projection_succeeded] / effect_bind_depformer_audio_input_embedding,
        "state_step_error_out_decision"_s <= "state_depformer_input_projection_result_decision"_s + completion<StepRun> [guard_depformer_input_projection_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_input_projection_result_decision,
        "state_step_error_out_decision"_s <= "state_depformer_input_projection_result_decision"_s + completion<StepRun> [guard_depformer_input_projection_embedding_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_input_projection_result_decision,
        "state_depformer_input_embedding_bind_result_decision"_s <= "state_depformer_input_embedding_bind"_s + completion<StepRun>,
        "state_depformer_input_embedding_row"_s <= "state_depformer_input_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bound] / effect_run_embedding_row_fetch_from_state_depformer_input_embedding_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_depformer_input_embedding_bind_result_decision"_s + completion<StepRun> [guard_embedding_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_bind_result_decision,
        "state_depformer_input_embedding_row_result_decision"_s <= "state_depformer_input_embedding_row"_s + completion<StepRun>,
        "state_depformer_input_embedding_apply"_s <= "state_depformer_input_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_succeeded] / effect_apply_depformer_input_embedding_row,
        "state_step_error_out_decision"_s <= "state_depformer_input_embedding_row_result_decision"_s + completion<StepRun> [guard_embedding_row_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_row_result_decision,
        "state_depformer_input_result_decision"_s <= "state_depformer_input_embedding_apply"_s + completion<StepRun> [guard_depformer_input_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_input_embedding_apply"_s + completion<StepRun> [guard_depformer_input_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_apply,
        "state_depformer_layer_norm"_s <= "state_depformer_input_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm_supported] / effect_run_depformer_layer_norm_rms_from_state_depformer_input_result_decision,
        "state_step_error_out_decision"_s <= "state_depformer_input_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_input_result_decision,
        "state_depformer_layer_norm_rms_result_decision"_s <= "state_depformer_layer_norm"_s + completion<StepRun>,
        "state_depformer_layer_norm_scale"_s <= "state_depformer_layer_norm_rms_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm_rms_succeeded] / effect_run_depformer_layer_norm_scale,
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm_rms_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm_rms_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_rms_result_decision,
        "state_depformer_layer_norm_result_decision"_s <= "state_depformer_layer_norm_scale"_s + completion<StepRun> [guard_depformer_layer_norm_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm_scale"_s + completion<StepRun> [guard_depformer_layer_norm_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_scale,
        "state_depformer_layer_projection"_s <= "state_depformer_layer_norm_result_decision"_s + completion<StepRun> [guard_depformer_layer_projection_supported] / effect_bind_depformer_layer_projection,
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm_result_decision"_s + completion<StepRun> [guard_depformer_layer_projection_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_result_decision,
        "state_depformer_layer_projection_bind_result_decision"_s <= "state_depformer_layer_projection"_s + completion<StepRun>,
        "state_depformer_layer_projection_run"_s <= "state_depformer_layer_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_depformer_layer_projection,
        "state_step_error_out_decision"_s <= "state_depformer_layer_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_bind_result_decision,
        "state_depformer_layer_projection_result_decision"_s <= "state_depformer_layer_projection_run"_s + completion<StepRun> [guard_depformer_layer_projection_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_projection_run"_s + completion<StepRun> [guard_depformer_layer_projection_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_run,
        "state_depformer_layer_cache_write"_s <= "state_depformer_layer_projection_result_decision"_s + completion<StepRun> [guard_depformer_layer_cache_write_supported] / effect_write_depformer_layer_kv_cache,
        "state_step_error_out_decision"_s <= "state_depformer_layer_projection_result_decision"_s + completion<StepRun> [guard_depformer_layer_cache_write_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_result_decision,
        "state_depformer_layer_cache_write_result_decision"_s <= "state_depformer_layer_cache_write"_s + completion<StepRun> [guard_depformer_layer_cache_write_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_cache_write"_s + completion<StepRun> [guard_depformer_layer_cache_write_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_cache_write,
        "state_depformer_layer_attention"_s <= "state_depformer_layer_cache_write_result_decision"_s + completion<StepRun> [guard_depformer_layer_attention_supported] / effect_run_depformer_layer_attention,
        "state_step_error_out_decision"_s <= "state_depformer_layer_cache_write_result_decision"_s + completion<StepRun> [guard_depformer_layer_attention_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_cache_write_result_decision,
        "state_depformer_layer_attention_result_decision"_s <= "state_depformer_layer_attention"_s + completion<StepRun> [guard_depformer_layer_attention_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_attention"_s + completion<StepRun> [guard_depformer_layer_attention_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_attention,
        "state_depformer_layer_out_projection"_s <= "state_depformer_layer_attention_result_decision"_s + completion<StepRun> [guard_depformer_layer_out_projection_supported] / effect_bind_depformer_layer_out_projection,
        "state_step_error_out_decision"_s <= "state_depformer_layer_attention_result_decision"_s + completion<StepRun> [guard_depformer_layer_out_projection_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_attention_result_decision,
        "state_depformer_layer_out_projection_bind_result_decision"_s <= "state_depformer_layer_out_projection"_s + completion<StepRun>,
        "state_depformer_layer_out_projection_run"_s <= "state_depformer_layer_out_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_depformer_layer_out_projection,
        "state_step_error_out_decision"_s <= "state_depformer_layer_out_projection_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_out_projection_bind_result_decision,
        "state_depformer_layer_out_projection_result_decision"_s <= "state_depformer_layer_out_projection_run"_s + completion<StepRun> [guard_depformer_layer_out_projection_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_out_projection_run"_s + completion<StepRun> [guard_depformer_layer_out_projection_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_out_projection_run,
        "state_depformer_layer_residual"_s <= "state_depformer_layer_out_projection_result_decision"_s + completion<StepRun> / effect_apply_depformer_layer_attention_residual,
        "state_depformer_layer_residual_result_decision"_s <= "state_depformer_layer_residual"_s + completion<StepRun> [guard_depformer_layer_residual_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_residual"_s + completion<StepRun> [guard_depformer_layer_residual_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_residual,
        "state_depformer_layer_norm2"_s <= "state_depformer_layer_residual_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm2_supported] / effect_run_depformer_layer_norm2_rms,
        "state_step_error_out_decision"_s <= "state_depformer_layer_residual_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm2_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_residual_result_decision,
        "state_depformer_layer_norm2_rms_result_decision"_s <= "state_depformer_layer_norm2"_s + completion<StepRun>,
        "state_depformer_layer_norm2_scale"_s <= "state_depformer_layer_norm2_rms_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm2_rms_succeeded] / effect_run_depformer_layer_norm2_scale,
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm2_rms_result_decision"_s + completion<StepRun> [guard_depformer_layer_norm2_rms_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_rms_result_decision,
        "state_depformer_layer_norm2_result_decision"_s <= "state_depformer_layer_norm2_scale"_s + completion<StepRun> [guard_depformer_layer_norm2_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm2_scale"_s + completion<StepRun> [guard_depformer_layer_norm2_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_scale,
        "state_depformer_layer_gating_in"_s <= "state_depformer_layer_norm2_result_decision"_s + completion<StepRun> [guard_depformer_layer_gating_in_supported] / effect_bind_depformer_layer_gating_in,
        "state_step_error_out_decision"_s <= "state_depformer_layer_norm2_result_decision"_s + completion<StepRun> [guard_depformer_layer_gating_in_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_result_decision,
        "state_depformer_layer_gating_in_bind_result_decision"_s <= "state_depformer_layer_gating_in"_s + completion<StepRun>,
        "state_depformer_layer_gating_in_run"_s <= "state_depformer_layer_gating_in_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_depformer_layer_gating_in,
        "state_step_error_out_decision"_s <= "state_depformer_layer_gating_in_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_bind_result_decision,
        "state_depformer_layer_gating_in_result_decision"_s <= "state_depformer_layer_gating_in_run"_s + completion<StepRun> [guard_depformer_layer_gating_in_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_gating_in_run"_s + completion<StepRun> [guard_depformer_layer_gating_in_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_run,
        "state_depformer_layer_silu_gate"_s <= "state_depformer_layer_gating_in_result_decision"_s + completion<StepRun> [guard_depformer_layer_silu_gate_supported] / effect_run_depformer_layer_silu_gate_silu,
        "state_step_error_out_decision"_s <= "state_depformer_layer_gating_in_result_decision"_s + completion<StepRun> [guard_depformer_layer_silu_gate_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_result_decision,
        "state_depformer_layer_silu_gate_silu_result_decision"_s <= "state_depformer_layer_silu_gate"_s + completion<StepRun>,
        "state_depformer_layer_silu_gate_mul"_s <= "state_depformer_layer_silu_gate_silu_result_decision"_s + completion<StepRun> [guard_depformer_layer_silu_gate_silu_succeeded] / effect_run_depformer_layer_silu_gate_mul,
        "state_step_error_out_decision"_s <= "state_depformer_layer_silu_gate_silu_result_decision"_s + completion<StepRun> [guard_depformer_layer_silu_gate_silu_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_silu_result_decision,
        "state_depformer_layer_silu_gate_result_decision"_s <= "state_depformer_layer_silu_gate_mul"_s + completion<StepRun> [guard_depformer_layer_silu_gate_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_silu_gate_mul"_s + completion<StepRun> [guard_depformer_layer_silu_gate_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_mul,
        "state_depformer_layer_gating_out"_s <= "state_depformer_layer_silu_gate_result_decision"_s + completion<StepRun> [guard_depformer_layer_gating_out_supported] / effect_bind_depformer_layer_gating_out,
        "state_step_error_out_decision"_s <= "state_depformer_layer_silu_gate_result_decision"_s + completion<StepRun> [guard_depformer_layer_gating_out_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_result_decision,
        "state_depformer_layer_gating_out_bind_result_decision"_s <= "state_depformer_layer_gating_out"_s + completion<StepRun>,
        "state_depformer_layer_gating_out_run"_s <= "state_depformer_layer_gating_out_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bound] / effect_run_depformer_layer_gating_out,
        "state_step_error_out_decision"_s <= "state_depformer_layer_gating_out_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_out_bind_result_decision,
        "state_depformer_layer_gating_out_result_decision"_s <= "state_depformer_layer_gating_out_run"_s + completion<StepRun> [guard_depformer_layer_gating_out_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_gating_out_run"_s + completion<StepRun> [guard_depformer_layer_gating_out_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_out_run,
        "state_depformer_layer_ff_residual"_s <= "state_depformer_layer_gating_out_result_decision"_s + completion<StepRun> / effect_apply_depformer_layer_ff_residual,
        "state_depformer_layer_ff_residual_result_decision"_s <= "state_depformer_layer_ff_residual"_s + completion<StepRun> [guard_depformer_layer_ff_residual_succeeded],
        "state_step_error_out_decision"_s <= "state_depformer_layer_ff_residual"_s + completion<StepRun> [guard_depformer_layer_ff_residual_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_ff_residual,
        "state_depformer_layer_advance"_s <= "state_depformer_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_more_depformer_layers] / effect_advance_depformer_layer,
        "state_depformer_logits"_s <= "state_depformer_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_depformer_logits_supported] / effect_bind_depformer_token_logits,
        "state_step_error_out_decision"_s <= "state_depformer_layer_ff_residual_result_decision"_s + completion<StepRun> [guard_depformer_logits_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_ff_residual_result_decision,
        "state_depformer_logits_bind_result_decision"_s <= "state_depformer_logits"_s + completion<StepRun>,
        "state_depformer_logits_run"_s <= "state_depformer_logits_bind_result_decision"_s + completion<StepRun> [guard_depformer_logits_projection_bound_argmax] / effect_select_depformer_token,
        "state_depformer_logits_sample_projection"_s <= "state_depformer_logits_bind_result_decision"_s + completion<StepRun> [guard_depformer_logits_projection_bound_sampling] / effect_compute_depformer_token_logits,
        "state_step_error_out_decision"_s <= "state_depformer_logits_bind_result_decision"_s + completion<StepRun> [guard_depformer_logits_projection_bound_sampling_invalid] / effect_mark_graph_execution_unsupported_from_state_depformer_logits_bind_result_decision,
        "state_step_error_out_decision"_s <= "state_depformer_logits_bind_result_decision"_s + completion<StepRun> [guard_projection_view_bind_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_logits_bind_result_decision,
        "state_depformer_logits_sample_projection_result_decision"_s <= "state_depformer_logits_sample_projection"_s + completion<StepRun>,
        "state_depformer_logits_sample_select"_s <= "state_depformer_logits_sample_projection_result_decision"_s + completion<StepRun> [guard_depformer_logits_matmul_succeeded] / effect_select_depformer_sampling_token,
        "state_step_error_out_decision"_s <= "state_depformer_logits_sample_projection_result_decision"_s + completion<StepRun> [guard_depformer_logits_matmul_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_logits_sample_projection_result_decision,
        "state_depformer_layer_norm"_s <= "state_depformer_layer_advance"_s + completion<StepRun> [guard_depformer_layer_norm_supported] / effect_run_depformer_layer_norm_rms_from_state_depformer_layer_advance,
        "state_step_error_out_decision"_s <= "state_depformer_layer_advance"_s + completion<StepRun> [guard_depformer_layer_norm_unsupported] / effect_mark_graph_execution_unsupported_from_state_depformer_layer_advance,
        "state_depformer_token_publish"_s <= "state_depformer_logits_run"_s + completion<StepRun> [guard_depformer_logits_succeeded] / effect_publish_depformer_token_from_state_depformer_logits_run,
        "state_step_error_out_decision"_s <= "state_depformer_logits_run"_s + completion<StepRun> [guard_depformer_logits_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_logits_run,
        "state_depformer_token_publish"_s <= "state_depformer_logits_sample_select"_s + completion<StepRun> [guard_depformer_sampling_succeeded] / effect_publish_depformer_token_from_state_depformer_logits_sample_select,
        "state_step_error_out_decision"_s <= "state_depformer_logits_sample_select"_s + completion<StepRun> [guard_depformer_sampling_failed] / effect_mark_graph_execution_unsupported_from_state_depformer_logits_sample_select,
        "state_depformer_logits_result_decision"_s <= "state_depformer_token_publish"_s + completion<StepRun>,
        "state_depformer_codebook_advance"_s <= "state_depformer_logits_result_decision"_s + completion<StepRun> [guard_more_depformer_codebooks] / effect_advance_depformer_codebook,
        "state_ready"_s <= "state_depformer_logits_result_decision"_s + completion<StepRun> [guard_depformer_codebooks_complete],
        "state_depformer_position_advance_result_decision"_s <= "state_depformer_codebook_advance"_s + completion<StepRun> / effect_advance_depformer_position_from_state_depformer_codebook_advance,
        "state_ready"_s <= "state_step_error_out_decision"_s + completion<StepRun> [guard_has_error_out_step_run] / effect_store_error_out_step_run_from_state_step_error_out_decision,
        "state_ready"_s <= "state_step_error_out_decision"_s + completion<StepRun> [guard_no_error_out_step_run],
        "state_uninit_step_error_out_decision"_s <= "state_uninitialized"_s + event<StepRun> / effect_mark_not_initialized,
        "state_uninitialized"_s <= "state_uninit_step_error_out_decision"_s + completion<StepRun> [guard_has_error_out_step_run] / effect_store_error_out_step_run_from_state_uninit_step_error_out_decision,
        "state_uninitialized"_s <= "state_uninit_step_error_out_decision"_s + completion<StepRun> [guard_no_error_out_step_run],
        "state_reset_temporal_positions_decision"_s <= "state_ready"_s + event<ResetRun>,
        "state_reset_temporal_positions_decision"_s <= "state_sampling_ready"_s + event<ResetRun>,
        "state_reset_temporal_positions_result_decision"_s <= "state_reset_temporal_positions_decision"_s + completion<ResetRun> [guard_reset_temporal_positions_present] / effect_reset_temporal_positions,
        "state_reset_depformer_positions_decision"_s <= "state_reset_temporal_positions_decision"_s + completion<ResetRun> [guard_reset_temporal_positions_missing],
        "state_reset_depformer_positions_decision"_s <= "state_reset_temporal_positions_result_decision"_s + completion<ResetRun> [guard_reset_temporal_positions_succeeded],
        "state_reset_failed"_s <= "state_reset_temporal_positions_result_decision"_s + completion<ResetRun> [guard_reset_temporal_positions_failed] / effect_mark_reset_failed_from_state_reset_temporal_positions_result_decision,
        "state_reset_depformer_positions_result_decision"_s <= "state_reset_depformer_positions_decision"_s + completion<ResetRun> [guard_reset_depformer_positions_present] / effect_reset_bound_depformer_positions,
        "state_uninitialized"_s <= "state_reset_depformer_positions_decision"_s + completion<ResetRun> [guard_reset_depformer_positions_missing] / effect_reset_session_from_state_reset_depformer_positions_decision,
        "state_uninitialized"_s <= "state_reset_depformer_positions_result_decision"_s + completion<ResetRun> [guard_reset_depformer_positions_succeeded] / effect_reset_session_from_state_reset_depformer_positions_result_decision,
        "state_reset_failed"_s <= "state_reset_depformer_positions_result_decision"_s + completion<ResetRun> [guard_reset_depformer_positions_failed] / effect_mark_reset_failed_from_state_reset_depformer_positions_result_decision,
        "state_uninitialized"_s <= "state_reset_failed"_s + completion<ResetRun> / effect_reset_session_from_state_reset_failed,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_ready,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_ready,
        "state_sampling_ready"_s <= "state_sampling_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_sampling_ready,
        "state_sampling_ready"_s <= "state_sampling_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_sampling_ready,
    }
}

/// Context for `SpeechPredictorMoshiExecutor` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechPredictorMoshiExecutorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechPredictorMoshiExecutorStateMachineContext for SpeechPredictorMoshiExecutorContext {
    fn effect_advance_depformer_codebook(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_depformer_codebook
        todo!(
            "TODO: port action `effect_advance_depformer_codebook` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_depformer_layer(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_depformer_layer
        todo!(
            "TODO: port action `effect_advance_depformer_layer` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_depformer_position_from_state_depformer_codebook_advance(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_depformer_position
        todo!(
            "TODO: port action `effect_advance_depformer_position` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_depformer_position_from_state_depformer_position_reset_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_depformer_position
        todo!(
            "TODO: port action `effect_advance_depformer_position` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_input_audio_codebook(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_input_audio_codebook
        todo!(
            "TODO: port action `effect_advance_input_audio_codebook` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_temporal_layer(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_temporal_layer
        todo!(
            "TODO: port action `effect_advance_temporal_layer` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_advance_temporal_position(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_advance_temporal_position
        todo!(
            "TODO: port action `effect_advance_temporal_position` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_depformer_input_embedding_row(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_depformer_input_embedding_row
        todo!(
            "TODO: port action `effect_apply_depformer_input_embedding_row` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_depformer_layer_attention_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_depformer_layer_attention_residual
        todo!(
            "TODO: port action `effect_apply_depformer_layer_attention_residual` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_depformer_layer_ff_residual(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_depformer_layer_ff_residual
        todo!(
            "TODO: port action `effect_apply_depformer_layer_ff_residual` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_external_input_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_external_input_embedding
        todo!(
            "TODO: port action `effect_apply_external_input_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_input_audio_embedding_row(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_input_audio_embedding_row
        todo!(
            "TODO: port action `effect_apply_input_audio_embedding_row` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_input_text_embedding_row(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_input_text_embedding_row
        todo!(
            "TODO: port action `effect_apply_input_text_embedding_row` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_temporal_layer_attention_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_temporal_layer_attention_residual
        todo!(
            "TODO: port action `effect_apply_temporal_layer_attention_residual` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_apply_temporal_layer_ff_residual(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_apply_temporal_layer_ff_residual
        todo!(
            "TODO: port action `effect_apply_temporal_layer_ff_residual` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_begin_input_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_begin_input_embedding
        todo!(
            "TODO: port action `effect_begin_input_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_contract(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_contract
        todo!(
            "TODO: port action `effect_bind_contract` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_audio_input_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_audio_input_embedding
        todo!(
            "TODO: port action `effect_bind_depformer_audio_input_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_audio_input_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_audio_input_projection
        todo!(
            "TODO: port action `effect_bind_depformer_audio_input_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_fused_projection_layout(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_fused_projection_layout
        todo!(
            "TODO: port action `effect_bind_depformer_fused_projection_layout` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_kv(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_kv
        todo!(
            "TODO: port action `effect_bind_depformer_kv` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_layer_gating_in(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_layer_gating_in
        todo!(
            "TODO: port action `effect_bind_depformer_layer_gating_in` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_layer_gating_out(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_layer_gating_out
        todo!(
            "TODO: port action `effect_bind_depformer_layer_gating_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_layer_out_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_layer_out_projection
        todo!(
            "TODO: port action `effect_bind_depformer_layer_out_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_layer_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_layer_projection
        todo!(
            "TODO: port action `effect_bind_depformer_layer_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_split_projection_layout(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_split_projection_layout
        todo!(
            "TODO: port action `effect_bind_depformer_split_projection_layout` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_text_input_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_text_input_embedding
        todo!(
            "TODO: port action `effect_bind_depformer_text_input_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_text_input_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_text_input_projection
        todo!(
            "TODO: port action `effect_bind_depformer_text_input_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_depformer_token_logits(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_depformer_token_logits
        todo!(
            "TODO: port action `effect_bind_depformer_token_logits` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_full_card_audio_sampling_top_k(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_full_card_audio_sampling_top_k
        todo!(
            "TODO: port action `effect_bind_full_card_audio_sampling_top_k` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_full_card_text_sampling_top_k(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_full_card_text_sampling_top_k
        todo!(
            "TODO: port action `effect_bind_full_card_text_sampling_top_k` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_input_audio_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_input_audio_embedding
        todo!(
            "TODO: port action `effect_bind_input_audio_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_input_text_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_input_text_embedding
        todo!(
            "TODO: port action `effect_bind_input_text_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_nonzero_sampling_seed(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_nonzero_sampling_seed
        todo!(
            "TODO: port action `effect_bind_nonzero_sampling_seed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_requested_audio_sampling_top_k(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_requested_audio_sampling_top_k
        todo!(
            "TODO: port action `effect_bind_requested_audio_sampling_top_k` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_requested_text_sampling_top_k(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_requested_text_sampling_top_k
        todo!(
            "TODO: port action `effect_bind_requested_text_sampling_top_k` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_fused_projection_layout(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_fused_projection_layout
        todo!(
            "TODO: port action `effect_bind_temporal_fused_projection_layout` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_kv(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_kv
        todo!(
            "TODO: port action `effect_bind_temporal_kv` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_layer_gating_in(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_layer_gating_in
        todo!(
            "TODO: port action `effect_bind_temporal_layer_gating_in` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_layer_gating_out(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_layer_gating_out
        todo!(
            "TODO: port action `effect_bind_temporal_layer_gating_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_layer_out_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_layer_out_projection
        todo!(
            "TODO: port action `effect_bind_temporal_layer_out_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_layer_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_layer_projection
        todo!(
            "TODO: port action `effect_bind_temporal_layer_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_temporal_split_projection_layout(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_temporal_split_projection_layout
        todo!(
            "TODO: port action `effect_bind_temporal_split_projection_layout` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_text_token_logits(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_text_token_logits
        todo!(
            "TODO: port action `effect_bind_text_token_logits` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_bind_zero_sampling_seed(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_bind_zero_sampling_seed
        todo!(
            "TODO: port action `effect_bind_zero_sampling_seed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_compute_depformer_token_logits(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_compute_depformer_token_logits
        todo!(
            "TODO: port action `effect_compute_depformer_token_logits` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_compute_text_token_logits_from_state_text_logits_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_compute_text_token_logits
        todo!(
            "TODO: port action `effect_compute_text_token_logits` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_copy_temporal_layer_key_rope(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_copy_temporal_layer_key_rope
        todo!(
            "TODO: port action `effect_copy_temporal_layer_key_rope` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_copy_temporal_layer_query_rope(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_copy_temporal_layer_query_rope
        todo!(
            "TODO: port action `effect_copy_temporal_layer_query_rope` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_emit_initialize_done(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_emit_initialize_error(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_finish_input_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_finish_input_embedding
        todo!(
            "TODO: port action `effect_finish_input_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_bind_failed_from_state_bind_contract_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_bind_failed_from_state_depformer_projection_layout_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_bind_failed_from_state_input_embedding_contract_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_bind_failed_from_state_temporal_projection_layout_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_bind_depformer_kv(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_bind_temporal_kv(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_apply(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_embedding_row_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_projection_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_projection_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_input_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_advance(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_attention(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_attention_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_cache_write(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_cache_write_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_ff_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_ff_residual_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_in_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_out_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_gating_out_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_rms_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm2_scale(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_rms_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_norm_scale(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_out_projection_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_out_projection_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_projection_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_residual_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_mul(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_layer_silu_gate_silu_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_logits_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_logits_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_logits_sample_projection_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_logits_sample_select(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_position_advance_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_position_reset_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_weight_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_depformer_weight_route_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_apply(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_audio_embedding_row_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_audio_token_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_embedding(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_embedding_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_text_embedding_apply(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_text_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_input_text_embedding_row_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_step_execution_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_advance(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_attention(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_attention_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_cache_write(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_cache_write_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_ff_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_ff_residual_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_in_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_out_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_gating_out_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_key_rope(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_key_rope_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_rms_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm2_scale(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_rms_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_norm_scale(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_out_projection_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_out_projection_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_projection_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_residual(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_residual_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_rope(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_mul(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_layer_silu_gate_silu_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_out_norm_rms_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_out_norm_scale(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_position_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_temporal_position_valid_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_phase_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_sample_projection_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_graph_execution_unsupported_from_state_text_logits_sample_select(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_graph_execution_unsupported
        todo!(
            "TODO: port action `effect_mark_graph_execution_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_model_mismatch(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_model_mismatch
        todo!(
            "TODO: port action `effect_mark_model_mismatch` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_not_initialized(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_request_shape_from_state_sampling_ready(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_request_shape
        todo!(
            "TODO: port action `effect_mark_request_shape` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_request_shape_from_state_step_shape_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_request_shape
        todo!(
            "TODO: port action `effect_mark_request_shape` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_request_shape_from_state_text_logits_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_request_shape
        todo!(
            "TODO: port action `effect_mark_request_shape` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_reset_failed_from_state_reset_depformer_positions_result_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_reset_failed
        todo!(
            "TODO: port action `effect_mark_reset_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_reset_failed_from_state_reset_temporal_positions_result_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_reset_failed
        todo!(
            "TODO: port action `effect_mark_reset_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_sampling_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_sampling_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_depformer_token_from_state_depformer_logits_run(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_depformer_token
        todo!(
            "TODO: port action `effect_publish_depformer_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_depformer_token_from_state_depformer_logits_sample_select(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_depformer_token
        todo!(
            "TODO: port action `effect_publish_depformer_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_forced_text_token_from_state_text_logits_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_forced_text_token
        todo!(
            "TODO: port action `effect_publish_forced_text_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_forced_text_token_from_state_text_logits_sample_select(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_forced_text_token
        todo!(
            "TODO: port action `effect_publish_forced_text_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_prediction_state(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_prediction_state
        todo!(
            "TODO: port action `effect_publish_prediction_state` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_sampled_text_token(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_sampled_text_token
        todo!(
            "TODO: port action `effect_publish_sampled_text_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_publish_temporal_out_norm(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_publish_temporal_out_norm
        todo!(
            "TODO: port action `effect_publish_temporal_out_norm` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_bound_depformer_positions(&mut self, _event: &ResetRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_bound_depformer_positions
        todo!(
            "TODO: port action `effect_reset_bound_depformer_positions` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_depformer_positions(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_depformer_positions
        todo!(
            "TODO: port action `effect_reset_depformer_positions` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_session_from_state_reset_depformer_positions_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_session
        todo!(
            "TODO: port action `effect_reset_session` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_session_from_state_reset_depformer_positions_result_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_session
        todo!(
            "TODO: port action `effect_reset_session` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_session_from_state_reset_failed(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_session
        todo!(
            "TODO: port action `effect_reset_session` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_reset_temporal_positions(&mut self, _event: &ResetRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_reset_temporal_positions
        todo!(
            "TODO: port action `effect_reset_temporal_positions` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_restore_prediction_state_and_bind_text_logits(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_restore_prediction_state_and_bind_text_logits
        todo!(
            "TODO: port action `effect_restore_prediction_state_and_bind_text_logits` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_input_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_input_projection
        todo!(
            "TODO: port action `effect_run_depformer_input_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_attention(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_attention
        todo!(
            "TODO: port action `effect_run_depformer_layer_attention` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_gating_in(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_gating_in
        todo!(
            "TODO: port action `effect_run_depformer_layer_gating_in` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_gating_out(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_gating_out
        todo!(
            "TODO: port action `effect_run_depformer_layer_gating_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_norm2_rms(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_norm2_rms
        todo!(
            "TODO: port action `effect_run_depformer_layer_norm2_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_norm2_scale(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_norm2_scale
        todo!(
            "TODO: port action `effect_run_depformer_layer_norm2_scale` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_norm_rms_from_state_depformer_input_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_norm_rms
        todo!(
            "TODO: port action `effect_run_depformer_layer_norm_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_norm_rms_from_state_depformer_layer_advance(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_norm_rms
        todo!(
            "TODO: port action `effect_run_depformer_layer_norm_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_norm_scale(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_norm_scale
        todo!(
            "TODO: port action `effect_run_depformer_layer_norm_scale` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_out_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_out_projection
        todo!(
            "TODO: port action `effect_run_depformer_layer_out_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_projection
        todo!(
            "TODO: port action `effect_run_depformer_layer_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_silu_gate_mul(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_silu_gate_mul
        todo!(
            "TODO: port action `effect_run_depformer_layer_silu_gate_mul` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_depformer_layer_silu_gate_silu(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_depformer_layer_silu_gate_silu
        todo!(
            "TODO: port action `effect_run_depformer_layer_silu_gate_silu` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_embedding_row_fetch_from_state_depformer_input_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_embedding_row_fetch
        todo!(
            "TODO: port action `effect_run_embedding_row_fetch` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_embedding_row_fetch_from_state_input_audio_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_embedding_row_fetch
        todo!(
            "TODO: port action `effect_run_embedding_row_fetch` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_embedding_row_fetch_from_state_input_text_embedding_bind_result_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_embedding_row_fetch
        todo!(
            "TODO: port action `effect_run_embedding_row_fetch` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_attention(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_attention
        todo!(
            "TODO: port action `effect_run_temporal_layer_attention` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_gating_in(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_gating_in
        todo!(
            "TODO: port action `effect_run_temporal_layer_gating_in` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_gating_out(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_gating_out
        todo!(
            "TODO: port action `effect_run_temporal_layer_gating_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_key_rope(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_key_rope
        todo!(
            "TODO: port action `effect_run_temporal_layer_key_rope` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_norm2_rms(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_norm2_rms
        todo!(
            "TODO: port action `effect_run_temporal_layer_norm2_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_norm2_scale(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_norm2_scale
        todo!(
            "TODO: port action `effect_run_temporal_layer_norm2_scale` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_norm_rms_from_state_temporal_layer_advance(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_norm_rms
        todo!(
            "TODO: port action `effect_run_temporal_layer_norm_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_norm_rms_from_state_temporal_position_valid_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_norm_rms
        todo!(
            "TODO: port action `effect_run_temporal_layer_norm_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_norm_scale(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_norm_scale
        todo!(
            "TODO: port action `effect_run_temporal_layer_norm_scale` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_out_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_out_projection
        todo!(
            "TODO: port action `effect_run_temporal_layer_out_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_projection(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_projection
        todo!(
            "TODO: port action `effect_run_temporal_layer_projection` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_query_rope(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_query_rope
        todo!(
            "TODO: port action `effect_run_temporal_layer_query_rope` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_silu_gate_mul(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_silu_gate_mul
        todo!(
            "TODO: port action `effect_run_temporal_layer_silu_gate_mul` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_layer_silu_gate_silu(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_layer_silu_gate_silu
        todo!(
            "TODO: port action `effect_run_temporal_layer_silu_gate_silu` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_out_norm_rms(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_out_norm_rms
        todo!(
            "TODO: port action `effect_run_temporal_out_norm_rms` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_run_temporal_out_norm_scale(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_run_temporal_out_norm_scale
        todo!(
            "TODO: port action `effect_run_temporal_out_norm_scale` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_select_depformer_sampling_token(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_select_depformer_sampling_token
        todo!(
            "TODO: port action `effect_select_depformer_sampling_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_select_depformer_token(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_select_depformer_token
        todo!(
            "TODO: port action `effect_select_depformer_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_select_text_sampling_token(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_select_text_sampling_token
        todo!(
            "TODO: port action `effect_select_text_sampling_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_select_text_token(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_select_text_token
        todo!(
            "TODO: port action `effect_select_text_token` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_skip_zero_input_audio_embedding(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_skip_zero_input_audio_embedding
        todo!(
            "TODO: port action `effect_skip_zero_input_audio_embedding` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_failed_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_store_error_out_step_run_from_state_step_error_out_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_store_error_out_step_run_from_state_uninit_step_error_out_decision(
        &mut self,
        _event: &StepRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_use_depformer_codebook_weight(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_use_depformer_codebook_weight
        todo!(
            "TODO: port action `effect_use_depformer_codebook_weight` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_use_depformer_scheduled_weight(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_use_depformer_scheduled_weight
        todo!(
            "TODO: port action `effect_use_depformer_scheduled_weight` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_write_depformer_layer_kv_cache(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_write_depformer_layer_kv_cache
        todo!(
            "TODO: port action `effect_write_depformer_layer_kv_cache` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn effect_write_temporal_layer_kv_cache(&mut self, _event: &StepRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp::effect_write_temporal_layer_kv_cache
        todo!(
            "TODO: port action `effect_write_temporal_layer_kv_cache` from emel.cpp/src/emel/speech/predictor/moshi/executor/actions.hpp"
        )
    }
    fn guard_audio_sampling_top_k_exceeds_card(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_audio_sampling_top_k_exceeds_card
        todo!(
            "TODO: port guard `guard_audio_sampling_top_k_exceeds_card` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_audio_sampling_top_k_within_card(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_audio_sampling_top_k_within_card
        todo!(
            "TODO: port guard `guard_audio_sampling_top_k_within_card` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_bind_contract_invalid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_bind_contract_invalid
        todo!(
            "TODO: port guard `guard_bind_contract_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_bind_contract_valid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_bind_contract_valid
        todo!(
            "TODO: port guard `guard_bind_contract_valid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_bound_root_operands_supported(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_bound_root_operands_supported
        todo!(
            "TODO: port guard `guard_bound_root_operands_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_bound_root_operands_unsupported(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_bound_root_operands_unsupported
        todo!(
            "TODO: port guard `guard_bound_root_operands_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_current_input_audio_token_present(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_current_input_audio_token_present
        todo!(
            "TODO: port guard `guard_current_input_audio_token_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_current_input_audio_token_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_current_input_audio_token_unsupported
        todo!(
            "TODO: port guard `guard_current_input_audio_token_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_current_input_audio_token_zero(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_current_input_audio_token_zero
        todo!(
            "TODO: port guard `guard_current_input_audio_token_zero` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_audio_input_projection_succeeded(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_audio_input_projection_succeeded
        todo!(
            "TODO: port guard `guard_depformer_audio_input_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_audio_input_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_audio_input_supported
        todo!(
            "TODO: port guard `guard_depformer_audio_input_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_codebooks_complete(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_codebooks_complete
        todo!(
            "TODO: port guard `guard_depformer_codebooks_complete` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_fused_projection_layout_supported(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_fused_projection_layout_supported
        todo!(
            "TODO: port guard `guard_depformer_fused_projection_layout_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_failed
        todo!(
            "TODO: port guard `guard_depformer_input_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_projection_bind_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_projection_bind_failed
        todo!(
            "TODO: port guard `guard_depformer_input_projection_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_projection_bound(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_projection_bound
        todo!(
            "TODO: port guard `guard_depformer_input_projection_bound` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_projection_embedding_unsupported(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_projection_embedding_unsupported
        todo!(
            "TODO: port guard `guard_depformer_input_projection_embedding_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_projection_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_projection_failed
        todo!(
            "TODO: port guard `guard_depformer_input_projection_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_succeeded
        todo!(
            "TODO: port guard `guard_depformer_input_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_input_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_input_unsupported
        todo!(
            "TODO: port guard `guard_depformer_input_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_kv_bind_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_kv_bind_failed
        todo!(
            "TODO: port guard `guard_depformer_kv_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_kv_binding_missing(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_kv_binding_missing
        todo!(
            "TODO: port guard `guard_depformer_kv_binding_missing` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_kv_binding_present(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_kv_binding_present
        todo!(
            "TODO: port guard `guard_depformer_kv_binding_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_kv_bound(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_kv_bound
        todo!(
            "TODO: port guard `guard_depformer_kv_bound` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_attention_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_attention_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_attention_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_attention_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_attention_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_attention_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_attention_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_attention_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_attention_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_attention_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_attention_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_attention_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_cache_write_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_cache_write_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_cache_write_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_cache_write_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_cache_write_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_cache_write_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_cache_write_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_cache_write_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_cache_write_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_cache_write_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_cache_write_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_cache_write_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_ff_residual_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_ff_residual_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_ff_residual_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_ff_residual_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_ff_residual_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_ff_residual_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_in_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_in_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_in_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_in_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_in_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_in_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_in_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_in_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_in_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_in_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_in_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_in_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_out_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_out_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_out_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_out_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_out_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_out_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_out_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_out_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_out_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_gating_out_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_gating_out_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_gating_out_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_rms_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_rms_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_rms_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_rms_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_rms_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_rms_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm2_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm2_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_norm2_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_rms_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_rms_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_rms_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_rms_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_rms_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_rms_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_norm_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_norm_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_norm_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_out_projection_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_out_projection_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_out_projection_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_out_projection_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_out_projection_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_out_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_out_projection_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_out_projection_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_out_projection_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_out_projection_unsupported(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_out_projection_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_out_projection_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_projection_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_projection_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_projection_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_projection_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_projection_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_projection_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_projection_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_projection_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_projection_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_projection_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_projection_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_residual_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_residual_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_residual_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_residual_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_residual_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_residual_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_silu_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_silu_failed
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_silu_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_silu_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_silu_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_silu_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_succeeded
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_supported
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_layer_silu_gate_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_layer_silu_gate_unsupported
        todo!(
            "TODO: port guard `guard_depformer_layer_silu_gate_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_failed
        todo!(
            "TODO: port guard `guard_depformer_logits_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_matmul_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_matmul_failed
        todo!(
            "TODO: port guard `guard_depformer_logits_matmul_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_matmul_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_matmul_succeeded
        todo!(
            "TODO: port guard `guard_depformer_logits_matmul_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_projection_bound_argmax(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_projection_bound_argmax
        todo!(
            "TODO: port guard `guard_depformer_logits_projection_bound_argmax` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_projection_bound_sampling(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_projection_bound_sampling
        todo!(
            "TODO: port guard `guard_depformer_logits_projection_bound_sampling` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_projection_bound_sampling_invalid(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_projection_bound_sampling_invalid
        todo!(
            "TODO: port guard `guard_depformer_logits_projection_bound_sampling_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_succeeded
        todo!(
            "TODO: port guard `guard_depformer_logits_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_supported
        todo!(
            "TODO: port guard `guard_depformer_logits_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_logits_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_logits_unsupported
        todo!(
            "TODO: port guard `guard_depformer_logits_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_position_advance_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_position_advance_failed
        todo!(
            "TODO: port guard `guard_depformer_position_advance_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_position_advance_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_position_advance_succeeded
        todo!(
            "TODO: port guard `guard_depformer_position_advance_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_position_reset_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_position_reset_failed
        todo!(
            "TODO: port guard `guard_depformer_position_reset_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_position_reset_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_position_reset_succeeded
        todo!(
            "TODO: port guard `guard_depformer_position_reset_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_projection_layout_unsupported(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_projection_layout_unsupported
        todo!(
            "TODO: port guard `guard_depformer_projection_layout_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_sampling_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_sampling_failed
        todo!(
            "TODO: port guard `guard_depformer_sampling_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_sampling_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_sampling_succeeded
        todo!(
            "TODO: port guard `guard_depformer_sampling_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_scheduled_weight_absent(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_scheduled_weight_absent
        todo!(
            "TODO: port guard `guard_depformer_scheduled_weight_absent` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_scheduled_weight_invalid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_scheduled_weight_invalid
        todo!(
            "TODO: port guard `guard_depformer_scheduled_weight_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_scheduled_weight_present(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_scheduled_weight_present
        todo!(
            "TODO: port guard `guard_depformer_scheduled_weight_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_split_projection_layout_supported(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_split_projection_layout_supported
        todo!(
            "TODO: port guard `guard_depformer_split_projection_layout_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_text_input_projection_succeeded(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_text_input_projection_succeeded
        todo!(
            "TODO: port guard `guard_depformer_text_input_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_depformer_text_input_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_depformer_text_input_supported
        todo!(
            "TODO: port guard `guard_depformer_text_input_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_embedding_row_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_embedding_row_failed
        todo!(
            "TODO: port guard `guard_embedding_row_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_embedding_row_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_embedding_row_succeeded
        todo!(
            "TODO: port guard `guard_embedding_row_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_embedding_view_bind_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_embedding_view_bind_failed
        todo!(
            "TODO: port guard `guard_embedding_view_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_embedding_view_bound(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_embedding_view_bound
        todo!(
            "TODO: port guard `guard_embedding_view_bound` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_external_input_embedding_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_external_input_embedding_supported
        todo!(
            "TODO: port guard `guard_external_input_embedding_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_forced_text_sampling_config_invalid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_forced_text_sampling_config_invalid
        todo!(
            "TODO: port guard `guard_forced_text_sampling_config_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_forced_text_sampling_consumed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_forced_text_sampling_consumed
        todo!(
            "TODO: port guard `guard_forced_text_sampling_consumed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_forced_text_token_invalid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_forced_text_token_invalid
        todo!(
            "TODO: port guard `guard_forced_text_token_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_forced_text_token_valid_and_sampling_consume(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_forced_text_token_valid_and_sampling_consume
        todo!(
            "TODO: port guard `guard_forced_text_token_valid_and_sampling_consume` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_forced_text_token_valid_without_sampling_consume(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_forced_text_token_valid_without_sampling_consume
        todo!(
            "TODO: port guard `guard_forced_text_token_valid_without_sampling_consume` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_full_graph_phase(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_full_graph_phase
        todo!(
            "TODO: port guard `guard_full_graph_phase` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_has_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_has_error_out_step_run(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_has_initialize_done_callback(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_has_initialize_done_callback
        todo!(
            "TODO: port guard `guard_has_initialize_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_has_initialize_error_callback(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_has_initialize_error_callback
        todo!(
            "TODO: port guard `guard_has_initialize_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_audio_codebooks_complete(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_audio_codebooks_complete
        todo!(
            "TODO: port guard `guard_input_audio_codebooks_complete` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_audio_embedding_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_audio_embedding_failed
        todo!(
            "TODO: port guard `guard_input_audio_embedding_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_audio_embedding_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_audio_embedding_succeeded
        todo!(
            "TODO: port guard `guard_input_audio_embedding_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_embedding_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_embedding_failed
        todo!(
            "TODO: port guard `guard_input_embedding_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_embedding_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_embedding_supported
        todo!(
            "TODO: port guard `guard_input_embedding_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_embedding_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_embedding_unsupported
        todo!(
            "TODO: port guard `guard_input_embedding_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_text_embedding_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_text_embedding_failed
        todo!(
            "TODO: port guard `guard_input_text_embedding_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_input_text_embedding_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_input_text_embedding_succeeded
        todo!(
            "TODO: port guard `guard_input_text_embedding_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_more_depformer_codebooks(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_more_depformer_codebooks
        todo!(
            "TODO: port guard `guard_more_depformer_codebooks` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_more_depformer_layers(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_more_depformer_layers
        todo!(
            "TODO: port guard `guard_more_depformer_layers` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_more_input_audio_codebooks(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_more_input_audio_codebooks
        todo!(
            "TODO: port guard `guard_more_input_audio_codebooks` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_more_temporal_layers(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_more_temporal_layers
        todo!(
            "TODO: port guard `guard_more_temporal_layers` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_no_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_no_error_out_step_run(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_no_initialize_done_callback(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_no_initialize_done_callback
        todo!(
            "TODO: port guard `guard_no_initialize_done_callback` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_no_initialize_error_callback(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_no_initialize_error_callback
        todo!(
            "TODO: port guard `guard_no_initialize_error_callback` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_prediction_graph_phase(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_prediction_graph_phase
        todo!(
            "TODO: port guard `guard_prediction_graph_phase` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_projection_view_bind_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_projection_view_bind_failed
        todo!(
            "TODO: port guard `guard_projection_view_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_projection_view_bound(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_projection_view_bound
        todo!(
            "TODO: port guard `guard_projection_view_bound` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_depformer_positions_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_depformer_positions_failed
        todo!(
            "TODO: port guard `guard_reset_depformer_positions_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_depformer_positions_missing(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_depformer_positions_missing
        todo!(
            "TODO: port guard `guard_reset_depformer_positions_missing` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_depformer_positions_present(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_depformer_positions_present
        todo!(
            "TODO: port guard `guard_reset_depformer_positions_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_depformer_positions_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_depformer_positions_succeeded
        todo!(
            "TODO: port guard `guard_reset_depformer_positions_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_temporal_positions_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_temporal_positions_failed
        todo!(
            "TODO: port guard `guard_reset_temporal_positions_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_temporal_positions_missing(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_temporal_positions_missing
        todo!(
            "TODO: port guard `guard_reset_temporal_positions_missing` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_temporal_positions_present(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_temporal_positions_present
        todo!(
            "TODO: port guard `guard_reset_temporal_positions_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_reset_temporal_positions_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_reset_temporal_positions_succeeded
        todo!(
            "TODO: port guard `guard_reset_temporal_positions_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_sampled_text_token_ready(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_sampled_text_token_ready
        todo!(
            "TODO: port guard `guard_sampled_text_token_ready` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_sampling_seed_nonzero(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_sampling_seed_nonzero
        todo!(
            "TODO: port guard `guard_sampling_seed_nonzero` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_sampling_seed_zero(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_sampling_seed_zero
        todo!(
            "TODO: port guard `guard_sampling_seed_zero` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_sampling_step_invalid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_sampling_step_invalid
        todo!(
            "TODO: port guard `guard_sampling_step_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_sampling_step_valid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_sampling_step_valid
        todo!(
            "TODO: port guard `guard_sampling_step_valid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_step_model_matches(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_step_model_matches
        todo!(
            "TODO: port guard `guard_step_model_matches` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_step_model_mismatch(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_step_model_mismatch
        todo!(
            "TODO: port guard `guard_step_model_mismatch` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_step_shape_invalid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_step_shape_invalid
        todo!(
            "TODO: port guard `guard_step_shape_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_step_shape_valid(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_step_shape_valid
        todo!(
            "TODO: port guard `guard_step_shape_valid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_fused_projection_layout_supported(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_fused_projection_layout_supported
        todo!(
            "TODO: port guard `guard_temporal_fused_projection_layout_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_kv_bind_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_kv_bind_failed
        todo!(
            "TODO: port guard `guard_temporal_kv_bind_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_kv_binding_missing(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_kv_binding_missing
        todo!(
            "TODO: port guard `guard_temporal_kv_binding_missing` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_kv_binding_present(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_kv_binding_present
        todo!(
            "TODO: port guard `guard_temporal_kv_binding_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_kv_bound(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_kv_bound
        todo!(
            "TODO: port guard `guard_temporal_kv_bound` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_attention_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_attention_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_attention_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_attention_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_attention_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_attention_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_attention_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_attention_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_attention_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_attention_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_attention_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_attention_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_cache_write_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_cache_write_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_cache_write_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_cache_write_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_cache_write_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_cache_write_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_cache_write_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_cache_write_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_cache_write_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_cache_write_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_cache_write_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_cache_write_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_ff_residual_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_ff_residual_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_ff_residual_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_ff_residual_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_ff_residual_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_ff_residual_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_in_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_in_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_in_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_in_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_in_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_in_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_in_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_in_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_in_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_in_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_in_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_in_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_out_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_out_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_out_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_out_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_out_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_out_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_out_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_out_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_out_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_gating_out_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_gating_out_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_gating_out_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_rms_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_rms_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_rms_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_rms_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_rms_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_rms_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm2_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm2_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_norm2_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_rms_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_rms_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_rms_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_rms_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_rms_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_rms_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_norm_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_norm_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_norm_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_out_projection_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_out_projection_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_out_projection_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_out_projection_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_out_projection_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_out_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_out_projection_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_out_projection_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_out_projection_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_out_projection_unsupported(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_out_projection_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_out_projection_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_projection_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_projection_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_projection_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_projection_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_projection_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_projection_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_projection_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_projection_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_projection_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_projection_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_projection_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_projection_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_residual_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_residual_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_residual_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_residual_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_residual_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_residual_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_rope_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_rope_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_rope_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_rope_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_rope_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_rope_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_rope_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_rope_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_rope_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_rope_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_rope_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_rope_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_silu_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_silu_failed
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_silu_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_silu_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_silu_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_silu_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_succeeded
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_supported
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layer_silu_gate_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layer_silu_gate_unsupported
        todo!(
            "TODO: port guard `guard_temporal_layer_silu_gate_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_layers_complete(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_layers_complete
        todo!(
            "TODO: port guard `guard_temporal_layers_complete` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_out_norm_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_out_norm_failed
        todo!(
            "TODO: port guard `guard_temporal_out_norm_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_out_norm_rms_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_out_norm_rms_failed
        todo!(
            "TODO: port guard `guard_temporal_out_norm_rms_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_out_norm_rms_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_out_norm_rms_succeeded
        todo!(
            "TODO: port guard `guard_temporal_out_norm_rms_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_out_norm_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_out_norm_succeeded
        todo!(
            "TODO: port guard `guard_temporal_out_norm_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_out_norm_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_out_norm_unsupported
        todo!(
            "TODO: port guard `guard_temporal_out_norm_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_position_advance_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_position_advance_failed
        todo!(
            "TODO: port guard `guard_temporal_position_advance_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_position_advance_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_position_advance_succeeded
        todo!(
            "TODO: port guard `guard_temporal_position_advance_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_projection_layout_unsupported(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_projection_layout_unsupported
        todo!(
            "TODO: port guard `guard_temporal_projection_layout_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_temporal_split_projection_layout_supported(
        &self,
        _event: &InitRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_temporal_split_projection_layout_supported
        todo!(
            "TODO: port guard `guard_temporal_split_projection_layout_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_failed
        todo!(
            "TODO: port guard `guard_text_logits_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_matmul_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_matmul_failed
        todo!(
            "TODO: port guard `guard_text_logits_matmul_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_matmul_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_matmul_succeeded
        todo!(
            "TODO: port guard `guard_text_logits_matmul_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_projection_bound_and_no_forced_token_argmax(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_projection_bound_and_no_forced_token_argmax
        todo!(
            "TODO: port guard `guard_text_logits_projection_bound_and_no_forced_token_argmax` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_projection_bound_and_no_forced_token_sampling(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_projection_bound_and_no_forced_token_sampling
        todo!(
            "TODO: port guard `guard_text_logits_projection_bound_and_no_forced_token_sampling` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid(
        &self,
        _event: &StepRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid
        todo!(
            "TODO: port guard `guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_succeeded(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_succeeded
        todo!(
            "TODO: port guard `guard_text_logits_succeeded` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_supported
        todo!(
            "TODO: port guard `guard_text_logits_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_logits_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_logits_unsupported
        todo!(
            "TODO: port guard `guard_text_logits_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_sampling_failed(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_sampling_failed
        todo!(
            "TODO: port guard `guard_text_sampling_failed` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_sampling_top_k_exceeds_card(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_sampling_top_k_exceeds_card
        todo!(
            "TODO: port guard `guard_text_sampling_top_k_exceeds_card` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_text_sampling_top_k_within_card(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_text_sampling_top_k_within_card
        todo!(
            "TODO: port guard `guard_text_sampling_top_k_within_card` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_token_input_embedding_supported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_token_input_embedding_supported
        todo!(
            "TODO: port guard `guard_token_input_embedding_supported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_token_input_embedding_unsupported(&self, _event: &StepRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_token_input_embedding_unsupported
        todo!(
            "TODO: port guard `guard_token_input_embedding_unsupported` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_absent(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_unexpected_error_out_absent
        todo!(
            "TODO: port guard `guard_unexpected_error_out_absent` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_present(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp::guard_unexpected_error_out_present
        todo!(
            "TODO: port guard `guard_unexpected_error_out_present` from emel.cpp/src/emel/speech/predictor/moshi/executor/guards.hpp"
        )
    }
}
