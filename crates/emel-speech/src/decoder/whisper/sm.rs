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

// --- machine SpeechDecoderWhisper from emel.cpp/src/emel/speech/decoder/whisper/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventDecodeRun;

sml! {
    SpeechDecoderWhisper {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<EventDecodeRun> / effect_begin_decode,
        "state_encoder_state_decision"_s <= "state_model_contract_decision"_s + completion<EventDecodeRun> [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventDecodeRun> [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_decode_policy_decision"_s <= "state_encoder_state_decision"_s + completion<EventDecodeRun> [guard_encoder_state_valid],
        "state_error_error_out_decision"_s <= "state_encoder_state_decision"_s + completion<EventDecodeRun> [guard_encoder_state_invalid] / effect_mark_encoder_state_invalid,
        "state_generated_token_capacity_decision"_s <= "state_decode_policy_decision"_s + completion<EventDecodeRun> [guard_decode_policy_supported],
        "state_error_error_out_decision"_s <= "state_decode_policy_decision"_s + completion<EventDecodeRun> [guard_decode_policy_unsupported] / effect_mark_decode_policy_invalid,
        "state_logits_capacity_decision"_s <= "state_generated_token_capacity_decision"_s + completion<EventDecodeRun> [guard_generated_token_capacity_valid],
        "state_error_error_out_decision"_s <= "state_generated_token_capacity_decision"_s + completion<EventDecodeRun> [guard_generated_token_capacity_invalid] / effect_mark_generated_token_capacity_invalid,
        "state_workspace_capacity_decision"_s <= "state_logits_capacity_decision"_s + completion<EventDecodeRun> [guard_logits_capacity_valid],
        "state_error_error_out_decision"_s <= "state_logits_capacity_decision"_s + completion<EventDecodeRun> [guard_logits_capacity_invalid] / effect_mark_logits_capacity_invalid,
        "state_variant_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventDecodeRun> [guard_workspace_capacity_valid],
        "state_error_error_out_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventDecodeRun> [guard_workspace_capacity_invalid] / effect_mark_workspace_capacity_invalid,
        "state_running_q8_0_f32_aux"_s <= "state_variant_decision"_s + completion<EventDecodeRun> [guard_q8_0_f32_aux_variant] / effect_run_decoder_q8_0_f32_aux,
        "state_running_q8_0"_s <= "state_variant_decision"_s + completion<EventDecodeRun> [guard_q8_0_variant] / effect_run_decoder_q8_0,
        "state_running_q4_0"_s <= "state_variant_decision"_s + completion<EventDecodeRun> [guard_q4_0_variant] / effect_run_decoder_q4_0,
        "state_running_q4_1"_s <= "state_variant_decision"_s + completion<EventDecodeRun> [guard_q4_1_variant] / effect_run_decoder_q4_1,
        "state_error_error_out_decision"_s <= "state_variant_decision"_s + completion<EventDecodeRun> [guard_unsupported_variant] / effect_mark_unsupported_variant,
        "state_success_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<EventDecodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q8_0"_s + completion<EventDecodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q4_0"_s + completion<EventDecodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q4_1"_s + completion<EventDecodeRun>,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun> [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun> [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun> [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun> [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun> [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun> [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun> [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun> [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDecodeRun>,
        "state_ready"_s <= "state_errored"_s + completion<EventDecodeRun>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_encoder_state_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encoder_state_decision,
        "state_ready"_s <= "state_decode_policy_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_policy_decision,
        "state_ready"_s <= "state_generated_token_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_generated_token_capacity_decision,
        "state_ready"_s <= "state_logits_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_logits_capacity_decision,
        "state_ready"_s <= "state_workspace_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_workspace_capacity_decision,
        "state_ready"_s <= "state_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_variant_decision,
        "state_ready"_s <= "state_running_q8_0"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q8_0,
        "state_ready"_s <= "state_running_q8_0_f32_aux"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q8_0_f32_aux,
        "state_ready"_s <= "state_running_q4_0"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q4_0,
        "state_ready"_s <= "state_running_q4_1"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q4_1,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Context for `SpeechDecoderWhisper` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechDecoderWhisperContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechDecoderWhisperStateMachineContext for SpeechDecoderWhisperContext {
    fn effect_begin_decode(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_begin_decode
        todo!(
            "TODO: port action `effect_begin_decode` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_emit_done(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_emit_done
        todo!(
            "TODO: port action `effect_emit_done` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_emit_error(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_emit_error
        todo!(
            "TODO: port action `effect_emit_error` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_decode_policy_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_decode_policy_invalid
        todo!(
            "TODO: port action `effect_mark_decode_policy_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_encoder_state_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_encoder_state_invalid
        todo!(
            "TODO: port action `effect_mark_encoder_state_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_generated_token_capacity_invalid(
        &mut self,
        _event: &EventDecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_generated_token_capacity_invalid
        todo!(
            "TODO: port action `effect_mark_generated_token_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_logits_capacity_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_logits_capacity_invalid
        todo!(
            "TODO: port action `effect_mark_logits_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_model_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_model_invalid
        todo!(
            "TODO: port action `effect_mark_model_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_unsupported_variant(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_unsupported_variant
        todo!(
            "TODO: port action `effect_mark_unsupported_variant` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_mark_workspace_capacity_invalid(
        &mut self,
        _event: &EventDecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_mark_workspace_capacity_invalid
        todo!(
            "TODO: port action `effect_mark_workspace_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_policy_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encoder_state_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_generated_token_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_logits_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_running_q4_0(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_running_q4_1(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_running_q8_0(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_running_q8_0_f32_aux(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_variant_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_workspace_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_run_decoder_q4_0(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_run_decoder_q4_0
        todo!(
            "TODO: port action `effect_run_decoder_q4_0` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_run_decoder_q4_1(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_run_decoder_q4_1
        todo!(
            "TODO: port action `effect_run_decoder_q4_1` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_run_decoder_q8_0(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_run_decoder_q8_0
        todo!(
            "TODO: port action `effect_run_decoder_q8_0` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_run_decoder_q8_0_f32_aux(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_run_decoder_q8_0_f32_aux
        todo!(
            "TODO: port action `effect_run_decoder_q8_0_f32_aux` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_store_error_error(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_store_error_error
        todo!(
            "TODO: port action `effect_store_error_error` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn effect_store_success_error(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp::effect_store_success_error
        todo!(
            "TODO: port action `effect_store_success_error` from emel.cpp/src/emel/speech/decoder/whisper/actions.hpp"
        )
    }
    fn guard_decode_policy_supported(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_decode_policy_supported
        todo!(
            "TODO: port guard `guard_decode_policy_supported` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_decode_policy_unsupported(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_decode_policy_unsupported
        todo!(
            "TODO: port guard `guard_decode_policy_unsupported` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_encoder_state_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_encoder_state_invalid
        todo!(
            "TODO: port guard `guard_encoder_state_invalid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_encoder_state_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_encoder_state_valid
        todo!(
            "TODO: port guard `guard_encoder_state_valid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_generated_token_capacity_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_generated_token_capacity_invalid
        todo!(
            "TODO: port guard `guard_generated_token_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_generated_token_capacity_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_generated_token_capacity_valid
        todo!(
            "TODO: port guard `guard_generated_token_capacity_valid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_has_done_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_has_error_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_has_error_out(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_logits_capacity_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_logits_capacity_invalid
        todo!(
            "TODO: port guard `guard_logits_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_logits_capacity_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_logits_capacity_valid
        todo!(
            "TODO: port guard `guard_logits_capacity_valid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_model_contract_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_model_contract_invalid
        todo!(
            "TODO: port guard `guard_model_contract_invalid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_model_contract_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_model_contract_valid
        todo!(
            "TODO: port guard `guard_model_contract_valid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_no_done_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_no_error_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_no_error_out(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_q4_0_variant(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_q4_0_variant
        todo!(
            "TODO: port guard `guard_q4_0_variant` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_q4_1_variant(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_q4_1_variant
        todo!(
            "TODO: port guard `guard_q4_1_variant` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_q8_0_f32_aux_variant(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_q8_0_f32_aux_variant
        todo!(
            "TODO: port guard `guard_q8_0_f32_aux_variant` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_q8_0_variant(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_q8_0_variant
        todo!(
            "TODO: port guard `guard_q8_0_variant` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_unsupported_variant(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_unsupported_variant
        todo!(
            "TODO: port guard `guard_unsupported_variant` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_workspace_capacity_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_workspace_capacity_invalid
        todo!(
            "TODO: port guard `guard_workspace_capacity_invalid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
    fn guard_workspace_capacity_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp::guard_workspace_capacity_valid
        todo!(
            "TODO: port guard `guard_workspace_capacity_valid` from emel.cpp/src/emel/speech/decoder/whisper/guards.hpp"
        )
    }
}
