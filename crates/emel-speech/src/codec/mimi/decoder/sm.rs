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

// --- machine SpeechCodecMimiDecoder from emel.cpp/src/emel/speech/codec/mimi/decoder/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventDecodeRun;

sml! {
    SpeechCodecMimiDecoder {
        "state_runtime_decision"_s <= *"state_ready"_s + event<EventDecodeRun>,
        "state_shape_decision"_s <= "state_runtime_decision"_s + completion<EventDecodeRun> [guard_runtime_bound],
        "state_error_error_out_decision"_s <= "state_runtime_decision"_s + completion<EventDecodeRun> [guard_runtime_unbound] / effect_mark_runtime_unbound,
        "state_capacity_decision"_s <= "state_shape_decision"_s + completion<EventDecodeRun> [guard_request_shape_valid],
        "state_error_error_out_decision"_s <= "state_shape_decision"_s + completion<EventDecodeRun> [guard_request_shape_invalid] / effect_mark_request_shape_invalid,
        "state_upsample_running"_s <= "state_capacity_decision"_s + completion<EventDecodeRun> [guard_buffer_capacity_valid] / effect_run_upsample,
        "state_error_error_out_decision"_s <= "state_capacity_decision"_s + completion<EventDecodeRun> [guard_buffer_capacity_invalid] / effect_mark_buffer_capacity_invalid,
        "state_transformer_variant_decision"_s <= "state_upsample_running"_s + completion<EventDecodeRun>,
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventDecodeRun> [guard_proj_f32] / effect_run_transformer_false,
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventDecodeRun> [guard_proj_q8] / effect_run_transformer_true,
        "state_backend_variant_decision"_s <= "state_transformer_running"_s + completion<EventDecodeRun>,
        "state_backend_running"_s <= "state_backend_variant_decision"_s + completion<EventDecodeRun> [guard_conv_f32] / effect_run_backend_false,
        "state_backend_running"_s <= "state_backend_variant_decision"_s + completion<EventDecodeRun> [guard_conv_f16] / effect_run_backend_true,
        "state_success_error_out_decision"_s <= "state_backend_running"_s + completion<EventDecodeRun>,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun> [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun> [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun> [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun> [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun> [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun> [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun> [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun> [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDecodeRun>,
        "state_ready"_s <= "state_errored"_s + completion<EventDecodeRun>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_runtime_decision,
        "state_ready"_s <= "state_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_shape_decision,
        "state_ready"_s <= "state_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_capacity_decision,
        "state_ready"_s <= "state_upsample_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_upsample_running,
        "state_ready"_s <= "state_transformer_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_variant_decision,
        "state_ready"_s <= "state_transformer_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_running,
        "state_ready"_s <= "state_backend_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_backend_variant_decision,
        "state_ready"_s <= "state_backend_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_backend_running,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Context for `SpeechCodecMimiDecoder` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechCodecMimiDecoderContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechCodecMimiDecoderStateMachineContext for SpeechCodecMimiDecoderContext {
    fn effect_emit_done(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_emit_done
        todo!(
            "TODO: port action `effect_emit_done` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_emit_error(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_emit_error
        todo!(
            "TODO: port action `effect_emit_error` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_mark_buffer_capacity_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_mark_buffer_capacity_invalid
        todo!(
            "TODO: port action `effect_mark_buffer_capacity_invalid` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_mark_request_shape_invalid(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_mark_request_shape_invalid
        todo!(
            "TODO: port action `effect_mark_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_mark_runtime_unbound(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_mark_runtime_unbound
        todo!(
            "TODO: port action `effect_mark_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_backend_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_backend_variant_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_runtime_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_shape_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_transformer_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_transformer_variant_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_upsample_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_run_backend_false(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_run_backend
        todo!(
            "TODO: port action `effect_run_backend` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_run_backend_true(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_run_backend
        todo!(
            "TODO: port action `effect_run_backend` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_run_transformer_false(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_run_transformer
        todo!(
            "TODO: port action `effect_run_transformer` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_run_transformer_true(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_run_transformer
        todo!(
            "TODO: port action `effect_run_transformer` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_run_upsample(&mut self, _event: &EventDecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_run_upsample
        todo!(
            "TODO: port action `effect_run_upsample` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_store_error_out_from_state_error_error_out_decision(
        &mut self,
        _event: &EventDecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn effect_store_error_out_from_state_success_error_out_decision(
        &mut self,
        _event: &EventDecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/decoder/actions.hpp"
        )
    }
    fn guard_buffer_capacity_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_buffer_capacity_invalid
        todo!(
            "TODO: port guard `guard_buffer_capacity_invalid` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_buffer_capacity_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_buffer_capacity_valid
        todo!(
            "TODO: port guard `guard_buffer_capacity_valid` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_conv_f16(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_conv_f16
        todo!(
            "TODO: port guard `guard_conv_f16` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_conv_f32(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_conv_f32
        todo!(
            "TODO: port guard `guard_conv_f32` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_has_done_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_has_error_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_has_error_out(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_no_done_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_no_error_callback(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_no_error_out(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_proj_f32(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_proj_f32
        todo!(
            "TODO: port guard `guard_proj_f32` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_proj_q8(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_proj_q8
        todo!(
            "TODO: port guard `guard_proj_q8` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_request_shape_invalid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_request_shape_invalid
        todo!(
            "TODO: port guard `guard_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_request_shape_valid(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_request_shape_valid
        todo!(
            "TODO: port guard `guard_request_shape_valid` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_runtime_bound(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_runtime_bound
        todo!(
            "TODO: port guard `guard_runtime_bound` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
    fn guard_runtime_unbound(&self, _event: &EventDecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp::guard_runtime_unbound
        todo!(
            "TODO: port guard `guard_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/decoder/guards.hpp"
        )
    }
}
