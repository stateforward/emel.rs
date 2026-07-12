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

// --- machine SpeechCodecMimiQuantizer from emel.cpp/src/emel/speech/codec/mimi/quantizer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DecodeRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EncodeRun;

sml! {
    SpeechCodecMimiQuantizer {
        "state_encode_runtime_decision"_s <= *"state_ready"_s + event<EncodeRun>,
        "state_encode_shape_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun> [guard_runtime_bound_encode_run],
        "state_encode_error_error_out_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun> [guard_runtime_unbound_encode_run] / effect_mark_runtime_unbound_encode_run,
        "state_encode_variant_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun> [guard_encode_shape_valid],
        "state_encode_error_error_out_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun> [guard_encode_shape_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun> [guard_class_f32_encode_run] / effect_run_quantize_false_false,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun> [guard_class_f16_encode_run] / effect_run_quantize_true_false,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun> [guard_class_q8_encode_run] / effect_run_quantize_true_true,
        "state_encode_success_error_out_decision"_s <= "state_encode_running"_s + completion<EncodeRun>,
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun> [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_success_error_out_decision,
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun> [guard_no_error_out_encode_run],
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun> [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_error_error_out_decision,
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun> [guard_no_error_out_encode_run],
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun> [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun> [guard_no_done_callback_encode_run],
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun> [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun> [guard_no_error_callback_encode_run],
        "state_ready"_s <= "state_encode_done"_s + completion<EncodeRun>,
        "state_ready"_s <= "state_encode_errored"_s + completion<EncodeRun>,
        "state_decode_runtime_decision"_s <= "state_ready"_s + event<DecodeRun>,
        "state_decode_shape_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun> [guard_runtime_bound_decode_run],
        "state_decode_error_error_out_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun> [guard_runtime_unbound_decode_run] / effect_mark_runtime_unbound_decode_run,
        "state_decode_codes_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun> [guard_decode_shape_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun> [guard_decode_shape_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_decode_variant_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun> [guard_decode_codes_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun> [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun> [guard_class_f32_decode_run] / effect_run_dequantize_false_false,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun> [guard_class_f16_decode_run] / effect_run_dequantize_true_false,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun> [guard_class_q8_decode_run] / effect_run_dequantize_true_true,
        "state_decode_success_error_out_decision"_s <= "state_decode_running"_s + completion<DecodeRun>,
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun> [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_success_error_out_decision,
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun> [guard_no_error_out_decode_run],
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun> [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_error_error_out_decision,
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun> [guard_no_error_out_decode_run],
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun> [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun> [guard_no_done_callback_decode_run],
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun> [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun> [guard_no_error_callback_decode_run],
        "state_ready"_s <= "state_decode_done"_s + completion<DecodeRun>,
        "state_ready"_s <= "state_decode_errored"_s + completion<DecodeRun>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_encode_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_runtime_decision,
        "state_ready"_s <= "state_encode_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_shape_decision,
        "state_ready"_s <= "state_encode_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_variant_decision,
        "state_ready"_s <= "state_encode_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_running,
        "state_ready"_s <= "state_encode_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_success_error_out_decision,
        "state_ready"_s <= "state_encode_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_success_callback_decision,
        "state_ready"_s <= "state_encode_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_error_error_out_decision,
        "state_ready"_s <= "state_encode_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_error_callback_decision,
        "state_ready"_s <= "state_encode_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_done,
        "state_ready"_s <= "state_encode_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_errored,
        "state_ready"_s <= "state_decode_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_runtime_decision,
        "state_ready"_s <= "state_decode_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_shape_decision,
        "state_ready"_s <= "state_decode_codes_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_codes_decision,
        "state_ready"_s <= "state_decode_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_variant_decision,
        "state_ready"_s <= "state_decode_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_running,
        "state_ready"_s <= "state_decode_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_success_error_out_decision,
        "state_ready"_s <= "state_decode_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_success_callback_decision,
        "state_ready"_s <= "state_decode_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_error_error_out_decision,
        "state_ready"_s <= "state_decode_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_error_callback_decision,
        "state_ready"_s <= "state_decode_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_done,
        "state_ready"_s <= "state_decode_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_errored,
    }
}

/// Context for `SpeechCodecMimiQuantizer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechCodecMimiQuantizerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechCodecMimiQuantizerStateMachineContext for SpeechCodecMimiQuantizerContext {
    fn effect_emit_decode_done(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_emit_decode_done
        todo!(
            "TODO: port action `effect_emit_decode_done` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_emit_decode_error(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_emit_decode_error
        todo!(
            "TODO: port action `effect_emit_decode_error` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_emit_encode_done(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_emit_encode_done
        todo!(
            "TODO: port action `effect_emit_encode_done` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_emit_encode_error(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_emit_encode_error
        todo!(
            "TODO: port action `effect_emit_encode_error` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_mark_code_range_invalid(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_mark_code_range_invalid
        todo!(
            "TODO: port action `effect_mark_code_range_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_mark_request_shape_invalid_decode_run(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_mark_request_shape_invalid
        todo!(
            "TODO: port action `effect_mark_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_mark_request_shape_invalid_encode_run(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_mark_request_shape_invalid
        todo!(
            "TODO: port action `effect_mark_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_mark_runtime_unbound_decode_run(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_mark_runtime_unbound
        todo!(
            "TODO: port action `effect_mark_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_mark_runtime_unbound_encode_run(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_mark_runtime_unbound
        todo!(
            "TODO: port action `effect_mark_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_codes_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_error_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_error_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_runtime_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_shape_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_success_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_success_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_decode_variant_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_error_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_error_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_runtime_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_shape_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_success_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_success_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_encode_variant_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_dequantize_false_false(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_dequantize
        todo!(
            "TODO: port action `effect_run_dequantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_dequantize_true_false(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_dequantize
        todo!(
            "TODO: port action `effect_run_dequantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_dequantize_true_true(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_dequantize
        todo!(
            "TODO: port action `effect_run_dequantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_quantize_false_false(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_quantize
        todo!(
            "TODO: port action `effect_run_quantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_quantize_true_false(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_quantize
        todo!(
            "TODO: port action `effect_run_quantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_run_quantize_true_true(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_run_quantize
        todo!(
            "TODO: port action `effect_run_quantize` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_store_error_out_decode_run_from_state_decode_error_error_out_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_store_error_out_decode_run_from_state_decode_success_error_out_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_store_error_out_encode_run_from_state_encode_error_error_out_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn effect_store_error_out_encode_run_from_state_encode_success_error_out_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/actions.hpp"
        )
    }
    fn guard_class_f16_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_f16
        todo!(
            "TODO: port guard `guard_class_f16` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_class_f16_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_f16
        todo!(
            "TODO: port guard `guard_class_f16` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_class_f32_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_f32
        todo!(
            "TODO: port guard `guard_class_f32` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_class_f32_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_f32
        todo!(
            "TODO: port guard `guard_class_f32` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_class_q8_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_q8
        todo!(
            "TODO: port guard `guard_class_q8` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_class_q8_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_class_q8
        todo!(
            "TODO: port guard `guard_class_q8` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_decode_codes_invalid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_decode_codes_invalid
        todo!(
            "TODO: port guard `guard_decode_codes_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_decode_codes_valid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_decode_codes_valid
        todo!(
            "TODO: port guard `guard_decode_codes_valid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_decode_shape_invalid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_decode_shape_invalid
        todo!(
            "TODO: port guard `guard_decode_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_decode_shape_valid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_decode_shape_valid
        todo!(
            "TODO: port guard `guard_decode_shape_valid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_encode_shape_invalid(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_encode_shape_invalid
        todo!(
            "TODO: port guard `guard_encode_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_encode_shape_valid(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_encode_shape_valid
        todo!(
            "TODO: port guard `guard_encode_shape_valid` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_done_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_done_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_error_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_error_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_error_out_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_has_error_out_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_done_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_done_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_error_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_error_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_error_out_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_no_error_out_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_runtime_bound_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_runtime_bound
        todo!(
            "TODO: port guard `guard_runtime_bound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_runtime_bound_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_runtime_bound
        todo!(
            "TODO: port guard `guard_runtime_bound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_runtime_unbound_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_runtime_unbound
        todo!(
            "TODO: port guard `guard_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
    fn guard_runtime_unbound_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp::guard_runtime_unbound
        todo!(
            "TODO: port guard `guard_runtime_unbound` from emel.cpp/src/emel/speech/codec/mimi/quantizer/guards.hpp"
        )
    }
}
