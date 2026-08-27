//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::enum_variant_names,
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

// --- machine SpeechCodecMimi from emel.cpp/src/emel/speech/codec/mimi/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DecodeRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EncodeRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventResetStreamRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct InitRun;

sml! {
    SpeechCodecMimi {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + event<InitRun>,
        "state_bind_capacity_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_valid],
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_binding"_s <= "state_bind_capacity_decision"_s + completion<InitRun> [guard_arena_capacity_valid] / effect_bind,
        "state_init_failed_error_out_decision"_s <= "state_bind_capacity_decision"_s + completion<InitRun> [guard_arena_capacity_invalid] / effect_mark_arena_capacity_invalid,
        "state_init_error_out_decision"_s <= "state_binding"_s + completion<InitRun>,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_error_out_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_failed_error_out_decision,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_has_done_callback_init_run] / effect_emit_initialize_done,
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_no_done_callback_init_run],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_has_error_callback_init_run] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_no_error_callback_init_run],
        "state_encode_request_decision"_s <= "state_session_ready"_s + event<EncodeRun>,
        "state_encoding"_s <= "state_encode_request_decision"_s + completion<EncodeRun> [guard_encode_request_valid] / effect_run_frontend_child,
        "state_encode_failed_error_out_decision"_s <= "state_encode_request_decision"_s + completion<EncodeRun> [guard_encode_request_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_quantizing"_s <= "state_encoding"_s + completion<EncodeRun> / effect_run_quantize_child,
        "state_encode_error_out_decision"_s <= "state_quantizing"_s + completion<EncodeRun>,
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EncodeRun> [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_error_out_decision,
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EncodeRun> [guard_no_error_out_encode_run],
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EncodeRun> [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_failed_error_out_decision,
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EncodeRun> [guard_no_error_out_encode_run],
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EncodeRun> [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EncodeRun> [guard_no_done_callback_encode_run],
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EncodeRun> [guard_has_error_callback_encode_run] / effect_emit_encode_error_from_state_encode_failed_callback_decision,
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EncodeRun> [guard_no_error_callback_encode_run],
        "state_decode_request_decision"_s <= "state_session_ready"_s + event<DecodeRun>,
        "state_decode_codes_decision"_s <= "state_decode_request_decision"_s + completion<DecodeRun> [guard_decode_request_valid],
        "state_decode_failed_error_out_decision"_s <= "state_decode_request_decision"_s + completion<DecodeRun> [guard_decode_request_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_dequantizing"_s <= "state_decode_codes_decision"_s + completion<DecodeRun> [guard_decode_codes_valid] / effect_run_dequantize_child,
        "state_decode_failed_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun> [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decoding"_s <= "state_dequantizing"_s + completion<DecodeRun> / effect_run_backend_child,
        "state_decode_error_out_decision"_s <= "state_decoding"_s + completion<DecodeRun>,
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<DecodeRun> [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_error_out_decision,
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<DecodeRun> [guard_no_error_out_decode_run],
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<DecodeRun> [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_failed_error_out_decision,
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<DecodeRun> [guard_no_error_out_decode_run],
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<DecodeRun> [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<DecodeRun> [guard_no_done_callback_decode_run],
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<DecodeRun> [guard_has_error_callback_decode_run] / effect_emit_decode_error_from_state_decode_failed_callback_decision,
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<DecodeRun> [guard_no_error_callback_decode_run],
        "state_session_ready"_s <= "state_session_ready"_s + event<EventResetStreamRun> / effect_reset_stream,
        "state_uninit_encode_error_out_decision"_s <= "state_uninitialized"_s + event<EncodeRun> / effect_mark_not_initialized_encode_run,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EncodeRun> [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_uninit_encode_error_out_decision,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EncodeRun> [guard_no_error_out_encode_run],
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EncodeRun> [guard_has_error_callback_encode_run] / effect_emit_encode_error_from_state_uninit_encode_callback_decision,
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EncodeRun> [guard_no_error_callback_encode_run],
        "state_uninit_decode_error_out_decision"_s <= "state_uninitialized"_s + event<DecodeRun> / effect_mark_not_initialized_decode_run,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<DecodeRun> [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_uninit_decode_error_out_decision,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<DecodeRun> [guard_no_error_out_decode_run],
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<DecodeRun> [guard_has_error_callback_decode_run] / effect_emit_decode_error_from_state_uninit_decode_callback_decision,
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<DecodeRun> [guard_no_error_callback_decode_run],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninitialized,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_session_ready,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_session_ready,
        "state_session_ready"_s <= "state_encode_request_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encode_request_decision,
        "state_session_ready"_s <= "state_encode_request_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encode_request_decision,
        "state_session_ready"_s <= "state_decode_request_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_request_decision,
        "state_session_ready"_s <= "state_decode_request_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_request_decision,
        "state_session_ready"_s <= "state_decode_codes_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_codes_decision,
        "state_session_ready"_s <= "state_decode_codes_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_codes_decision,
        "state_session_ready"_s <= "state_encoding"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encoding,
        "state_session_ready"_s <= "state_encoding"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encoding,
        "state_session_ready"_s <= "state_quantizing"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_quantizing,
        "state_session_ready"_s <= "state_quantizing"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_quantizing,
        "state_session_ready"_s <= "state_dequantizing"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_dequantizing,
        "state_session_ready"_s <= "state_dequantizing"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_dequantizing,
        "state_session_ready"_s <= "state_decoding"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decoding,
        "state_session_ready"_s <= "state_decoding"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decoding,
        "state_session_ready"_s <= "state_encode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encode_error_out_decision,
        "state_session_ready"_s <= "state_encode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encode_error_out_decision,
        "state_session_ready"_s <= "state_encode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encode_callback_decision,
        "state_session_ready"_s <= "state_encode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encode_callback_decision,
        "state_session_ready"_s <= "state_encode_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encode_failed_error_out_decision,
        "state_session_ready"_s <= "state_encode_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encode_failed_error_out_decision,
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_encode_failed_callback_decision,
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_encode_failed_callback_decision,
        "state_session_ready"_s <= "state_decode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_error_out_decision,
        "state_session_ready"_s <= "state_decode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_error_out_decision,
        "state_session_ready"_s <= "state_decode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_callback_decision,
        "state_session_ready"_s <= "state_decode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_callback_decision,
        "state_session_ready"_s <= "state_decode_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_failed_error_out_decision,
        "state_session_ready"_s <= "state_decode_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_failed_error_out_decision,
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_decode_failed_callback_decision,
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_decode_failed_callback_decision,
        "state_uninitialized"_s <= "state_bind_contract_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_bind_contract_decision,
        "state_uninitialized"_s <= "state_bind_contract_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_bind_contract_decision,
        "state_uninitialized"_s <= "state_bind_capacity_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_bind_capacity_decision,
        "state_uninitialized"_s <= "state_bind_capacity_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_bind_capacity_decision,
        "state_uninitialized"_s <= "state_binding"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_binding,
        "state_uninitialized"_s <= "state_binding"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_binding,
        "state_uninitialized"_s <= "state_init_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_init_error_out_decision,
        "state_uninitialized"_s <= "state_init_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_init_error_out_decision,
        "state_uninitialized"_s <= "state_init_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_init_callback_decision,
        "state_uninitialized"_s <= "state_init_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_init_callback_decision,
        "state_uninitialized"_s <= "state_init_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_init_failed_error_out_decision,
        "state_uninitialized"_s <= "state_init_failed_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_init_failed_error_out_decision,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_init_failed_callback_decision,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_init_failed_callback_decision,
        "state_uninitialized"_s <= "state_uninit_encode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninit_encode_error_out_decision,
        "state_uninitialized"_s <= "state_uninit_encode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninit_encode_error_out_decision,
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninit_encode_callback_decision,
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninit_encode_callback_decision,
        "state_uninitialized"_s <= "state_uninit_decode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninit_decode_error_out_decision,
        "state_uninitialized"_s <= "state_uninit_decode_error_out_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninit_decode_error_out_decision,
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninit_decode_callback_decision,
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninit_decode_callback_decision,
    }
}

/// Context for `SpeechCodecMimi` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechCodecMimiContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechCodecMimiStateMachineContext for SpeechCodecMimiContext {
    fn effect_bind(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_bind
        todo!(
            "TODO: port action `effect_bind` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_decode_done(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_decode_done
        todo!(
            "TODO: port action `effect_emit_decode_done` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_decode_error_from_state_decode_failed_callback_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_decode_error
        todo!(
            "TODO: port action `effect_emit_decode_error` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_decode_error_from_state_uninit_decode_callback_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_decode_error
        todo!(
            "TODO: port action `effect_emit_decode_error` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_encode_done(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_encode_done
        todo!(
            "TODO: port action `effect_emit_encode_done` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_encode_error_from_state_encode_failed_callback_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_encode_error
        todo!(
            "TODO: port action `effect_emit_encode_error` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_encode_error_from_state_uninit_encode_callback_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_encode_error
        todo!(
            "TODO: port action `effect_emit_encode_error` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_initialize_done(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_emit_initialize_error(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_arena_capacity_invalid(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_arena_capacity_invalid
        todo!(
            "TODO: port action `effect_mark_arena_capacity_invalid` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_bind_failed(&mut self, _event: &InitRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_bind_failed
        todo!(
            "TODO: port action `effect_mark_bind_failed` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_code_range_invalid(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_code_range_invalid
        todo!(
            "TODO: port action `effect_mark_code_range_invalid` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_decode_run(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_not_initialized_encode_run(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_not_initialized
        todo!(
            "TODO: port action `effect_mark_not_initialized` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_request_shape_invalid_decode_run(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_request_shape_invalid
        todo!(
            "TODO: port action `effect_mark_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_request_shape_invalid_encode_run(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_request_shape_invalid
        todo!(
            "TODO: port action `effect_mark_request_shape_invalid` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_bind_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_bind_contract_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_binding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_codes_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decode_request_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_decoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_dequantizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encode_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encode_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encode_request_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_encoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_init_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_init_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_init_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_init_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_quantizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_session_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninit_decode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninit_decode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninit_encode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninit_encode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_and_store_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected_and_store
        todo!(
            "TODO: port action `effect_mark_unexpected_and_store` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_bind_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_bind_contract_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_binding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_codes_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decode_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_decoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_dequantizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encode_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encode_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encode_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encode_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encode_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_encoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_init_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_init_error_out_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_init_failed_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_init_failed_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_quantizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_session_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninit_decode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninit_decode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninit_encode_callback_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninit_encode_error_out_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_mark_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_mark_unexpected
        todo!(
            "TODO: port action `effect_mark_unexpected` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_reset_stream(&mut self, _event: &EventResetStreamRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_reset_stream
        todo!(
            "TODO: port action `effect_reset_stream` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_run_backend_child(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_run_backend_child
        todo!(
            "TODO: port action `effect_run_backend_child` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_run_dequantize_child(&mut self, _event: &DecodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_run_dequantize_child
        todo!(
            "TODO: port action `effect_run_dequantize_child` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_run_frontend_child(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_run_frontend_child
        todo!(
            "TODO: port action `effect_run_frontend_child` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_run_quantize_child(&mut self, _event: &EncodeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_run_quantize_child
        todo!(
            "TODO: port action `effect_run_quantize_child` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_decode_run_from_state_decode_error_out_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_decode_run_from_state_decode_failed_error_out_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_decode_run_from_state_uninit_decode_error_out_decision(
        &mut self,
        _event: &DecodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_encode_run_from_state_encode_error_out_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_encode_run_from_state_encode_failed_error_out_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_encode_run_from_state_uninit_encode_error_out_decision(
        &mut self,
        _event: &EncodeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn effect_store_error_out_init_run_from_state_init_failed_error_out_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/actions.hpp::effect_store_error_out
        todo!(
            "TODO: port action `effect_store_error_out` from emel.cpp/src/emel/speech/codec/mimi/actions.hpp"
        )
    }
    fn guard_arena_capacity_invalid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_arena_capacity_invalid
        todo!(
            "TODO: port guard `guard_arena_capacity_invalid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_arena_capacity_valid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_arena_capacity_valid
        todo!(
            "TODO: port guard `guard_arena_capacity_valid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_bind_contract_invalid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_bind_contract_invalid
        todo!(
            "TODO: port guard `guard_bind_contract_invalid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_bind_contract_valid(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_bind_contract_valid
        todo!(
            "TODO: port guard `guard_bind_contract_valid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_decode_codes_invalid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_decode_codes_invalid
        todo!(
            "TODO: port guard `guard_decode_codes_invalid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_decode_codes_valid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_decode_codes_valid
        todo!(
            "TODO: port guard `guard_decode_codes_valid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_decode_request_invalid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_decode_request_invalid
        todo!(
            "TODO: port guard `guard_decode_request_invalid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_decode_request_valid(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_decode_request_valid
        todo!(
            "TODO: port guard `guard_decode_request_valid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_encode_request_invalid(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_encode_request_invalid
        todo!(
            "TODO: port guard `guard_encode_request_invalid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_encode_request_valid(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_encode_request_valid
        todo!(
            "TODO: port guard `guard_encode_request_valid` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_done_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_done_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_done_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_done_callback
        todo!(
            "TODO: port guard `guard_has_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_callback
        todo!(
            "TODO: port guard `guard_has_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_out_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_out_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_has_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_has_error_out
        todo!(
            "TODO: port guard `guard_has_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_done_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_done_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_done_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_done_callback
        todo!(
            "TODO: port guard `guard_no_done_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_callback_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_callback_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_callback_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_callback
        todo!(
            "TODO: port guard `guard_no_error_callback` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_out_decode_run(&self, _event: &DecodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_out_encode_run(&self, _event: &EncodeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_no_error_out_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_no_error_out
        todo!(
            "TODO: port guard `guard_no_error_out` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_absent(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_unexpected_error_out_absent
        todo!(
            "TODO: port guard `guard_unexpected_error_out_absent` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
    fn guard_unexpected_error_out_present(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/codec/mimi/guards.hpp::guard_unexpected_error_out_present
        todo!(
            "TODO: port guard `guard_unexpected_error_out_present` from emel.cpp/src/emel/speech/codec/mimi/guards.hpp"
        )
    }
}
