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

// --- machine SpeechTranscriber from emel.cpp/src/emel/speech/transcriber/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitializeRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRecognizeRun;

sml! {
    SpeechTranscriber {
        "state_initializing"_s <= *"state_uninitialized"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_uninitialized,
        "state_initialize_error_out_decision"_s <= "state_uninitialized"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_uninitialized,
        "state_initializing"_s <= "state_ready"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_ready,
        "state_initialize_error_out_decision"_s <= "state_ready"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_ready,
        "state_initialize_error_out_decision"_s <= "state_errored"_s + event<EventInitializeRun> / effect_reject_initialize_from_state_errored,
        "state_tokenizer_decision"_s <= "state_initializing"_s + completion<EventInitializeRun>,
        "state_tokenizer_validation_decision"_s <= "state_tokenizer_decision"_s + completion<EventInitializeRun> [guard_initialize_tokenizer_supported] / effect_validate_tokenizer_assets,
        "state_initialize_error_out_decision"_s <= "state_tokenizer_decision"_s + completion<EventInitializeRun> [guard_initialize_tokenizer_unsupported] / effect_mark_tokenizer_invalid_from_state_tokenizer_decision,
        "state_model_support_decision"_s <= "state_tokenizer_validation_decision"_s + completion<EventInitializeRun> [guard_tokenizer_validation_accepted],
        "state_initialize_error_out_decision"_s <= "state_tokenizer_validation_decision"_s + completion<EventInitializeRun> [guard_tokenizer_validation_rejected] / effect_mark_tokenizer_invalid_from_state_tokenizer_validation_decision,
        "state_initialize_success"_s <= "state_model_support_decision"_s + completion<EventInitializeRun> [guard_initialize_model_supported],
        "state_initialize_error_out_decision"_s <= "state_model_support_decision"_s + completion<EventInitializeRun> [guard_initialize_unsupported_model] / effect_mark_unsupported_model,
        "state_initialize_done_callback_decision"_s <= "state_initialize_success"_s + completion<EventInitializeRun> [guard_has_initialize_error_out] / effect_store_initialize_success,
        "state_initialize_done_callback_decision"_s <= "state_initialize_success"_s + completion<EventInitializeRun> [guard_no_initialize_error_out],
        "state_initialize_error_callback_decision"_s <= "state_initialize_error_out_decision"_s + completion<EventInitializeRun> [guard_has_initialize_error_out] / effect_store_initialize_error,
        "state_initialize_error_callback_decision"_s <= "state_initialize_error_out_decision"_s + completion<EventInitializeRun> [guard_no_initialize_error_out],
        "state_ready"_s <= "state_initialize_done_callback_decision"_s + completion<EventInitializeRun> [guard_has_initialize_done_callback] / effect_emit_initialize_done,
        "state_ready"_s <= "state_initialize_done_callback_decision"_s + completion<EventInitializeRun> [guard_no_initialize_done_callback],
        "state_errored"_s <= "state_initialize_error_callback_decision"_s + completion<EventInitializeRun> [guard_has_initialize_error_callback] / effect_emit_initialize_error,
        "state_errored"_s <= "state_initialize_error_callback_decision"_s + completion<EventInitializeRun> [guard_no_initialize_error_callback],
        "state_recognize_support_decision"_s <= "state_ready"_s + event<EventRecognizeRun> [guard_valid_recognize] / effect_begin_recognize,
        "state_recognize_error_out_decision"_s <= "state_ready"_s + event<EventRecognizeRun> [guard_invalid_recognize] / effect_reject_recognize,
        "state_recognize_uninitialized_error_out_decision"_s <= "state_uninitialized"_s + event<EventRecognizeRun> / effect_mark_uninitialized_from_state_uninitialized,
        "state_recognize_errored_error_out_decision"_s <= "state_errored"_s + event<EventRecognizeRun> / effect_mark_uninitialized_from_state_errored,
        "state_recognize_error_out_decision"_s <= "state_recognize_support_decision"_s + completion<EventRecognizeRun> [guard_transcriber_unsupported] / effect_mark_uninitialized_from_state_recognize_support_decision,
        "state_encoding"_s <= "state_recognize_support_decision"_s + completion<EventRecognizeRun> [guard_transcriber_ready] / effect_encode,
        "state_encoder_decision"_s <= "state_encoding"_s + completion<EventRecognizeRun>,
        "state_decoding"_s <= "state_encoder_decision"_s + completion<EventRecognizeRun> [guard_encoder_success] / effect_decode,
        "state_recognize_error_out_decision"_s <= "state_encoder_decision"_s + completion<EventRecognizeRun> [guard_encoder_failure] / effect_mark_backend_error_from_state_encoder_decision,
        "state_decoder_decision"_s <= "state_decoding"_s + completion<EventRecognizeRun>,
        "state_detokenizing"_s <= "state_decoder_decision"_s + completion<EventRecognizeRun> [guard_decoder_success] / effect_detokenize,
        "state_recognize_error_out_decision"_s <= "state_decoder_decision"_s + completion<EventRecognizeRun> [guard_decoder_failure] / effect_mark_backend_error_from_state_decoder_decision,
        "state_detokenize_decision"_s <= "state_detokenizing"_s + completion<EventRecognizeRun>,
        "state_recognize_success"_s <= "state_detokenize_decision"_s + completion<EventRecognizeRun> [guard_detokenize_success] / effect_publish_recognition_outputs,
        "state_recognize_error_out_decision"_s <= "state_detokenize_decision"_s + completion<EventRecognizeRun> [guard_detokenize_failure] / effect_mark_backend_error_from_state_detokenize_decision,
        "state_recognize_done_callback_decision"_s <= "state_recognize_success"_s + completion<EventRecognizeRun> [guard_has_recognize_error_out] / effect_store_recognize_success,
        "state_recognize_done_callback_decision"_s <= "state_recognize_success"_s + completion<EventRecognizeRun> [guard_no_recognize_error_out],
        "state_recognize_error_callback_decision"_s <= "state_recognize_error_out_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_error_out_decision,
        "state_recognize_error_callback_decision"_s <= "state_recognize_error_out_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_out],
        "state_done"_s <= "state_recognize_done_callback_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_done_callback] / effect_emit_recognize_done,
        "state_done"_s <= "state_recognize_done_callback_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_done_callback],
        "state_ready"_s <= "state_recognize_error_callback_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_error_callback_decision,
        "state_ready"_s <= "state_recognize_error_callback_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventRecognizeRun>,
        "state_recognize_uninitialized_error_callback_decision"_s <= "state_recognize_uninitialized_error_out_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_uninitialized_error_out_decision,
        "state_recognize_uninitialized_error_callback_decision"_s <= "state_recognize_uninitialized_error_out_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_out],
        "state_uninitialized"_s <= "state_recognize_uninitialized_error_callback_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_uninitialized_error_callback_decision,
        "state_uninitialized"_s <= "state_recognize_uninitialized_error_callback_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_callback],
        "state_recognize_errored_error_callback_decision"_s <= "state_recognize_errored_error_out_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_out] / effect_store_recognize_error_from_state_recognize_errored_error_out_decision,
        "state_recognize_errored_error_callback_decision"_s <= "state_recognize_errored_error_out_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_out],
        "state_errored"_s <= "state_recognize_errored_error_callback_decision"_s + completion<EventRecognizeRun> [guard_has_recognize_error_callback] / effect_emit_recognize_error_from_state_recognize_errored_error_callback_decision,
        "state_errored"_s <= "state_recognize_errored_error_callback_decision"_s + completion<EventRecognizeRun> [guard_no_recognize_error_callback],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_on_unexpected_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Context for `SpeechTranscriber` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechTranscriberContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechTranscriberStateMachineContext for SpeechTranscriberContext {
    fn effect_begin_initialize_from_state_ready(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_begin_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_begin_recognize(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_begin_recognize
        todo!(
            "TODO: port action `effect_begin_recognize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_decode(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_decode
        todo!(
            "TODO: port action `effect_decode` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_detokenize(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_detokenize
        todo!(
            "TODO: port action `effect_detokenize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_initialize_done(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_initialize_error(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_recognize_done(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_recognize_done
        todo!(
            "TODO: port action `effect_emit_recognize_done` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_recognize_error_from_state_recognize_error_callback_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_recognize_error
        todo!(
            "TODO: port action `effect_emit_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_recognize_error_from_state_recognize_errored_error_callback_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_recognize_error
        todo!(
            "TODO: port action `effect_emit_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_emit_recognize_error_from_state_recognize_uninitialized_error_callback_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_emit_recognize_error
        todo!(
            "TODO: port action `effect_emit_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_encode(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_encode
        todo!(
            "TODO: port action `effect_encode` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_backend_error_from_state_decoder_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_backend_error
        todo!(
            "TODO: port action `effect_mark_backend_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_backend_error_from_state_detokenize_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_backend_error
        todo!(
            "TODO: port action `effect_mark_backend_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_backend_error_from_state_encoder_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_backend_error
        todo!(
            "TODO: port action `effect_mark_backend_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_tokenizer_invalid_from_state_tokenizer_decision(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_tokenizer_invalid
        todo!(
            "TODO: port action `effect_mark_tokenizer_invalid` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_tokenizer_invalid_from_state_tokenizer_validation_decision(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_tokenizer_invalid
        todo!(
            "TODO: port action `effect_mark_tokenizer_invalid` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_uninitialized_from_state_errored(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_uninitialized
        todo!(
            "TODO: port action `effect_mark_uninitialized` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_uninitialized_from_state_recognize_support_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_uninitialized
        todo!(
            "TODO: port action `effect_mark_uninitialized` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_uninitialized_from_state_uninitialized(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_uninitialized
        todo!(
            "TODO: port action `effect_mark_uninitialized` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_mark_unsupported_model(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_mark_unsupported_model
        todo!(
            "TODO: port action `effect_mark_unsupported_model` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_publish_recognition_outputs(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_publish_recognition_outputs
        todo!(
            "TODO: port action `effect_publish_recognition_outputs` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_errored(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_ready(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_reject_recognize(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_reject_recognize
        todo!(
            "TODO: port action `effect_reject_recognize` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_initialize_error(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_initialize_error
        todo!(
            "TODO: port action `effect_store_initialize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_initialize_success(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_initialize_success
        todo!(
            "TODO: port action `effect_store_initialize_success` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_recognize_error_from_state_recognize_error_out_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_recognize_error
        todo!(
            "TODO: port action `effect_store_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_recognize_error_from_state_recognize_errored_error_out_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_recognize_error
        todo!(
            "TODO: port action `effect_store_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_recognize_error_from_state_recognize_uninitialized_error_out_decision(
        &mut self,
        _event: &EventRecognizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_recognize_error
        todo!(
            "TODO: port action `effect_store_recognize_error` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_store_recognize_success(&mut self, _event: &EventRecognizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_store_recognize_success
        todo!(
            "TODO: port action `effect_store_recognize_success` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn effect_validate_tokenizer_assets(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/actions.hpp::effect_validate_tokenizer_assets
        todo!(
            "TODO: port action `effect_validate_tokenizer_assets` from emel.cpp/src/emel/speech/transcriber/actions.hpp"
        )
    }
    fn guard_decoder_failure(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_decoder_failure
        todo!(
            "TODO: port guard `guard_decoder_failure` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_decoder_success(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_decoder_success
        todo!(
            "TODO: port guard `guard_decoder_success` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_detokenize_failure(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_detokenize_failure
        todo!(
            "TODO: port guard `guard_detokenize_failure` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_detokenize_success(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_detokenize_success
        todo!(
            "TODO: port guard `guard_detokenize_success` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_encoder_failure(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_encoder_failure
        todo!(
            "TODO: port guard `guard_encoder_failure` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_encoder_success(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_encoder_success
        todo!(
            "TODO: port guard `guard_encoder_success` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_initialize_done_callback
        todo!(
            "TODO: port guard `guard_has_initialize_done_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_initialize_error_callback
        todo!(
            "TODO: port guard `guard_has_initialize_error_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_initialize_error_out(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_initialize_error_out
        todo!(
            "TODO: port guard `guard_has_initialize_error_out` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_recognize_done_callback(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_recognize_done_callback
        todo!(
            "TODO: port guard `guard_has_recognize_done_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_recognize_error_callback(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_recognize_error_callback
        todo!(
            "TODO: port guard `guard_has_recognize_error_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_has_recognize_error_out(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_has_recognize_error_out
        todo!(
            "TODO: port guard `guard_has_recognize_error_out` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_initialize_model_supported(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_initialize_model_supported
        todo!(
            "TODO: port guard `guard_initialize_model_supported` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_initialize_tokenizer_supported(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_initialize_tokenizer_supported
        todo!(
            "TODO: port guard `guard_initialize_tokenizer_supported` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_initialize_tokenizer_unsupported(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_initialize_tokenizer_unsupported
        todo!(
            "TODO: port guard `guard_initialize_tokenizer_unsupported` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_initialize_unsupported_model(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_initialize_unsupported_model
        todo!(
            "TODO: port guard `guard_initialize_unsupported_model` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_invalid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_invalid_initialize
        todo!(
            "TODO: port guard `guard_invalid_initialize` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_invalid_recognize(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_invalid_recognize
        todo!(
            "TODO: port guard `guard_invalid_recognize` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_initialize_done_callback
        todo!(
            "TODO: port guard `guard_no_initialize_done_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_initialize_error_callback
        todo!(
            "TODO: port guard `guard_no_initialize_error_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_initialize_error_out(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_initialize_error_out
        todo!(
            "TODO: port guard `guard_no_initialize_error_out` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_recognize_done_callback(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_recognize_done_callback
        todo!(
            "TODO: port guard `guard_no_recognize_done_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_recognize_error_callback(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_recognize_error_callback
        todo!(
            "TODO: port guard `guard_no_recognize_error_callback` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_no_recognize_error_out(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_no_recognize_error_out
        todo!(
            "TODO: port guard `guard_no_recognize_error_out` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_tokenizer_validation_accepted(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_tokenizer_validation_accepted
        todo!(
            "TODO: port guard `guard_tokenizer_validation_accepted` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_tokenizer_validation_rejected(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_tokenizer_validation_rejected
        todo!(
            "TODO: port guard `guard_tokenizer_validation_rejected` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_transcriber_ready(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_transcriber_ready
        todo!(
            "TODO: port guard `guard_transcriber_ready` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_transcriber_unsupported(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_transcriber_unsupported
        todo!(
            "TODO: port guard `guard_transcriber_unsupported` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_valid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_valid_initialize
        todo!(
            "TODO: port guard `guard_valid_initialize` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
    fn guard_valid_recognize(&self, _event: &EventRecognizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/transcriber/guards.hpp::guard_valid_recognize
        todo!(
            "TODO: port guard `guard_valid_recognize` from emel.cpp/src/emel/speech/transcriber/guards.hpp"
        )
    }
}
