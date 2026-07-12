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

// --- machine EmbeddingsGenerator from emel.cpp/src/emel/embeddings/generator/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventEmbedAudioRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventEmbedImageRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventEmbedTextRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitializeRun;

sml! {
    EmbeddingsGenerator {
        "state_initializing"_s <= *"state_uninitialized"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_uninitialized,
        "state_initialize_publish_error"_s <= "state_uninitialized"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_uninitialized,
        "state_initializing"_s <= "state_idle"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_idle,
        "state_initialize_publish_error"_s <= "state_idle"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_idle,
        "state_conditioning"_s <= "state_idle"_s + event<EventEmbedTextRun> [guard_valid_embed_full] / effect_begin_embed_text_from_state_idle,
        "state_conditioning"_s <= "state_idle"_s + event<EventEmbedTextRun> [guard_valid_embed_truncate] / effect_begin_embed_text_from_state_idle,
        "state_embed_publish_error"_s <= "state_idle"_s + event<EventEmbedTextRun> [guard_invalid_embed] / effect_reject_embed_text_from_state_idle,
        "state_image_preparing"_s <= "state_idle"_s + event<EventEmbedImageRun> [guard_valid_embed_image_full] / effect_begin_embed_image_from_state_idle,
        "state_image_preparing"_s <= "state_idle"_s + event<EventEmbedImageRun> [guard_valid_embed_image_truncate] / effect_begin_embed_image_from_state_idle,
        "state_embed_publish_error"_s <= "state_idle"_s + event<EventEmbedImageRun> [guard_invalid_embed_image] / effect_reject_embed_image_from_state_idle,
        "state_audio_preparing"_s <= "state_idle"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_full] / effect_begin_embed_audio_from_state_idle,
        "state_audio_preparing"_s <= "state_idle"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_truncate] / effect_begin_embed_audio_from_state_idle,
        "state_embed_publish_error"_s <= "state_idle"_s + event<EventEmbedAudioRun> [guard_invalid_embed_audio] / effect_reject_embed_audio_from_state_idle,
        "state_initializing"_s <= "state_done"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_done,
        "state_initialize_publish_error"_s <= "state_done"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_done,
        "state_conditioning"_s <= "state_done"_s + event<EventEmbedTextRun> [guard_valid_embed_full] / effect_begin_embed_text_from_state_done,
        "state_conditioning"_s <= "state_done"_s + event<EventEmbedTextRun> [guard_valid_embed_truncate] / effect_begin_embed_text_from_state_done,
        "state_embed_publish_error"_s <= "state_done"_s + event<EventEmbedTextRun> [guard_invalid_embed] / effect_reject_embed_text_from_state_done,
        "state_image_preparing"_s <= "state_done"_s + event<EventEmbedImageRun> [guard_valid_embed_image_full] / effect_begin_embed_image_from_state_done,
        "state_image_preparing"_s <= "state_done"_s + event<EventEmbedImageRun> [guard_valid_embed_image_truncate] / effect_begin_embed_image_from_state_done,
        "state_embed_publish_error"_s <= "state_done"_s + event<EventEmbedImageRun> [guard_invalid_embed_image] / effect_reject_embed_image_from_state_done,
        "state_audio_preparing"_s <= "state_done"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_full] / effect_begin_embed_audio_from_state_done,
        "state_audio_preparing"_s <= "state_done"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_truncate] / effect_begin_embed_audio_from_state_done,
        "state_embed_publish_error"_s <= "state_done"_s + event<EventEmbedAudioRun> [guard_invalid_embed_audio] / effect_reject_embed_audio_from_state_done,
        "state_initializing"_s <= "state_errored"_s + event<EventInitializeRun> [guard_valid_initialize] / effect_begin_initialize_from_state_errored,
        "state_initialize_publish_error"_s <= "state_errored"_s + event<EventInitializeRun> [guard_invalid_initialize] / effect_reject_initialize_from_state_errored,
        "state_conditioning"_s <= "state_errored"_s + event<EventEmbedTextRun> [guard_valid_embed_full] / effect_begin_embed_text_from_state_errored,
        "state_conditioning"_s <= "state_errored"_s + event<EventEmbedTextRun> [guard_valid_embed_truncate] / effect_begin_embed_text_from_state_errored,
        "state_embed_publish_error"_s <= "state_errored"_s + event<EventEmbedTextRun> [guard_invalid_embed] / effect_reject_embed_text_from_state_errored,
        "state_image_preparing"_s <= "state_errored"_s + event<EventEmbedImageRun> [guard_valid_embed_image_full] / effect_begin_embed_image_from_state_errored,
        "state_image_preparing"_s <= "state_errored"_s + event<EventEmbedImageRun> [guard_valid_embed_image_truncate] / effect_begin_embed_image_from_state_errored,
        "state_embed_publish_error"_s <= "state_errored"_s + event<EventEmbedImageRun> [guard_invalid_embed_image] / effect_reject_embed_image_from_state_errored,
        "state_audio_preparing"_s <= "state_errored"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_full] / effect_begin_embed_audio_from_state_errored,
        "state_audio_preparing"_s <= "state_errored"_s + event<EventEmbedAudioRun> [guard_valid_embed_audio_truncate] / effect_begin_embed_audio_from_state_errored,
        "state_embed_publish_error"_s <= "state_errored"_s + event<EventEmbedAudioRun> [guard_invalid_embed_audio] / effect_reject_embed_audio_from_state_errored,
        "state_initialize_decision"_s <= "state_initializing"_s + completion<EventInitializeRun> / effect_dispatch_bind_conditioner,
        "state_initialize_publish_success"_s <= "state_initialize_decision"_s + completion<EventInitializeRun> [guard_initialize_success] / effect_mark_initialized,
        "state_initialize_publish_error"_s <= "state_initialize_decision"_s + completion<EventInitializeRun> [guard_initialize_model_invalid] / effect_set_initialize_model_invalid,
        "state_initialize_publish_error"_s <= "state_initialize_decision"_s + completion<EventInitializeRun> [guard_initialize_backend_error] / effect_set_initialize_backend_error,
        "state_idle"_s <= "state_initialize_publish_success"_s + completion<EventInitializeRun> [guard_has_initialize_done_callback] / effect_emit_initialize_done,
        "state_idle"_s <= "state_initialize_publish_success"_s + completion<EventInitializeRun> [guard_no_initialize_done_callback],
        "state_initialize_error_channel_decision"_s <= "state_initialize_publish_error"_s + completion<EventInitializeRun> / effect_write_initialize_error_out,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<EventInitializeRun> [guard_has_initialize_error_callback] / effect_emit_initialize_error,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + completion<EventInitializeRun> [guard_no_initialize_error_callback],
        "state_conditioning_decision"_s <= "state_conditioning"_s + completion<EventEmbedTextRun> / effect_dispatch_condition_text,
        "state_embed_publish_error"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_invalid_request] / effect_set_embed_invalid_request,
        "state_embed_publish_error"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_model_invalid] / effect_set_embed_model_invalid,
        "state_embed_publish_error"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_backend_error] / effect_set_embed_backend_error_event_embed_text_run,
        "state_encoding"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_success],
        "state_embed_publish_error"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_route_unsupported] / effect_set_embed_backend_error_event_embed_text_run,
        "state_embed_publish_error"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_encode_unready] / effect_set_embed_backend_error_event_embed_text_run,
        "state_embedding_decision"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_encode_ready],
        "state_embed_publish_error"_s <= "state_embedding_decision"_s + completion<EventEmbedTextRun> [guard_embedding_failed_event_embed_text_run],
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedTextRun> [guard_embedding_succeeded_full_event_embed_text_run] / effect_publish_full_embedding_event_embed_text_run,
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedTextRun> [guard_embedding_succeeded_truncate_event_embed_text_run] / effect_publish_truncated_embedding_event_embed_text_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedTextRun> [guard_has_embed_done_callback_event_embed_text_run] / effect_emit_embed_done_event_embed_text_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedTextRun> [guard_no_embed_done_callback_event_embed_text_run],
        "state_embed_error_channel_decision"_s <= "state_embed_publish_error"_s + completion<EventEmbedTextRun> / effect_write_embed_error_out_event_embed_text_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedTextRun> [guard_has_embed_error_callback_event_embed_text_run] / effect_emit_embed_error_event_embed_text_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedTextRun> [guard_no_embed_error_callback_event_embed_text_run],
        "state_embed_publish_error"_s <= "state_image_preparing"_s + completion<EventEmbedImageRun> [guard_image_route_unsupported] / effect_set_embed_backend_error_event_embed_image_run,
        "state_embed_publish_error"_s <= "state_image_preparing"_s + completion<EventEmbedImageRun> [guard_image_prepare_unready] / effect_set_embed_backend_error_event_embed_image_run,
        "state_image_encoding"_s <= "state_image_preparing"_s + completion<EventEmbedImageRun> [guard_image_prepare_ready],
        "state_embed_publish_error"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_route_unsupported] / effect_set_embed_backend_error_event_embed_image_run,
        "state_embed_publish_error"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_encode_unready] / effect_set_embed_backend_error_event_embed_image_run,
        "state_embedding_decision"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_encode_ready],
        "state_embed_publish_error"_s <= "state_embedding_decision"_s + completion<EventEmbedImageRun> [guard_embedding_failed_event_embed_image_run],
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedImageRun> [guard_embedding_succeeded_full_event_embed_image_run] / effect_publish_full_embedding_event_embed_image_run,
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedImageRun> [guard_embedding_succeeded_truncate_event_embed_image_run] / effect_publish_truncated_embedding_event_embed_image_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedImageRun> [guard_has_embed_done_callback_event_embed_image_run] / effect_emit_embed_done_event_embed_image_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedImageRun> [guard_no_embed_done_callback_event_embed_image_run],
        "state_embed_error_channel_decision"_s <= "state_embed_publish_error"_s + completion<EventEmbedImageRun> / effect_write_embed_error_out_event_embed_image_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedImageRun> [guard_has_embed_error_callback_event_embed_image_run] / effect_emit_embed_error_event_embed_image_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedImageRun> [guard_no_embed_error_callback_event_embed_image_run],
        "state_embed_publish_error"_s <= "state_audio_preparing"_s + completion<EventEmbedAudioRun> [guard_audio_route_unsupported] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_embed_publish_error"_s <= "state_audio_preparing"_s + completion<EventEmbedAudioRun> [guard_audio_prepare_unready] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_audio_encoding"_s <= "state_audio_preparing"_s + completion<EventEmbedAudioRun> [guard_audio_prepare_ready],
        "state_embed_publish_error"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_route_unsupported] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_embed_publish_error"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_encode_unready] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_embedding_decision"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_encode_ready],
        "state_embed_publish_error"_s <= "state_embedding_decision"_s + completion<EventEmbedAudioRun> [guard_embedding_failed_event_embed_audio_run],
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedAudioRun> [guard_embedding_succeeded_full_event_embed_audio_run] / effect_publish_full_embedding_event_embed_audio_run,
        "state_embed_publish_success"_s <= "state_embedding_decision"_s + completion<EventEmbedAudioRun> [guard_embedding_succeeded_truncate_event_embed_audio_run] / effect_publish_truncated_embedding_event_embed_audio_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedAudioRun> [guard_has_embed_done_callback_event_embed_audio_run] / effect_emit_embed_done_event_embed_audio_run,
        "state_done"_s <= "state_embed_publish_success"_s + completion<EventEmbedAudioRun> [guard_no_embed_done_callback_event_embed_audio_run],
        "state_embed_error_channel_decision"_s <= "state_embed_publish_error"_s + completion<EventEmbedAudioRun> / effect_write_embed_error_out_event_embed_audio_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedAudioRun> [guard_has_embed_error_callback_event_embed_audio_run] / effect_emit_embed_error_event_embed_audio_run,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + completion<EventEmbedAudioRun> [guard_no_embed_error_callback_event_embed_audio_run],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_reject_unexpected_from_state_uninitialized,
        "state_uninitialized"_s <= "state_initializing"_s + unexpected_event<_> / effect_reject_unexpected_from_state_initializing,
        "state_uninitialized"_s <= "state_initialize_decision"_s + unexpected_event<_> / effect_reject_unexpected_from_state_initialize_decision,
        "state_idle"_s <= "state_initialize_publish_success"_s + unexpected_event<_> / effect_reject_unexpected_from_state_initialize_publish_success,
        "state_errored"_s <= "state_initialize_publish_error"_s + unexpected_event<_> / effect_reject_unexpected_from_state_initialize_publish_error,
        "state_errored"_s <= "state_initialize_error_channel_decision"_s + unexpected_event<_> / effect_reject_unexpected_from_state_initialize_error_channel_decision,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_reject_unexpected_from_state_idle,
        "state_idle"_s <= "state_conditioning"_s + unexpected_event<_> / effect_reject_unexpected_from_state_conditioning,
        "state_idle"_s <= "state_conditioning_decision"_s + unexpected_event<_> / effect_reject_unexpected_from_state_conditioning_decision,
        "state_idle"_s <= "state_encoding"_s + unexpected_event<_> / effect_reject_unexpected_from_state_encoding,
        "state_idle"_s <= "state_image_preparing"_s + unexpected_event<_> / effect_reject_unexpected_from_state_image_preparing,
        "state_idle"_s <= "state_image_encoding"_s + unexpected_event<_> / effect_reject_unexpected_from_state_image_encoding,
        "state_idle"_s <= "state_audio_preparing"_s + unexpected_event<_> / effect_reject_unexpected_from_state_audio_preparing,
        "state_idle"_s <= "state_audio_encoding"_s + unexpected_event<_> / effect_reject_unexpected_from_state_audio_encoding,
        "state_idle"_s <= "state_embedding_decision"_s + unexpected_event<_> / effect_reject_unexpected_from_state_embedding_decision,
        "state_done"_s <= "state_embed_publish_success"_s + unexpected_event<_> / effect_reject_unexpected_from_state_embed_publish_success,
        "state_errored"_s <= "state_embed_publish_error"_s + unexpected_event<_> / effect_reject_unexpected_from_state_embed_publish_error,
        "state_errored"_s <= "state_embed_error_channel_decision"_s + unexpected_event<_> / effect_reject_unexpected_from_state_embed_error_channel_decision,
        "state_done"_s <= "state_done"_s + unexpected_event<_> / effect_reject_unexpected_from_state_done,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_reject_unexpected_from_state_errored,
    }
}

/// Context for `EmbeddingsGenerator` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct EmbeddingsGeneratorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl EmbeddingsGeneratorStateMachineContext for EmbeddingsGeneratorContext {
    fn effect_begin_embed_audio_from_state_done(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_audio
        todo!(
            "TODO: port action `effect_begin_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_audio_from_state_errored(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_audio
        todo!(
            "TODO: port action `effect_begin_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_audio_from_state_idle(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_audio
        todo!(
            "TODO: port action `effect_begin_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_image_from_state_done(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_image
        todo!(
            "TODO: port action `effect_begin_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_image_from_state_errored(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_image
        todo!(
            "TODO: port action `effect_begin_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_image_from_state_idle(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_image
        todo!(
            "TODO: port action `effect_begin_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_text_from_state_done(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_text
        todo!(
            "TODO: port action `effect_begin_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_text_from_state_errored(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_text
        todo!(
            "TODO: port action `effect_begin_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_embed_text_from_state_idle(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_embed_text
        todo!(
            "TODO: port action `effect_begin_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_initialize_from_state_done(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_initialize_from_state_errored(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_initialize_from_state_idle(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_begin_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_begin_initialize
        todo!(
            "TODO: port action `effect_begin_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_dispatch_bind_conditioner(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_dispatch_bind_conditioner
        todo!(
            "TODO: port action `effect_dispatch_bind_conditioner` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_dispatch_condition_text(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_dispatch_condition_text
        todo!(
            "TODO: port action `effect_dispatch_condition_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_done_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_done
        todo!(
            "TODO: port action `effect_emit_embed_done` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_done_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_done
        todo!(
            "TODO: port action `effect_emit_embed_done` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_done_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_done
        todo!(
            "TODO: port action `effect_emit_embed_done` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_error_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_error
        todo!(
            "TODO: port action `effect_emit_embed_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_error_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_error
        todo!(
            "TODO: port action `effect_emit_embed_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_embed_error_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_embed_error
        todo!(
            "TODO: port action `effect_emit_embed_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_initialize_done(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_initialize_done
        todo!(
            "TODO: port action `effect_emit_initialize_done` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_emit_initialize_error(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_emit_initialize_error
        todo!(
            "TODO: port action `effect_emit_initialize_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_mark_initialized(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_mark_initialized
        todo!(
            "TODO: port action `effect_mark_initialized` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_full_embedding_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_full_embedding
        todo!(
            "TODO: port action `effect_publish_full_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_full_embedding_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_full_embedding
        todo!(
            "TODO: port action `effect_publish_full_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_full_embedding_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_full_embedding
        todo!(
            "TODO: port action `effect_publish_full_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_truncated_embedding_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_truncated_embedding
        todo!(
            "TODO: port action `effect_publish_truncated_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_truncated_embedding_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_truncated_embedding
        todo!(
            "TODO: port action `effect_publish_truncated_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_publish_truncated_embedding_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_publish_truncated_embedding
        todo!(
            "TODO: port action `effect_publish_truncated_embedding` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_audio_from_state_done(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_audio
        todo!(
            "TODO: port action `effect_reject_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_audio_from_state_errored(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_audio
        todo!(
            "TODO: port action `effect_reject_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_audio_from_state_idle(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_audio
        todo!(
            "TODO: port action `effect_reject_embed_audio` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_image_from_state_done(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_image
        todo!(
            "TODO: port action `effect_reject_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_image_from_state_errored(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_image
        todo!(
            "TODO: port action `effect_reject_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_image_from_state_idle(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_image
        todo!(
            "TODO: port action `effect_reject_embed_image` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_text_from_state_done(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_text
        todo!(
            "TODO: port action `effect_reject_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_text_from_state_errored(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_text
        todo!(
            "TODO: port action `effect_reject_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_embed_text_from_state_idle(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_embed_text
        todo!(
            "TODO: port action `effect_reject_embed_text` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_done(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_errored(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_idle(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_initialize
        todo!(
            "TODO: port action `effect_reject_initialize` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_audio_encoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_audio_preparing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_conditioning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_conditioning_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_embed_error_channel_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_embed_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_embed_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_embedding_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_encoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_image_encoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_image_preparing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_initialize_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_initialize_error_channel_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_initialize_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_initialize_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_initializing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_reject_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_reject_unexpected
        todo!(
            "TODO: port action `effect_reject_unexpected` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_embed_backend_error_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_embed_backend_error
        todo!(
            "TODO: port action `effect_set_embed_backend_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_embed_backend_error_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_embed_backend_error
        todo!(
            "TODO: port action `effect_set_embed_backend_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_embed_backend_error_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_embed_backend_error
        todo!(
            "TODO: port action `effect_set_embed_backend_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_embed_invalid_request(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_embed_invalid_request
        todo!(
            "TODO: port action `effect_set_embed_invalid_request` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_embed_model_invalid(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_embed_model_invalid
        todo!(
            "TODO: port action `effect_set_embed_model_invalid` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_initialize_backend_error(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_initialize_backend_error
        todo!(
            "TODO: port action `effect_set_initialize_backend_error` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_set_initialize_model_invalid(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_set_initialize_model_invalid
        todo!(
            "TODO: port action `effect_set_initialize_model_invalid` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_write_embed_error_out_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_write_embed_error_out
        todo!(
            "TODO: port action `effect_write_embed_error_out` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_write_embed_error_out_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_write_embed_error_out
        todo!(
            "TODO: port action `effect_write_embed_error_out` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_write_embed_error_out_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_write_embed_error_out
        todo!(
            "TODO: port action `effect_write_embed_error_out` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn effect_write_initialize_error_out(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/actions.hpp::effect_write_initialize_error_out
        todo!(
            "TODO: port action `effect_write_initialize_error_out` from emel.cpp/src/emel/embeddings/generator/actions.hpp"
        )
    }
    fn guard_audio_encode_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_audio_encode_ready
        todo!(
            "TODO: port guard `guard_audio_encode_ready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_audio_encode_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_audio_encode_unready
        todo!(
            "TODO: port guard `guard_audio_encode_unready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_audio_prepare_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_audio_prepare_ready
        todo!(
            "TODO: port guard `guard_audio_prepare_ready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_audio_prepare_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_audio_prepare_unready
        todo!(
            "TODO: port guard `guard_audio_prepare_unready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_audio_route_unsupported(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_audio_route_unsupported
        todo!(
            "TODO: port guard `guard_audio_route_unsupported` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_failed_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_failed
        todo!(
            "TODO: port guard `guard_embedding_failed` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_failed_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_failed
        todo!(
            "TODO: port guard `guard_embedding_failed` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_failed_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_failed
        todo!(
            "TODO: port guard `guard_embedding_failed` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_full_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_full
        todo!(
            "TODO: port guard `guard_embedding_succeeded_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_full_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_full
        todo!(
            "TODO: port guard `guard_embedding_succeeded_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_full_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_full
        todo!(
            "TODO: port guard `guard_embedding_succeeded_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_truncate_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_truncate
        todo!(
            "TODO: port guard `guard_embedding_succeeded_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_truncate_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_truncate
        todo!(
            "TODO: port guard `guard_embedding_succeeded_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_embedding_succeeded_truncate_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_embedding_succeeded_truncate
        todo!(
            "TODO: port guard `guard_embedding_succeeded_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_done_callback_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_done_callback
        todo!(
            "TODO: port guard `guard_has_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_done_callback_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_done_callback
        todo!(
            "TODO: port guard `guard_has_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_done_callback_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_done_callback
        todo!(
            "TODO: port guard `guard_has_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_error_callback_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_error_callback
        todo!(
            "TODO: port guard `guard_has_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_error_callback_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_error_callback
        todo!(
            "TODO: port guard `guard_has_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_embed_error_callback_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_embed_error_callback
        todo!(
            "TODO: port guard `guard_has_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_initialize_done_callback
        todo!(
            "TODO: port guard `guard_has_initialize_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_has_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_has_initialize_error_callback
        todo!(
            "TODO: port guard `guard_has_initialize_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_image_encode_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_image_encode_ready
        todo!(
            "TODO: port guard `guard_image_encode_ready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_image_encode_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_image_encode_unready
        todo!(
            "TODO: port guard `guard_image_encode_unready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_image_prepare_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_image_prepare_ready
        todo!(
            "TODO: port guard `guard_image_prepare_ready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_image_prepare_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_image_prepare_unready
        todo!(
            "TODO: port guard `guard_image_prepare_unready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_image_route_unsupported(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_image_route_unsupported
        todo!(
            "TODO: port guard `guard_image_route_unsupported` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_initialize_backend_error(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_initialize_backend_error
        todo!(
            "TODO: port guard `guard_initialize_backend_error` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_initialize_model_invalid(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_initialize_model_invalid
        todo!(
            "TODO: port guard `guard_initialize_model_invalid` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_initialize_success(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_initialize_success
        todo!(
            "TODO: port guard `guard_initialize_success` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_invalid_embed(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_invalid_embed
        todo!(
            "TODO: port guard `guard_invalid_embed` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_invalid_embed_audio(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_invalid_embed_audio
        todo!(
            "TODO: port guard `guard_invalid_embed_audio` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_invalid_embed_image(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_invalid_embed_image
        todo!(
            "TODO: port guard `guard_invalid_embed_image` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_invalid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_invalid_initialize
        todo!(
            "TODO: port guard `guard_invalid_initialize` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_done_callback_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_done_callback
        todo!(
            "TODO: port guard `guard_no_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_done_callback_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_done_callback
        todo!(
            "TODO: port guard `guard_no_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_done_callback_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_done_callback
        todo!(
            "TODO: port guard `guard_no_embed_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_error_callback_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_error_callback
        todo!(
            "TODO: port guard `guard_no_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_error_callback_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_error_callback
        todo!(
            "TODO: port guard `guard_no_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_embed_error_callback_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_embed_error_callback
        todo!(
            "TODO: port guard `guard_no_embed_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_initialize_done_callback
        todo!(
            "TODO: port guard `guard_no_initialize_done_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_no_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_no_initialize_error_callback
        todo!(
            "TODO: port guard `guard_no_initialize_error_callback` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_prepare_backend_error(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_prepare_backend_error
        todo!(
            "TODO: port guard `guard_prepare_backend_error` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_prepare_invalid_request(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_prepare_invalid_request
        todo!(
            "TODO: port guard `guard_prepare_invalid_request` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_prepare_model_invalid(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_prepare_model_invalid
        todo!(
            "TODO: port guard `guard_prepare_model_invalid` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_prepare_success(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_prepare_success
        todo!(
            "TODO: port guard `guard_prepare_success` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_text_encode_ready(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_text_encode_ready
        todo!(
            "TODO: port guard `guard_text_encode_ready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_text_encode_unready(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_text_encode_unready
        todo!(
            "TODO: port guard `guard_text_encode_unready` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_text_route_unsupported(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_text_route_unsupported
        todo!(
            "TODO: port guard `guard_text_route_unsupported` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_audio_full(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_audio_full
        todo!(
            "TODO: port guard `guard_valid_embed_audio_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_audio_truncate(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_audio_truncate
        todo!(
            "TODO: port guard `guard_valid_embed_audio_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_full(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_full
        todo!(
            "TODO: port guard `guard_valid_embed_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_image_full(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_image_full
        todo!(
            "TODO: port guard `guard_valid_embed_image_full` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_image_truncate(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_image_truncate
        todo!(
            "TODO: port guard `guard_valid_embed_image_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_embed_truncate(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_embed_truncate
        todo!(
            "TODO: port guard `guard_valid_embed_truncate` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
    fn guard_valid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/guards.hpp::guard_valid_initialize
        todo!(
            "TODO: port guard `guard_valid_initialize` from emel.cpp/src/emel/embeddings/generator/guards.hpp"
        )
    }
}
