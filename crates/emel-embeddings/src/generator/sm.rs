//! Source-aligned bounded EmbeddingsGenerator actor.
#![allow(
    dead_code,
    missing_docs,
    unused_imports,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::struct_excessive_bools,
    clippy::struct_field_names,
    clippy::missing_const_for_fn,
    reason = "SML-generated names and fixed-capacity event/context layouts are contract-stable and allocation-free"
)]

use emel_text::{ChatMessage, Conditioner, ConditionerError, ConditioningDone};
use sml::sml;

/// Maximum copied text messages retained by one dispatch.
pub const MAX_MESSAGES: usize = 16;
/// Maximum copied bytes retained for one message.
pub const MAX_MESSAGE_BYTES: usize = 4096;
/// Maximum copied image payload.
pub const MAX_RGBA_BYTES: usize = 1024 * 1024;
/// Fixed audio input required by the source contract.
pub const AUDIO_SAMPLE_COUNT: usize = 4000;
/// Maximum embedding scratch dimension.
pub const MAX_EMBEDDING_DIMENSION: usize = 4096;
/// Maximum text token scratch capacity.
pub const MAX_TOKEN_POSITIONS: usize = 4096;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmbeddingsGeneratorStatus {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ModelInvalid = 2,
    Backend = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextRouteKind {
    #[default]
    None,
    Encoder,
}
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageRouteKind {
    #[default]
    None,
    Encoder,
}
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioRouteKind {
    #[default]
    None,
    Encoder,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BenchmarkStageTimings {
    pub prepare_ns: u64,
    pub encode_ns: u64,
    pub publish_ns: u64,
    pub total_ns: u64,
}
pub type BenchmarkNowFn = fn() -> u64;
pub type InitializeBindFn = fn(&EventInitializeRun) -> bool;
pub type TextPrepareFn = fn(&EventEmbedTextRun, &mut [i32], &mut usize) -> TextPrepareResult;
pub type TextEncodeFn = fn(&EventEmbedTextRun, &mut [f32]) -> bool;
pub type ImagePrepareFn = fn(&EventEmbedImageRun, &mut [f32]) -> bool;
pub type ImageEncodeFn = fn(&EventEmbedImageRun, &mut [f32]) -> bool;
pub type AudioPrepareFn = fn(&EventEmbedAudioRun, &mut [f32]) -> bool;
pub type AudioEncodeFn = fn(&EventEmbedAudioRun, &mut [f32]) -> bool;
pub type InitDoneFn = fn();
pub type InitErrorFn = fn(EmbeddingsGeneratorStatus);
pub type EmbedPublishFn = fn(&[f32], usize);
pub type EmbedDoneFn = fn(&[f32], usize);
pub type EmbedErrorFn = fn(EmbeddingsGeneratorStatus);
pub type ErrorOutFn = fn(EmbeddingsGeneratorStatus);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextPrepareResult {
    Success,
    InvalidRequest,
    ModelInvalid,
    Backend,
}

#[derive(Clone, Copy, Debug)]
pub struct CopiedMessage {
    pub bytes: [u8; MAX_MESSAGE_BYTES],
    pub len: usize,
    /// Set when the caller supplied more bytes than the bounded request arena.
    ///
    /// The source contract rejects an over-capacity request; it must not silently
    /// embed a truncated message.
    pub oversized: bool,
}
impl Default for CopiedMessage {
    fn default() -> Self {
        Self {
            bytes: [0; MAX_MESSAGE_BYTES],
            len: 0,
            oversized: false,
        }
    }
}
impl CopiedMessage {
    pub fn new(input: &[u8]) -> Self {
        let len = input.len().min(MAX_MESSAGE_BYTES);
        let mut bytes = [0; MAX_MESSAGE_BYTES];
        bytes[..len].copy_from_slice(&input[..len]);
        Self {
            bytes,
            len,
            oversized: input.len() > MAX_MESSAGE_BYTES,
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EventInitializeRun {
    pub tokenizer_sm: usize,
    pub preprocessor_variant: u8,
    pub encoder_variant: u8,
    pub add_special: bool,
    pub parse_special: bool,
    pub bind: Option<InitializeBindFn>,
    pub on_done: Option<InitDoneFn>,
    pub on_error: Option<InitErrorFn>,
    pub error_out: Option<ErrorOutFn>,
}
impl Default for EventInitializeRun {
    fn default() -> Self {
        Self {
            tokenizer_sm: 0,
            preprocessor_variant: 0,
            encoder_variant: 0,
            add_special: true,
            parse_special: false,
            bind: None,
            on_done: None,
            on_error: None,
            error_out: None,
        }
    }
}
impl EventInitializeRun {
    pub fn new(tokenizer_sm: usize, bind: Option<InitializeBindFn>) -> Self {
        Self {
            tokenizer_sm,
            bind,
            ..Self::default()
        }
    }
}
#[derive(Clone, Debug)]
pub struct EventEmbedTextRun {
    pub messages: [CopiedMessage; MAX_MESSAGES],
    pub message_count: usize,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub truncate_dimension: usize,
    pub output_capacity: usize,
    pub prepare: Option<TextPrepareFn>,
    pub encode: Option<TextEncodeFn>,
    pub publish: Option<EmbedPublishFn>,
    pub on_done: Option<EmbedDoneFn>,
    pub on_error: Option<EmbedErrorFn>,
    pub error_out: Option<ErrorOutFn>,
    pub benchmark_now: BenchmarkNowFn,
    pub benchmark_timings: BenchmarkStageTimings,
}
impl Default for EventEmbedTextRun {
    fn default() -> Self {
        Self {
            messages: [CopiedMessage::default(); MAX_MESSAGES],
            message_count: 0,
            add_generation_prompt: false,
            enable_thinking: false,
            truncate_dimension: 0,
            output_capacity: 0,
            prepare: None,
            encode: None,
            publish: None,
            on_done: None,
            on_error: None,
            error_out: None,
            benchmark_now: || 0,
            benchmark_timings: BenchmarkStageTimings::default(),
        }
    }
}
impl EventEmbedTextRun {
    pub fn new(text: &[u8], output_capacity: usize) -> Self {
        let mut messages = [CopiedMessage::default(); MAX_MESSAGES];
        messages[0] = CopiedMessage::new(text);
        Self {
            messages,
            message_count: usize::from(!text.is_empty()),
            output_capacity,
            ..Self::default()
        }
    }
    fn has_messages(&self) -> bool {
        self.message_count > 0
            && self.message_count <= MAX_MESSAGES
            && self.messages[..self.message_count]
                .iter()
                .all(|m| m.len > 0 && !m.oversized)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EventEmbedImageRun {
    pub rgba: [u8; MAX_RGBA_BYTES],
    pub rgba_len: usize,
    pub width: i32,
    pub height: i32,
    pub truncate_dimension: usize,
    pub output_capacity: usize,
    pub prepare: Option<ImagePrepareFn>,
    pub encode: Option<ImageEncodeFn>,
    pub publish: Option<EmbedPublishFn>,
    pub on_done: Option<EmbedDoneFn>,
    pub on_error: Option<EmbedErrorFn>,
    pub error_out: Option<ErrorOutFn>,
    pub benchmark_now: BenchmarkNowFn,
    pub benchmark_timings: BenchmarkStageTimings,
}
impl Default for EventEmbedImageRun {
    fn default() -> Self {
        Self {
            rgba: [0; MAX_RGBA_BYTES],
            rgba_len: 0,
            width: 0,
            height: 0,
            truncate_dimension: 0,
            output_capacity: 0,
            prepare: None,
            encode: None,
            publish: None,
            on_done: None,
            on_error: None,
            error_out: None,
            benchmark_now: || 0,
            benchmark_timings: BenchmarkStageTimings::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EventEmbedAudioRun {
    pub pcm: [f32; AUDIO_SAMPLE_COUNT],
    pub pcm_len: usize,
    pub sample_rate: i32,
    pub truncate_dimension: usize,
    pub output_capacity: usize,
    pub prepare: Option<AudioPrepareFn>,
    pub encode: Option<AudioEncodeFn>,
    pub publish: Option<EmbedPublishFn>,
    pub on_done: Option<EmbedDoneFn>,
    pub on_error: Option<EmbedErrorFn>,
    pub error_out: Option<ErrorOutFn>,
    pub benchmark_now: BenchmarkNowFn,
    pub benchmark_timings: BenchmarkStageTimings,
}
impl Default for EventEmbedAudioRun {
    fn default() -> Self {
        Self {
            pcm: [0.0; AUDIO_SAMPLE_COUNT],
            pcm_len: 0,
            sample_rate: 0,
            truncate_dimension: 0,
            output_capacity: 0,
            prepare: None,
            encode: None,
            publish: None,
            on_done: None,
            on_error: None,
            error_out: None,
            benchmark_now: || 0,
            benchmark_timings: BenchmarkStageTimings::default(),
        }
    }
}

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
        "state_embed_publish_error"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_untracked_error] / effect_set_embed_backend_error_event_embed_text_run,
        "state_encoding"_s <= "state_conditioning_decision"_s + completion<EventEmbedTextRun> [guard_prepare_success],
        "state_embed_publish_error"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_route_unsupported] / effect_set_embed_backend_error_event_embed_text_run,
        "state_embed_publish_error"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_encode_unready] / effect_set_embed_backend_error_event_embed_text_run,
        "state_embedding_decision"_s <= "state_encoding"_s + completion<EventEmbedTextRun> [guard_text_encode_ready] / effect_encode_text,
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
        "state_image_encoding"_s <= "state_image_preparing"_s + completion<EventEmbedImageRun> [guard_image_prepare_ready] / effect_prepare_image,
        "state_embed_publish_error"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_route_unsupported] / effect_set_embed_backend_error_event_embed_image_run,
        "state_embed_publish_error"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_encode_unready] / effect_set_embed_backend_error_event_embed_image_run,
        "state_embedding_decision"_s <= "state_image_encoding"_s + completion<EventEmbedImageRun> [guard_image_encode_ready] / effect_encode_image,
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
        "state_audio_encoding"_s <= "state_audio_preparing"_s + completion<EventEmbedAudioRun> [guard_audio_prepare_ready] / effect_prepare_audio,
        "state_embed_publish_error"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_route_unsupported] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_embed_publish_error"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_encode_unready] / effect_set_embed_backend_error_event_embed_audio_run,
        "state_embedding_decision"_s <= "state_audio_encoding"_s + completion<EventEmbedAudioRun> [guard_audio_encode_ready] / effect_encode_audio,
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
    }
}

#[derive(Debug)]
pub struct EmbeddingsGeneratorContext {
    pub model_ready: bool,
    pub conditioner_ready: bool,
    pub initialized: bool,
    pub text_route: TextRouteKind,
    pub image_route: ImageRouteKind,
    pub audio_route: AudioRouteKind,
    pub text_ready: bool,
    pub image_ready: bool,
    pub audio_ready: bool,
    pub scratch_ready: bool,
    pub embedding_length: usize,
    pub image_encoder_length: usize,
    pub audio_encoder_length: usize,
    pub max_positions: usize,
    pub matryoshka_dimensions: [usize; 32],
    pub matryoshka_dimension_count: usize,
    pub scratch: [f32; MAX_EMBEDDING_DIMENSION],
    pub token_ids: [i32; MAX_TOKEN_POSITIONS],
    pub error: EmbeddingsGeneratorStatus,
    pub bind_accepted: bool,
    pub bind_err_code: i32,
    /// Typed conditioner completion retained until the payload-less SML completion
    /// event is consumed by guard-selected transitions.
    pub prepare_result: Result<ConditioningDone, ConditionerError>,
    /// Whether a native binding is available for the owned context.
    ///
    /// Native tensor borrows are caller-owned and are consumed by the explicit
    /// synchronous handoff rather than retained here.
    native_execution_available: bool,
    pub token_count: usize,
    pub output_dimension: usize,
    pub conditioner: Conditioner,
}
impl Default for EmbeddingsGeneratorContext {
    fn default() -> Self {
        let mut conditioner = Conditioner::default();
        conditioner.set_dependencies(emel_text::raw_formatter, unavailable_tokenizer);
        Self {
            model_ready: false,
            conditioner_ready: false,
            initialized: false,
            text_route: TextRouteKind::None,
            image_route: ImageRouteKind::None,
            audio_route: AudioRouteKind::None,
            text_ready: false,
            image_ready: false,
            audio_ready: false,
            scratch_ready: false,
            embedding_length: 0,
            image_encoder_length: 0,
            audio_encoder_length: 0,
            max_positions: 0,
            matryoshka_dimensions: [0; 32],
            matryoshka_dimension_count: 0,
            scratch: [0.0; MAX_EMBEDDING_DIMENSION],
            token_ids: [0; MAX_TOKEN_POSITIONS],
            error: EmbeddingsGeneratorStatus::None,
            bind_accepted: false,
            bind_err_code: 0,
            prepare_result: Err(ConditionerError::None),
            native_execution_available: false,
            token_count: 0,
            output_dimension: 0,
            conditioner,
        }
    }
}
impl EmbeddingsGeneratorContext {
    pub fn configure(&mut self, embedding_length: usize, max_positions: usize) {
        self.embedding_length = embedding_length.min(MAX_EMBEDDING_DIMENSION);
        self.max_positions = max_positions.min(MAX_TOKEN_POSITIONS);
        self.model_ready = embedding_length > 0 && embedding_length <= MAX_EMBEDDING_DIMENSION;
        self.scratch_ready =
            self.model_ready && max_positions > 0 && max_positions <= MAX_TOKEN_POSITIONS;
    }
    pub fn set_routes(
        &mut self,
        text: TextRouteKind,
        image: ImageRouteKind,
        audio: AudioRouteKind,
    ) {
        self.text_route = text;
        self.image_route = image;
        self.audio_route = audio;
        self.text_ready = text == TextRouteKind::Encoder;
        self.image_ready = image == ImageRouteKind::Encoder;
        self.audio_ready = audio == AudioRouteKind::Encoder;
        self.conditioner_ready = self.text_ready;
    }
    pub fn state_error(&self) -> EmbeddingsGeneratorStatus {
        self.error
    }
    /// Marks native execution unavailable for this owned generated context.
    pub fn set_native_execution_unavailable(&mut self) {
        self.native_execution_available = false;
    }
    #[must_use]
    pub fn native_execution_ready(&self) -> bool {
        self.native_execution_available
    }
    fn reset(&mut self) {
        self.error = EmbeddingsGeneratorStatus::None;
        self.bind_accepted = false;
        self.bind_err_code = 0;
        self.prepare_result = Err(ConditionerError::None);
        self.token_count = 0;
        self.output_dimension = 0;
    }
    fn set_error(&mut self, error: EmbeddingsGeneratorStatus) {
        self.error = error;
        self.output_dimension = 0;
    }
    fn requested_dimension(&self, dimension: usize) -> usize {
        if dimension == 0 {
            self.embedding_length
        } else {
            dimension
        }
    }
    fn valid_dim(&self, dimension: usize) -> bool {
        let count = self
            .matryoshka_dimension_count
            .min(self.matryoshka_dimensions.len());
        dimension > 0
            && dimension <= self.embedding_length
            && (dimension == self.embedding_length
                || self.matryoshka_dimensions[..count].contains(&dimension))
    }
    fn valid_image(event: &EventEmbedImageRun) -> bool {
        if event.width <= 0 || event.height <= 0 || event.rgba_len > MAX_RGBA_BYTES {
            return false;
        }
        usize::try_from(event.width)
            .ok()
            .and_then(|width| {
                usize::try_from(event.height)
                    .ok()
                    .map(|height| (width, height))
            })
            .and_then(|(width, height)| width.checked_mul(height))
            .and_then(|pixels| pixels.checked_mul(4))
            == Some(event.rgba_len)
    }
}

fn normalize(values: &mut [f32]) -> bool {
    if values.is_empty() || values.len() > MAX_EMBEDDING_DIMENSION {
        return false;
    }
    let mut sum = 0.0f32;
    for value in values.iter() {
        if !value.is_finite() {
            return false;
        }
        sum = (*value).mul_add(*value, sum);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return false;
    }
    let inv = 1.0 / sum.sqrt();
    for value in values {
        *value *= inv;
    }
    true
}
fn unavailable_tokenizer(
    _text: &[u8],
    _add_special: bool,
    _parse_special: bool,
    _token_ids: &mut [i32],
) -> Result<usize, ConditionerError> {
    Err(ConditionerError::Backend)
}

impl EmbeddingsGeneratorStateMachineContext for EmbeddingsGeneratorContext {
    fn effect_encode_text(&mut self, event: &EventEmbedTextRun) -> Result<(), ()> {
        self.error = if event
            .encode
            .is_some_and(|encode| encode(event, &mut self.scratch[..self.embedding_length]))
        {
            EmbeddingsGeneratorStatus::None
        } else {
            EmbeddingsGeneratorStatus::Backend
        };
        Ok(())
    }
    fn effect_prepare_image(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> {
        self.error = EmbeddingsGeneratorStatus::Backend;
        Ok(())
    }
    fn effect_encode_image(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> {
        self.error = EmbeddingsGeneratorStatus::Backend;
        Ok(())
    }
    fn effect_prepare_audio(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> {
        self.error = EmbeddingsGeneratorStatus::Backend;
        Ok(())
    }
    fn effect_begin_embed_audio_from_state_done(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_audio_from_state_errored(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_audio_from_state_idle(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_image_from_state_done(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_image_from_state_errored(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_image_from_state_idle(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_text_from_state_done(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_text_from_state_errored(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_embed_text_from_state_idle(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_initialize_from_state_done(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_initialize_from_state_errored(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_initialize_from_state_idle(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_begin_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn effect_encode_audio(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> {
        self.error = EmbeddingsGeneratorStatus::Backend;
        Ok(())
    }
    fn effect_dispatch_condition_text(&mut self, event: &EventEmbedTextRun) -> Result<(), ()> {
        self.token_count = 0;

        let mut messages = [ChatMessage {
            role: &[],
            content: &[],
        }; MAX_MESSAGES];
        for (destination, source) in messages
            .iter_mut()
            .zip(event.messages[..event.message_count].iter())
        {
            destination.content = source.as_bytes();
        }
        self.prepare_result = self
            .conditioner
            .prepare(emel_text::conditioner_event::Prepare {
                messages: &messages[..event.message_count],
                formatter_available: true,
                tokenizer_available: self.conditioner_ready,
                model_valid: self.model_ready,
                token_capacity: self.max_positions,
                token_ids: &mut self.token_ids[..self.max_positions],
                token_count: &mut self.token_count,
                add_generation_prompt: event.add_generation_prompt,
                enable_thinking: event.enable_thinking,
                add_special: true,
                parse_special: false,
            });
        Ok(())
    }
    fn effect_dispatch_bind_conditioner(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        let result = self
            .conditioner
            .process_event(emel_text::conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: self.model_ready,
            });
        self.bind_accepted = result.is_ok();
        self.conditioner_ready = self.bind_accepted;
        self.bind_err_code = match result {
            Ok(()) => 0,
            Err(ConditionerError::ModelInvalid) => 2,
            Err(_) => 3,
        };
        Ok(())
    }
    fn effect_emit_embed_done_event_embed_image_run(
        &mut self,
        event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f(
                &self.scratch[..self.output_dimension],
                self.output_dimension,
            );
        }
        Ok(())
    }
    fn effect_emit_embed_done_event_embed_text_run(
        &mut self,
        event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f(
                &self.scratch[..self.output_dimension],
                self.output_dimension,
            );
        }
        Ok(())
    }
    fn effect_emit_embed_done_event_embed_audio_run(
        &mut self,
        event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f(
                &self.scratch[..self.output_dimension],
                self.output_dimension,
            );
        }
        Ok(())
    }

    fn effect_emit_embed_error_event_embed_audio_run(
        &mut self,
        event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_embed_error_event_embed_image_run(
        &mut self,
        event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_embed_error_event_embed_text_run(
        &mut self,
        event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_emit_initialize_done(&mut self, event: &EventInitializeRun) -> Result<(), ()> {
        if let Some(f) = event.on_done {
            f();
        }
        Ok(())
    }
    fn effect_emit_initialize_error(&mut self, event: &EventInitializeRun) -> Result<(), ()> {
        if let Some(f) = event.on_error {
            f(self.error);
        }
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_mark_initialized(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.initialized = true;
        self.error = EmbeddingsGeneratorStatus::None;
        Ok(())
    }
    fn effect_publish_full_embedding_event_embed_audio_run(
        &mut self,
        event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.output_dimension = self.embedding_length;
        if let Some(f) = event.publish {
            f(
                &self.scratch[..self.embedding_length],
                self.embedding_length,
            );
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_publish_full_embedding_event_embed_image_run(
        &mut self,
        event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.output_dimension = self.embedding_length;
        if let Some(f) = event.publish {
            f(
                &self.scratch[..self.embedding_length],
                self.embedding_length,
            );
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_publish_full_embedding_event_embed_text_run(
        &mut self,
        event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.output_dimension = self.embedding_length;
        if let Some(f) = event.publish {
            f(
                &self.scratch[..self.embedding_length],
                self.embedding_length,
            );
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_publish_truncated_embedding_event_embed_audio_run(
        &mut self,
        event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        if !normalize(&mut self.scratch[..dimension]) {
            self.error = EmbeddingsGeneratorStatus::Backend;
            return Ok(());
        }
        self.output_dimension = dimension;
        if let Some(f) = event.publish {
            f(&self.scratch[..dimension], dimension);
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_publish_truncated_embedding_event_embed_image_run(
        &mut self,
        event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        if !normalize(&mut self.scratch[..dimension]) {
            self.error = EmbeddingsGeneratorStatus::Backend;
            return Ok(());
        }
        self.output_dimension = dimension;
        if let Some(f) = event.publish {
            f(&self.scratch[..dimension], dimension);
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_publish_truncated_embedding_event_embed_text_run(
        &mut self,
        event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        if !normalize(&mut self.scratch[..dimension]) {
            self.error = EmbeddingsGeneratorStatus::Backend;
            return Ok(());
        }
        self.output_dimension = dimension;
        if let Some(f) = event.publish {
            f(&self.scratch[..dimension], dimension);
        }
        if let Some(f) = event.error_out {
            f(EmbeddingsGeneratorStatus::None);
        }
        Ok(())
    }
    fn effect_reject_embed_image_from_state_errored(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_image_from_state_idle(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_text_from_state_done(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_text_from_state_errored(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_text_from_state_idle(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_initialize_from_state_done(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_initialize_from_state_errored(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_initialize_from_state_idle(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_image_from_state_done(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_initialize_publish_success(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_initialize_publish_error(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_initializing(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn guard_no_initialize_done_callback(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_no_initialize_error_callback(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn guard_no_embed_done_callback_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_no_embed_done_callback_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_no_embed_done_callback_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_no_embed_error_callback_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn guard_no_embed_error_callback_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn guard_no_embed_error_callback_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn guard_prepare_invalid_request(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(matches!(
            self.prepare_result,
            Err(ConditionerError::InvalidArgument | ConditionerError::Capacity)
        ) || matches!(self.prepare_result, Ok(outcome) if outcome.token_count == 0))
    }
    fn guard_prepare_model_invalid(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(matches!(
            self.prepare_result,
            Err(ConditionerError::ModelInvalid)
        ))
    }
    fn guard_prepare_backend_error(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(matches!(
            self.prepare_result,
            Err(ConditionerError::Backend)
        ))
    }
    fn guard_prepare_untracked_error(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(matches!(
            self.prepare_result,
            Err(ConditionerError::Untracked | ConditionerError::None)
        ))
    }
    fn guard_prepare_success(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(matches!(self.prepare_result, Ok(outcome) if outcome.token_count > 0))
    }
    fn effect_reject_unexpected_from_state_audio_encoding(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_audio_preparing(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_conditioning(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_conditioning_decision(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_audio_from_state_done(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_audio_from_state_errored(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_embed_audio_from_state_idle(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_embed_error_channel_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_embed_publish_error(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_embed_publish_success(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_embedding_decision(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_encoding(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_idle(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_image_encoding(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_image_preparing(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_initialize_decision(&mut self) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_reject_unexpected_from_state_initialize_error_channel_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_write_embed_error_out_event_embed_audio_run(
        &mut self,
        event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_write_embed_error_out_event_embed_image_run(
        &mut self,
        event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_write_embed_error_out_event_embed_text_run(
        &mut self,
        event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_write_initialize_error_out(&mut self, event: &EventInitializeRun) -> Result<(), ()> {
        if let Some(f) = event.error_out {
            f(self.error);
        }
        Ok(())
    }
    fn effect_set_embed_backend_error_event_embed_audio_run(
        &mut self,
        _event: &EventEmbedAudioRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::Backend);
        Ok(())
    }
    fn effect_set_embed_backend_error_event_embed_image_run(
        &mut self,
        _event: &EventEmbedImageRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::Backend);
        Ok(())
    }
    fn effect_set_embed_backend_error_event_embed_text_run(
        &mut self,
        _event: &EventEmbedTextRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::Backend);
        Ok(())
    }
    fn effect_set_embed_invalid_request(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::InvalidRequest);
        Ok(())
    }
    fn effect_set_embed_model_invalid(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::ModelInvalid);
        Ok(())
    }
    fn effect_set_initialize_backend_error(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::Backend);
        Ok(())
    }
    fn effect_set_initialize_model_invalid(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        self.set_error(EmbeddingsGeneratorStatus::ModelInvalid);
        Ok(())
    }
    fn guard_audio_encode_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(self.audio_ready && self.scratch_ready)
    }
    fn guard_audio_encode_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(!(self.audio_ready && self.scratch_ready))
    }
    fn guard_audio_prepare_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(self.audio_route == AudioRouteKind::Encoder
            && self.model_ready
            && self.audio_ready
            && self.scratch_ready)
    }
    fn guard_audio_prepare_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(!(self.audio_route == AudioRouteKind::Encoder
            && self.model_ready
            && self.audio_ready
            && self.scratch_ready))
    }
    fn guard_audio_route_unsupported(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(self.audio_route != AudioRouteKind::Encoder)
    }
    fn guard_embedding_failed_event_embed_audio_run(
        &self,
        _event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(self.error != EmbeddingsGeneratorStatus::None)
    }
    fn guard_embedding_failed_event_embed_image_run(
        &self,
        _event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(self.error != EmbeddingsGeneratorStatus::None)
    }
    fn guard_embedding_failed_event_embed_text_run(
        &self,
        _event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(self.error != EmbeddingsGeneratorStatus::None)
    }
    fn guard_embedding_succeeded_full_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && self.requested_dimension(event.truncate_dimension) == self.embedding_length)
    }
    fn guard_embedding_succeeded_full_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && self.requested_dimension(event.truncate_dimension) == self.embedding_length)
    }
    fn guard_embedding_succeeded_full_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && self.requested_dimension(event.truncate_dimension) == self.embedding_length)
    }
    fn guard_embedding_succeeded_truncate_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && dimension > 0
            && dimension < self.embedding_length)
    }
    fn guard_embedding_succeeded_truncate_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && dimension > 0
            && dimension < self.embedding_length)
    }
    fn guard_embedding_succeeded_truncate_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.error == EmbeddingsGeneratorStatus::None
            && dimension > 0
            && dimension < self.embedding_length)
    }
    fn guard_has_embed_done_callback_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_has_embed_done_callback_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_has_embed_done_callback_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_has_embed_error_callback_event_embed_audio_run(
        &self,
        event: &EventEmbedAudioRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_has_embed_error_callback_event_embed_image_run(
        &self,
        event: &EventEmbedImageRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_has_embed_error_callback_event_embed_text_run(
        &self,
        event: &EventEmbedTextRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_has_initialize_done_callback(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_has_initialize_error_callback(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_image_encode_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(self.image_ready && self.scratch_ready)
    }
    fn guard_image_encode_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(!(self.image_ready && self.scratch_ready))
    }
    fn guard_image_prepare_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(self.image_route == ImageRouteKind::Encoder
            && self.model_ready
            && self.image_ready
            && self.scratch_ready)
    }
    fn guard_image_prepare_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(!(self.image_route == ImageRouteKind::Encoder
            && self.model_ready
            && self.image_ready
            && self.scratch_ready))
    }
    fn guard_image_route_unsupported(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(self.image_route != ImageRouteKind::Encoder)
    }
    fn guard_initialize_backend_error(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!self.guard_initialize_success(event)?
            && !self.guard_initialize_model_invalid(event)?
            && self.error == EmbeddingsGeneratorStatus::None)
    }
    fn guard_initialize_model_invalid(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!self.text_ready || self.bind_err_code == 2)
    }
    fn guard_initialize_success(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(self.text_ready
            && self.scratch_ready
            && self.bind_accepted
            && self.bind_err_code == 0
            && self.error == EmbeddingsGeneratorStatus::None)
    }
    fn guard_invalid_embed(&self, event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(!self.guard_valid_embed_full(event)? && !self.guard_valid_embed_truncate(event)?)
    }
    fn guard_invalid_embed_audio(&self, event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(!self.guard_valid_embed_audio_full(event)?
            && !self.guard_valid_embed_audio_truncate(event)?)
    }
    fn guard_invalid_embed_image(&self, event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(!self.guard_valid_embed_image_full(event)?
            && !self.guard_valid_embed_image_truncate(event)?)
    }
    fn guard_invalid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!self.guard_valid_initialize(event)?)
    }
    fn guard_valid_embed_audio_full(&self, event: &EventEmbedAudioRun) -> Result<bool, ()> {
        Ok(self.initialized
            && self.audio_ready
            && event.publish.is_some()
            && event.sample_rate == 16000
            && event.pcm_len == AUDIO_SAMPLE_COUNT
            && event.output_capacity >= self.embedding_length
            && (event.truncate_dimension == 0 || event.truncate_dimension == self.embedding_length))
    }
    fn guard_valid_embed_audio_truncate(&self, event: &EventEmbedAudioRun) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.initialized
            && self.audio_ready
            && event.publish.is_some()
            && event.sample_rate == 16000
            && event.pcm_len == AUDIO_SAMPLE_COUNT
            && event.truncate_dimension != 0
            && event.truncate_dimension != self.embedding_length
            && self.valid_dim(dimension)
            && event.output_capacity >= dimension)
    }
    fn guard_valid_embed_full(&self, event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(self.initialized
            && self.text_ready
            && event.publish.is_some()
            && event.has_messages()
            && event.output_capacity >= self.embedding_length
            && (event.truncate_dimension == 0 || event.truncate_dimension == self.embedding_length))
    }
    fn guard_valid_embed_image_full(&self, event: &EventEmbedImageRun) -> Result<bool, ()> {
        Ok(self.initialized
            && self.image_ready
            && event.publish.is_some()
            && Self::valid_image(event)
            && event.output_capacity >= self.embedding_length
            && (event.truncate_dimension == 0 || event.truncate_dimension == self.embedding_length))
    }
    fn guard_valid_embed_image_truncate(&self, event: &EventEmbedImageRun) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.initialized
            && self.image_ready
            && event.publish.is_some()
            && Self::valid_image(event)
            && event.truncate_dimension != 0
            && event.truncate_dimension != self.embedding_length
            && self.valid_dim(dimension)
            && event.output_capacity >= dimension)
    }
    fn guard_valid_embed_truncate(&self, event: &EventEmbedTextRun) -> Result<bool, ()> {
        let dimension = self.requested_dimension(event.truncate_dimension);
        Ok(self.initialized
            && self.text_ready
            && event.publish.is_some()
            && event.has_messages()
            && event.truncate_dimension != 0
            && event.truncate_dimension != self.embedding_length
            && self.valid_dim(dimension)
            && event.output_capacity >= dimension)
    }
    fn guard_text_encode_ready(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(self.text_ready
            && self.scratch_ready
            && self.token_count > 0
            && self.token_count <= self.max_positions)
    }
    fn guard_text_encode_unready(&self, event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(!self.guard_text_encode_ready(event)?)
    }
    fn guard_text_route_unsupported(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> {
        Ok(self.text_route != TextRouteKind::Encoder)
    }
    fn guard_valid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.tokenizer_sm != 0
            && self.model_ready
            && self.conditioner_ready
            && event.preprocessor_variant != 0
            && event.encoder_variant != 0)
    }
}

/// Synchronous bounded actor wrapper.
#[allow(
    missing_debug_implementations,
    reason = "generated state-machine wrapper has no stable Debug contract"
)]
pub struct EmbeddingsGeneratorActor {
    machine: EmbeddingsGeneratorStateMachine<EmbeddingsGeneratorContext>,
}
impl Default for EmbeddingsGeneratorActor {
    fn default() -> Self {
        Self::new()
    }
}
impl EmbeddingsGeneratorActor {
    pub fn new() -> Self {
        Self {
            machine: EmbeddingsGeneratorStateMachine::new(EmbeddingsGeneratorContext::default()),
        }
    }
    /// Processes one initialization request through the generated state machine.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingsGeneratorStatus::InvalidRequest`] when the request is
    /// invalid for its fields or current machine state,
    /// [`EmbeddingsGeneratorStatus::ModelInvalid`] when model or conditioner
    /// binding validation fails, or [`EmbeddingsGeneratorStatus::Backend`] when
    /// initialization reports a backend failure.
    pub fn process_initialize(
        &mut self,
        event: EventInitializeRun,
    ) -> Result<(), EmbeddingsGeneratorStatus> {
        self.machine
            .process_event(EmbeddingsGeneratorEvents::EventInitializeRun(event))
            .map_err(|_| EmbeddingsGeneratorStatus::InvalidRequest)?;
        match self.machine.context().state_error() {
            EmbeddingsGeneratorStatus::None => Ok(()),
            error => Err(error),
        }
    }
    /// Processes one text embedding request through the generated state machine.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingsGeneratorStatus::InvalidRequest`] when the request is
    /// invalid for its fields or current machine state,
    /// [`EmbeddingsGeneratorStatus::ModelInvalid`] when text preparation reports
    /// an invalid model, or [`EmbeddingsGeneratorStatus::Backend`] when the text
    /// encoder callback, route, or embedding backend fails.
    pub fn process_text(
        &mut self,
        event: EventEmbedTextRun,
    ) -> Result<(), EmbeddingsGeneratorStatus> {
        self.machine
            .process_event(EmbeddingsGeneratorEvents::EventEmbedTextRun(event))
            .map_err(|_| EmbeddingsGeneratorStatus::InvalidRequest)?;
        match self.machine.context().state_error() {
            EmbeddingsGeneratorStatus::None => Ok(()),
            error => Err(error),
        }
    }
    /// Processes one image embedding request through the generated state machine.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingsGeneratorStatus::InvalidRequest`] when the request is
    /// invalid for its fields or current machine state, or
    /// [`EmbeddingsGeneratorStatus::Backend`] when the image route,
    /// preparation, encoding path, or embedding backend is unavailable or fails.
    #[allow(
        clippy::large_types_passed_by_value,
        reason = "this public consuming wrapper preserves fixed-capacity event ownership and allocation-free dispatch"
    )]
    pub fn process_image(
        &mut self,
        event: EventEmbedImageRun,
    ) -> Result<(), EmbeddingsGeneratorStatus> {
        self.machine
            .process_event(EmbeddingsGeneratorEvents::EventEmbedImageRun(event))
            .map_err(|_| EmbeddingsGeneratorStatus::InvalidRequest)?;
        match self.machine.context().state_error() {
            EmbeddingsGeneratorStatus::None => Ok(()),
            error => Err(error),
        }
    }
    /// Processes one audio embedding request through the generated state machine.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingsGeneratorStatus::InvalidRequest`] when the request is
    /// invalid for its fields or current machine state, or
    /// [`EmbeddingsGeneratorStatus::Backend`] when the audio route,
    /// preparation, encoding path, or embedding backend is unavailable or fails.
    #[allow(
        clippy::large_types_passed_by_value,
        reason = "this public consuming wrapper preserves fixed-capacity event ownership and allocation-free dispatch"
    )]
    pub fn process_audio(
        &mut self,
        event: EventEmbedAudioRun,
    ) -> Result<(), EmbeddingsGeneratorStatus> {
        self.machine
            .process_event(EmbeddingsGeneratorEvents::EventEmbedAudioRun(event))
            .map_err(|_| EmbeddingsGeneratorStatus::InvalidRequest)?;
        match self.machine.context().state_error() {
            EmbeddingsGeneratorStatus::None => Ok(()),
            error => Err(error),
        }
    }
    /// Executes the maintained native OmniEmbed text path synchronously.
    ///
    /// The model binding and output remain caller-owned; this actor retains no
    /// borrow and performs no deferred work.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingsGeneratorStatus::InvalidRequest`] when token or output
    /// inputs are empty, exceed the fixed capacity, or are rejected by native
    /// execution. Returns [`EmbeddingsGeneratorStatus::ModelInvalid`] when the
    /// native model tensors are unsupported, invalid, or exceed model capacity.
    pub fn process_native_text(
        &mut self,
        binding: &crate::generator::omniembed::NativeTensorBinding<'_>,
        token_ids: &[i32],
        output: &mut [f32],
    ) -> Result<usize, EmbeddingsGeneratorStatus> {
        if token_ids.is_empty() || token_ids.len() > MAX_TOKEN_POSITIONS || output.is_empty() {
            return Err(EmbeddingsGeneratorStatus::InvalidRequest);
        }
        match crate::generator::omniembed::detail::execute_text(binding, token_ids, output) {
            Ok(embedding_length) => Ok(embedding_length),
            Err(crate::generator::omniembed::detail::Error::InvalidRequest) => {
                Err(EmbeddingsGeneratorStatus::InvalidRequest)
            }
            Err(
                crate::generator::omniembed::detail::Error::UnsupportedTensor
                | crate::generator::omniembed::detail::Error::ModelInvalid
                | crate::generator::omniembed::detail::Error::Capacity,
            ) => Err(EmbeddingsGeneratorStatus::ModelInvalid),
        }
    }

    pub fn is(&self, state: &EmbeddingsGeneratorStates) -> bool {
        self.machine.is(state)
    }
}
