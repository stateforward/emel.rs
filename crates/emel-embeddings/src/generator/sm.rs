//! Source-aligned bounded EmbeddingsGenerator actor.
#![allow(dead_code, missing_docs, unused_imports, clippy::too_many_arguments, clippy::type_complexity)]

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
pub enum EmbeddingsGeneratorError { None = 0, InvalidRequest = 1, ModelInvalid = 2, Backend = 4 }

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextRouteKind { None, Encoder }
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageRouteKind { None, Encoder }
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioRouteKind { None, Encoder }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BenchmarkStageTimings { pub prepare_ns: u64, pub encode_ns: u64, pub publish_ns: u64, pub total_ns: u64 }
pub type BenchmarkNowFn = fn() -> u64;
pub type InitializeBindFn = fn(&EventInitializeRun) -> bool;
pub type TextEncodeFn = fn(&EventEmbedTextRun, &mut [f32]) -> bool;
pub type ImagePrepareFn = fn(&EventEmbedImageRun, &mut [f32]) -> bool;
pub type ImageEncodeFn = fn(&EventEmbedImageRun, &mut [f32]) -> bool;
pub type AudioPrepareFn = fn(&EventEmbedAudioRun, &mut [f32]) -> bool;
pub type AudioEncodeFn = fn(&EventEmbedAudioRun, &mut [f32]) -> bool;
pub type InitDoneFn = fn();
pub type InitErrorFn = fn(EmbeddingsGeneratorError);
pub type EmbedDoneFn = fn(&[f32], usize);
pub type EmbedErrorFn = fn(EmbeddingsGeneratorError);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CopiedMessage { pub bytes: [u8; MAX_MESSAGE_BYTES], pub len: usize }
impl CopiedMessage { pub fn new(input: &[u8]) -> Self { let mut out = Self::default(); out.len = input.len().min(MAX_MESSAGE_BYTES); out.bytes[..out.len].copy_from_slice(&input[..out.len]); out } pub fn as_bytes(&self) -> &[u8] { &self.bytes[..self.len] } }

#[derive(Clone, Copy, Debug)]
pub struct EventInitializeRun { pub tokenizer_sm: usize, pub preprocessor_variant: u8, pub encoder_variant: u8, pub add_special: bool, pub parse_special: bool, pub bind: Option<InitializeBindFn>, pub on_done: Option<InitDoneFn>, pub on_error: Option<InitErrorFn> }
impl Default for EventInitializeRun { fn default() -> Self { Self { tokenizer_sm: 0, preprocessor_variant: 0, encoder_variant: 0, add_special: true, parse_special: false, bind: None, on_done: None, on_error: None } } }
impl EventInitializeRun { pub fn new(tokenizer_sm: usize, bind: Option<InitializeBindFn>) -> Self { Self { tokenizer_sm, bind, ..Self::default() } } }

#[derive(Clone, Copy, Debug)]
pub struct EventEmbedTextRun { pub messages: [CopiedMessage; MAX_MESSAGES], pub message_count: usize, pub add_generation_prompt: bool, pub enable_thinking: bool, pub truncate_dimension: usize, pub output_capacity: usize, pub encode: Option<TextEncodeFn>, pub on_done: Option<EmbedDoneFn>, pub on_error: Option<EmbedErrorFn>, pub benchmark_now: BenchmarkNowFn, pub benchmark_timings: BenchmarkStageTimings }
impl Default for EventEmbedTextRun { fn default() -> Self { Self { messages: [CopiedMessage::default(); MAX_MESSAGES], message_count: 0, add_generation_prompt: false, enable_thinking: false, truncate_dimension: 0, output_capacity: 0, encode: None, on_done: None, on_error: None, benchmark_now: || 0, benchmark_timings: BenchmarkStageTimings::default() } } }
impl EventEmbedTextRun { pub fn new(text: &[u8], output_capacity: usize) -> Self { let mut out = Self::default(); out.messages[0] = CopiedMessage::new(text); out.message_count = usize::from(!text.is_empty()); out.output_capacity = output_capacity; out } fn has_messages(&self) -> bool { self.message_count > 0 } }

#[derive(Clone, Copy, Debug)]
pub struct EventEmbedImageRun { pub rgba: [u8; MAX_RGBA_BYTES], pub rgba_len: usize, pub width: i32, pub height: i32, pub truncate_dimension: usize, pub output_capacity: usize, pub prepare: Option<ImagePrepareFn>, pub encode: Option<ImageEncodeFn>, pub on_done: Option<EmbedDoneFn>, pub on_error: Option<EmbedErrorFn>, pub benchmark_now: BenchmarkNowFn, pub benchmark_timings: BenchmarkStageTimings }
impl Default for EventEmbedImageRun { fn default() -> Self { Self { rgba: [0; MAX_RGBA_BYTES], rgba_len: 0, width: 0, height: 0, truncate_dimension: 0, output_capacity: 0, prepare: None, encode: None, on_done: None, on_error: None, benchmark_now: || 0, benchmark_timings: BenchmarkStageTimings::default() } } }

#[derive(Clone, Copy, Debug)]
pub struct EventEmbedAudioRun { pub pcm: [f32; AUDIO_SAMPLE_COUNT], pub pcm_len: usize, pub sample_rate: i32, pub truncate_dimension: usize, pub output_capacity: usize, pub prepare: Option<AudioPrepareFn>, pub encode: Option<AudioEncodeFn>, pub on_done: Option<EmbedDoneFn>, pub on_error: Option<EmbedErrorFn>, pub benchmark_now: BenchmarkNowFn, pub benchmark_timings: BenchmarkStageTimings }
impl Default for EventEmbedAudioRun { fn default() -> Self { Self { pcm: [0.0; AUDIO_SAMPLE_COUNT], pcm_len: 0, sample_rate: 0, truncate_dimension: 0, output_capacity: 0, prepare: None, encode: None, on_done: None, on_error: None, benchmark_now: || 0, benchmark_timings: BenchmarkStageTimings::default() } } }

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
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_reject_unexpected_from_state_errored,
    }
}

#[derive(Debug)]
pub struct EmbeddingsGeneratorContext {
    pub model_ready: bool, pub conditioner_ready: bool, pub initialized: bool,
    pub text_route: TextRouteKind, pub image_route: ImageRouteKind, pub audio_route: AudioRouteKind,
    pub text_ready: bool, pub image_ready: bool, pub audio_ready: bool, pub scratch_ready: bool,
    pub embedding_length: usize, pub image_encoder_length: usize, pub audio_encoder_length: usize, pub max_positions: usize,
    pub matryoshka_dimensions: [usize; 32], pub matryoshka_dimension_count: usize,
    pub scratch: [f32; MAX_EMBEDDING_DIMENSION], pub token_ids: [i32; MAX_TOKEN_POSITIONS],
    pub error: EmbeddingsGeneratorError, pub bind_accepted: bool, pub bind_err_code: i32, pub prepare_accepted: bool, pub prepare_err_code: i32, pub token_count: usize, pub output_dimension: usize,
}
impl Default for EmbeddingsGeneratorContext { fn default() -> Self { Self { model_ready:false, conditioner_ready:false, initialized:false, text_route:TextRouteKind::None, image_route:ImageRouteKind::None, audio_route:AudioRouteKind::None, text_ready:false, image_ready:false, audio_ready:false, scratch_ready:false, embedding_length:0, image_encoder_length:0, audio_encoder_length:0, max_positions:0, matryoshka_dimensions:[0;32], matryoshka_dimension_count:0, scratch:[0.0;MAX_EMBEDDING_DIMENSION], token_ids:[0;MAX_TOKEN_POSITIONS], error:EmbeddingsGeneratorError::None, bind_accepted:false, bind_err_code:0, prepare_accepted:false, prepare_err_code:0, token_count:0, output_dimension:0 } } }
impl EmbeddingsGeneratorContext {
    pub fn configure(&mut self, embedding_length: usize, max_positions: usize) { self.embedding_length = embedding_length.min(MAX_EMBEDDING_DIMENSION); self.max_positions = max_positions.min(MAX_TOKEN_POSITIONS); self.model_ready = self.embedding_length > 0; self.scratch_ready = true; }
    pub fn set_routes(&mut self, text: TextRouteKind, image: ImageRouteKind, audio: AudioRouteKind) { self.text_route = text; self.image_route = image; self.audio_route = audio; self.text_ready = text == TextRouteKind::Encoder; self.image_ready = image == ImageRouteKind::Encoder; self.audio_ready = audio == AudioRouteKind::Encoder; self.conditioner_ready = self.text_ready; }
    pub fn state_error(&self) -> EmbeddingsGeneratorError { self.error }
    fn reset(&mut self) { self.error = EmbeddingsGeneratorError::None; self.bind_accepted = false; self.bind_err_code = 0; self.prepare_accepted = false; self.prepare_err_code = 0; self.token_count = 0; self.output_dimension = 0; }
    fn set_error(&mut self, error: EmbeddingsGeneratorError) { self.error = error; self.output_dimension = 0; }
    fn requested_dimension(&self, dimension: usize) -> usize { if dimension == 0 { self.embedding_length } else { dimension } }
    fn valid_dim(&self, dimension: usize) -> bool { dimension > 0 && dimension <= self.embedding_length && (dimension == self.embedding_length || self.matryoshka_dimensions[..self.matryoshka_dimension_count.min(32)].contains(&dimension)) }
    fn valid_image(&self, event: &EventEmbedImageRun) -> bool { event.width > 0 && event.height > 0 && event.rgba_len == (event.width as usize).saturating_mul(event.height as usize).saturating_mul(4) && event.rgba_len <= MAX_RGBA_BYTES }
}

fn normalize(values: &mut [f32]) -> bool { let mut sum = 0.0; for value in values.iter() { sum += *value * *value; } if sum <= 0.0 { return false; } let inv = 1.0 / sum.sqrt(); for value in values { *value *= inv; } true }
impl EmbeddingsGeneratorStateMachineContext {
    fn effect_encode_text(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { let ok = _event.encode.map_or(false, |f| f(_event, &mut self.scratch[..self.embedding_length.min(MAX_EMBEDDING_DIMENSION)])); if !ok { self.error = EmbeddingsGeneratorError::Backend; } Ok(()) }
    fn effect_prepare_image(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { let ok = _event.prepare.map_or(true, |f| f(_event, &mut self.scratch)); if !ok { self.error = EmbeddingsGeneratorError::Backend; } Ok(()) }
    fn effect_encode_image(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { let ok = _event.encode.map_or(false, |f| f(_event, &mut self.scratch[..self.embedding_length.min(MAX_EMBEDDING_DIMENSION)])); if !ok { self.error = EmbeddingsGeneratorError::Backend; } Ok(()) }
    fn effect_prepare_audio(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { let ok = _event.prepare.map_or(true, |f| f(_event, &mut self.scratch)); if !ok { self.error = EmbeddingsGeneratorError::Backend; } Ok(()) }
    fn effect_encode_audio(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { let ok = _event.encode.map_or(false, |f| f(_event, &mut self.scratch[..self.embedding_length.min(MAX_EMBEDDING_DIMENSION)])); if !ok { self.error = EmbeddingsGeneratorError::Backend; } Ok(()) }
    fn effect_begin_embed_audio_from_state_done(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_audio_from_state_errored(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_audio_from_state_idle(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_image_from_state_done(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_image_from_state_errored(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_image_from_state_idle(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_text_from_state_done(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_text_from_state_errored(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_embed_text_from_state_idle(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_initialize_from_state_done(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_initialize_from_state_errored(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_initialize_from_state_idle(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_begin_initialize_from_state_uninitialized(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.reset(); Ok(()) }
    fn effect_dispatch_bind_conditioner(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.bind_accepted = _event.bind.map_or(false, |f| f(_event)); self.bind_err_code = if self.bind_accepted { 0 } else { 1 }; Ok(()) }
    fn effect_dispatch_condition_text(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.prepare_accepted = _event.has_messages() && self.text_ready; self.prepare_err_code = if self.prepare_accepted { 0 } else { 1 }; self.token_count = _event.messages[.._event.message_count.min(MAX_MESSAGES)].iter().map(|m| m.len).sum::<usize>().min(self.max_positions); Ok(()) }
    fn effect_emit_embed_done_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { if let Some(f) = _event.on_done { f(&self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)], self.output_dimension); } Ok(()) }
    fn effect_emit_embed_done_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { if let Some(f) = _event.on_done { f(&self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)], self.output_dimension); } Ok(()) }
    fn effect_emit_embed_done_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { if let Some(f) = _event.on_done { f(&self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)], self.output_dimension); } Ok(()) }
    fn effect_emit_embed_error_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { if let Some(f) = _event.on_error { f(self.error); } Ok(()) }
    fn effect_emit_embed_error_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { if let Some(f) = _event.on_error { f(self.error); } Ok(()) }
    fn effect_emit_embed_error_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { if let Some(f) = _event.on_error { f(self.error); } Ok(()) }
    fn effect_emit_initialize_done(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { if let Some(f) = _event.on_done { f(); } Ok(()) }
    fn effect_emit_initialize_error(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { if let Some(f) = _event.on_error { f(self.error); } Ok(()) }
    fn effect_mark_initialized(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.initialized = true; self.error = EmbeddingsGeneratorError::None; Ok(()) }
    fn effect_publish_full_embedding_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.output_dimension = self.embedding_length; Ok(()) }
    fn effect_publish_full_embedding_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.output_dimension = self.embedding_length; Ok(()) }
    fn effect_publish_full_embedding_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.output_dimension = self.embedding_length; Ok(()) }
    fn effect_publish_truncated_embedding_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.output_dimension = self.requested_dimension(_event.truncate_dimension); let _ = normalize(&mut self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)]); Ok(()) }
    fn effect_publish_truncated_embedding_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.output_dimension = self.requested_dimension(_event.truncate_dimension); let _ = normalize(&mut self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)]); Ok(()) }
    fn effect_publish_truncated_embedding_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.output_dimension = self.requested_dimension(_event.truncate_dimension); let _ = normalize(&mut self.scratch[..self.output_dimension.min(MAX_EMBEDDING_DIMENSION)]); Ok(()) }
    fn effect_reject_embed_audio_from_state_done(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_audio_from_state_errored(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_audio_from_state_idle(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_image_from_state_done(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_image_from_state_errored(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_image_from_state_idle(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_text_from_state_done(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_text_from_state_errored(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_embed_text_from_state_idle(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_initialize_from_state_done(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_initialize_from_state_errored(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_initialize_from_state_idle(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_initialize_from_state_uninitialized(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_audio_encoding(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_audio_preparing(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_conditioning(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_conditioning_decision(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_embed_error_channel_decision(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_embed_publish_error(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_embed_publish_success(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_embedding_decision(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_encoding(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_idle(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_image_encoding(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_image_preparing(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_initialize_decision(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_initialize_error_channel_decision(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_initialize_publish_error(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_initialize_publish_success(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_initializing(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_reject_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_set_embed_backend_error_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::Backend); Ok(()) }
    fn effect_set_embed_backend_error_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::Backend); Ok(()) }
    fn effect_set_embed_backend_error_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::Backend); Ok(()) }
    fn effect_set_embed_invalid_request(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::InvalidRequest); Ok(()) }
    fn effect_set_embed_model_invalid(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::ModelInvalid); Ok(()) }
    fn effect_set_initialize_backend_error(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::Backend); Ok(()) }
    fn effect_set_initialize_model_invalid(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { self.set_error(EmbeddingsGeneratorError::ModelInvalid); Ok(()) }
    fn effect_write_embed_error_out_event_embed_audio_run(&mut self, _event: &EventEmbedAudioRun) -> Result<(), ()> { Ok(()) }
    fn effect_write_embed_error_out_event_embed_image_run(&mut self, _event: &EventEmbedImageRun) -> Result<(), ()> { Ok(()) }
    fn effect_write_embed_error_out_event_embed_text_run(&mut self, _event: &EventEmbedTextRun) -> Result<(), ()> { Ok(()) }
    fn effect_write_initialize_error_out(&mut self, _event: &EventInitializeRun) -> Result<(), ()> { Ok(()) }
    fn guard_audio_encode_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.audio_ready && self.scratch_ready) }
    fn guard_audio_encode_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(!(self.audio_ready && self.scratch_ready)) }
    fn guard_audio_prepare_ready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.audio_route == AudioRouteKind::Encoder && self.model_ready && self.audio_ready && self.scratch_ready) }
    fn guard_audio_prepare_unready(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(!(self.audio_route == AudioRouteKind::Encoder && self.model_ready && self.audio_ready && self.scratch_ready)) }
    fn guard_audio_route_unsupported(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.audio_route != AudioRouteKind::Encoder) }
    fn guard_embedding_failed_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.error != EmbeddingsGeneratorError::None) }
    fn guard_embedding_failed_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.error != EmbeddingsGeneratorError::None) }
    fn guard_embedding_failed_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.error != EmbeddingsGeneratorError::None) }
    fn guard_embedding_succeeded_full_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) == self.embedding_length) }
    fn guard_embedding_succeeded_full_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) == self.embedding_length) }
    fn guard_embedding_succeeded_full_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) == self.embedding_length) }
    fn guard_embedding_succeeded_truncate_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) > 0 && self.requested_dimension(_event.truncate_dimension) < self.embedding_length) }
    fn guard_embedding_succeeded_truncate_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) > 0 && self.requested_dimension(_event.truncate_dimension) < self.embedding_length) }
    fn guard_embedding_succeeded_truncate_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.error == EmbeddingsGeneratorError::None && self.requested_dimension(_event.truncate_dimension) > 0 && self.requested_dimension(_event.truncate_dimension) < self.embedding_length) }
    fn guard_has_embed_done_callback_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(_event.on_done.is_some()) }
    fn guard_has_embed_done_callback_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(_event.on_done.is_some()) }
    fn guard_has_embed_done_callback_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(_event.on_done.is_some()) }
    fn guard_has_embed_error_callback_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(_event.on_error.is_some()) }
    fn guard_has_embed_error_callback_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(_event.on_error.is_some()) }
    fn guard_has_embed_error_callback_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(_event.on_error.is_some()) }
    fn guard_has_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(_event.on_done.is_some()) }
    fn guard_has_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(_event.on_error.is_some()) }
    fn guard_image_encode_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.image_ready && self.scratch_ready) }
    fn guard_image_encode_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(!(self.image_ready && self.scratch_ready)) }
    fn guard_image_prepare_ready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.image_route == ImageRouteKind::Encoder && self.model_ready && self.image_ready && self.scratch_ready) }
    fn guard_image_prepare_unready(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(!(self.image_route == ImageRouteKind::Encoder && self.model_ready && self.image_ready && self.scratch_ready)) }
    fn guard_image_route_unsupported(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.image_route != ImageRouteKind::Encoder) }
    fn guard_initialize_backend_error(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(!self.guard_initialize_success(_event)? && !self.guard_initialize_model_invalid(_event)? && self.error == EmbeddingsGeneratorError::None) }
    fn guard_initialize_model_invalid(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(!self.text_ready || self.bind_err_code == 2) }
    fn guard_initialize_success(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(self.text_ready && self.scratch_ready && self.bind_accepted && self.bind_err_code == 0 && self.error == EmbeddingsGeneratorError::None) }
    fn guard_invalid_embed(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(!self.guard_valid_embed_full(_event)? && !self.guard_valid_embed_truncate(_event)?) }
    fn guard_invalid_embed_audio(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(!self.guard_valid_embed_audio_full(_event)? && !self.guard_valid_embed_audio_truncate(_event)?) }
    fn guard_invalid_embed_image(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(!self.guard_valid_embed_image_full(_event)? && !self.guard_valid_embed_image_truncate(_event)?) }
    fn guard_invalid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(!self.guard_valid_initialize(_event)?) }
    fn guard_no_embed_done_callback_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(_event.on_done.is_none()) }
    fn guard_no_embed_done_callback_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(_event.on_done.is_none()) }
    fn guard_no_embed_done_callback_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(_event.on_done.is_none()) }
    fn guard_no_embed_error_callback_event_embed_audio_run(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(_event.on_error.is_none()) }
    fn guard_no_embed_error_callback_event_embed_image_run(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(_event.on_error.is_none()) }
    fn guard_no_embed_error_callback_event_embed_text_run(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(_event.on_error.is_none()) }
    fn guard_no_initialize_done_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(_event.on_done.is_none()) }
    fn guard_no_initialize_error_callback(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(_event.on_error.is_none()) }
    fn guard_prepare_backend_error(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(!self.guard_prepare_success(_event)? && !self.guard_prepare_invalid_request(_event)? && !self.guard_prepare_model_invalid(_event)? && self.error == EmbeddingsGeneratorError::None) }
    fn guard_prepare_invalid_request(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.prepare_err_code == 1) }
    fn guard_prepare_model_invalid(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.prepare_err_code == 2) }
    fn guard_prepare_success(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.prepare_accepted && self.prepare_err_code == 0 && self.token_count > 0 && self.error == EmbeddingsGeneratorError::None) }
    fn guard_text_encode_ready(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.text_ready && self.scratch_ready && self.token_count > 0 && self.token_count <= self.max_positions) }
    fn guard_text_encode_unready(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(!(self.text_ready && self.scratch_ready && self.token_count > 0 && self.token_count <= self.max_positions)) }
    fn guard_text_route_unsupported(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.text_route != TextRouteKind::Encoder) }
    fn guard_valid_embed_audio_full(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.initialized && self.audio_ready && _event.sample_rate == 16000 && _event.pcm_len == AUDIO_SAMPLE_COUNT && _event.output_capacity >= self.embedding_length && (_event.truncate_dimension == 0 || _event.truncate_dimension == self.embedding_length)) }
    fn guard_valid_embed_audio_truncate(&self, _event: &EventEmbedAudioRun) -> Result<bool, ()> { Ok(self.initialized && self.audio_ready && _event.sample_rate == 16000 && _event.pcm_len == AUDIO_SAMPLE_COUNT && self.valid_dim(self.requested_dimension(_event.truncate_dimension)) && _event.truncate_dimension != 0 && _event.truncate_dimension != self.embedding_length && _event.output_capacity >= self.requested_dimension(_event.truncate_dimension)) }
    fn guard_valid_embed_full(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.initialized && _event.has_messages() && _event.output_capacity >= self.embedding_length && (_event.truncate_dimension == 0 || _event.truncate_dimension == self.embedding_length)) }
    fn guard_valid_embed_image_full(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.initialized && self.image_ready && self.valid_image(_event) && _event.output_capacity >= self.embedding_length && (_event.truncate_dimension == 0 || _event.truncate_dimension == self.embedding_length)) }
    fn guard_valid_embed_image_truncate(&self, _event: &EventEmbedImageRun) -> Result<bool, ()> { Ok(self.initialized && self.image_ready && self.valid_image(_event) && self.valid_dim(self.requested_dimension(_event.truncate_dimension)) && _event.truncate_dimension != 0 && _event.truncate_dimension != self.embedding_length && _event.output_capacity >= self.requested_dimension(_event.truncate_dimension)) }
    fn guard_valid_embed_truncate(&self, _event: &EventEmbedTextRun) -> Result<bool, ()> { Ok(self.initialized && _event.has_messages() && self.valid_dim(self.requested_dimension(_event.truncate_dimension)) && _event.truncate_dimension != 0 && _event.truncate_dimension != self.embedding_length && _event.output_capacity >= self.requested_dimension(_event.truncate_dimension)) }
    fn guard_valid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> { Ok(_event.tokenizer_sm != 0 && self.model_ready && self.conditioner_ready && _event.preprocessor_variant != 0 && _event.encoder_variant != 0) }
}

/// Synchronous bounded actor wrapper.
pub struct EmbeddingsGeneratorActor { machine: EmbeddingsGeneratorStateMachine }
impl Default for EmbeddingsGeneratorActor { fn default() -> Self { Self::new() } }
impl EmbeddingsGeneratorActor {
    pub fn new() -> Self { Self { machine: EmbeddingsGeneratorStateMachine::new(EmbeddingsGeneratorContext::default()) } }
    pub fn process_initialize(&mut self, event: EventInitializeRun) -> Result<(), ()> { self.machine.process_event(EmbeddingsGeneratorEvents::EventInitializeRun(event)).map(|_| ()) }
    pub fn process_text(&mut self, event: EventEmbedTextRun) -> Result<(), ()> { self.machine.process_event(EmbeddingsGeneratorEvents::EventEmbedTextRun(event)).map(|_| ()) }
    pub fn process_image(&mut self, event: EventEmbedImageRun) -> Result<(), ()> { self.machine.process_event(EmbeddingsGeneratorEvents::EventEmbedImageRun(event)).map(|_| ()) }
    pub fn process_audio(&mut self, event: EventEmbedAudioRun) -> Result<(), ()> { self.machine.process_event(EmbeddingsGeneratorEvents::EventEmbedAudioRun(event)).map(|_| ()) }
    pub fn context(&self) -> &EmbeddingsGeneratorContext { self.machine.context() }
    pub fn context_mut(&mut self) -> &mut EmbeddingsGeneratorContext { self.machine.context_mut() }
    pub fn state(&self) -> &EmbeddingsGeneratorStates { self.machine.state() }
    pub fn is(&self, state: &EmbeddingsGeneratorStates) -> bool { self.machine.is(state) }
}
