//! Source-aligned bounded Mimi speech codec facade.
//!
//! The facade owns lifecycle and sequencing only. Binding metadata and arena
//! capacities are validated before the generated machine enters its bind
//! states. Frame data, streaming state, and numeric stage contracts remain
//! caller-owned and are dispatched synchronously through the child actors.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    dead_code,
    missing_docs,
    private_interfaces
)]

use core::fmt;
use sml::sml;
use super::{binding, decoder, encoder, quantizer};
use binding::PreparedMimiBinding;

const MAX_LATENT_FLOATS: usize = 512;

/// Top-level Mimi errors corresponding to `mimi::error`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MimiError {
    /// No error was selected.
    #[default]
    None = 0,
    /// A frame operation was requested before initialization.
    NotInitialized = 1,
    /// The prepared model binding does not satisfy the Mimi contract.
    BindFailed = 2,
    /// A caller-owned arena is smaller than the binding contract.
    ArenaCapacity = 3,
    /// A frame request has invalid shape or capacity.
    RequestShape = 4,
    /// A decode code is outside the bound codebook range.
    CodeRange = 5,
    /// The event was not modeled in the current state.
    UnexpectedEvent = 6,
}

impl fmt::Display for MimiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "no error",
            Self::NotInitialized => "Mimi codec is not initialized",
            Self::BindFailed => "Mimi codec binding failed",
            Self::ArenaCapacity => "Mimi codec arena capacity is invalid",
            Self::RequestShape => "Mimi codec request shape is invalid",
            Self::CodeRange => "Mimi codec code is out of range",
            Self::UnexpectedEvent => "unexpected Mimi codec event",
        })
    }
}
impl std::error::Error for MimiError {}

/// Bounded initialize completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeDone { pub frame_samples: u32, pub n_q: u32 }
/// Bounded initialize failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeError { pub error: MimiError }
/// Bounded encode completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeDone { pub n_q: u32 }
/// Bounded encode failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeError { pub error: MimiError }
/// Bounded decode completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeDone { pub frame_samples: u32 }
/// Bounded decode failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError { pub error: MimiError }

/// Synchronous initialize completion callback.
pub type InitializeDoneCallback = fn(InitializeDone) -> bool;
/// Synchronous initialize failure callback.
pub type InitializeErrorCallback = fn(InitializeError) -> bool;
/// Synchronous encode completion callback.
pub type EncodeDoneCallback = fn(EncodeDone) -> bool;
/// Synchronous encode failure callback.
pub type EncodeErrorCallback = fn(EncodeError) -> bool;
/// Synchronous decode completion callback.
pub type DecodeDoneCallback = fn(DecodeDone) -> bool;
/// Synchronous decode failure callback.
pub type DecodeErrorCallback = fn(DecodeError) -> bool;

/// One-time binding request. All four arenas remain caller-owned.
#[derive(Debug)]
pub struct InitRun<'a> {
    /// Prepared immutable binding produced by the public binding factory.
    pub binding: PreparedMimiBinding<'a>,
    /// Prepared-weight arena supplied by the caller.
    pub prepared: &'a mut [f32],
    /// Persistent streaming-state arena supplied by the caller.
    pub state_arena: &'a mut [f32],
    /// Per-dispatch workspace arena supplied by the caller.
    pub workspace: &'a mut [f32],
    /// One-frame staging arena supplied by the caller.
    pub frame: &'a mut [f32],
    /// Optional synchronous error destination.
    pub error_out: Option<&'a mut MimiError>,
    /// Optional synchronous success callback.
    pub on_done: Option<InitializeDoneCallback>,
    /// Optional synchronous failure callback.
    pub on_error: Option<InitializeErrorCallback>,
}
impl<'a> InitRun<'a> {
    /// Creates a binding request over caller-owned arenas.
    #[must_use]
    pub const fn new(binding: PreparedMimiBinding<'a>, prepared: &'a mut [f32], state_arena: &'a mut [f32], workspace: &'a mut [f32], frame: &'a mut [f32]) -> Self {
        Self { binding, prepared, state_arena, workspace, frame, error_out: None, on_done: None, on_error: None }
    }
    /// Installs an optional error destination.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self { self.error_out = Some(error_out); self }
    /// Installs optional synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(mut self, on_done: Option<InitializeDoneCallback>, on_error: Option<InitializeErrorCallback>) -> Self { self.on_done = on_done; self.on_error = on_error; self }
}

/// One caller-owned PCM-to-code frame request.
#[derive(Debug)]
pub struct EncodeRun<'a> {
    /// Encoder runtime contract and stage callbacks.
    pub encoder_runtime: &'a encoder::EncoderRuntime,
    /// Quantizer runtime contract and stage callbacks.
    pub quantizer_runtime: &'a quantizer::QuantizerRuntime,
    /// Caller-owned encoder streaming state.
    pub streaming: &'a mut encoder::EncoderStreamingState<'a>,
    /// Mono PCM input frame.
    pub pcm: &'a [f32],
    /// Caller-owned frame staging arena.
    pub frame: &'a mut [f32],
    /// Caller-owned workspace arena.
    pub workspace: &'a mut [f32],
    /// Caller-owned destination code values.
    pub codes_out: &'a mut [i32],
    /// Optional synchronous error destination.
    pub error_out: Option<&'a mut MimiError>,
    /// Optional synchronous success callback.
    pub on_done: Option<EncodeDoneCallback>,
    /// Optional synchronous failure callback.
    pub on_error: Option<EncodeErrorCallback>,
}
impl<'a> EncodeRun<'a> {
    /// Creates one encode request over caller-owned buffers.
    #[must_use]
    pub const fn new(encoder_runtime: &'a encoder::EncoderRuntime, quantizer_runtime: &'a quantizer::QuantizerRuntime, streaming: &'a mut encoder::EncoderStreamingState<'a>, pcm: &'a [f32], frame: &'a mut [f32], workspace: &'a mut [f32], codes_out: &'a mut [i32]) -> Self {
        Self { encoder_runtime, quantizer_runtime, streaming, pcm, frame, workspace, codes_out, error_out: None, on_done: None, on_error: None }
    }
    /// Installs an optional error destination.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self { self.error_out = Some(error_out); self }
    /// Installs optional synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(mut self, on_done: Option<EncodeDoneCallback>, on_error: Option<EncodeErrorCallback>) -> Self { self.on_done = on_done; self.on_error = on_error; self }
}

/// One caller-owned code-to-PCM frame request.
#[derive(Debug)]
pub struct DecodeRun<'a> {
    /// Decoder runtime contract and stage callbacks.
    pub runtime: binding::CodecRuntime<'a>,
    /// Caller-owned decoder streaming state.
    pub streaming: &'a mut decoder::CodecStreamingState<'a>,
    /// Code prefix, from semantic through the requested active levels.
    pub codes: &'a [i32],
    /// Caller-owned decoder frame staging arena.
    pub frame: &'a mut [f32],
    /// Caller-owned workspace arena.
    pub workspace: &'a mut [f32],
    /// Caller-owned PCM destination frame.
    pub pcm_out: &'a mut [f32],
    /// Optional synchronous error destination.
    pub error_out: Option<&'a mut MimiError>,
    /// Optional synchronous success callback.
    pub on_done: Option<DecodeDoneCallback>,
    /// Optional synchronous failure callback.
    pub on_error: Option<DecodeErrorCallback>,
}
impl<'a> DecodeRun<'a> {
    /// Creates one decode request over caller-owned buffers.
    #[must_use]
    pub const fn new(runtime: binding::CodecRuntime<'a>, streaming: &'a mut decoder::CodecStreamingState<'a>, codes: &'a [i32], frame: &'a mut [f32], workspace: &'a mut [f32], pcm_out: &'a mut [f32]) -> Self {
        Self { runtime, streaming, codes, frame, workspace, pcm_out, error_out: None, on_done: None, on_error: None }
    }
    /// Installs an optional error destination.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self { self.error_out = Some(error_out); self }
    /// Installs optional synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(mut self, on_done: Option<DecodeDoneCallback>, on_error: Option<DecodeErrorCallback>) -> Self { self.on_done = on_done; self.on_error = on_error; self }
}

/// Caller-owned streaming state reset request.
#[derive(Debug)]
pub struct EventResetStreamRun<'a> {
    /// Encoder state to rewind.
    pub encoder: &'a mut encoder::EncoderStreamingState<'a>,
    /// Decoder state to rewind.
    pub decoder: &'a mut decoder::CodecStreamingState<'a>,
}
impl<'a> EventResetStreamRun<'a> {
    /// Creates a reset request over both direction states.
    #[must_use]
    pub const fn new(encoder: &'a mut encoder::EncoderStreamingState<'a>, decoder: &'a mut decoder::CodecStreamingState<'a>) -> Self { Self { encoder, decoder } }
}
/// Public top-level event accepted by the synchronous facade.
pub enum MimiEvent<'a> {
    /// Bind the prepared model and caller-owned arenas.
    Init(InitRun<'a>),
    /// Encode one PCM frame.
    Encode(EncodeRun<'a>),
    /// Decode one code prefix into one PCM frame.
    Decode(DecodeRun<'a>),
    /// Reset both streaming directions.
    Reset(EventResetStreamRun<'a>),
}

sml! {
    SpeechCodecMimi {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + InitRun(&'dispatch InitRun<'dispatch>),
        "state_bind_capacity_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_bind_contract_valid],
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_binding"_s <= "state_bind_capacity_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_arena_capacity_valid] / effect_bind,
        "state_init_failed_error_out_decision"_s <= "state_bind_capacity_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_arena_capacity_invalid] / effect_mark_arena_capacity_invalid,
        "state_init_error_out_decision"_s <= "state_binding"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>),
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_has_error_out_init_run] / effect_store_error_out_init_run,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_no_error_out_init_run],
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_has_error_out_init_run] / effect_store_error_out_init_run,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_no_error_out_init_run],
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_has_done_callback_init_run] / effect_emit_initialize_done,
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_no_done_callback_init_run],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_has_error_callback_init_run] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun>(&'dispatch InitRun<'dispatch>) [guard_no_error_callback_init_run],
        "state_encode_request_decision"_s <= "state_session_ready"_s + EncodeRun(&'dispatch EncodeRun<'dispatch>),
        "state_encoding"_s <= "state_encode_request_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_encode_request_valid] / effect_run_frontend_child,
        "state_encode_failed_error_out_decision"_s <= "state_encode_request_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_encode_request_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_quantizing"_s <= "state_encoding"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) / effect_run_quantize_child,
        "state_encode_error_out_decision"_s <= "state_quantizing"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>),
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_out_encode_run],
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_out_encode_run],
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_done_callback_encode_run],
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_callback_encode_run],
        "state_decode_request_decision"_s <= "state_session_ready"_s + DecodeRun(&'dispatch DecodeRun<'dispatch>),
        "state_decode_codes_decision"_s <= "state_decode_request_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_decode_request_valid],
        "state_decode_failed_error_out_decision"_s <= "state_decode_request_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_decode_request_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_dequantizing"_s <= "state_decode_codes_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_decode_codes_valid] / effect_run_dequantize_child,
        "state_decode_failed_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decoding"_s <= "state_dequantizing"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) / effect_run_backend_child,
        "state_decode_error_out_decision"_s <= "state_decoding"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>),
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_out_decode_run],
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_out_decode_run],
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_done_callback_decode_run],
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_callback_decode_run],
        "state_session_ready"_s <= "state_session_ready"_s + EventResetStreamRun(&'dispatch EventResetStreamRun<'dispatch>) / effect_reset_stream,
        "state_uninit_encode_error_out_decision"_s <= "state_uninitialized"_s + EncodeRun(&'dispatch EncodeRun<'dispatch>) / effect_mark_not_initialized_encode_run,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_out_encode_run],
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_callback_encode_run],
        "state_uninit_decode_error_out_decision"_s <= "state_uninitialized"_s + DecodeRun(&'dispatch DecodeRun<'dispatch>) / effect_mark_not_initialized_decode_run,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_out_decode_run],
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_callback_decode_run],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_mark_unexpected,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> / effect_mark_unexpected,
    }
}

/// Persistent top-level control-plane state.
#[derive(Debug)]
pub struct SpeechCodecMimiContext {
    initialized: bool,
    frame_samples: u32,
    n_q: u32,
    dim: usize,
    card: usize,
    semantic_n_q: usize,
    error: MimiError,
    latent: [f32; MAX_LATENT_FLOATS],
    frontend: encoder::SpeechCodecMimiEncoder,
    quantizer: quantizer::SpeechCodecMimiQuantizer,
    backend: decoder::SpeechCodecMimiDecoder,
}
impl Default for SpeechCodecMimiContext {
    fn default() -> Self {
        Self { initialized: false, frame_samples: 0, n_q: 0, dim: 0, card: 0, semantic_n_q: 0, error: MimiError::None, latent: [0.0; MAX_LATENT_FLOATS], frontend: encoder::SpeechCodecMimiEncoder::new(), quantizer: quantizer::SpeechCodecMimiQuantizer::new(), backend: decoder::SpeechCodecMimiDecoder::new() }
    }
}
impl SpeechCodecMimiContext {
    fn mark(&mut self, error: MimiError) -> Result<(), ()> { self.error = error; Ok(()) }
    fn clear(&mut self) { self.error = MimiError::None; }
}

impl SpeechCodecMimiStateMachineContext for SpeechCodecMimiContext {
    fn effect_bind(&mut self, event: &InitRun<'_>) -> Result<(), ()> {
        let runtime = event.binding.codec_runtime();
        let h = runtime.model().hparams();
        self.initialized = true;
        self.frame_samples = runtime.frame_samples();
        self.n_q = runtime.n_q();
        self.dim = usize::try_from(h.dim()).map_err(|_| ())?;
        self.card = usize::try_from(h.card()).map_err(|_| ())?;
        self.semantic_n_q = usize::try_from(h.semantic_n_q()).map_err(|_| ())?;
        self.error = MimiError::None;
        Ok(())
    }
    fn effect_emit_initialize_done(&mut self, event: &InitRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { callback(InitializeDone { frame_samples: self.frame_samples, n_q: self.n_q }); } Ok(()) }
    fn effect_emit_initialize_error(&mut self, event: &InitRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { callback(InitializeError { error: self.error }); } Ok(()) }
    fn effect_emit_encode_done(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { callback(EncodeDone { n_q: self.n_q }); } Ok(()) }
    fn effect_emit_encode_error(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { callback(EncodeError { error: self.error }); } Ok(()) }
    fn effect_emit_decode_done(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { callback(DecodeDone { frame_samples: self.frame_samples }); } Ok(()) }
    fn effect_emit_decode_error(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { callback(DecodeError { error: self.error }); } Ok(()) }
    fn effect_mark_bind_failed(&mut self, _: &InitRun<'_>) -> Result<(), ()> { self.mark(MimiError::BindFailed) }
    fn effect_mark_arena_capacity_invalid(&mut self, _: &InitRun<'_>) -> Result<(), ()> { self.mark(MimiError::ArenaCapacity) }
    fn effect_mark_request_shape_invalid_encode_run(&mut self, _: &EncodeRun<'_>) -> Result<(), ()> { self.mark(MimiError::RequestShape) }
    fn effect_mark_request_shape_invalid_decode_run(&mut self, _: &DecodeRun<'_>) -> Result<(), ()> { self.mark(MimiError::RequestShape) }
    fn effect_mark_code_range_invalid(&mut self, _: &DecodeRun<'_>) -> Result<(), ()> { self.mark(MimiError::CodeRange) }
    fn effect_mark_not_initialized_encode_run(&mut self, _: &EncodeRun<'_>) -> Result<(), ()> { self.mark(MimiError::NotInitialized) }
    fn effect_mark_not_initialized_decode_run(&mut self, _: &DecodeRun<'_>) -> Result<(), ()> { self.mark(MimiError::NotInitialized) }
    fn effect_run_frontend_child(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> {
        let request = encoder::Encode::new(event.encoder_runtime, event.streaming, event.pcm, event.frame, event.workspace, &mut self.latent[..self.dim]);
        if self.frontend.process_event(request).is_err() { self.mark(MimiError::RequestShape)?; }
        Ok(())
    }
    fn effect_run_quantize_child(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> {
        let request = quantizer::Encode::new(event.quantizer_runtime, &self.latent[..self.dim], event.codes_out, event.workspace);
        if self.quantizer.process_encode(request).is_err() { self.mark(MimiError::RequestShape)?; }
        Ok(())
    }
    fn effect_run_dequantize_child(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> {
        let h = event.runtime.model().hparams();
        let runtime = quantizer::QuantizerRuntime::new(self.n_q as usize, self.dim, self.semantic_n_q, self.card, self.card, matches!(event.runtime.variant(), binding::RuntimeVariant::F16), matches!(event.runtime.variant(), binding::RuntimeVariant::Q8));
        let request = quantizer::Decode::new(&runtime, event.codes, &mut self.latent[..self.dim], event.workspace);
        if self.quantizer.process_decode(request).is_err() { self.mark(MimiError::CodeRange)?; }
        let _ = h;
        Ok(())
    }
    fn effect_run_backend_child(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> {
        let request = decoder::DecodeRequest::new(event.runtime, event.streaming, &self.latent[..self.dim], event.frame, event.workspace, event.pcm_out);
        if self.backend.process_event(decoder::EventDecodeRun::new(request)).is_err() { self.mark(MimiError::RequestShape)?; }
        Ok(())
    }
    fn effect_reset_stream(&mut self, event: &EventResetStreamRun<'_>) -> Result<(), ()> { event.encoder.encoder_positions = 0; event.decoder.decoder_positions = 0; event.encoder.arena.fill(0.0); event.decoder.arena.fill(0.0); Ok(()) }
    fn effect_store_error_out_init_run(&mut self, event: &InitRun<'_>) -> Result<(), ()> { if let Some(out) = event.error_out.as_deref_mut() { **out = self.error; } Ok(()) }
    fn effect_store_error_out_encode_run(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { if let Some(out) = event.error_out.as_deref_mut() { **out = self.error; } Ok(()) }
    fn effect_store_error_out_decode_run(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { if let Some(out) = event.error_out.as_deref_mut() { **out = self.error; } Ok(()) }

    fn guard_bind_contract_valid(&self, event: &InitRun<'_>) -> Result<bool, ()> { let runtime = event.binding.codec_runtime(); let h = runtime.model().hparams(); Ok(runtime.model().component() == emel_model::bridge::MoshiComponent::Mimi && runtime.model().architecture_name() == b"moshi" && runtime.model().tensor_count() > 0 && h.validate().is_ok() && h.dim() > 0 && h.n_q() > 0) }
    fn guard_bind_contract_invalid(&self, event: &InitRun<'_>) -> Result<bool, ()> { Ok(!self.guard_bind_contract_valid(event)?) }
    fn guard_arena_capacity_valid(&self, event: &InitRun<'_>) -> Result<bool, ()> { let needed = event.binding.codec_runtime().arenas(); let h = event.binding.codec_runtime().model().hparams(); Ok(event.prepared.len() >= needed.prepared_floats() && event.state_arena.len() >= needed.state_floats() && event.workspace.len() >= needed.workspace_floats() && event.frame.len() >= needed.frame_floats() && h.dim() > 0 && h.dim() as usize <= MAX_LATENT_FLOATS) }
    fn guard_arena_capacity_invalid(&self, event: &InitRun<'_>) -> Result<bool, ()> { Ok(!self.guard_arena_capacity_valid(event)?) }
    fn guard_encode_request_valid(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(self.initialized && event.pcm.len() == self.frame_samples as usize && !event.pcm.is_empty() && event.codes_out.len() >= self.n_q as usize && event.encoder_runtime.model_bound && event.encoder_runtime.frame_samples == self.frame_samples as usize && event.encoder_runtime.dim == self.dim && event.quantizer_runtime.model_bound && event.quantizer_runtime.n_q == self.n_q as usize && event.quantizer_runtime.dim == self.dim) }
    fn guard_encode_request_invalid(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_encode_request_valid(event)?) }
    fn guard_decode_request_valid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(self.initialized && !event.codes.is_empty() && event.codes.len() >= self.semantic_n_q && event.codes.len() <= self.n_q as usize && event.pcm_out.len() >= self.frame_samples as usize) }
    fn guard_decode_request_invalid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_decode_request_valid(event)?) }
    fn guard_decode_codes_valid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.codes.iter().all(|code| *code >= 0 && usize::try_from(*code).ok().is_some_and(|code| code < self.card))) }
    fn guard_decode_codes_invalid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_decode_codes_valid(event)?) }
    fn guard_has_error_out_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_some()) }
    fn guard_no_error_out_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_none()) }
    fn guard_has_error_out_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_some()) }
    fn guard_no_error_out_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_none()) }
    fn guard_has_error_out_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_some()) }
    fn guard_no_error_out_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.error_out.is_none()) }
    fn guard_has_done_callback_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_some()) }
    fn guard_no_done_callback_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_none()) }
    fn guard_has_error_callback_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_some()) }
    fn guard_no_error_callback_init_run(&self, e: &InitRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_none()) }
    fn guard_has_done_callback_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_some()) }
    fn guard_no_done_callback_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_none()) }
    fn guard_has_error_callback_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_some()) }
    fn guard_no_error_callback_encode_run(&self, e: &EncodeRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_none()) }
    fn guard_has_done_callback_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_some()) }
    fn guard_no_done_callback_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_none()) }
    fn guard_has_error_callback_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_some()) }
    fn guard_no_error_callback_decode_run(&self, e: &DecodeRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_none()) }
    fn guard_unexpected_error_out_present(&self) -> Result<bool, ()> { Ok(false) }
    fn guard_unexpected_error_out_absent(&self) -> Result<bool, ()> { Ok(true) }
    fn effect_mark_unexpected(&mut self) -> Result<(), ()> { self.mark(MimiError::UnexpectedEvent) }
}

/// Single-writer synchronous top-level Mimi actor.
pub struct SpeechCodecMimi { machine: SpeechCodecMimiStateMachine<SpeechCodecMimiContext> }
impl Default for SpeechCodecMimi { fn default() -> Self { Self::new() } }
impl SpeechCodecMimi {
    /// Constructs an actor in generated `state_uninitialized`.
    #[must_use]
    pub fn new() -> Self { Self { machine: SpeechCodecMimiStateMachine::new(SpeechCodecMimiContext::default()) } }
    /// Dispatches one top-level event synchronously to completion.
    pub fn process_event(&mut self, event: MimiEvent<'_>) -> Result<(), MimiError> {
        match event {
            MimiEvent::Init(event) => self.process_init(event),
            MimiEvent::Encode(event) => self.process_encode(event),
            MimiEvent::Decode(event) => self.process_decode(event),
            MimiEvent::Reset(event) => self.process_reset(event),
        }
    }
    pub fn process_init(&mut self, event: InitRun<'_>) -> Result<(), MimiError> { self.machine.context_mut().clear(); if self.machine.process_event(SpeechCodecMimiEvents::InitRun(&event)).is_err() { self.machine.context_mut().error = MimiError::UnexpectedEvent; } self.result() }
    /// Dispatches encode synchronously.
    pub fn process_encode(&mut self, event: EncodeRun<'_>) -> Result<(), MimiError> { self.machine.context_mut().clear(); if self.machine.process_event(SpeechCodecMimiEvents::EncodeRun(&event)).is_err() { self.machine.context_mut().error = MimiError::UnexpectedEvent; } self.result() }
    /// Dispatches decode synchronously.
    pub fn process_decode(&mut self, event: DecodeRun<'_>) -> Result<(), MimiError> { self.machine.context_mut().clear(); if self.machine.process_event(SpeechCodecMimiEvents::DecodeRun(&event)).is_err() { self.machine.context_mut().error = MimiError::UnexpectedEvent; } self.result() }
    /// Dispatches reset synchronously.
    pub fn process_reset(&mut self, event: EventResetStreamRun<'_>) -> Result<(), MimiError> { self.machine.context_mut().clear(); if self.machine.process_event(SpeechCodecMimiEvents::EventResetStreamRun(&event)).is_err() { self.machine.context_mut().error = MimiError::UnexpectedEvent; } self.result() }
    fn result(&self) -> Result<(), MimiError> { let error = self.machine.context().error; if error == MimiError::None { Ok(()) } else { Err(error) } }
    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiStates { self.machine.state() }
    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: &SpeechCodecMimiStates) -> bool { self.machine.is(state) }
    /// Returns bounded context for inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiContext { self.machine.context() }
}
/// Short alias matching the pinned C++ `sm` surface.
pub type Mimi = SpeechCodecMimi;
