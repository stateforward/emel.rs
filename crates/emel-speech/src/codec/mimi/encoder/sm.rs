//! Source-aligned, bounded Mimi encoder actor.
//!
//! The actor owns only control-plane state.  Requests, streaming state, and
//! all stage buffers remain caller-owned; stage computation is supplied by
//! already-bound synchronous callbacks and selected by explicit transitions.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    dead_code,
    missing_docs
)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors published by the Mimi encoder boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum EncoderError {
    /// No error was reported.
    #[default]
    None = 0,
    /// The runtime has not been bound completely.
    RuntimeUnbound = 1,
    /// PCM or latent output shape is invalid.
    RequestShape = 2,
    /// Caller-owned frame or workspace storage is too small.
    BufferCapacity = 3,
    /// The event was not accepted by the generated machine.
    UnexpectedEvent = 4,
}

impl fmt::Display for EncoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "no error",
            Self::RuntimeUnbound => "encoder runtime is unbound",
            Self::RequestShape => "encoder request shape is invalid",
            Self::BufferCapacity => "encoder buffer capacity is invalid",
            Self::UnexpectedEvent => "unexpected encoder event",
        })
    }
}

/// Caller-owned streaming state for one bound encoder runtime.
#[derive(Debug)]
pub struct EncoderStreamingState<'a> {
    /// Persistent arena used by streaming convolutions and transformer state.
    pub arena: &'a mut [f32],
    /// Number of frames already consumed by the encoder transformer.
    pub encoder_positions: i64,
}

impl<'a> EncoderStreamingState<'a> {
    /// Creates streaming state over a caller-owned arena.
    #[must_use]
    pub const fn new(arena: &'a mut [f32]) -> Self {
        Self { arena, encoder_positions: 0 }
    }
}

/// Synchronous callback used for one numeric encoder stage.
///
/// The final boolean is the operand variant selected by the transition row:
/// `false` is the canonical f32 path and `true` is the reference f16/q8 path.
pub type EncoderStage = fn(
    &EncoderRuntime,
    &mut EncoderStreamingState<'_>,
    &mut [f32],
    &mut [f32],
    bool,
);

/// Bound runtime facts and synchronous numeric-stage entry points.
#[derive(Clone, Copy, Debug)]
pub struct EncoderRuntime {
    /// Whether model weights and runtime storage have been bound.
    pub model_bound: bool,
    /// Samples in one 80 ms input frame.
    pub frame_samples: usize,
    /// Number of floats in one latent column.
    pub dim: usize,
    /// Minimum frame-buffer capacity in floats.
    pub frame_floats: usize,
    /// Minimum workspace capacity in floats.
    pub workspace_floats: usize,
    /// Selects the f16 convolution paths.
    pub conv_f16: bool,
    /// Selects the q8 projection path.
    pub proj_q8: bool,
    /// Bound SEANet encoder callback.
    pub frontend: Option<EncoderStage>,
    /// Bound codec-transformer callback.
    pub transformer: Option<EncoderStage>,
    /// Bound downsample callback.
    pub downsample: Option<EncoderStage>,
}

impl Default for EncoderRuntime {
    fn default() -> Self {
        Self {
            model_bound: false,
            frame_samples: 0,
            dim: 0,
            frame_floats: 0,
            workspace_floats: 0,
            conv_f16: false,
            proj_q8: false,
            frontend: None,
            transformer: None,
            downsample: None,
        }
    }
}

impl EncoderRuntime {
    /// Creates bound runtime facts without allocating.
    #[must_use]
    pub const fn new(
        frame_samples: usize,
        dim: usize,
        frame_floats: usize,
        workspace_floats: usize,
        conv_f16: bool,
        proj_q8: bool,
    ) -> Self {
        Self {
            model_bound: true,
            frame_samples,
            dim,
            frame_floats,
            workspace_floats,
            conv_f16,
            proj_q8,
            frontend: None,
            transformer: None,
            downsample: None,
        }
    }

    /// Installs the already-bound synchronous numeric stage callbacks.
    #[must_use]
    pub const fn with_stages(
        mut self,
        frontend: EncoderStage,
        transformer: EncoderStage,
        downsample: EncoderStage,
    ) -> Self {
        self.frontend = Some(frontend);
        self.transformer = Some(transformer);
        self.downsample = Some(downsample);
        self
    }
}

/// Successful synchronous encoder result.
#[derive(Clone, Copy, Debug)]
pub struct EncodeDone<'a> {
    /// Request completed by this dispatch.
    pub request: &'a Encode<'a>,
}

/// Failed synchronous encoder result.
#[derive(Clone, Copy, Debug)]
pub struct EncodeError<'a> {
    /// Request that produced the error.
    pub request: &'a Encode<'a>,
    /// Error selected by validation or unexpected handling.
    pub error: EncoderError,
}

/// One caller-owned 80 ms PCM frame encode request.
#[derive(Debug)]
pub struct Encode<'a> {
    /// Bound runtime facts and stage callbacks.
    pub runtime: &'a EncoderRuntime,
    /// Persistent caller-owned streaming state.
    pub streaming: &'a mut EncoderStreamingState<'a>,
    /// Input mono PCM frame.
    pub pcm: &'a [f32],
    /// Caller-owned frame staging buffer.
    pub frame: &'a mut [f32],
    /// Caller-owned per-dispatch workspace.
    pub workspace: &'a mut [f32],
    /// Destination latent column.
    pub latent_out: &'a mut [f32],
    /// Optional error output channel.
    pub error_out: Option<&'a Cell<EncoderError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'r> fn(EncodeDone<'r>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'r> fn(EncodeError<'r>)>,
}

impl<'a> Encode<'a> {
    /// Creates a request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'a EncoderRuntime,
        streaming: &'a mut EncoderStreamingState<'a>,
        pcm: &'a [f32],
        frame: &'a mut [f32],
        workspace: &'a mut [f32],
        latent_out: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            streaming,
            pcm,
            frame,
            workspace,
            latent_out,
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
}

/// Runtime event corresponding to the pinned `encode_run` event.
#[derive(Debug)]
pub struct EventEncodeRun<'a> {
    /// Caller-owned request for this dispatch.
    pub request: Encode<'a>,
}

impl<'a> EventEncodeRun<'a> {
    /// Wraps one encode request.
    #[must_use]
    pub const fn new(request: Encode<'a>) -> Self { Self { request } }
}

sml! {
    SpeechCodecMimiEncoder {
        "state_runtime_decision"_s <= *"state_ready"_s + EventEncodeRun(&'dispatch EventEncodeRun<'dispatch>),
        "state_shape_decision"_s <= "state_runtime_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_runtime_bound],
        "state_error_error_out_decision"_s <= "state_runtime_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_runtime_unbound] / effect_mark_runtime_unbound,
        "state_capacity_decision"_s <= "state_shape_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_request_shape_valid],
        "state_error_error_out_decision"_s <= "state_shape_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_request_shape_invalid] / effect_mark_request_shape_invalid,
        "state_frontend_variant_decision"_s <= "state_capacity_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_buffer_capacity_valid],
        "state_error_error_out_decision"_s <= "state_capacity_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_buffer_capacity_invalid] / effect_mark_buffer_capacity_invalid,
        "state_frontend_running"_s <= "state_frontend_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_conv_f32] / effect_run_frontend_false,
        "state_frontend_running"_s <= "state_frontend_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_conv_f16] / effect_run_frontend_true,
        "state_transformer_variant_decision"_s <= "state_frontend_running"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>),
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_proj_f32] / effect_run_transformer_false,
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_proj_q8] / effect_run_transformer_true,
        "state_downsample_variant_decision"_s <= "state_transformer_running"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>),
        "state_downsample_running"_s <= "state_downsample_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_conv_f32] / effect_run_downsample_false,
        "state_downsample_running"_s <= "state_downsample_variant_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_conv_f16] / effect_run_downsample_true,
        "state_success_error_out_decision"_s <= "state_downsample_running"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>),
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>),
        "state_ready"_s <= "state_errored"_s + completion<EventEncodeRun<'dispatch>>(&'dispatch EventEncodeRun<'dispatch>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_runtime_decision,
        "state_ready"_s <= "state_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_shape_decision,
        "state_ready"_s <= "state_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_capacity_decision,
        "state_ready"_s <= "state_frontend_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_frontend_variant_decision,
        "state_ready"_s <= "state_frontend_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_frontend_running,
        "state_ready"_s <= "state_downsample_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_downsample_variant_decision,
        "state_ready"_s <= "state_transformer_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_variant_decision,
        "state_ready"_s <= "state_transformer_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_running,
        "state_ready"_s <= "state_downsample_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_downsample_running,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Mutable per-actor context matching the pinned stateless action context.
#[derive(Debug, Default)]
pub struct SpeechCodecMimiEncoderContext {
    error: EncoderError,
}

impl SpeechCodecMimiEncoderContext {
    /// Returns the most recently selected error.
    #[must_use]
    pub const fn error(&self) -> EncoderError { self.error }

    fn clear(&mut self) { self.error = EncoderError::None; }
    fn mark(&mut self, error: EncoderError) { self.error = error; }
    fn unexpected(&mut self) -> Result<(), ()> {
        self.mark(EncoderError::UnexpectedEvent);
        Ok(())
    }
}

impl SpeechCodecMimiEncoderStateMachineContext for SpeechCodecMimiEncoderContext {
    fn effect_emit_done(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_done {
            callback(EncodeDone { request: &event.request });
        }
        Ok(())
    }

    fn effect_emit_error(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_error {
            callback(EncodeError { request: &event.request, error: self.error });
        }
        Ok(())
    }

    fn effect_mark_buffer_capacity_invalid(&mut self, _event: &EventEncodeRun<'_>) -> Result<(), ()> {
        self.mark(EncoderError::BufferCapacity);
        Ok(())
    }

    fn effect_mark_request_shape_invalid(&mut self, _event: &EventEncodeRun<'_>) -> Result<(), ()> {
        self.mark(EncoderError::RequestShape);
        Ok(())
    }

    fn effect_mark_runtime_unbound(&mut self, _event: &EventEncodeRun<'_>) -> Result<(), ()> {
        self.mark(EncoderError::RuntimeUnbound);
        Ok(())
    }

    fn effect_on_unexpected_from_state_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_downsample_running(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_downsample_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_frontend_running(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_frontend_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_runtime_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_shape_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_running(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }

    fn effect_run_downsample_false(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(stage) = event.request.runtime.downsample {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, false);
        }
        let count = event.request.runtime.dim.min(event.request.frame.len()).min(event.request.latent_out.len());
        event.request.latent_out[..count].copy_from_slice(&event.request.frame[..count]);
        Ok(())
    }

    fn effect_run_downsample_true(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(stage) = event.request.runtime.downsample {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, true);
        }
        let count = event.request.runtime.dim.min(event.request.frame.len()).min(event.request.latent_out.len());
        event.request.latent_out[..count].copy_from_slice(&event.request.frame[..count]);
        Ok(())
    }

    fn effect_run_frontend_false(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        event.request.frame[..event.request.pcm.len()].copy_from_slice(event.request.pcm);
        if let Some(stage) = event.request.runtime.frontend {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, false);
        }
        Ok(())
    }

    fn effect_run_frontend_true(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        event.request.frame[..event.request.pcm.len()].copy_from_slice(event.request.pcm);
        if let Some(stage) = event.request.runtime.frontend {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, true);
        }
        Ok(())
    }

    fn effect_run_transformer_false(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(stage) = event.request.runtime.transformer {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, false);
        }
        Ok(())
    }

    fn effect_run_transformer_true(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(stage) = event.request.runtime.transformer {
            stage(event.request.runtime, event.request.streaming, event.request.frame, event.request.workspace, true);
        }
        Ok(())
    }

    fn effect_store_error_out_from_state_error_error_out_decision(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(error_out) = event.request.error_out { error_out.set(self.error); }
        Ok(())
    }

    fn effect_store_error_out_from_state_success_error_out_decision(&mut self, event: &EventEncodeRun<'_>) -> Result<(), ()> {
        if let Some(error_out) = event.request.error_out { error_out.set(EncoderError::None); }
        Ok(())
    }

    fn guard_buffer_capacity_invalid(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_buffer_capacity_valid(event)?) }
    fn guard_buffer_capacity_valid(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> {
        let request = &event.request;
        Ok(!request.frame.is_empty()
            && !request.workspace.is_empty()
            && request.runtime.frame_floats > 0
            && request.frame.len() >= request.runtime.frame_floats
            && request.workspace.len() >= request.runtime.workspace_floats)
    }
    fn guard_conv_f16(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.conv_f16) }
    fn guard_conv_f32(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.runtime.conv_f16) }
    fn guard_has_done_callback(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_some()) }
    fn guard_has_error_callback(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_some()) }
    fn guard_has_error_out(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_no_done_callback(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_none()) }
    fn guard_no_error_callback(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_none()) }
    fn guard_no_error_out(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_proj_f32(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.runtime.proj_q8) }
    fn guard_proj_q8(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.proj_q8) }
    fn guard_request_shape_invalid(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_request_shape_valid(event)?) }
    fn guard_request_shape_valid(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> {
        let request = &event.request;
        Ok(!request.pcm.is_empty()
            && !request.latent_out.is_empty()
            && request.pcm.len() == request.runtime.frame_samples
            && request.latent_out.len() >= request.runtime.dim)
    }
    fn guard_runtime_bound(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> {
        let runtime = event.request.runtime;
        Ok(runtime.model_bound
            && runtime.frame_samples > 0
            && runtime.dim > 0
            && !event.request.streaming.arena.is_empty())
    }
    fn guard_runtime_unbound(&self, event: &EventEncodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_runtime_bound(event)?) }
}

/// Single-writer synchronous Mimi encoder actor.
pub struct SpeechCodecMimiEncoder {
    machine: SpeechCodecMimiEncoderStateMachine<SpeechCodecMimiEncoderContext>,
}

impl Default for SpeechCodecMimiEncoder {
    fn default() -> Self { Self::new() }
}

impl SpeechCodecMimiEncoder {
    /// Constructs an encoder in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: SpeechCodecMimiEncoderStateMachine::new(SpeechCodecMimiEncoderContext::default()) }
    }

    /// Dispatches one bounded encode request synchronously.
    pub fn process_event(&mut self, request: Encode<'_>) -> Result<(), EncoderError> {
        let event = EventEncodeRun::new(request);
        if self.machine.process_event(SpeechCodecMimiEncoderEvents::EventEncodeRun(&event)).is_err() {
            self.machine.context_mut().mark(EncoderError::UnexpectedEvent);
            return Err(EncoderError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == EncoderError::None { Ok(()) } else { Err(error) }
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiEncoderStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &SpeechCodecMimiEncoderStates) -> bool { self.machine.is(state) }

    /// Returns bounded actor context for inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiEncoderContext { self.machine.context() }
}
