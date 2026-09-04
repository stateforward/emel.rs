//! Source-aligned, bounded Mimi encoder actor.
//!
//! The actor owns only control-plane state.  Requests, streaming state, and
//! all stage buffers remain caller-owned; stage computation is supplied by
//! already-bound synchronous callbacks and selected by explicit transitions.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    missing_docs
)]

use core::cell::{Cell, RefCell};
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
    /// `PCM` or latent output shape is invalid.
    RequestShape = 2,
    /// Caller-owned frame or workspace storage is too small.
    BufferCapacity = 3,
    /// The native frontend, transformer, or downsample stage was not bound.
    StageUnavailable = 5,
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
            Self::StageUnavailable => "encoder stage is unavailable",
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
        Self {
            arena,
            encoder_positions: 0,
        }
    }
}

/// The final boolean is the operand variant selected by the transition row:
/// `false` is the canonical `f32` path and `true` is the reference `f16`/`q8` path.
pub type EncoderStage =
    fn(&EncoderRuntime, &mut EncoderStreamingState<'_>, &mut [f32], &mut [f32], bool);

/// Bound runtime facts and synchronous numeric-stage entry points.
#[derive(Clone, Copy, Debug, Default)]
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
    /// Selects the `f16` convolution paths.
    pub conv_f16: bool,
    /// Selects the `q8` projection path.
    pub proj_q8: bool,
    /// Bound `SEANet` encoder callback.
    pub frontend: Option<EncoderStage>,
    /// Bound codec-transformer callback.
    pub transformer: Option<EncoderStage>,
    /// Bound downsample callback.
    pub downsample: Option<EncoderStage>,
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
pub struct EncodeDone<'dispatch, 'event, 'state> {
    /// Request completed by this dispatch.
    pub request: &'dispatch Encode<'event, 'state>,
}

/// Failed synchronous encoder result.
#[derive(Clone, Copy, Debug)]
pub struct EncodeError<'dispatch, 'event, 'state> {
    /// Request that produced the error.
    pub request: &'dispatch Encode<'event, 'state>,
    /// Error selected by validation or unexpected handling.
    pub error: EncoderError,
}

/// One caller-owned 80 ms `PCM` frame encode request.
#[derive(Debug)]
pub struct Encode<'event, 'state> {
    /// Bound runtime facts and stage callbacks.
    pub runtime: &'event EncoderRuntime,
    /// Persistent caller-owned streaming state.
    pub streaming: RefCell<&'event mut EncoderStreamingState<'state>>,
    /// Input mono `PCM` frame.
    pub pcm: &'event [f32],
    /// Caller-owned frame staging buffer.
    pub frame: RefCell<&'event mut [f32]>,
    /// Caller-owned per-dispatch workspace.
    pub workspace: RefCell<&'event mut [f32]>,
    /// Destination latent column.
    pub latent_out: RefCell<&'event mut [f32]>,
    /// Optional error output channel.
    pub error_out: Option<&'event Cell<EncoderError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'d, 'e, 's> fn(EncodeDone<'d, 'e, 's>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'d, 'e, 's> fn(EncodeError<'d, 'e, 's>)>,
}

impl<'event, 'state> Encode<'event, 'state> {
    /// Creates a request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'event EncoderRuntime,
        streaming: &'event mut EncoderStreamingState<'state>,
        pcm: &'event [f32],
        frame: &'event mut [f32],
        workspace: &'event mut [f32],
        latent_out: &'event mut [f32],
    ) -> Self {
        Self {
            runtime,
            streaming: RefCell::new(streaming),
            pcm,
            frame: RefCell::new(frame),
            workspace: RefCell::new(workspace),
            latent_out: RefCell::new(latent_out),
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
}

/// Runtime event corresponding to the pinned `encode_run` event.
#[derive(Debug)]
pub struct EventEncodeRun<'event, 'state> {
    /// Caller-owned request for this dispatch.
    pub request: Encode<'event, 'state>,
}

impl<'event, 'state> EventEncodeRun<'event, 'state> {
    /// Wraps one encode request.
    #[must_use]
    pub const fn new(request: Encode<'event, 'state>) -> Self {
        Self { request }
    }
}
// The event is shared throughout the generated run-to-completion dispatch.
// Mutable request lanes are borrowed from their `RefCell`s only by synchronous actions.
// `sml!` derives comparison support for generated state/event types without `Eq`.
sml! {
    SpeechCodecMimiEncoder<'dispatch, 'event, 'state>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        "state_runtime_decision"_s <= *"state_ready"_s + EventEncodeRun(&'dispatch EventEncodeRun<'event, 'state>),
        "state_shape_decision"_s <= "state_runtime_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_runtime_bound],
        "state_error_error_out_decision"_s <= "state_runtime_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_runtime_unbound] / effect_mark_runtime_unbound,
        "state_capacity_decision"_s <= "state_shape_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_request_shape_valid],
        "state_error_error_out_decision"_s <= "state_shape_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_request_shape_invalid] / effect_mark_request_shape_invalid,
        "state_error_error_out_decision"_s <= "state_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_buffer_capacity_invalid] / effect_mark_buffer_capacity_invalid,
        "state_error_error_out_decision"_s <= "state_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_stages_unbound] / effect_mark_stage_unavailable,
        "state_frontend_variant_decision"_s <= "state_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_stages_bound],
        "state_frontend_running"_s <= "state_frontend_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_conv_f32] / effect_run_frontend_false,
        "state_frontend_running"_s <= "state_frontend_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_conv_f16] / effect_run_frontend_true,
        "state_transformer_variant_decision"_s <= "state_frontend_running"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>),
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_proj_f32] / effect_run_transformer_false,
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_proj_q8] / effect_run_transformer_true,
        "state_downsample_variant_decision"_s <= "state_transformer_running"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>),
        "state_downsample_running"_s <= "state_downsample_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_conv_f32] / effect_run_downsample_false,
        "state_downsample_running"_s <= "state_downsample_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_conv_f16] / effect_run_downsample_true,
        "state_success_error_out_decision"_s <= "state_downsample_running"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>),
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>),
        "state_ready"_s <= "state_errored"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event, 'state>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_runtime_decision,
        "state_ready"_s <= "state_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_shape_decision,
        "state_ready"_s <= "state_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_capacity_decision,
        "state_ready"_s <= "state_frontend_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_frontend_variant_decision,
        "state_ready"_s <= "state_frontend_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_frontend_running,
        "state_ready"_s <= "state_transformer_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_variant_decision,
        "state_ready"_s <= "state_transformer_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_running,
        "state_ready"_s <= "state_downsample_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_downsample_variant_decision,
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
    pub const fn error(&self) -> EncoderError {
        self.error
    }

    fn clear(&mut self) {
        self.error = EncoderError::None;
    }
    fn mark(&mut self, error: EncoderError) {
        self.error = error;
    }
    #[allow(clippy::unnecessary_wraps)]
    fn unexpected(&mut self) -> Result<(), ()> {
        self.mark(EncoderError::UnexpectedEvent);
        Ok(())
    }
}

impl SpeechCodecMimiEncoderStateMachineContext for SpeechCodecMimiEncoderContext {
    fn effect_emit_done<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        if let Some(callback) = event.request.on_done {
            callback(EncodeDone {
                request: &event.request,
            });
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        if let Some(callback) = event.request.on_error {
            callback(EncodeError {
                request: &event.request,
                error: self.error,
            });
        }
        Ok(())
    }
    fn effect_mark_stage_unavailable<'dispatch, 'event, 'state>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        self.mark(EncoderError::StageUnavailable);
        Ok(())
    }

    fn effect_mark_buffer_capacity_invalid<'dispatch, 'event, 'state>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        self.mark(EncoderError::BufferCapacity);
        Ok(())
    }
    fn effect_mark_request_shape_invalid<'dispatch, 'event, 'state>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        self.mark(EncoderError::RequestShape);
        Ok(())
    }
    fn effect_mark_runtime_unbound<'dispatch, 'event, 'state>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        self.mark(EncoderError::RuntimeUnbound);
        Ok(())
    }

    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_frontend_running(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_frontend_variant_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_runtime_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_shape_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_running(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_variant_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn effect_on_unexpected_from_state_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_downsample_running(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_downsample_variant_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn guard_runtime_bound<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        let runtime = event.request.runtime;
        Ok(runtime.model_bound
            && runtime.frame_samples > 0
            && runtime.dim > 0
            && !event.request.streaming.borrow().arena.is_empty())
    }
    fn guard_runtime_unbound<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(!self.guard_runtime_bound(event)?)
    }

    fn effect_run_downsample_false<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_downsample(event, false);
        Ok(())
    }
    fn effect_run_downsample_true<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_downsample(event, true);
        Ok(())
    }
    fn effect_run_frontend_false<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_frontend(event, false);
        Ok(())
    }
    fn effect_run_frontend_true<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_frontend(event, true);
        Ok(())
    }
    fn effect_run_transformer_false<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_transformer(event, false);
        Ok(())
    }
    fn effect_run_transformer_true<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Self::run_transformer(event, true);
        Ok(())
    }

    fn effect_store_error_out_from_state_error_error_out_decision<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(self.error);
        }
        Ok(())
    }
    fn effect_store_error_out_from_state_success_error_out_decision<'dispatch, 'event, 'state>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(EncoderError::None);
        }
        Ok(())
    }

    fn guard_buffer_capacity_invalid<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        let request = &event.request;
        let frame = request.frame.borrow();
        let workspace = request.workspace.borrow();
        Ok(frame.is_empty()
            || workspace.is_empty()
            || request.runtime.frame_floats == 0
            || frame.len() < request.runtime.frame_floats
            || frame.len() < request.pcm.len()
            || workspace.len() < request.runtime.workspace_floats)
    }
    fn guard_stages_bound<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        let runtime = event.request.runtime;
        Ok(runtime.frontend.is_some()
            && runtime.transformer.is_some()
            && runtime.downsample.is_some())
    }
    fn guard_stages_unbound<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(!self.guard_stages_bound(event)?)
    }
    fn guard_conv_f16<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.runtime.conv_f16)
    }
    fn guard_conv_f32<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(!event.request.runtime.conv_f16)
    }
    fn guard_has_done_callback<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_has_error_callback<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_has_error_out<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_no_done_callback<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_no_error_callback<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_no_error_out<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_proj_f32<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(!event.request.runtime.proj_q8)
    }
    fn guard_proj_q8<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(event.request.runtime.proj_q8)
    }
    fn guard_request_shape_invalid<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        Ok(!self.guard_request_shape_valid(event)?)
    }
    fn guard_request_shape_valid<'dispatch, 'event, 'state>(
        &self,
        event: &'dispatch EventEncodeRun<'event, 'state>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
        'state: 'dispatch,
        'state: 'event,
    {
        let request = &event.request;
        let latent_out = request.latent_out.borrow();
        Ok(!request.pcm.is_empty()
            && !latent_out.is_empty()
            && request.pcm.len() == request.runtime.frame_samples
            && latent_out.len() >= request.runtime.dim)
    }
}

impl SpeechCodecMimiEncoderContext {
    fn run_downsample(event: &EventEncodeRun<'_, '_>, variant: bool) {
        let runtime = event.request.runtime;
        let stage = runtime.downsample.expect("stage availability guard");
        {
            let mut streaming = event.request.streaming.borrow_mut();
            let mut frame = event.request.frame.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            stage(runtime, &mut streaming, &mut frame, &mut workspace, variant);
        }
        let count = {
            let frame = event.request.frame.borrow();
            let latent_out = event.request.latent_out.borrow();
            runtime.dim.min(frame.len()).min(latent_out.len())
        };
        let frame = event.request.frame.borrow();
        let mut latent_out = event.request.latent_out.borrow_mut();
        latent_out[..count].copy_from_slice(&frame[..count]);
    }
    fn run_frontend(event: &EventEncodeRun<'_, '_>, variant: bool) {
        {
            let mut frame = event.request.frame.borrow_mut();
            frame[..event.request.pcm.len()].copy_from_slice(event.request.pcm);
        }
        let stage = event
            .request
            .runtime
            .frontend
            .expect("stage availability guard");
        let mut streaming = event.request.streaming.borrow_mut();
        let mut frame = event.request.frame.borrow_mut();
        let mut workspace = event.request.workspace.borrow_mut();
        stage(
            event.request.runtime,
            &mut streaming,
            &mut frame,
            &mut workspace,
            variant,
        );
    }
    fn run_transformer(event: &EventEncodeRun<'_, '_>, variant: bool) {
        let stage = event
            .request
            .runtime
            .transformer
            .expect("stage availability guard");
        let mut streaming = event.request.streaming.borrow_mut();
        let mut frame = event.request.frame.borrow_mut();
        let mut workspace = event.request.workspace.borrow_mut();
        stage(
            event.request.runtime,
            &mut streaming,
            &mut frame,
            &mut workspace,
            variant,
        );
    }
}

/// Single-writer synchronous Mimi encoder actor.
pub struct SpeechCodecMimiEncoder {
    machine: SpeechCodecMimiEncoderStateMachine<SpeechCodecMimiEncoderContext>,
}

impl Default for SpeechCodecMimiEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechCodecMimiEncoder {
    /// Constructs an encoder in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechCodecMimiEncoderStateMachine::new(
                SpeechCodecMimiEncoderContext::default(),
            ),
        }
    }

    /// Dispatches one bounded encode request synchronously.
    pub fn process_event(&mut self, request: Encode<'_, '_>) -> Result<(), EncoderError> {
        self.machine.context_mut().clear();
        let event = EventEncodeRun::new(request);
        if self
            .machine
            .process_event(SpeechCodecMimiEncoderEvents::EventEncodeRun(&event))
            .is_err()
        {
            self.machine
                .context_mut()
                .mark(EncoderError::UnexpectedEvent);
            return Err(EncoderError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == EncoderError::None {
            Ok(())
        } else {
            Err(error)
        }
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiEncoderStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &SpeechCodecMimiEncoderStates) -> bool {
        self.machine.is(state)
    }

    /// Returns bounded actor context for inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiEncoderContext {
        self.machine.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_frame_shorter_than_pcm_before_frontend_slice() {
        let runtime = EncoderRuntime::new(
            4, // frame_samples
            1, // dim
            2, // frame_floats: intentionally below the PCM frame length
            1, // workspace_floats
            false, false,
        );
        let pcm = [0.0_f32; 4];
        let mut frame = [0.0_f32; 2];
        let mut workspace = [0.0_f32; 1];
        let mut latent_out = [0.0_f32; 1];
        let mut arena = [0.0_f32; 1];
        let mut streaming = EncoderStreamingState::new(&mut arena);
        let request = Encode::new(
            &runtime,
            &mut streaming,
            &pcm,
            &mut frame,
            &mut workspace,
            &mut latent_out,
        );

        let mut encoder = SpeechCodecMimiEncoder::new();
        assert_eq!(
            encoder.process_event(request),
            Err(EncoderError::BufferCapacity)
        );
        assert!(encoder.is(&SpeechCodecMimiEncoderStates::StateReady));
    }

    use core::sync::atomic::{AtomicUsize, Ordering};

    static STAGE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ERROR_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn stage(
        _: &EncoderRuntime,
        _: &mut EncoderStreamingState<'_>,
        _: &mut [f32],
        _: &mut [f32],
        _: bool,
    ) {
        STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
    }
    fn noop_stage(
        _: &EncoderRuntime,
        _: &mut EncoderStreamingState<'_>,
        _: &mut [f32],
        _: &mut [f32],
        _: bool,
    ) {
    }

    fn done(_: EncodeDone<'_, '_, '_>) {
        DONE_CALLS.fetch_add(1, Ordering::SeqCst);
    }
    fn error(_: EncodeError<'_, '_, '_>) {
        ERROR_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn missing_each_stage_rejects_before_actions_or_success() {
        for missing in 0..3 {
            STAGE_CALLS.store(0, Ordering::SeqCst);
            DONE_CALLS.store(0, Ordering::SeqCst);
            ERROR_CALLS.store(0, Ordering::SeqCst);
            let mut runtime = EncoderRuntime::new(4, 1, 4, 1, false, false);
            if missing != 0 {
                runtime.frontend = Some(stage);
            }
            if missing != 1 {
                runtime.transformer = Some(stage);
            }
            if missing != 2 {
                runtime.downsample = Some(stage);
            }
            let pcm = [1.0_f32; 4];
            let mut frame = [7.0_f32; 4];
            let mut workspace = [8.0_f32; 1];
            let mut latent_out = [9.0_f32; 1];
            let mut arena = [0.0_f32; 1];
            let mut streaming = EncoderStreamingState::new(&mut arena);
            let error_out = Cell::new(EncoderError::None);
            let mut request = Encode::new(
                &runtime,
                &mut streaming,
                &pcm,
                &mut frame,
                &mut workspace,
                &mut latent_out,
            );
            request.error_out = Some(&error_out);
            request.on_done = Some(done);
            request.on_error = Some(error);

            let mut encoder = SpeechCodecMimiEncoder::new();
            assert_eq!(
                encoder.process_event(request),
                Err(EncoderError::StageUnavailable)
            );
            assert!(encoder.is(&SpeechCodecMimiEncoderStates::StateReady));
            assert_eq!(error_out.get(), EncoderError::StageUnavailable);
            assert_eq!(STAGE_CALLS.load(Ordering::SeqCst), 0);
            assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 0);
            assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
            assert!(
                latent_out
                    .iter()
                    .zip([9.0_f32; 1])
                    .all(|(actual, expected)| (actual - expected).abs() < f32::EPSILON)
            );
        }
    }

    #[test]
    fn invalid_then_valid_dispatch_clears_prior_error() {
        let runtime = EncoderRuntime::new(4, 1, 4, 1, false, false)
            .with_stages(noop_stage, noop_stage, noop_stage);
        let pcm = [1.0_f32; 4];
        let mut frame = [0.0_f32; 4];
        let mut workspace = [0.0_f32; 1];
        let mut latent_out = [0.0_f32; 1];
        let mut arena = [0.0_f32; 1];
        let mut streaming = EncoderStreamingState::new(&mut arena);
        let error_out = Cell::new(EncoderError::None);
        let invalid = Encode::new(
            &runtime,
            &mut streaming,
            &[],
            &mut frame,
            &mut workspace,
            &mut latent_out,
        );
        assert_eq!(
            SpeechCodecMimiEncoder::new().process_event(invalid),
            Err(EncoderError::RequestShape)
        );

        let mut encoder = SpeechCodecMimiEncoder::new();
        let invalid = Encode::new(
            &runtime,
            &mut streaming,
            &[],
            &mut frame,
            &mut workspace,
            &mut latent_out,
        );
        assert_eq!(
            encoder.process_event(invalid),
            Err(EncoderError::RequestShape)
        );
        let mut valid = Encode::new(
            &runtime,
            &mut streaming,
            &pcm,
            &mut frame,
            &mut workspace,
            &mut latent_out,
        );
        valid.error_out = Some(&error_out);
        assert_eq!(encoder.process_event(valid), Ok(()));
        assert_eq!(error_out.get(), EncoderError::None);
        assert!(encoder.is(&SpeechCodecMimiEncoderStates::StateReady));
    }
}
