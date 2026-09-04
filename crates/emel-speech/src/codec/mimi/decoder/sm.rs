//! Source-aligned, bounded Mimi decoder actor.
//!
//! The transition topology mirrors the pinned C++ decoder.  Requests and all
//! storage are caller-owned; dispatch is synchronous and does not allocate.
//! Numeric decoder kernels are deliberately outside this orchestration actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::enum_variant_names,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

use core::cell::RefCell;

use sml::sml;

use super::super::binding::{CodecRuntime, RuntimeVariant};

/// Executes Mimi's source-aligned depthwise stride-2 transposed-convolution
/// upsample leaf through the owning kernel package. Weight bytes remain
/// model-owned; all state, output, and workspace storage are caller-owned.
pub fn native_upsample_stage(
    runtime: &CodecRuntime<'_>,
    streaming: &mut CodecStreamingState<'_>,
    latent: &[f32],
    frame: &mut [f32],
    workspace: &mut [f32],
) -> bool {
    let upsample = runtime.upsample();
    emel_kernels::any::conv_transpose_1d::native_depthwise_stride2_f32(
        upsample.bytes(),
        upsample.dim(),
        upsample.taps(),
        streaming.arena,
        latent,
        frame,
        workspace,
    )
}

/// Decoder errors from the pinned `decoder::error` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DecoderError {
    /// No error occurred.
    #[default]
    None = 0,
    /// The runtime binding or streaming state was absent.
    RuntimeUnbound = 1,
    /// The latent or PCM request shape was invalid.
    RequestShape = 2,
    /// A caller-owned frame or workspace buffer was too small.
    BufferCapacity = 3,
    /// A required numeric stage callback was not bound.
    StageUnavailable = 4,
    /// A bound numeric stage callback rejected its request.
    StageFailed = 5,
}

/// Successful decoder callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeDone {
    /// Number of PCM samples in the published frame.
    pub frame_samples: u32,
}

/// Failed decoder callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError {
    /// Error selected by validation or orchestration.
    pub error: DecoderError,
}

/// Synchronous completion callback. The callback must not retain or re-enter
/// the actor.
pub type DoneCallback = fn(DecodeDone) -> bool;
/// Synchronous error callback. The callback must not retain or re-enter the
/// actor.
pub type ErrorCallback = fn(DecodeError) -> bool;

/// Synchronous upsample callback. It consumes the dequantized latent column,
/// writes the intermediate decoder frame, and uses caller-owned workspaces.
/// Returning `false` selects the explicit stage-failure route.
pub type DecoderUpsampleStage = for<'a, 'b, 'c, 'd> fn(
    &CodecRuntime<'a>,
    &mut CodecStreamingState<'b>,
    &[f32],
    &mut [f32],
    &mut [f32],
) -> bool;

/// Synchronous decoder-transformer callback. `variant` selects the explicit
/// f32 (`false`) or q8 projection (`true`) route.
pub type DecoderTransformerStage = for<'a, 'b> fn(
    &CodecRuntime<'a>,
    &mut CodecStreamingState<'b>,
    &mut [f32],
    &mut [f32],
    bool,
) -> bool;

/// Synchronous decoder backend callback. It consumes the transformed frame,
/// writes PCM, and uses caller-owned workspaces. `variant` selects the
/// explicit f32 (`false`) or f16 convolution (`true`) route.
pub type DecoderBackendStage = for<'a, 'b> fn(
    &CodecRuntime<'a>,
    &mut CodecStreamingState<'b>,
    &[f32],
    &mut [f32],
    &mut [f32],
    bool,
) -> bool;

/// Caller-owned persistent Mimi decoder state.
#[derive(Debug)]
pub struct CodecStreamingState<'a> {
    /// Persistent state arena supplied at bind time.
    pub arena: &'a mut [f32],
    /// Decoder transformer position, advanced by the numeric backend.
    pub decoder_positions: i64,
}

impl<'a> CodecStreamingState<'a> {
    /// Creates streaming state over a caller-owned arena.
    #[must_use]
    pub const fn new(arena: &'a mut [f32]) -> Self {
        Self {
            arena,
            decoder_positions: 0,
        }
    }
}

/// One bounded, caller-owned decoder request.
#[derive(Debug)]
pub struct DecodeRequest<'dispatch, 'state, 'model> {
    /// Borrowed prepared runtime binding. `None` represents an unbound runtime.
    pub runtime: Option<&'dispatch CodecRuntime<'model>>,
    /// Persistent streaming state for this decoder session.
    pub streaming: RefCell<Option<&'dispatch mut CodecStreamingState<'state>>>,
    /// One 12.5 Hz latent column.
    pub latent: &'dispatch [f32],
    /// Caller-owned intermediate frame arena.
    pub frame: RefCell<&'dispatch mut [f32]>,
    /// Caller-owned per-dispatch workspace arena.
    pub workspace: RefCell<&'dispatch mut [f32]>,
    /// Caller-owned 24 kHz PCM output frame.
    pub pcm_out: RefCell<&'dispatch mut [f32]>,
    /// Synchronous upsample callback.
    pub upsample: Option<DecoderUpsampleStage>,
    /// Synchronous decoder-transformer callback.
    pub transformer: Option<DecoderTransformerStage>,
    /// Synchronous decoder backend callback.
    pub backend: Option<DecoderBackendStage>,
    /// Optional synchronous error output channel.
    pub error_out: Option<&'dispatch core::cell::Cell<DecoderError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<DoneCallback>,
    /// Optional synchronous error callback.
    pub on_error: Option<ErrorCallback>,
}

impl<'dispatch, 'state, 'model> DecodeRequest<'dispatch, 'state, 'model> {
    /// Creates a request with caller-owned buffers and no stage callbacks.
    #[must_use]
    pub const fn new(
        runtime: &'dispatch CodecRuntime<'model>,
        streaming: &'dispatch mut CodecStreamingState<'state>,
        latent: &'dispatch [f32],
        frame: &'dispatch mut [f32],
        workspace: &'dispatch mut [f32],
        pcm_out: &'dispatch mut [f32],
    ) -> Self {
        Self {
            runtime: Some(runtime),
            streaming: RefCell::new(Some(streaming)),
            latent,
            frame: RefCell::new(frame),
            workspace: RefCell::new(workspace),
            pcm_out: RefCell::new(pcm_out),
            upsample: Some(native_upsample_stage),
            transformer: None,
            backend: None,
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }

    /// Installs the source-aligned native Mimi upsample stage.
    #[must_use]
    pub const fn with_stages(
        mut self,
        upsample: DecoderUpsampleStage,
        transformer: DecoderTransformerStage,
        backend: DecoderBackendStage,
    ) -> Self {
        self.upsample = Some(upsample);
        self.transformer = Some(transformer);
        self.backend = Some(backend);
        self
    }

    /// Installs an optional error output destination.
    #[must_use]
    pub const fn with_error_out(
        mut self,
        error_out: &'dispatch core::cell::Cell<DecoderError>,
    ) -> Self {
        self.error_out = Some(error_out);
        self
    }

    /// Installs optional synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

/// Runtime event corresponding to C++ `event::decode`/`event::decode_run`.
#[derive(Debug)]
pub struct EventDecodeRun<'dispatch, 'state, 'model> {
    /// Caller-owned request for this dispatch.
    pub request: DecodeRequest<'dispatch, 'state, 'model>,
}

impl<'dispatch, 'state, 'model> EventDecodeRun<'dispatch, 'state, 'model> {
    /// Wraps a decoder request as a runtime event.
    #[must_use]
    pub const fn new(request: DecodeRequest<'dispatch, 'state, 'model>) -> Self {
        Self { request }
    }
}

sml! {
    SpeechCodecMimiDecoder<'dispatch, 'state, 'model>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        "state_runtime_decision"_s <= *"state_ready"_s + EventDecodeRun(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>),
        "state_shape_decision"_s <= "state_runtime_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_runtime_bound],
        "state_error_error_out_decision"_s <= "state_runtime_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_runtime_unbound] / effect_mark_runtime_unbound,
        "state_capacity_decision"_s <= "state_shape_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_request_shape_valid],
        "state_error_error_out_decision"_s <= "state_shape_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_request_shape_invalid] / effect_mark_request_shape_invalid,
        "state_upsample_running"_s <= "state_capacity_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_buffer_capacity_valid] / effect_run_upsample,
        "state_error_error_out_decision"_s <= "state_capacity_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_buffer_capacity_invalid] / effect_mark_buffer_capacity_invalid,
        "state_transformer_variant_decision"_s <= "state_upsample_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_success],
        "state_error_error_out_decision"_s <= "state_upsample_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_failed],
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_proj_f32] / effect_run_transformer_false,
        "state_transformer_running"_s <= "state_transformer_variant_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_proj_q8] / effect_run_transformer_true,
        "state_backend_variant_decision"_s <= "state_transformer_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_success],
        "state_error_error_out_decision"_s <= "state_transformer_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_failed],
        "state_backend_running"_s <= "state_backend_variant_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_conv_f32] / effect_run_backend_false,
        "state_backend_running"_s <= "state_backend_variant_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_conv_f16] / effect_run_backend_true,
        "state_success_error_out_decision"_s <= "state_backend_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_success],
        "state_error_error_out_decision"_s <= "state_backend_running"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_stage_failed],
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_has_error_out] / effect_store_error_out_from_state_success_error_out_decision,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_has_error_out] / effect_store_error_out_from_state_error_error_out_decision,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>),
        "state_ready"_s <= "state_errored"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'dispatch, 'state, 'model>),
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

/// Bounded per-dispatch context. Numeric work is delegated to backend actors;
/// this context only carries the error selected by the guarded route.
/// Bounded per-dispatch context. Numeric work is delegated to explicit stage
/// callbacks; this context records the selected error and callback outcome.
#[derive(Debug, Default)]
pub struct SpeechCodecMimiDecoderContext {
    /// Error selected by validation or stage dispatch.
    pub err: DecoderError,
    stage_ok: bool,
}

impl SpeechCodecMimiDecoderContext {
    fn clear(&mut self) {
        self.err = DecoderError::None;
        self.stage_ok = false;
    }
    #[allow(
        clippy::unnecessary_wraps,
        reason = "SML effect helpers require the generated Result<(), ()> callback contract"
    )]
    fn mark(&mut self, error: DecoderError) -> Result<(), ()> {
        self.err = error;
        self.stage_ok = false;
        Ok(())
    }
    #[allow(
        clippy::unnecessary_wraps,
        reason = "SML effect helpers require the generated Result<(), ()> callback contract"
    )]
    fn stage(&mut self, ok: bool) -> Result<(), ()> {
        self.stage_ok = ok;
        if !ok {
            self.err = DecoderError::StageFailed;
        }
        Ok(())
    }
    #[allow(
        clippy::unnecessary_wraps,
        reason = "SML effect helpers require the generated Result<(), ()> callback contract"
    )]
    fn unexpected(&mut self) -> Result<(), ()> {
        self.mark(DecoderError::StageFailed)
    }
}
impl SpeechCodecMimiDecoderStateMachineContext for SpeechCodecMimiDecoderContext {
    fn effect_emit_done<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            callback(DecodeDone {
                frame_samples: event
                    .request
                    .runtime
                    .map_or(0, |runtime| runtime.frame_samples()),
            });
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(DecodeError { error: self.err });
        }
        Ok(())
    }
    fn effect_mark_buffer_capacity_invalid<'dispatch, 'state, 'model>(
        &mut self,
        _: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.mark(DecoderError::BufferCapacity)
    }
    fn effect_mark_request_shape_invalid<'dispatch, 'state, 'model>(
        &mut self,
        _: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.mark(DecoderError::RequestShape)
    }
    fn effect_mark_runtime_unbound<'dispatch, 'state, 'model>(
        &mut self,
        _: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.mark(DecoderError::RuntimeUnbound)
    }

    fn effect_run_upsample<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(stage) = event.request.upsample else {
            return self.mark(DecoderError::StageUnavailable);
        };
        let Some(runtime) = event.request.runtime else {
            return self.mark(DecoderError::RuntimeUnbound);
        };
        let streaming = event.request.streaming.borrow_mut().take().ok_or(())?;
        let result = {
            let mut frame = event.request.frame.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            stage(
                runtime,
                &mut *streaming,
                event.request.latent,
                &mut frame[..],
                &mut workspace[..],
            )
        };
        *event.request.streaming.borrow_mut() = Some(streaming);
        self.stage(result)
    }
    fn effect_run_transformer_false<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.run_transformer(event, false)
    }
    fn effect_run_transformer_true<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.run_transformer(event, true)
    }
    fn effect_run_backend_false<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.run_backend(event, false)
    }
    fn effect_run_backend_true<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        self.run_backend(event, true)
    }

    fn effect_store_error_out_from_state_success_error_out_decision<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(DecoderError::None);
        }
        Ok(())
    }
    fn effect_store_error_out_from_state_error_error_out_decision<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(self.err);
        }
        Ok(())
    }

    fn guard_stage_success<'dispatch, 'state, 'model>(
        &self,
        _: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(self.stage_ok)
    }
    fn guard_stage_failed<'dispatch, 'state, 'model>(
        &self,
        _: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(!self.stage_ok)
    }
    #[allow(
        clippy::suspicious_operation_groupings,
        reason = "frame_floats is the caller-owned frame arena capacity; this comparison intentionally pairs frame.len() with that capacity"
    )]
    fn guard_buffer_capacity_invalid<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(runtime) = event.request.runtime else {
            return Ok(true);
        };
        let capacities = runtime.arenas();
        let frame = event.request.frame.borrow();
        let workspace = event.request.workspace.borrow();
        Ok(frame.is_empty()
            || workspace.is_empty()
            || frame.len() < capacities.frame_floats()
            || workspace.len() < capacities.workspace_floats())
    }
    fn guard_buffer_capacity_valid<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(!self.guard_buffer_capacity_invalid(event)?)
    }
    fn guard_conv_f16<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event
            .request
            .runtime
            .is_some_and(|r| r.variant() == RuntimeVariant::F16))
    }
    fn guard_conv_f32<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event
            .request
            .runtime
            .is_some_and(|r| r.variant() != RuntimeVariant::F16))
    }
    fn guard_proj_q8<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event
            .request
            .runtime
            .is_some_and(|r| r.variant() == RuntimeVariant::Q8))
    }
    fn guard_proj_f32<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event
            .request
            .runtime
            .is_some_and(|r| r.variant() != RuntimeVariant::Q8))
    }
    fn guard_has_error_out<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_no_error_out<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_has_done_callback<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_no_done_callback<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_has_error_callback<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_no_error_callback<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_request_shape_invalid<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(runtime) = event.request.runtime else {
            return Ok(true);
        };
        let dim = usize::try_from(runtime.model().hparams().dim()).map_err(|_| ())?;
        let pcm_out = event.request.pcm_out.borrow();
        Ok(event.request.latent.is_empty()
            || pcm_out.is_empty()
            || event.request.latent.len() < dim
            || pcm_out.len() < usize::try_from(runtime.frame_samples()).map_err(|_| ())?)
    }
    fn guard_request_shape_valid<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(runtime) = event.request.runtime else {
            return Ok(false);
        };
        let dim = usize::try_from(runtime.model().hparams().dim()).map_err(|_| ())?;
        let pcm_out = event.request.pcm_out.borrow();
        Ok(!event.request.latent.is_empty()
            && !pcm_out.is_empty()
            && event.request.latent.len() >= dim
            && pcm_out.len() >= usize::try_from(runtime.frame_samples()).map_err(|_| ())?)
    }
    fn guard_runtime_bound<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(runtime) = event.request.runtime else {
            return Ok(false);
        };
        let hparams = runtime.model().hparams();
        Ok(runtime.model().tensor_count() > 0
            && hparams.dim() > 0
            && hparams.frame_samples().is_some()
            && event
                .request
                .streaming
                .borrow()
                .as_ref()
                .is_some_and(|state| !state.arena.is_empty()))
    }
    fn guard_runtime_unbound<'dispatch, 'state, 'model>(
        &self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<bool, ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(runtime) = event.request.runtime else {
            return Ok(true);
        };
        let hparams = runtime.model().hparams();
        Ok(runtime.model().tensor_count() == 0
            || hparams.dim() == 0
            || hparams.frame_samples().is_none()
            || event
                .request
                .streaming
                .borrow()
                .as_ref()
                .is_none_or(|state| state.arena.is_empty()))
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_backend_running(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_backend_variant_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
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
    fn effect_on_unexpected_from_state_upsample_running(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
}
impl SpeechCodecMimiDecoderContext {
    fn run_transformer<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
        variant: bool,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(stage) = event.request.transformer else {
            return self.mark(DecoderError::StageUnavailable);
        };
        let Some(runtime) = event.request.runtime else {
            return self.mark(DecoderError::RuntimeUnbound);
        };
        let streaming = event.request.streaming.borrow_mut().take().ok_or(())?;
        let result = {
            let mut frame = event.request.frame.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            stage(
                runtime,
                &mut *streaming,
                &mut frame[..],
                &mut workspace[..],
                variant,
            )
        };
        *event.request.streaming.borrow_mut() = Some(streaming);
        self.stage(result)
    }
    fn run_backend<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
        variant: bool,
    ) -> Result<(), ()>
    where
        'model: 'dispatch,
        'state: 'dispatch,
    {
        let Some(stage) = event.request.backend else {
            return self.mark(DecoderError::StageUnavailable);
        };
        let Some(runtime) = event.request.runtime else {
            return self.mark(DecoderError::RuntimeUnbound);
        };
        let streaming = event.request.streaming.borrow_mut().take().ok_or(())?;
        let result = {
            let frame = event.request.frame.borrow();
            let mut workspace = event.request.workspace.borrow_mut();
            let mut pcm_out = event.request.pcm_out.borrow_mut();
            stage(
                runtime,
                &mut *streaming,
                &frame[..],
                &mut pcm_out[..],
                &mut workspace[..],
                variant,
            )
        };
        *event.request.streaming.borrow_mut() = Some(streaming);
        self.stage(result)
    }
}
/// Single-writer synchronous decoder actor.
pub struct SpeechCodecMimiDecoder {
    machine: SpeechCodecMimiDecoderStateMachine<SpeechCodecMimiDecoderContext>,
}

impl Default for SpeechCodecMimiDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechCodecMimiDecoder {
    /// Creates an actor in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechCodecMimiDecoderStateMachine::new(
                SpeechCodecMimiDecoderContext::default(),
            ),
        }
    }

    /// Dispatches one bounded request synchronously to completion.
    pub fn process_event<'dispatch, 'state, 'model>(
        &mut self,
        event: &'dispatch EventDecodeRun<'dispatch, 'state, 'model>,
    ) -> Result<(), DecoderError>
    where
        'model: 'dispatch,
    {
        self.machine.context_mut().clear();
        if !self.machine.is(&SpeechCodecMimiDecoderStates::StateReady) {
            return Err(DecoderError::RuntimeUnbound);
        }
        if self
            .machine
            .process_event(SpeechCodecMimiDecoderEvents::EventDecodeRun(event))
            .is_err()
        {
            return Err(DecoderError::RequestShape);
        }
        let error = self.machine.context().err;
        if error == DecoderError::None {
            Ok(())
        } else {
            Err(error)
        }
    }

    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiDecoderStates {
        self.machine.state()
    }

    /// Returns the bounded context for error inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiDecoderContext {
        self.machine.context()
    }
}

/// Short actor alias matching the C++ `sm` surface.
pub type Decoder = SpeechCodecMimiDecoder;

#[cfg(test)]
mod tests {
    use super::super::super::binding::{ArenaCapacities, UpsampleBinding};
    use super::*;
    use emel_model::bridge::{
        Data, MimiDataInput, MimiHParams, MimiHParamsInput, TensorInput, TensorMetadata,
        TensorMetadataInput,
    };
    use emel_tensor::dtype::SerializedType;

    #[test]
    fn native_upsample_overlaps_persistent_tail_and_emits_time_major() {
        // Pinned [taps, 1, channels] layout: channel 0 taps are 1, 2, 3;
        // channel 1 taps are 10, 20, 30. The non-interleaved values make a
        // tap-major/channel-interleaved read observably produce wrong output.
        let bytes: [u8; 24] = [
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 32, 65, 0, 0, 160, 65, 0, 0, 240, 65,
        ];
        let h = MimiHParams::try_new(MimiHParamsInput {
            sample_rate: 24_000,
            frame_rate: 12.5,
            n_q: 2,
            card: 2,
            dim: 2,
            semantic_n_q: 1,
            codebook_dim: 2,
            transformer_num_layers: 1,
            transformer_num_heads: 1,
            transformer_context: 1,
            transformer_max_period: 1,
        })
        .unwrap();
        let tensor = TensorInput::with_bytes(
            b"mimi.upsample.convtr.convtr.convtr.weight",
            TensorMetadata::new(TensorMetadataInput {
                tensor_type: SerializedType::F32,
                dimension_count: 3,
                dimensions: [3, 1, 2, 1],
                data_offset: 0,
                file_offset: 0,
                data_size: 24,
                file_index: 0,
                storage: None,
            }),
            &bytes,
        );
        let data = Data::try_from_mimi(MimiDataInput {
            hparams: h,
            tensors: &[tensor],
        })
        .unwrap();
        let runtime = CodecRuntime::for_test(
            data.mimi_binding_input(),
            ArenaCapacities::new(6, 6, 8, 4),
            UpsampleBinding::for_test(&bytes, 2, 3),
        );
        let mut state_arena = [0.0; 6];
        let mut state = CodecStreamingState::new(&mut state_arena);
        let mut frame = [0.0; 4];
        let mut workspace = [0.0; 8];
        assert!(native_upsample_stage(
            &runtime,
            &mut state,
            &[1.0, 2.0],
            &mut frame,
            &mut workspace
        ));
        assert!(
            frame
                .iter()
                .zip([1.0, 20.0, 2.0, 40.0])
                .all(|(actual, expected)| (actual - expected).abs() < f32::EPSILON)
        );
        assert!(native_upsample_stage(
            &runtime,
            &mut state,
            &[2.0, 1.0],
            &mut frame,
            &mut workspace
        ));
        assert!(
            frame
                .iter()
                .zip([5.0, 70.0, 4.0, 20.0])
                .all(|(actual, expected)| (actual - expected).abs() < f32::EPSILON)
        );
    }
}
