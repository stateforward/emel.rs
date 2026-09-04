//! Source-aligned bounded Mimi residual-vector-quantizer actor.
//!
//! The actor retains only control-plane state. Requests, code buffers, latent
//! buffers, workspaces, and numeric stage callbacks remain caller-owned. Every
//! dispatch is synchronous and run-to-completion; variant selection is made by
//! explicit generated transition guards before a stage callback is invoked.

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

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

/// Errors represented by the pinned Mimi quantizer boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum QuantizerError {
    /// No error was selected.
    #[default]
    None = 0,
    /// The bound codec runtime is incomplete.
    RuntimeUnbound = 1,
    /// A request buffer has invalid shape or capacity.
    RequestShape = 2,
    /// A caller-owned buffer does not meet the runtime capacity contract.
    BufferCapacity = 3,
    /// A decode code does not address a codebook entry.
    CodeRange = 4,
    /// A required numeric stage callback was not bound.
    StageUnavailable = 5,
    /// The generated machine rejected the event.
    UnexpectedEvent = 6,
}

impl fmt::Display for QuantizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "no error",
            Self::RuntimeUnbound => "quantizer runtime is unbound",
            Self::RequestShape => "quantizer request shape is invalid",
            Self::BufferCapacity => "quantizer buffer capacity is invalid",
            Self::CodeRange => "quantizer code is out of range",
            Self::StageUnavailable => "quantizer stage is unavailable",
            Self::UnexpectedEvent => "unexpected quantizer event",
        })
    }
}

/// Synchronous numeric encode stage selected by a quantizer transition.
pub type EncodeStage =
    for<'a> fn(&QuantizerRuntime<'a>, &[f32], &mut [i32], &mut [f32], bool, bool);

/// Synchronous numeric decode stage selected by a quantizer transition.
pub type DecodeStage =
    for<'a> fn(&QuantizerRuntime<'a>, &[i32], &mut [f32], &mut [f32], bool, bool);

/// Model-owned F32 RVQ weights retained by a prepared Mimi binding.
#[derive(Clone, Copy, Debug)]
pub struct NativeF32Binding<'a> {
    /// Semantic and acoustic input projections, stored as `[codebook_dim, dim]`.
    pub input_projections: [Option<&'a [u8]>; 2],
    /// Semantic and acoustic output projections, stored as `[dim, codebook_dim]`.
    pub output_projections: [Option<&'a [u8]>; 2],
    /// Codebooks in deterministic split/level order, stored as `[codebook_dim, entries]`.
    pub codebooks: [[Option<&'a [u8]>; 32]; 2],
}

/// Bound runtime facts consumed by one quantizer request.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantizerRuntime<'a> {
    pub model_bound: bool,
    pub n_q: usize,
    pub dim: usize,
    pub semantic_n_q: usize,
    pub codebook_dim: usize,
    pub codebook_entries: usize,
    pub conv_f16: bool,
    pub proj_q8: bool,
    pub encode_stage: Option<EncodeStage>,
    pub decode_stage: Option<DecodeStage>,
    /// Prepared model-owned native F32 decode operands.
    pub native_f32: Option<NativeF32Binding<'a>>,
}

impl<'a> QuantizerRuntime<'a> {
    /// Creates bound runtime facts without allocating.
    #[must_use]
    pub const fn new(
        n_q: usize,
        dim: usize,
        semantic_n_q: usize,
        codebook_dim: usize,
        codebook_entries: usize,
        conv_f16: bool,
        proj_q8: bool,
    ) -> Self {
        Self {
            model_bound: true,
            n_q,
            dim,
            semantic_n_q,
            codebook_dim,
            codebook_entries,
            conv_f16,
            proj_q8,
            encode_stage: None,
            decode_stage: None,
            native_f32: None,
        }
    }

    /// Installs caller-provided synchronous numeric stages.
    #[must_use]
    pub const fn with_stages(
        mut self,
        encode_stage: EncodeStage,
        decode_stage: DecodeStage,
    ) -> Self {
        self.encode_stage = Some(encode_stage);
        self.decode_stage = Some(decode_stage);
        self
    }

    /// Installs model-owned native F32 RVQ encode and decode operands.
    #[must_use]
    pub const fn with_native_f32(mut self, native_f32: NativeF32Binding<'a>) -> Self {
        self.native_f32 = Some(native_f32);
        self
    }

    /// Returns the minimum workspace length required by the pinned guard.
    #[must_use]
    pub const fn workspace_floats(&self) -> usize {
        self.codebook_dim
            .saturating_mul(3)
            .saturating_add(self.dim)
            .saturating_add(8)
    }
}

/// One caller-owned latent column to codebook-index request.
#[derive(Debug)]
pub struct Encode<'a> {
    /// Bound quantizer runtime.
    pub runtime: &'a QuantizerRuntime<'a>,
    /// Latent vector of at least `runtime.dim` values.
    pub latent: &'a [f32],
    pub codes_out: RefCell<&'a mut [i32]>,
    /// Caller-owned quantizer workspace.
    pub workspace: RefCell<&'a mut [f32]>,
    /// Optional synchronous error output channel.
    pub error_out: Option<&'a Cell<QuantizerError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'dispatch, 'event> fn(EncodeDone<'dispatch, 'event>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'dispatch, 'event> fn(EncodeError<'dispatch, 'event>)>,
}

impl<'a> Encode<'a> {
    /// Creates an encode request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'a QuantizerRuntime<'a>,
        latent: &'a [f32],
        codes_out: &'a mut [i32],
        workspace: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            latent,
            codes_out: RefCell::new(codes_out),
            workspace: RefCell::new(workspace),
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
}

/// One caller-owned code prefix to latent-column request.
#[derive(Debug)]
pub struct Decode<'a> {
    /// Bound quantizer runtime.
    pub runtime: &'a QuantizerRuntime<'a>,
    /// Codebook indexes, from semantic through the requested active prefix.
    pub codes: &'a [i32],
    pub latent_out: RefCell<&'a mut [f32]>,
    /// Caller-owned quantizer workspace.
    pub workspace: RefCell<&'a mut [f32]>,
    /// Optional synchronous error output channel.
    pub error_out: Option<&'a Cell<QuantizerError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'dispatch, 'event> fn(DecodeDone<'dispatch, 'event>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'dispatch, 'event> fn(DecodeError<'dispatch, 'event>)>,
}

impl<'a> Decode<'a> {
    /// Creates a decode request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'a QuantizerRuntime<'a>,
        codes: &'a [i32],
        latent_out: &'a mut [f32],
        workspace: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            codes,
            latent_out: RefCell::new(latent_out),
            workspace: RefCell::new(workspace),
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
}

/// Runtime event carrying an encode request.
#[derive(Debug)]
pub struct EncodeRun<'a> {
    /// Request for this synchronous dispatch.
    pub request: Encode<'a>,
}

impl<'a> EncodeRun<'a> {
    /// Wraps an encode request as a runtime event.
    #[must_use]
    pub const fn new(request: Encode<'a>) -> Self {
        Self { request }
    }
}

/// Runtime event carrying a decode request.
#[derive(Debug)]
pub struct DecodeRun<'a> {
    /// Request for this synchronous dispatch.
    pub request: Decode<'a>,
}

impl<'a> DecodeRun<'a> {
    /// Wraps a decode request as a runtime event.
    #[must_use]
    pub const fn new(request: Decode<'a>) -> Self {
        Self { request }
    }
}

/// Successful encode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct EncodeDone<'dispatch, 'event> {
    /// Request completed by this dispatch.
    pub request: &'dispatch Encode<'event>,
}

/// Failed encode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct EncodeError<'dispatch, 'event> {
    /// Request that produced the error.
    pub request: &'dispatch Encode<'event>,
    /// Error selected by the validation path.
    pub error: QuantizerError,
}

/// Successful decode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DecodeDone<'dispatch, 'event> {
    /// Request completed by this dispatch.
    pub request: &'dispatch Decode<'event>,
}

/// Failed decode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DecodeError<'dispatch, 'event> {
    /// Request that produced the error.
    pub request: &'dispatch Decode<'event>,
    /// Error selected by the validation path.
    pub error: QuantizerError,
}
sml! {
    SpeechCodecMimiQuantizer<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_encode_runtime_decision"_s <= *"state_ready"_s + event<EncodeRun>(&'dispatch EncodeRun<'event>),
        "state_encode_shape_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_runtime_bound_encode_run],
        "state_encode_error_error_out_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_runtime_unbound_encode_run] / effect_mark_runtime_unbound_encode_run,
        "state_encode_variant_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_encode_shape_valid],
        "state_encode_error_error_out_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_encode_shape_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_encode_stage_decision"_s <= "state_encode_variant_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_encode_stage_available_encode_run],
        "state_encode_error_error_out_decision"_s <= "state_encode_variant_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_encode_stage_unavailable_encode_run] / effect_mark_encode_stage_unavailable,
        "state_encode_running"_s <= "state_encode_stage_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_class_f32_encode_run] / effect_run_quantize_false_false,
        "state_encode_running"_s <= "state_encode_stage_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_class_f16_encode_run] / effect_run_quantize_true_false,
        "state_encode_running"_s <= "state_encode_stage_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_class_q8_encode_run] / effect_run_quantize_true_true,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_class_f32_native_encode_run] / effect_run_native_quantize_f32,
        "state_encode_success_error_out_decision"_s <= "state_encode_running"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>),
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_success_error_out_decision,
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_no_error_out_encode_run],
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_error_error_out_decision,
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_no_error_out_encode_run],
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_no_done_callback_encode_run],
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>) [guard_no_error_callback_encode_run],
        "state_ready"_s <= "state_encode_done"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>),
        "state_ready"_s <= "state_encode_errored"_s + completion<EncodeRun>(&'dispatch EncodeRun<'event>),
        "state_decode_runtime_decision"_s <= "state_ready"_s + event<DecodeRun>(&'dispatch DecodeRun<'event>),
        "state_decode_shape_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_runtime_bound_decode_run],
        "state_decode_error_error_out_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_runtime_unbound_decode_run] / effect_mark_runtime_unbound_decode_run,
        "state_decode_codes_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_shape_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_shape_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_decode_variant_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_codes_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decode_stage_decision"_s <= "state_decode_variant_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_stage_available_decode_run],
        "state_decode_error_error_out_decision"_s <= "state_decode_variant_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_decode_stage_unavailable_decode_run] / effect_mark_decode_stage_unavailable,
        "state_decode_running"_s <= "state_decode_stage_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_class_f32_decode_run] / effect_run_dequantize_false_false,
        "state_decode_running"_s <= "state_decode_stage_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_class_f16_decode_run] / effect_run_dequantize_true_false,
        "state_decode_running"_s <= "state_decode_stage_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_class_q8_decode_run] / effect_run_dequantize_true_true,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_class_f32_native_decode_run] / effect_run_native_dequantize_f32,
        "state_decode_success_error_out_decision"_s <= "state_decode_running"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>),
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_success_error_out_decision,
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_no_error_out_decode_run],
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_error_error_out_decision,
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_no_error_out_decode_run],
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_no_done_callback_decode_run],
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>) [guard_no_error_callback_decode_run],
        "state_ready"_s <= "state_decode_done"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>),
        "state_ready"_s <= "state_decode_errored"_s + completion<DecodeRun>(&'dispatch DecodeRun<'event>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
    }
}
// SML compares generated values by discriminant only, so these marker impls
// accurately complete its generated `PartialEq` contracts.
impl Eq for SpeechCodecMimiQuantizerStates {}

impl<'dispatch, 'event> Eq for SpeechCodecMimiQuantizerEvents<'dispatch, 'event> where
    'event: 'dispatch
{
}

impl<T: Eq> Eq for SpeechCodecMimiQuantizerError<T> {}

/// Stateless generated-machine context with the per-dispatch selected error.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeechCodecMimiQuantizerContext {
    error: QuantizerError,
}

impl SpeechCodecMimiQuantizerContext {
    /// Returns the most recently selected dispatch error.
    #[must_use]
    pub const fn error(self) -> QuantizerError {
        self.error
    }
    fn clear(&mut self) {
        self.error = QuantizerError::None;
    }
    fn mark(&mut self, error: QuantizerError) {
        self.error = error;
    }
    fn unexpected(&mut self) {
        self.mark(QuantizerError::UnexpectedEvent);
    }
}

fn decode_codes_valid(event: &DecodeRun<'_>) -> bool {
    event.request.codes.iter().all(|code| {
        *code >= 0
            && usize::try_from(*code)
                .is_ok_and(|code| code < event.request.runtime.codebook_entries)
    })
}

fn decode_shape_valid(event: &DecodeRun<'_>) -> bool {
    let request = &event.request;
    let latent_out = request.latent_out.borrow();
    let workspace = request.workspace.borrow();
    !request.codes.is_empty()
        && request.codes.len() >= request.runtime.semantic_n_q
        && request.codes.len() <= request.runtime.n_q
        && !latent_out.is_empty()
        && !workspace.is_empty()
        && latent_out.len() >= request.runtime.dim
        && workspace.len() >= request.runtime.workspace_floats()
}

fn encode_shape_valid(event: &EncodeRun<'_>) -> bool {
    let request = &event.request;
    let codes_out = request.codes_out.borrow();
    let workspace = request.workspace.borrow();
    !request.latent.is_empty()
        && !codes_out.is_empty()
        && !workspace.is_empty()
        && request.latent.len() >= request.runtime.dim
        && codes_out.len() >= request.runtime.n_q
        && workspace.len() >= request.runtime.workspace_floats()
}

impl SpeechCodecMimiQuantizerStateMachineContext for SpeechCodecMimiQuantizerContext {
    fn effect_emit_decode_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            callback(DecodeDone {
                request: &event.request,
            });
        }
        Ok(())
    }
    fn effect_emit_decode_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(DecodeError {
                request: &event.request,
                error: self.error,
            });
        }
        Ok(())
    }
    fn effect_emit_encode_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            callback(EncodeDone {
                request: &event.request,
            });
        }
        Ok(())
    }
    fn effect_emit_encode_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(EncodeError {
                request: &event.request,
                error: self.error,
            });
        }
        Ok(())
    }
    fn effect_mark_encode_stage_unavailable<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::StageUnavailable);
        Ok(())
    }
    fn effect_mark_decode_stage_unavailable<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::StageUnavailable);
        Ok(())
    }
    fn effect_mark_code_range_invalid<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::CodeRange);
        Ok(())
    }
    fn effect_mark_request_shape_invalid_decode_run<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::RequestShape);
        Ok(())
    }
    fn effect_mark_request_shape_invalid_encode_run<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::RequestShape);
        Ok(())
    }
    fn effect_mark_runtime_unbound_decode_run<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::RuntimeUnbound);
        Ok(())
    }
    fn effect_mark_runtime_unbound_encode_run<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(QuantizerError::RuntimeUnbound);
        Ok(())
    }
    fn effect_store_error_out_encode_run_from_state_encode_success_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(QuantizerError::None);
        }
        Ok(())
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn effect_run_quantize_false_false<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_encode(event, false, false);
        Ok(())
    }
    fn effect_run_quantize_true_false<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_encode(event, true, false);
        Ok(())
    }
    fn effect_run_quantize_true_true<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_encode(event, true, true);
        Ok(())
    }
    fn effect_run_dequantize_false_false<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_decode(event, false, false);
        Ok(())
    }
    fn effect_run_dequantize_true_false<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_decode(event, true, false);
        Ok(())
    }
    fn effect_run_dequantize_true_true<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Self::run_decode(event, true, true);
        Ok(())
    }
    fn effect_run_native_quantize_f32<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut codes_out = event.request.codes_out.borrow_mut();
        let mut workspace = event.request.workspace.borrow_mut();
        run_native_quantize_f32(
            event.request.runtime,
            event.request.latent,
            &mut codes_out,
            &mut workspace,
        );
        Ok(())
    }
    fn effect_run_native_dequantize_f32<'dispatch, 'event>(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut latent_out = event.request.latent_out.borrow_mut();
        let mut workspace = event.request.workspace.borrow_mut();
        run_native_dequantize_f32(
            event.request.runtime,
            event.request.codes,
            &mut latent_out,
            &mut workspace,
        );
        Ok(())
    }

    fn effect_store_error_out_decode_run_from_state_decode_error_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(self.error);
        }
        Ok(())
    }
    fn effect_store_error_out_decode_run_from_state_decode_success_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(QuantizerError::None);
        }
        Ok(())
    }
    fn effect_store_error_out_encode_run_from_state_encode_error_error_out_decision<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.request.error_out {
            error_out.set(self.error);
        }
        Ok(())
    }
    fn guard_encode_stage_available_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.runtime.encode_stage.is_some()
                && !native_f32_route(event.request.runtime),
        )
    }
    fn guard_encode_stage_unavailable_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.runtime.encode_stage.is_none()
                && !native_f32_route(event.request.runtime),
        )
    }
    fn guard_decode_stage_available_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.runtime.decode_stage.is_some()
                && !native_f32_route(event.request.runtime),
        )
    }
    fn guard_decode_stage_unavailable_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.runtime.decode_stage.is_none()
                && !native_f32_route(event.request.runtime),
        )
    }
    fn guard_class_f32_native_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(native_f32_route(event.request.runtime))
    }
    fn guard_class_f32_native_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(native_f32_route(event.request.runtime))
    }
    fn guard_class_f16_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.runtime.conv_f16 && !event.request.runtime.proj_q8)
    }
    fn guard_class_f16_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.runtime.conv_f16 && !event.request.runtime.proj_q8)
    }
    fn guard_class_f32_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.request.runtime.conv_f16
            && !event.request.runtime.proj_q8
            && !native_f32_route(event.request.runtime))
    }
    fn guard_class_f32_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.request.runtime.conv_f16
            && !event.request.runtime.proj_q8
            && !native_f32_route(event.request.runtime))
    }
    fn guard_class_q8_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.runtime.proj_q8)
    }
    fn guard_class_q8_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.runtime.proj_q8)
    }

    fn guard_decode_codes_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_decode_codes_valid(event)?)
    }
    fn guard_decode_codes_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(decode_codes_valid(event))
    }
    fn guard_decode_shape_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_decode_shape_valid(event)?)
    }
    fn guard_decode_shape_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(decode_shape_valid(event))
    }
    fn guard_encode_shape_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_encode_shape_valid(event)?)
    }
    fn guard_encode_shape_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(encode_shape_valid(event))
    }
    fn guard_has_done_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_has_done_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_has_error_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_has_error_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_has_error_out_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_has_error_out_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_some())
    }
    fn guard_no_done_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_no_done_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_no_error_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_no_error_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_no_error_out_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_no_error_out_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.is_none())
    }
    fn guard_runtime_bound_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(runtime_bound(event.request.runtime))
    }
    fn guard_runtime_bound_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(runtime_bound(event.request.runtime))
    }
    fn guard_runtime_unbound_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch DecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!runtime_bound(event.request.runtime))
    }
    fn guard_runtime_unbound_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!runtime_bound(event.request.runtime))
    }
}
impl SpeechCodecMimiQuantizerContext {
    fn run_encode(event: &EncodeRun<'_>, conv_f16: bool, proj_q8: bool) {
        if let Some(stage) = event.request.runtime.encode_stage {
            let mut codes_out = event.request.codes_out.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            stage(
                event.request.runtime,
                event.request.latent,
                &mut codes_out,
                &mut workspace,
                conv_f16,
                proj_q8,
            );
        }
    }
    fn run_decode(event: &DecodeRun<'_>, conv_f16: bool, proj_q8: bool) {
        if let Some(stage) = event.request.runtime.decode_stage {
            let mut latent_out = event.request.latent_out.borrow_mut();
            let mut workspace = event.request.workspace.borrow_mut();
            stage(
                event.request.runtime,
                event.request.codes,
                &mut latent_out,
                &mut workspace,
                conv_f16,
                proj_q8,
            );
        }
    }
}
fn native_f32_route(runtime: &QuantizerRuntime<'_>) -> bool {
    let Some(native) = runtime.native_f32 else {
        return false;
    };
    if !runtime.model_bound
        || runtime.dim == 0
        || runtime.codebook_dim == 0
        || runtime.codebook_entries == 0
        || runtime.conv_f16
        || runtime.proj_q8
        || runtime.semantic_n_q == 0
        || runtime.semantic_n_q > 32
        || runtime.n_q < runtime.semantic_n_q
        || runtime.n_q - runtime.semantic_n_q > 32
    {
        return false;
    }
    let projection_bytes = runtime
        .dim
        .saturating_mul(runtime.codebook_dim)
        .saturating_mul(4);
    let codebook_bytes = runtime
        .codebook_entries
        .saturating_mul(runtime.codebook_dim)
        .saturating_mul(4);
    native
        .input_projections
        .iter()
        .chain(native.output_projections.iter())
        .all(|projection| projection.is_some_and(|bytes| bytes.len() >= projection_bytes))
        && native.codebooks[0][..runtime.semantic_n_q]
            .iter()
            .all(|codebook| codebook.is_some_and(|bytes| bytes.len() >= codebook_bytes))
        && native.codebooks[1][..runtime.n_q - runtime.semantic_n_q]
            .iter()
            .all(|codebook| codebook.is_some_and(|bytes| bytes.len() >= codebook_bytes))
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let value = bytes.get(start..start.checked_add(4)?)?;
    Some(f32::from_le_bytes(value.try_into().ok()?))
}

#[allow(clippy::suboptimal_flops)]
fn run_native_quantize_f32(
    runtime: &QuantizerRuntime<'_>,
    latent: &[f32],
    codes_out: &mut [i32],
    workspace: &mut [f32],
) {
    let Some(native) = runtime.native_f32 else {
        return;
    };
    let (residual, _) = workspace.split_at_mut(runtime.codebook_dim);
    let mut code_row = 0usize;
    for (split, level_count) in [runtime.semantic_n_q, runtime.n_q - runtime.semantic_n_q]
        .into_iter()
        .enumerate()
    {
        let Some(projection) = native.input_projections[split] else {
            return;
        };
        for (output, residual_value) in residual.iter_mut().enumerate() {
            let mut value = 0.0;
            for (input, latent_value) in latent[..runtime.dim].iter().enumerate() {
                let Some(weight) = read_f32(
                    projection,
                    output
                        .checked_mul(runtime.dim)
                        .and_then(|row| row.checked_add(input))
                        .unwrap_or(usize::MAX),
                ) else {
                    return;
                };
                value = latent_value.mul_add(weight, value);
            }
            *residual_value = value;
        }
        for level in 0..level_count {
            let Some(codebook) = native.codebooks[split][level] else {
                return;
            };
            let mut best_index = 0usize;
            let mut best_distance = 0.0;
            for (column, residual_value) in
                residual.iter_mut().take(runtime.codebook_dim).enumerate()
            {
                let Some(candidate) = read_f32(codebook, column) else {
                    return;
                };
                let difference = candidate - *residual_value;
                best_distance += difference * difference;
            }
            for entry in 1..runtime.codebook_entries {
                let mut distance = 0.0;
                for (column, residual_value) in
                    residual.iter_mut().take(runtime.codebook_dim).enumerate()
                {
                    let Some(candidate) = read_f32(
                        codebook,
                        entry
                            .checked_mul(runtime.codebook_dim)
                            .and_then(|row| row.checked_add(column))
                            .unwrap_or(usize::MAX),
                    ) else {
                        return;
                    };
                    let difference = candidate - *residual_value;
                    distance += difference * difference;
                }
                if distance < best_distance {
                    best_distance = distance;
                    best_index = entry;
                }
            }
            let Ok(best_index) = i32::try_from(best_index) else {
                return;
            };
            codes_out[code_row] = best_index;
            code_row += 1;
            if level + 1 < level_count {
                for (column, residual_value) in residual.iter_mut().enumerate() {
                    let Some(chosen) = read_f32(
                        codebook,
                        usize::try_from(best_index)
                            .ok()
                            .and_then(|entry| entry.checked_mul(runtime.codebook_dim))
                            .and_then(|row| row.checked_add(column))
                            .unwrap_or(usize::MAX),
                    ) else {
                        return;
                    };
                    *residual_value -= chosen;
                }
            }
        }
    }
}
fn run_native_dequantize_f32(
    runtime: &QuantizerRuntime<'_>,
    codes: &[i32],
    latent: &mut [f32],
    workspace: &mut [f32],
) {
    let Some(native) = runtime.native_f32 else {
        return;
    };
    let dim = runtime.dim;
    let codebook_dim = runtime.codebook_dim;
    let (summed, workspace) = workspace.split_at_mut(codebook_dim);
    let (chosen, workspace) = workspace.split_at_mut(codebook_dim);
    let (projected, _) = workspace.split_at_mut(dim);
    latent[..dim].fill(0.0);
    let mut code_row = 0usize;
    for (split, active) in [
        runtime.semantic_n_q,
        codes.len().saturating_sub(runtime.semantic_n_q),
    ]
    .into_iter()
    .enumerate()
    {
        if active == 0 {
            continue;
        }
        let Some(active_codes) = codes.get(code_row..code_row.saturating_add(active)) else {
            return;
        };
        summed.fill(0.0);
        for (level, code) in active_codes.iter().enumerate() {
            let Ok(code) = usize::try_from(*code) else {
                return;
            };
            let Some(bytes) = native.codebooks[split][level] else {
                return;
            };
            for (column, chosen_value) in chosen.iter_mut().enumerate() {
                let Some(value) = read_f32(
                    bytes,
                    code.checked_mul(codebook_dim)
                        .and_then(|row| row.checked_add(column))
                        .unwrap_or(usize::MAX),
                ) else {
                    return;
                };
                *chosen_value = value;
                summed[column] += value;
            }
        }
        code_row += active;
        let Some(projection) = native.output_projections[split] else {
            return;
        };
        for (output, (projected_value, latent_value)) in
            projected.iter_mut().zip(&mut latent[..dim]).enumerate()
        {
            let mut value = 0.0;
            for (input, summed_value) in summed.iter().enumerate() {
                let Some(weight) = read_f32(
                    projection,
                    output
                        .checked_mul(codebook_dim)
                        .and_then(|row| row.checked_add(input))
                        .unwrap_or(usize::MAX),
                ) else {
                    return;
                };
                value = summed_value.mul_add(weight, value);
            }
            *projected_value = value;
            *latent_value += value;
        }
    }
}

fn runtime_bound(runtime: &QuantizerRuntime<'_>) -> bool {
    runtime.model_bound
        && runtime.n_q > 0
        && runtime.dim > 0
        && runtime.codebook_dim > 0
        && runtime.codebook_entries > 0
}

/// Single-writer synchronous Mimi quantizer actor.
pub struct SpeechCodecMimiQuantizer {
    machine: SpeechCodecMimiQuantizerStateMachine<SpeechCodecMimiQuantizerContext>,
}

impl Default for SpeechCodecMimiQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechCodecMimiQuantizer {
    /// Constructs an actor in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechCodecMimiQuantizerStateMachine::new(
                SpeechCodecMimiQuantizerContext::default(),
            ),
        }
    }

    /// Dispatches one quantizer event synchronously to completion.
    pub fn process_event(&mut self, event: QuantizerEvent<'_>) -> Result<(), QuantizerError> {
        match event {
            QuantizerEvent::Encode(event) => self.process_encode(event),
            QuantizerEvent::Decode(event) => self.process_decode(event),
        }
    }

    /// Dispatches encode synchronously.
    pub fn process_encode(&mut self, event: Encode<'_>) -> Result<(), QuantizerError> {
        self.machine.context_mut().clear();
        let event = EncodeRun::new(event);
        if self
            .machine
            .process_event(SpeechCodecMimiQuantizerEvents::EncodeRun(&event))
            .is_err()
        {
            self.machine.context_mut().error = QuantizerError::UnexpectedEvent;
        }
        self.result()
    }

    /// Dispatches decode synchronously.
    pub fn process_decode(&mut self, event: Decode<'_>) -> Result<(), QuantizerError> {
        self.machine.context_mut().clear();
        let event = DecodeRun::new(event);
        if self
            .machine
            .process_event(SpeechCodecMimiQuantizerEvents::DecodeRun(&event))
            .is_err()
        {
            self.machine.context_mut().error = QuantizerError::UnexpectedEvent;
        }
        self.result()
    }

    fn result(&self) -> Result<(), QuantizerError> {
        let error = self.machine.context().error;
        if error == QuantizerError::None {
            Ok(())
        } else {
            Err(error)
        }
    }

    /// Returns whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &SpeechCodecMimiQuantizerStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the actor context for inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiQuantizerContext {
        self.machine.context()
    }
}

/// Public synchronous quantizer event.
pub enum QuantizerEvent<'a> {
    /// Encode one latent column.
    Encode(Encode<'a>),
    /// Decode one code prefix.
    Decode(Decode<'a>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[f32]) -> Vec<u8> {
        let mut output = Vec::with_capacity(core::mem::size_of_val(values));
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
        output
    }

    fn native<'a>(
        inputs: [&'a [u8]; 2],
        outputs: [&'a [u8]; 2],
        semantic: &[&'a [u8]],
        acoustic: &[&'a [u8]],
    ) -> NativeF32Binding<'a> {
        let mut codebooks = [[None; 32]; 2];
        for (level, codebook) in semantic.iter().enumerate() {
            codebooks[0][level] = Some(*codebook);
        }
        for (level, codebook) in acoustic.iter().enumerate() {
            codebooks[1][level] = Some(*codebook);
        }
        NativeF32Binding {
            input_projections: [Some(inputs[0]), Some(inputs[1])],
            output_projections: [Some(outputs[0]), Some(outputs[1])],
            codebooks,
        }
    }

    fn encode_callback_must_not_run(
        _: &QuantizerRuntime<'_>,
        _: &[f32],
        _: &mut [i32],
        _: &mut [f32],
        _: bool,
        _: bool,
    ) {
        panic!("native F32 encode must not invoke the callback stage");
    }

    fn decode_callback_must_not_run(
        _: &QuantizerRuntime<'_>,
        _: &[i32],
        _: &mut [f32],
        _: &mut [f32],
        _: bool,
        _: bool,
    ) {
        panic!("native F32 decode must not invoke the callback stage");
    }

    #[test]
    fn native_f32_encode_emits_semantic_acoustic_strict_first_residual_codes_and_returns_ready() {
        let input = bytes(&[1.0]);
        let output = bytes(&[1.0]);
        let semantic_first = bytes(&[2.0, 4.0]);
        let semantic_second = bytes(&[0.0, 1.0]);
        let acoustic = bytes(&[3.0, 4.0]);
        let runtime = QuantizerRuntime::new(3, 1, 2, 1, 2, false, false)
            .with_stages(encode_callback_must_not_run, decode_callback_must_not_run)
            .with_native_f32(native(
                [&input, &input],
                [&output, &output],
                &[&semantic_first, &semantic_second],
                &[&acoustic],
            ));
        let mut codes = [-1; 3];
        let mut workspace = [0.0; 12];
        let mut actor = SpeechCodecMimiQuantizer::new();

        assert_eq!(
            actor.process_encode(Encode::new(&runtime, &[3.0], &mut codes, &mut workspace)),
            Ok(())
        );
        assert_eq!(codes, [0, 1, 0]);
        assert!(actor.is(&SpeechCodecMimiQuantizerStates::StateReady));
    }

    #[test]
    fn native_f32_encode_uses_separate_distance_multiply_and_add_for_near_tie() {
        let projection = bytes(&[1.0, 0.0, 0.0, 1.0]);
        let first = [f32::from_bits(0x3f80_15d3), f32::from_bits(0x3f80_0410)];
        let second = [f32::from_bits(0x3f80_11fd), f32::from_bits(0x3f80_07e6)];
        let codebook = bytes(&[first[0], first[1], second[0], second[1]]);
        let runtime = QuantizerRuntime::new(1, 2, 1, 2, 2, false, false).with_native_f32(native(
            [&projection, &projection],
            [&projection, &projection],
            &[&codebook],
            &[],
        ));
        let mut codes = [-1];
        let mut workspace = [0.0; 16];
        let mut actor = SpeechCodecMimiQuantizer::new();

        assert_eq!(
            actor.process_encode(Encode::new(
                &runtime,
                &[0.0, 0.0],
                &mut codes,
                &mut workspace
            )),
            Ok(())
        );
        assert_eq!(codes, [1]);
        assert!(actor.is(&SpeechCodecMimiQuantizerStates::StateReady));
    }

    #[test]
    fn native_f32_decode_runs_through_public_actor_route() {
        let projection = bytes(&[1.0, 2.0, 3.0, 4.0]);
        let semantic_codebook = bytes(&[1.0, 2.0, 3.0, 4.0]);
        let acoustic_codebook = bytes(&[5.0, 6.0, 7.0, 8.0]);
        let runtime = QuantizerRuntime::new(2, 2, 1, 2, 2, false, false).with_native_f32(native(
            [&projection, &projection],
            [&projection, &projection],
            &[&semantic_codebook],
            &[&acoustic_codebook],
        ));
        let codes = [1, 0];
        let mut latent = [0.0; 2];
        let mut workspace = [0.0; 16];
        let mut actor = SpeechCodecMimiQuantizer::new();

        assert_eq!(
            actor.process_decode(Decode::new(&runtime, &codes, &mut latent, &mut workspace)),
            Ok(())
        );
        assert!((latent[0] - 28.0).abs() < f32::EPSILON);
        assert!((latent[1] - 64.0).abs() < f32::EPSILON);
        assert!(actor.is(&SpeechCodecMimiQuantizerStates::StateReady));
    }
}
