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

use core::cell::Cell;
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
    /// The generated machine rejected the event.
    UnexpectedEvent = 5,
}

impl fmt::Display for QuantizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "no error",
            Self::RuntimeUnbound => "quantizer runtime is unbound",
            Self::RequestShape => "quantizer request shape is invalid",
            Self::BufferCapacity => "quantizer buffer capacity is invalid",
            Self::CodeRange => "quantizer code is out of range",
            Self::UnexpectedEvent => "unexpected quantizer event",
        })
    }
}

/// Synchronous numeric encode stage selected by a quantizer transition.
///
/// The final two flags are the selected convolution/projection classes:
/// `(false, false)` is f32, `(true, false)` is f16, and `(true, true)` is q8.
pub type EncodeStage = fn(
    &QuantizerRuntime,
    &[f32],
    &mut [i32],
    &mut [f32],
    bool,
    bool,
);

/// Synchronous numeric decode stage selected by a quantizer transition.
pub type DecodeStage = fn(
    &QuantizerRuntime,
    &[i32],
    &mut [f32],
    &mut [f32],
    bool,
    bool,
);

/// Bound runtime facts consumed by one quantizer request.
#[derive(Clone, Copy, Debug)]
pub struct QuantizerRuntime {
    /// Whether model and quantizer storage have been bound.
    pub model_bound: bool,
    /// Total active codebook levels.
    pub n_q: usize,
    /// Latent vector width.
    pub dim: usize,
    /// Minimum active semantic code prefix.
    pub semantic_n_q: usize,
    /// Codebook vector width used by the workspace contract.
    pub codebook_dim: usize,
    /// Number of entries in every codebook.
    pub codebook_entries: usize,
    /// Selects the reference f16 projection/convolution class.
    pub conv_f16: bool,
    /// Selects the q8 projection class, which takes precedence over f16.
    pub proj_q8: bool,
    /// Bound numeric encode stage.
    pub encode_stage: Option<EncodeStage>,
    /// Bound numeric decode stage.
    pub decode_stage: Option<DecodeStage>,
}

impl Default for QuantizerRuntime {
    fn default() -> Self {
        Self {
            model_bound: false,
            n_q: 0,
            dim: 0,
            semantic_n_q: 0,
            codebook_dim: 0,
            codebook_entries: 0,
            conv_f16: false,
            proj_q8: false,
            encode_stage: None,
            decode_stage: None,
        }
    }
}

impl QuantizerRuntime {
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

    /// Returns the minimum workspace length required by the pinned guard.
    #[must_use]
    pub const fn workspace_floats(self) -> usize {
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
    pub runtime: &'a QuantizerRuntime,
    /// Latent vector of at least `runtime.dim` values.
    pub latent: &'a [f32],
    /// Destination code indexes of at least `runtime.n_q` values.
    pub codes_out: &'a mut [i32],
    /// Caller-owned quantizer workspace.
    pub workspace: &'a mut [f32],
    /// Optional synchronous error output channel.
    pub error_out: Option<&'a Cell<QuantizerError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'r> fn(EncodeDone<'r>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'r> fn(EncodeError<'r>)>,
}

impl<'a> Encode<'a> {
    /// Creates an encode request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'a QuantizerRuntime,
        latent: &'a [f32],
        codes_out: &'a mut [i32],
        workspace: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            latent,
            codes_out,
            workspace,
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
    pub runtime: &'a QuantizerRuntime,
    /// Codebook indexes, from semantic through the requested active prefix.
    pub codes: &'a [i32],
    /// Destination latent vector of at least `runtime.dim` values.
    pub latent_out: &'a mut [f32],
    /// Caller-owned quantizer workspace.
    pub workspace: &'a mut [f32],
    /// Optional synchronous error output channel.
    pub error_out: Option<&'a Cell<QuantizerError>>,
    /// Optional synchronous success callback.
    pub on_done: Option<for<'r> fn(DecodeDone<'r>)>,
    /// Optional synchronous error callback.
    pub on_error: Option<for<'r> fn(DecodeError<'r>)>,
}

impl<'a> Decode<'a> {
    /// Creates a decode request over caller-owned buffers.
    #[must_use]
    pub const fn new(
        runtime: &'a QuantizerRuntime,
        codes: &'a [i32],
        latent_out: &'a mut [f32],
        workspace: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            codes,
            latent_out,
            workspace,
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
    pub const fn new(request: Encode<'a>) -> Self { Self { request } }
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
    pub const fn new(request: Decode<'a>) -> Self { Self { request } }
}

/// Successful encode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct EncodeDone<'a> {
    /// Request completed by this dispatch.
    pub request: &'a Encode<'a>,
}

/// Failed encode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct EncodeError<'a> {
    /// Request that produced the error.
    pub request: &'a Encode<'a>,
    /// Error selected by the validation path.
    pub error: QuantizerError,
}

/// Successful decode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DecodeDone<'a> {
    /// Request completed by this dispatch.
    pub request: &'a Decode<'a>,
}

/// Failed decode callback payload.
#[derive(Clone, Copy, Debug)]
pub struct DecodeError<'a> {
    /// Request that produced the error.
    pub request: &'a Decode<'a>,
    /// Error selected by the validation path.
    pub error: QuantizerError,
}

sml! {
    SpeechCodecMimiQuantizer {
        "state_encode_runtime_decision"_s <= *"state_ready"_s + event<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>),
        "state_encode_shape_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_runtime_bound_encode_run],
        "state_encode_error_error_out_decision"_s <= "state_encode_runtime_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_runtime_unbound_encode_run] / effect_mark_runtime_unbound_encode_run,
        "state_encode_variant_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_encode_shape_valid],
        "state_encode_error_error_out_decision"_s <= "state_encode_shape_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_encode_shape_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_class_f32_encode_run] / effect_run_quantize_false_false,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_class_f16_encode_run] / effect_run_quantize_true_false,
        "state_encode_running"_s <= "state_encode_variant_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_class_q8_encode_run] / effect_run_quantize_true_true,
        "state_encode_success_error_out_decision"_s <= "state_encode_running"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>),
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_success_error_out_decision,
        "state_encode_success_callback_decision"_s <= "state_encode_success_error_out_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_out_encode_run],
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run_from_state_encode_error_error_out_decision,
        "state_encode_error_callback_decision"_s <= "state_encode_error_error_out_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_out_encode_run],
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_encode_done"_s <= "state_encode_success_callback_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_no_done_callback_encode_run],
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_encode_errored"_s <= "state_encode_error_callback_decision"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>) [guard_no_error_callback_encode_run],
        "state_ready"_s <= "state_encode_done"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>),
        "state_ready"_s <= "state_encode_errored"_s + completion<EncodeRun<'dispatch>>(&'dispatch EncodeRun<'dispatch>),
        "state_decode_runtime_decision"_s <= "state_ready"_s + event<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>),
        "state_decode_shape_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_runtime_bound_decode_run],
        "state_decode_error_error_out_decision"_s <= "state_decode_runtime_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_runtime_unbound_decode_run] / effect_mark_runtime_unbound_decode_run,
        "state_decode_codes_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_decode_shape_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_shape_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_decode_shape_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_decode_variant_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_decode_codes_valid],
        "state_decode_error_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_class_f32_decode_run] / effect_run_dequantize_false_false,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_class_f16_decode_run] / effect_run_dequantize_true_false,
        "state_decode_running"_s <= "state_decode_variant_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_class_q8_decode_run] / effect_run_dequantize_true_true,
        "state_decode_success_error_out_decision"_s <= "state_decode_running"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>),
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_success_error_out_decision,
        "state_decode_success_callback_decision"_s <= "state_decode_success_error_out_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_out_decode_run],
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run_from_state_decode_error_error_out_decision,
        "state_decode_error_callback_decision"_s <= "state_decode_error_error_out_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_out_decode_run],
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_decode_done"_s <= "state_decode_success_callback_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_no_done_callback_decode_run],
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_decode_errored"_s <= "state_decode_error_callback_decision"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>) [guard_no_error_callback_decode_run],
        "state_ready"_s <= "state_decode_done"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>),
        "state_ready"_s <= "state_decode_errored"_s + completion<DecodeRun<'dispatch>>(&'dispatch DecodeRun<'dispatch>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_encode_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_runtime_decision,
        "state_ready"_s <= "state_encode_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_shape_decision,
        "state_ready"_s <= "state_encode_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_variant_decision,
        "state_ready"_s <= "state_encode_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_running,
        "state_ready"_s <= "state_encode_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_success_error_out_decision,
        "state_ready"_s <= "state_encode_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_success_callback_decision,
        "state_ready"_s <= "state_encode_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_error_error_out_decision,
        "state_ready"_s <= "state_encode_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_error_callback_decision,
        "state_ready"_s <= "state_encode_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_done,
        "state_ready"_s <= "state_encode_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_encode_errored,
        "state_ready"_s <= "state_decode_runtime_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_runtime_decision,
        "state_ready"_s <= "state_decode_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_shape_decision,
        "state_ready"_s <= "state_decode_codes_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_codes_decision,
        "state_ready"_s <= "state_decode_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_variant_decision,
        "state_ready"_s <= "state_decode_running"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_running,
        "state_ready"_s <= "state_decode_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_success_error_out_decision,
        "state_ready"_s <= "state_decode_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_success_callback_decision,
        "state_ready"_s <= "state_decode_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_error_error_out_decision,
        "state_ready"_s <= "state_decode_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_error_callback_decision,
        "state_ready"_s <= "state_decode_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_done,
        "state_ready"_s <= "state_decode_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_errored,
    }
}

/// Stateless generated-machine context with the per-dispatch selected error.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeechCodecMimiQuantizerContext {
    error: QuantizerError,
}

impl SpeechCodecMimiQuantizerContext {
    /// Returns the most recently selected dispatch error.
    #[must_use]
    pub const fn error(&self) -> QuantizerError { self.error }
    fn clear(&mut self) { self.error = QuantizerError::None; }
    fn mark(&mut self, error: QuantizerError) { self.error = error; }
    fn unexpected(&mut self) -> Result<(), ()> { self.mark(QuantizerError::UnexpectedEvent); Ok(()) }
}

impl SpeechCodecMimiQuantizerStateMachineContext for SpeechCodecMimiQuantizerContext {
    fn effect_emit_decode_done(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_done { callback(DecodeDone { request: &event.request }); }
        Ok(())
    }
    fn effect_emit_decode_error(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_error { callback(DecodeError { request: &event.request, error: self.error }); }
        Ok(())
    }
    fn effect_emit_encode_done(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_done { callback(EncodeDone { request: &event.request }); }
        Ok(())
    }
    fn effect_emit_encode_error(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.on_error { callback(EncodeError { request: &event.request, error: self.error }); }
        Ok(())
    }
    fn effect_mark_code_range_invalid(&mut self, _event: &DecodeRun<'_>) -> Result<(), ()> { self.mark(QuantizerError::CodeRange); Ok(()) }
    fn effect_mark_request_shape_invalid_decode_run(&mut self, _event: &DecodeRun<'_>) -> Result<(), ()> { self.mark(QuantizerError::RequestShape); Ok(()) }
    fn effect_mark_request_shape_invalid_encode_run(&mut self, _event: &EncodeRun<'_>) -> Result<(), ()> { self.mark(QuantizerError::RequestShape); Ok(()) }
    fn effect_mark_runtime_unbound_decode_run(&mut self, _event: &DecodeRun<'_>) -> Result<(), ()> { self.mark(QuantizerError::RuntimeUnbound); Ok(()) }
    fn effect_mark_runtime_unbound_encode_run(&mut self, _event: &EncodeRun<'_>) -> Result<(), ()> { self.mark(QuantizerError::RuntimeUnbound); Ok(()) }

    fn effect_run_quantize_false_false(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { self.run_encode(event, false, false); Ok(()) }
    fn effect_run_quantize_true_false(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { self.run_encode(event, true, false); Ok(()) }
    fn effect_run_quantize_true_true(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { self.run_encode(event, true, true); Ok(()) }
    fn effect_run_dequantize_false_false(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { self.run_decode(event, false, false); Ok(()) }
    fn effect_run_dequantize_true_false(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { self.run_decode(event, true, false); Ok(()) }
    fn effect_run_dequantize_true_true(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { self.run_decode(event, true, true); Ok(()) }

    fn effect_store_error_out_decode_run_from_state_decode_error_error_out_decision(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { if let Some(error_out) = event.request.error_out { error_out.set(self.error); } Ok(()) }
    fn effect_store_error_out_decode_run_from_state_decode_success_error_out_decision(&mut self, event: &DecodeRun<'_>) -> Result<(), ()> { if let Some(error_out) = event.request.error_out { error_out.set(QuantizerError::None); } Ok(()) }
    fn effect_store_error_out_encode_run_from_state_encode_error_error_out_decision(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { if let Some(error_out) = event.request.error_out { error_out.set(self.error); } Ok(()) }
    fn effect_store_error_out_encode_run_from_state_encode_success_error_out_decision(&mut self, event: &EncodeRun<'_>) -> Result<(), ()> { if let Some(error_out) = event.request.error_out { error_out.set(QuantizerError::None); } Ok(()) }

    fn guard_class_f16_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.conv_f16 && !event.request.runtime.proj_q8) }
    fn guard_class_f16_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.conv_f16 && !event.request.runtime.proj_q8) }
    fn guard_class_f32_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.runtime.conv_f16 && !event.request.runtime.proj_q8) }
    fn guard_class_f32_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.runtime.conv_f16 && !event.request.runtime.proj_q8) }
    fn guard_class_q8_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.proj_q8) }
    fn guard_class_q8_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.runtime.proj_q8) }

    fn guard_decode_codes_invalid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_decode_codes_valid(event)?) }
    fn guard_decode_codes_valid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> {
        Ok(event.request.codes.iter().all(|code| *code >= 0 && usize::try_from(*code).ok().is_some_and(|code| code < event.request.runtime.codebook_entries)))
    }
    fn guard_decode_shape_invalid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_decode_shape_valid(event)?) }
    fn guard_decode_shape_valid(&self, event: &DecodeRun<'_>) -> Result<bool, ()> {
        let request = &event.request;
        Ok(!request.codes.is_empty() && request.codes.len() >= request.runtime.semantic_n_q && request.codes.len() <= request.runtime.n_q && !request.latent_out.is_empty() && !request.workspace.is_empty() && request.latent_out.len() >= request.runtime.dim && request.workspace.len() >= request.runtime.workspace_floats())
    }
    fn guard_encode_shape_invalid(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_encode_shape_valid(event)?) }
    fn guard_encode_shape_valid(&self, event: &EncodeRun<'_>) -> Result<bool, ()> {
        let request = &event.request;
        Ok(!request.latent.is_empty() && !request.codes_out.is_empty() && !request.workspace.is_empty() && request.latent.len() >= request.runtime.dim && request.codes_out.len() >= request.runtime.n_q && request.workspace.len() >= request.runtime.workspace_floats())
    }

    fn guard_has_done_callback_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_some()) }
    fn guard_has_done_callback_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_some()) }
    fn guard_has_error_callback_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_some()) }
    fn guard_has_error_callback_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_some()) }
    fn guard_has_error_out_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_has_error_out_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_no_done_callback_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_none()) }
    fn guard_no_done_callback_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_none()) }
    fn guard_no_error_callback_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_none()) }
    fn guard_no_error_callback_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_none()) }
    fn guard_no_error_out_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_no_error_out_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_runtime_bound_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(runtime_bound(event.request.runtime)) }
    fn guard_runtime_bound_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(runtime_bound(event.request.runtime)) }
    fn guard_runtime_unbound_decode_run(&self, event: &DecodeRun<'_>) -> Result<bool, ()> { Ok(!runtime_bound(event.request.runtime)) }
    fn guard_runtime_unbound_encode_run(&self, event: &EncodeRun<'_>) -> Result<bool, ()> { Ok(!runtime_bound(event.request.runtime)) }

    fn effect_on_unexpected_from_state_decode_codes_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_running(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_runtime_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_shape_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_running(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_runtime_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_shape_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encode_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
}

impl SpeechCodecMimiQuantizerContext {
    fn run_encode(&mut self, event: &EncodeRun<'_>, conv_f16: bool, proj_q8: bool) {
        if let Some(stage) = event.request.runtime.encode_stage {
            stage(event.request.runtime, event.request.latent, event.request.codes_out, event.request.workspace, conv_f16, proj_q8);
        }
    }
    fn run_decode(&mut self, event: &DecodeRun<'_>, conv_f16: bool, proj_q8: bool) {
        if let Some(stage) = event.request.runtime.decode_stage {
            stage(event.request.runtime, event.request.codes, event.request.latent_out, event.request.workspace, conv_f16, proj_q8);
        }
    }
}

fn runtime_bound(runtime: &QuantizerRuntime) -> bool {
    runtime.model_bound && runtime.n_q > 0 && runtime.dim > 0 && runtime.codebook_dim > 0 && runtime.codebook_entries > 0
}

/// Single-writer synchronous Mimi quantizer actor.
pub struct SpeechCodecMimiQuantizer {
    machine: SpeechCodecMimiQuantizerStateMachine<SpeechCodecMimiQuantizerContext>,
}

impl Default for SpeechCodecMimiQuantizer {
    fn default() -> Self { Self::new() }
}

impl SpeechCodecMimiQuantizer {
    /// Constructs an actor in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: SpeechCodecMimiQuantizerStateMachine::new(SpeechCodecMimiQuantizerContext::default()) }
    }

    /// Dispatches one bounded encode request synchronously.
    pub fn process_encode(&mut self, request: Encode<'_>) -> Result<(), QuantizerError> {
        self.machine.context_mut().clear();
        let event = EncodeRun::new(request);
        if self.machine.process_event(SpeechCodecMimiQuantizerEvents::EncodeRun(&event)).is_err() {
            self.machine.context_mut().mark(QuantizerError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == QuantizerError::None { Ok(()) } else { Err(error) }
    }

    /// Dispatches one bounded decode request synchronously.
    pub fn process_decode(&mut self, request: Decode<'_>) -> Result<(), QuantizerError> {
        self.machine.context_mut().clear();
        let event = DecodeRun::new(request);
        if self.machine.process_event(SpeechCodecMimiQuantizerEvents::DecodeRun(&event)).is_err() {
            self.machine.context_mut().mark(QuantizerError::UnexpectedEvent);
        }
        let error = self.machine.context().error();
        if error == QuantizerError::None { Ok(()) } else { Err(error) }
    }

    /// Dispatches either supported request type synchronously.
    pub fn process_event(&mut self, event: QuantizerEvent<'_>) -> Result<(), QuantizerError> {
        match event {
            QuantizerEvent::Encode(request) => self.process_encode(request),
            QuantizerEvent::Decode(request) => self.process_decode(request),
        }
    }

    /// Records an unsupported event as an explicit internal error.
    pub fn process_unexpected_event(&mut self) -> Result<(), QuantizerError> {
        self.machine.context_mut().unexpected().ok();
        Err(QuantizerError::UnexpectedEvent)
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiQuantizerStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: &SpeechCodecMimiQuantizerStates) -> bool { self.machine.is(state) }

    /// Returns the actor context for inspection.
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiQuantizerContext { self.machine.context() }
}

/// Public synchronous quantizer event.
pub enum QuantizerEvent<'a> {
    /// Encode one latent column.
    Encode(Encode<'a>),
    /// Decode one code prefix.
    Decode(Decode<'a>),
}
