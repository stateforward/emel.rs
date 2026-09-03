//! Source-aligned synchronous Sortformer pipeline actor.
//!
//! The pipeline owns only bounded per-dispatch bookkeeping. Model work is
//! supplied through caller-owned, synchronous function pointers; callbacks may
//! write only into the buffers supplied by the run event and must not retain or
//! re-enter the actor.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    missing_debug_implementations,
    dead_code,
    unused_imports,
    missing_docs
)]

use super::super::encoder::feature_extractor::{
    Error as NativeFeatureExtractorError, Route as NativeFeatureExtractorRoute,
};
use super::super::executor::native_projection::{
    Error as NativeProjectionError, Route as NativeProjectionRoute,
};
use super::super::executor::native_transformer::{
    Error as NativeTransformerError, Route as NativeTransformerRoute,
};
use super::super::{NativeOutputError, NativeOutputRoute};
use core::cell::RefCell;
use sml::sml;

/// Fixed dimensions from the maintained Sortformer pipeline contract.
pub const SAMPLE_RATE: i32 = 16_000;
pub const CHANNEL_COUNT: i32 = 1;
pub const FRAME_COUNT: i32 = 188;
pub const FEATURE_BIN_COUNT: i32 = 128;
pub const FEATURE_FRAME_COUNT: i32 = 1_504;
pub const HIDDEN_DIM: i32 = 192;
pub const SPEAKER_COUNT: i32 = 4;
pub const CHUNK_LEN: i32 = FRAME_COUNT;
pub const REQUIRED_SAMPLE_COUNT: usize = 240_640;
pub const REQUIRED_FEATURE_COUNT: usize = 1_504 * 128;
pub const REQUIRED_ENCODER_VALUE_COUNT: usize = 188 * 512;
pub const REQUIRED_HIDDEN_VALUE_COUNT: usize = 188 * 192;
pub const REQUIRED_PROBABILITY_VALUE_COUNT: usize = 188 * 4;
const REQUIRED_PROBABILITY_VALUE_COUNT_I32: i32 = 752;
const MAX_SEGMENT_COUNT_I32: i32 = REQUIRED_PROBABILITY_VALUE_COUNT_I32;
pub const MAX_SEGMENT_COUNT: usize = REQUIRED_PROBABILITY_VALUE_COUNT;

/// Pipeline errors from the pinned C++ error domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum PipelineError {
    #[default]
    None = 0,
    ModelInvalid = 1 << 0,
    SampleRate = 1 << 1,
    ChannelCount = 1 << 2,
    PcmShape = 1 << 3,
    ProbabilityCapacity = 1 << 4,
    SegmentCapacity = 1 << 5,
    TensorContract = 1 << 6,
    Request = 1 << 7,
    Executor = 1 << 8,
    Unexpected = 1 << 9,
    Kernel = 1 << 10,
}

/// A decoded diarization interval written by the segment callback.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SegmentRecord {
    pub speaker: i32,
    pub start_frame: i32,
    pub end_frame: i32,
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub max_probability: f32,
}

/// Caller-owned Sortformer model and runtime contract.
///
/// The four family counts and readiness bits are the bounded Rust equivalent
/// of the C++ model pointers and tensor records. Numeric operations are
/// intentionally callback-owned because the lower model/kernel actors may be
/// supplied by different integrations.
#[allow(clippy::struct_excessive_bools)]
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PipelineContract {
    pub model_present: bool,
    pub sample_rate: i32,
    pub speaker_count: i32,
    pub chunk_len: i32,
    pub feature_extractor_tensors: u32,
    pub encoder_tensors: u32,
    pub modules_tensors: u32,
    pub transformer_encoder_tensors: u32,
    pub feature_extractor_ready: bool,
    pub encoder_ready: bool,
    pub modules_ready: bool,
    pub transformer_encoder_ready: bool,
    pub extract_features: Option<ExtractFeaturesFn>,
    pub bind_encoder: Option<BindFn>,
    pub encode_frames: Option<EncodeFramesFn>,
    pub execute_hidden: Option<ExecuteHiddenFn>,
    pub bind_modules: Option<BindFn>,
    pub compute_probabilities: Option<ComputeProbabilitiesFn>,
    pub decode_segments: Option<DecodeSegmentsFn>,
}

/// Feature extraction callback.
pub type ExtractFeaturesFn = fn(&[f32], &PipelineContract, &mut [f32]) -> bool;
/// Contract binding callback for encoder and modules.
pub type BindFn = fn(&PipelineContract) -> bool;
/// Encoder callback: features to 512-wide encoder frames.
pub type EncodeFramesFn = fn(&[f32], &PipelineContract, &mut [f32]) -> bool;
/// Executor callback: encoder frames to 192-wide hidden frames.
pub type ExecuteHiddenFn = fn(&[f32], &PipelineContract, &mut [f32]) -> bool;
/// Probability callback: hidden frames to frame-major speaker probabilities.
pub type ComputeProbabilitiesFn = fn(&[f32], &PipelineContract, &mut [f32]) -> bool;
/// Segment decoder callback. It must set `segment_count` within `segments`.
pub type DecodeSegmentsFn = fn(&[f32], &mut [SegmentRecord], &mut i32) -> bool;

impl PipelineContract {
    /// Returns the pinned metadata contract without runtime callbacks.
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            model_present: true,
            sample_rate: SAMPLE_RATE,
            speaker_count: SPEAKER_COUNT,
            chunk_len: CHUNK_LEN,
            feature_extractor_tensors: 1,
            encoder_tensors: 1,
            modules_tensors: 1,
            transformer_encoder_tensors: 1,
            feature_extractor_ready: true,
            encoder_ready: true,
            modules_ready: true,
            transformer_encoder_ready: true,
            extract_features: None,
            bind_encoder: None,
            encode_frames: None,
            execute_hidden: None,
            bind_modules: None,
            compute_probabilities: None,
            decode_segments: None,
        }
    }

    #[must_use]
    pub const fn model_contract_valid(self) -> bool {
        self.model_present
            && self.sample_rate == SAMPLE_RATE
            && self.speaker_count == SPEAKER_COUNT
            && self.chunk_len == CHUNK_LEN
            && self.feature_extractor_tensors != 0
            && self.encoder_tensors != 0
            && self.modules_tensors != 0
            && self.transformer_encoder_tensors != 0
    }
}

/// Compatibility spelling for integrations that call the model contract an execution contract.
pub type ExecutionContract = PipelineContract;
/// Short error spelling retained for pipeline callers.
pub type Error = PipelineError;

/// Successful completion payload.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RunDone {
    pub frame_count: i32,
    pub probability_count: i32,
    pub segment_count: i32,
}
/// Failed completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunError {
    pub error: PipelineError,
}
/// Caller-owned completion callback.
pub type DoneCallback = fn(RunDone) -> bool;
/// Caller-owned error callback.
pub type ErrorCallback = fn(RunError) -> bool;
pub struct EventRunFlow<'a> {
    pub contract: &'a PipelineContract,
    pub pcm: &'a [f32],
    pub sample_rate: i32,
    pub channel_count: i32,
    pub features: RefCell<&'a mut [f32]>,
    pub encoder_frames: RefCell<&'a mut [f32]>,
    pub hidden: RefCell<&'a mut [f32]>,
    pub probabilities: RefCell<&'a mut [f32]>,
    pub segments: RefCell<&'a mut [SegmentRecord]>,
    pub frame_count_out: RefCell<&'a mut i32>,
    pub probability_count_out: RefCell<&'a mut i32>,
    pub segment_count_out: RefCell<&'a mut i32>,
    pub error_out: RefCell<&'a mut PipelineError>,
    /// Caller-owned native feature-extractor route and reusable workspace.
    pub native_feature_extractor_route: RefCell<Option<&'a mut NativeFeatureExtractorRoute<'a>>>,
    /// Caller-owned native encoder-projection route and reusable workspace.
    pub native_projection_route: RefCell<Option<&'a mut NativeProjectionRoute<'a>>>,
    /// Caller-owned native transformer route and reusable workspace.
    pub native_transformer_route: RefCell<Option<&'a mut NativeTransformerRoute<'a>>>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
    pub native_output_route: RefCell<Option<&'a mut NativeOutputRoute<'a>>>,
}

impl<'a> EventRunFlow<'a> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract: &'a PipelineContract,
        pcm: &'a [f32],
        sample_rate: i32,
        channel_count: i32,
        features: &'a mut [f32],
        encoder_frames: &'a mut [f32],
        hidden: &'a mut [f32],
        probabilities: &'a mut [f32],
        segments: &'a mut [SegmentRecord],
        frame_count_out: &'a mut i32,
        probability_count_out: &'a mut i32,
        segment_count_out: &'a mut i32,
        error_out: &'a mut PipelineError,
    ) -> Self {
        Self {
            contract,
            pcm,
            sample_rate,
            channel_count,
            features: RefCell::new(features),
            encoder_frames: RefCell::new(encoder_frames),
            hidden: RefCell::new(hidden),
            probabilities: RefCell::new(probabilities),
            segments: RefCell::new(segments),
            frame_count_out: RefCell::new(frame_count_out),
            probability_count_out: RefCell::new(probability_count_out),
            segment_count_out: RefCell::new(segment_count_out),
            error_out: RefCell::new(error_out),
            native_feature_extractor_route: RefCell::new(None),
            native_projection_route: RefCell::new(None),
            native_transformer_route: RefCell::new(None),
            native_output_route: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }
    #[must_use]
    pub fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
    #[must_use]
    pub fn with_native_output_route(mut self, route: &'a mut NativeOutputRoute<'a>) -> Self {
        self.native_output_route.get_mut().replace(route);
        self
    }
    #[must_use]
    pub fn with_native_feature_extractor_route(
        mut self,
        route: &'a mut NativeFeatureExtractorRoute<'a>,
    ) -> Self {
        *self.native_feature_extractor_route.get_mut() = Some(route);
        self
    }
    #[must_use]
    pub fn with_native_projection_route(
        mut self,
        route: &'a mut NativeProjectionRoute<'a>,
    ) -> Self {
        *self.native_projection_route.get_mut() = Some(route);
        self
    }
    #[must_use]
    pub fn with_native_transformer_route(
        mut self,
        route: &'a mut NativeTransformerRoute<'a>,
    ) -> Self {
        *self.native_transformer_route.get_mut() = Some(route);
        self
    }
}

sml! {
    DiarizationSortformerPipeline<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<EventRunFlow<'event>>(&'dispatch EventRunFlow<'event>) / effect_begin_run,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_model_contract_valid],
        "state_publish_error"_s <= "state_model_contract_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_sample_rate_valid],
        "state_publish_error"_s <= "state_sample_rate_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_channel_count_valid],
        "state_publish_error"_s <= "state_channel_count_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_feature_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_pcm_shape_valid],
        "state_publish_error"_s <= "state_pcm_shape_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_encoder_capacity_decision"_s <= "state_feature_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_feature_capacity_valid],
        "state_publish_error"_s <= "state_feature_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_feature_capacity_invalid] / effect_mark_feature_capacity_invalid,
        "state_hidden_capacity_decision"_s <= "state_encoder_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_capacity_valid],
        "state_publish_error"_s <= "state_encoder_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_capacity_invalid] / effect_mark_encoder_capacity_invalid,
        "state_probability_capacity_decision"_s <= "state_hidden_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_hidden_capacity_valid],
        "state_publish_error"_s <= "state_hidden_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_hidden_capacity_invalid] / effect_mark_hidden_capacity_invalid,
        "state_segment_capacity_decision"_s <= "state_probability_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_capacity_valid],
        "state_publish_error"_s <= "state_probability_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_capacity_invalid] / effect_mark_probability_capacity_invalid,
        "state_tensor_contract_decision"_s <= "state_segment_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_segment_capacity_valid],
        "state_publish_error"_s <= "state_segment_capacity_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_segment_capacity_invalid] / effect_mark_segment_capacity_invalid,
        "state_preparing_features"_s <= "state_tensor_contract_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_tensor_contract_valid],
        "state_publish_error"_s <= "state_tensor_contract_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_tensor_contract_invalid] / effect_mark_tensor_contract_invalid,
        "state_prepare_decision"_s <= "state_preparing_features"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_prepare_features,
        "state_binding_encoder"_s <= "state_prepare_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_no_error],
        "state_publish_error"_s <= "state_prepare_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_has_error],
        "state_computing_encoder"_s <= "state_binding_encoder"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_bind_encoder,
        "state_executing_hidden"_s <= "state_computing_encoder"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_kernel_ready] / effect_compute_encoder_frames,
        "state_publish_error"_s <= "state_computing_encoder"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_kernel_unavailable] / effect_mark_kernel_error_from_state_computing_encoder,
        "state_encoder_compute_decision"_s <= "state_executing_hidden"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_compute_succeeded],
        "state_publish_error"_s <= "state_executing_hidden"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_encoder_compute_failed],
        "state_execute_decision"_s <= "state_encoder_compute_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_execute_hidden,
        "state_binding_modules"_s <= "state_execute_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_callback_probability_route],
        "state_probability_decision"_s <= "state_execute_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_native_probability_route] / effect_compute_probabilities_native,
        "state_publish_error"_s <= "state_execute_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_route_missing] / effect_mark_kernel_error_from_state_execute_decision,
        "state_computing_probabilities"_s <= "state_binding_modules"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_bind_modules,
        "state_probability_decision"_s <= "state_computing_probabilities"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_kernel_ready] / effect_compute_probabilities,
        "state_publish_error"_s <= "state_computing_probabilities"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_kernel_unavailable] / effect_mark_kernel_error_from_state_computing_probabilities,
        "state_probability_compute_decision"_s <= "state_probability_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_compute_succeeded],
        "state_publish_error"_s <= "state_probability_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_probability_compute_failed],
        "state_segment_decode_decision"_s <= "state_probability_compute_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>),
        "state_native_decoding_segments"_s <= "state_segment_decode_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_native_segment_route] / effect_decode_segments_native,
        "state_callback_decoding_segments"_s <= "state_segment_decode_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_callback_segment_route] / effect_decode_segments,
        "state_publish_error"_s <= "state_segment_decode_decision"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_segment_route_missing] / effect_mark_kernel_error_from_state_segment_decode_decision,
        "state_publish_success"_s <= "state_native_decoding_segments"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_no_error],
        "state_publish_error"_s <= "state_native_decoding_segments"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_has_error],
        "state_publish_success"_s <= "state_callback_decoding_segments"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_no_error],
        "state_publish_error"_s <= "state_callback_decoding_segments"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) [guard_has_error],
        "state_done"_s <= "state_publish_success"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_publish_success,
        "state_errored"_s <= "state_publish_error"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>) / effect_publish_error,
        "state_ready"_s <= "state_done"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>),
        "state_ready"_s <= "state_errored"_s + completion<EventRunFlow>(&'dispatch EventRunFlow<'event>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_sample_rate_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_sample_rate_decision,
        "state_ready"_s <= "state_channel_count_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_channel_count_decision,
        "state_ready"_s <= "state_pcm_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_pcm_shape_decision,
        "state_ready"_s <= "state_probability_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_probability_capacity_decision,
        "state_ready"_s <= "state_segment_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_segment_capacity_decision,
        "state_ready"_s <= "state_tensor_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_tensor_contract_decision,
        "state_ready"_s <= "state_preparing_features"_s + unexpected_event<_> / effect_on_unexpected_from_state_preparing_features,
        "state_ready"_s <= "state_prepare_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_prepare_decision,
        "state_ready"_s <= "state_binding_encoder"_s + unexpected_event<_> / effect_on_unexpected_from_state_binding_encoder,
        "state_ready"_s <= "state_computing_encoder"_s + unexpected_event<_> / effect_on_unexpected_from_state_computing_encoder,
        "state_ready"_s <= "state_encoder_compute_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encoder_compute_decision,
        "state_ready"_s <= "state_executing_hidden"_s + unexpected_event<_> / effect_on_unexpected_from_state_executing_hidden,
        "state_ready"_s <= "state_execute_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_execute_decision,
        "state_ready"_s <= "state_binding_modules"_s + unexpected_event<_> / effect_on_unexpected_from_state_binding_modules,
        "state_ready"_s <= "state_computing_probabilities"_s + unexpected_event<_> / effect_on_unexpected_from_state_computing_probabilities,
        "state_ready"_s <= "state_feature_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_feature_capacity_decision,
        "state_ready"_s <= "state_encoder_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encoder_capacity_decision,
        "state_ready"_s <= "state_hidden_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_hidden_capacity_decision,
        "state_ready"_s <= "state_probability_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_probability_decision,
        "state_ready"_s <= "state_probability_compute_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_probability_compute_decision,
        "state_ready"_s <= "state_segment_decode_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_segment_decode_decision,
        "state_ready"_s <= "state_native_decoding_segments"_s + unexpected_event<_> / effect_on_unexpected_from_state_native_decoding_segments,
        "state_ready"_s <= "state_callback_decoding_segments"_s + unexpected_event<_> / effect_on_unexpected_from_state_callback_decoding_segments,
        "state_ready"_s <= "state_publish_success"_s + unexpected_event<_> / effect_on_unexpected_from_state_publish_success,
        "state_ready"_s <= "state_publish_error"_s + unexpected_event<_> / effect_on_unexpected_from_state_publish_error,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

#[derive(Debug, Default)]
pub struct DiarizationSortformerPipelineContext {
    pub err: PipelineError,
    encoder_bound: bool,
    modules_bound: bool,
}

impl DiarizationSortformerPipelineContext {
    #[allow(clippy::unnecessary_wraps)]
    fn unexpected(&mut self) -> Result<(), ()> {
        self.err = PipelineError::Unexpected;
        Ok(())
    }
    #[allow(clippy::unnecessary_wraps)]
    fn mark(&mut self, error: PipelineError) -> Result<(), ()> {
        self.err = error;
        Ok(())
    }
}

impl DiarizationSortformerPipelineStateMachineContext for DiarizationSortformerPipelineContext {
    fn effect_begin_run<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = PipelineError::None;
        self.encoder_bound = false;
        self.modules_bound = false;
        **event.frame_count_out.borrow_mut() = 0;
        **event.probability_count_out.borrow_mut() = 0;
        **event.segment_count_out.borrow_mut() = 0;
        **event.error_out.borrow_mut() = PipelineError::None;
        Ok(())
    }
    fn effect_bind_encoder<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.encoder_bound = event
            .contract
            .bind_encoder
            .is_some_and(|bind| bind(event.contract));
        Ok(())
    }
    fn effect_bind_modules<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.modules_bound = event
            .contract
            .bind_modules
            .is_some_and(|bind| bind(event.contract));
        Ok(())
    }
    fn effect_compute_encoder_frames<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(encode) = event.contract.encode_frames else {
            self.err = PipelineError::Kernel;
            return Ok(());
        };
        let mut output = event.encoder_frames.borrow_mut();
        if !encode(
            &event.features.borrow()[..REQUIRED_FEATURE_COUNT],
            event.contract,
            &mut output[..REQUIRED_ENCODER_VALUE_COUNT],
        ) {
            self.err = PipelineError::Kernel;
        }
        Ok(())
    }
    fn effect_compute_probabilities<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(compute) = event.contract.compute_probabilities else {
            self.err = PipelineError::Kernel;
            return Ok(());
        };
        let mut output = event.probabilities.borrow_mut();
        if !compute(
            &event.hidden.borrow()[..REQUIRED_HIDDEN_VALUE_COUNT],
            event.contract,
            &mut output[..REQUIRED_PROBABILITY_VALUE_COUNT],
        ) {
            self.err = PipelineError::Kernel;
        }
        Ok(())
    }
    fn effect_compute_probabilities_native<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let route = event
            .native_output_route
            .take()
            .expect("native probability route required by guard");
        let result = {
            let hidden = event.hidden.borrow();
            let mut output = event.probabilities.borrow_mut();
            route.compute(
                &hidden[..REQUIRED_HIDDEN_VALUE_COUNT],
                &mut output[..REQUIRED_PROBABILITY_VALUE_COUNT],
            )
        };
        event.native_output_route.replace(Some(route));
        if result.is_ok() {
            **event.probability_count_out.borrow_mut() = REQUIRED_PROBABILITY_VALUE_COUNT_I32;
        } else {
            self.err = PipelineError::Kernel;
        }
        Ok(())
    }
    fn guard_native_probability_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_output_route.borrow().is_some())
    }
    fn guard_callback_probability_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_output_route.borrow().is_none()
            && event.contract.bind_modules.is_some()
            && event.contract.compute_probabilities.is_some())
    }
    fn guard_probability_route_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_native_probability_route(event)?
            && !self.guard_callback_probability_route(event)?)
    }
    fn guard_probability_kernel_ready<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.modules_bound && event.contract.compute_probabilities.is_some())
    }
    fn guard_probability_kernel_unavailable<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!(self.modules_bound && event.contract.compute_probabilities.is_some()))
    }
    fn effect_mark_kernel_error_from_state_execute_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Kernel)
    }
    fn effect_execute_hidden<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(execute) = event.contract.execute_hidden else {
            self.err = PipelineError::Executor;
            return Ok(());
        };
        let mut output = event.hidden.borrow_mut();
        if !execute(
            &event.encoder_frames.borrow()[..REQUIRED_ENCODER_VALUE_COUNT],
            event.contract,
            &mut output[..REQUIRED_HIDDEN_VALUE_COUNT],
        ) {
            self.err = PipelineError::Executor;
        }
        Ok(())
    }
    fn effect_decode_segments<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(decode) = event.contract.decode_segments else {
            self.err = PipelineError::Kernel;
            return Ok(());
        };
        let mut segments = event.segments.borrow_mut();
        let mut count = event.segment_count_out.borrow_mut();
        if !decode(
            &event.probabilities.borrow()[..REQUIRED_PROBABILITY_VALUE_COUNT],
            &mut segments[..MAX_SEGMENT_COUNT],
            &mut count,
        ) || **count < 0
            || **count > MAX_SEGMENT_COUNT_I32
        {
            self.err = PipelineError::Kernel;
            **count = 0;
        } else {
            **event.frame_count_out.borrow_mut() = FRAME_COUNT;
        }
        Ok(())
    }
    fn effect_decode_segments_native<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let route = event
            .native_output_route
            .take()
            .expect("native segment route required by guard");
        let mut segments = event.segments.borrow_mut();
        let mut count = event.segment_count_out.borrow_mut();
        let result = route.decode_segments(
            &event.probabilities.borrow()[..REQUIRED_PROBABILITY_VALUE_COUNT],
            0.5,
            &mut segments[..MAX_SEGMENT_COUNT],
            &mut count,
        );
        event.native_output_route.replace(Some(route));
        match result {
            Ok(()) => **event.frame_count_out.borrow_mut() = FRAME_COUNT,
            Err(NativeOutputError::SegmentCapacity) => {
                self.err = PipelineError::SegmentCapacity;
                **count = 0;
            }
            Err(_) => {
                self.err = PipelineError::Kernel;
                **count = 0;
            }
        }
        Ok(())
    }
    fn guard_native_segment_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_output_route.borrow().is_some())
    }
    fn guard_callback_segment_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.native_output_route.borrow().is_none()
                && event.contract.decode_segments.is_some(),
        )
    }
    fn guard_segment_route_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            !self.guard_native_segment_route(event)?
                && !self.guard_callback_segment_route(event)?,
        )
    }
    fn effect_mark_kernel_error_from_state_segment_decode_decision<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Kernel)
    }
    fn guard_encoder_kernel_ready<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.encoder_bound && event.contract.encode_frames.is_some())
    }
    fn guard_encoder_kernel_unavailable<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_encoder_kernel_ready(event)?)
    }
    fn effect_mark_tensor_contract_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::TensorContract)
    }
    fn effect_mark_feature_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Request)
    }
    fn effect_mark_encoder_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Executor)
    }
    fn effect_mark_hidden_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Executor)
    }
    fn effect_mark_kernel_error_from_state_computing_encoder<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Kernel)
    }
    fn effect_mark_kernel_error_from_state_computing_probabilities<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::Kernel)
    }
    fn effect_mark_channel_count_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::ChannelCount)
    }
    fn effect_mark_model_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::ModelInvalid)
    }
    fn effect_mark_pcm_shape_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::PcmShape)
    }
    fn effect_mark_probability_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::ProbabilityCapacity)
    }
    fn effect_mark_sample_rate_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::SampleRate)
    }
    fn effect_mark_segment_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(PipelineError::SegmentCapacity)
    }
    fn effect_prepare_features<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let Some(extract) = event.contract.extract_features else {
            self.err = PipelineError::Request;
            return Ok(());
        };
        let mut features = event.features.borrow_mut();
        if !extract(
            event.pcm,
            event.contract,
            &mut features[..REQUIRED_FEATURE_COUNT],
        ) {
            self.err = PipelineError::Request;
        }
        Ok(())
    }
    fn effect_publish_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.error_out.borrow_mut() = self.err;
        if let Some(callback) = event.on_error {
            let _ = callback(RunError { error: self.err });
        }
        Ok(())
    }
    fn effect_publish_success<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.error_out.borrow_mut() = self.err;
        if let Some(callback) = event.on_done {
            let _ = callback(RunDone {
                frame_count: **event.frame_count_out.borrow(),
                probability_count: **event.probability_count_out.borrow(),
                segment_count: **event.segment_count_out.borrow(),
            });
        }
        Ok(())
    }
    fn guard_feature_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.features.borrow()).len() >= REQUIRED_FEATURE_COUNT)
    }
    fn guard_feature_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_feature_capacity_valid(event)?)
    }
    fn guard_encoder_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.encoder_frames.borrow()).len() >= REQUIRED_ENCODER_VALUE_COUNT)
    }
    fn guard_encoder_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_encoder_capacity_valid(event)?)
    }
    fn guard_hidden_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.hidden.borrow()).len() >= REQUIRED_HIDDEN_VALUE_COUNT)
    }
    fn guard_hidden_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_hidden_capacity_valid(event)?)
    }
    fn guard_channel_count_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count != CHANNEL_COUNT)
    }
    fn guard_channel_count_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count == CHANNEL_COUNT)
    }
    fn guard_probability_compute_succeeded<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err == PipelineError::None)
    }
    fn guard_probability_compute_failed<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err != PipelineError::None)
    }
    fn guard_encoder_compute_failed<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err != PipelineError::None)
    }
    fn guard_encoder_compute_succeeded<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err == PipelineError::None)
    }
    fn guard_has_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err != PipelineError::None)
    }
    fn guard_model_contract_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.contract.model_contract_valid())
    }
    fn guard_model_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.contract.model_contract_valid())
    }
    fn guard_no_error<'dispatch, 'event>(
        &self,
        _: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err == PipelineError::None)
    }
    fn guard_pcm_shape_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.pcm.len() != REQUIRED_SAMPLE_COUNT)
    }
    fn guard_pcm_shape_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.pcm.len() == REQUIRED_SAMPLE_COUNT)
    }
    fn guard_probability_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.probabilities.borrow()).len() < REQUIRED_PROBABILITY_VALUE_COUNT)
    }
    fn guard_probability_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.probabilities.borrow()).len() >= REQUIRED_PROBABILITY_VALUE_COUNT)
    }
    fn guard_sample_rate_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate != SAMPLE_RATE)
    }
    fn guard_sample_rate_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate == SAMPLE_RATE)
    }
    fn guard_tensor_contract_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_tensor_contract_valid(event)?)
    }
    fn guard_segment_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.segments.borrow()).len() < MAX_SEGMENT_COUNT)
    }
    fn guard_segment_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.segments.borrow()).len() >= MAX_SEGMENT_COUNT)
    }
    fn guard_tensor_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventRunFlow<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.contract.encoder_ready
            && event.contract.modules_ready
            && event.contract.transformer_encoder_ready
            && event.contract.encoder_tensors != 0
            && event.contract.modules_tensors != 0
            && event.contract.transformer_encoder_tensors != 0)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn effect_on_unexpected_from_state_binding_encoder(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_binding_modules(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_computing_encoder(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_computing_probabilities(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_segment_decode_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_native_decoding_segments(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_callback_decoding_segments(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_encoder_compute_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_execute_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_executing_hidden(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_prepare_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_preparing_features(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_probability_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_probability_compute_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_probability_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_feature_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_encoder_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_hidden_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_publish_error(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_publish_success(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_segment_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_tensor_contract_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
}

/// Public single-writer synchronous pipeline wrapper.
pub struct DiarizationSortformerPipeline {
    machine: DiarizationSortformerPipelineStateMachine<DiarizationSortformerPipelineContext>,
}

impl Default for DiarizationSortformerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DiarizationSortformerPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: DiarizationSortformerPipelineStateMachine::new(
                DiarizationSortformerPipelineContext::default(),
            ),
        }
    }
    pub fn run(&mut self, event: EventRunFlow<'_>) -> Result<(), PipelineError> {
        let EventRunFlow {
            contract,
            pcm,
            sample_rate,
            channel_count,
            features,
            encoder_frames,
            hidden,
            probabilities,
            segments,
            frame_count_out,
            probability_count_out,
            segment_count_out,
            error_out,
            native_feature_extractor_route,
            native_projection_route,
            native_transformer_route,
            on_done,
            on_error,
            native_output_route,
        } = event;
        let features = features.into_inner();
        let encoder_frames = encoder_frames.into_inner();
        let hidden = hidden.into_inner();
        let probabilities = probabilities.into_inner();
        let segments = segments.into_inner();
        let frame_count_out = frame_count_out.into_inner();
        let probability_count_out = probability_count_out.into_inner();
        let segment_count_out = segment_count_out.into_inner();
        let error_out = error_out.into_inner();
        let native_feature_extractor_route = native_feature_extractor_route.into_inner();
        let native_projection_route = native_projection_route.into_inner();
        let native_transformer_route = native_transformer_route.into_inner();
        let native_output_route = native_output_route.into_inner();
        let event = EventRunFlow {
            contract,
            pcm,
            sample_rate,
            channel_count,
            features: RefCell::new(features),
            encoder_frames: RefCell::new(encoder_frames),
            hidden: RefCell::new(hidden),
            probabilities: RefCell::new(probabilities),
            segments: RefCell::new(segments),
            frame_count_out: RefCell::new(frame_count_out),
            probability_count_out: RefCell::new(probability_count_out),
            segment_count_out: RefCell::new(segment_count_out),
            error_out: RefCell::new(error_out),
            native_feature_extractor_route: RefCell::new(native_feature_extractor_route),
            native_projection_route: RefCell::new(native_projection_route),
            native_transformer_route: RefCell::new(native_transformer_route),
            on_done,
            on_error,
            native_output_route: RefCell::new(native_output_route),
        };
        if self
            .machine
            .process_event(DiarizationSortformerPipelineEvents::EventRunFlow(&event))
            .is_err()
        {
            self.machine.context_mut().err = PipelineError::Unexpected;
            return Err(PipelineError::Unexpected);
        }
        let error = self.machine.context().err;
        if error == PipelineError::None {
            Ok(())
        } else {
            Err(error)
        }
    }
    #[must_use]
    pub fn state(&self) -> &DiarizationSortformerPipelineStates {
        self.machine.state()
    }

    #[must_use]
    pub fn is(&self, state: &DiarizationSortformerPipelineStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the current selected pipeline error.
    #[must_use]
    pub fn error(&self) -> PipelineError {
        self.machine.context().err
    }

    /// Returns whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine
            .is(&DiarizationSortformerPipelineStates::StateReady)
    }

    #[must_use]
    pub fn context(&self) -> &DiarizationSortformerPipelineContext {
        self.machine.context()
    }
}

/// Short actor alias matching the maintained C++ `Pipeline` spelling.
pub type Pipeline = DiarizationSortformerPipeline;
