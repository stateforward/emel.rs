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
    dead_code,
    unused_imports,
    missing_docs
)]

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
pub const REQUIRED_FEATURE_COUNT: usize = FEATURE_FRAME_COUNT as usize * FEATURE_BIN_COUNT as usize;
pub const REQUIRED_ENCODER_VALUE_COUNT: usize = FRAME_COUNT as usize * 512;
pub const REQUIRED_HIDDEN_VALUE_COUNT: usize = FRAME_COUNT as usize * HIDDEN_DIM as usize;
pub const REQUIRED_PROBABILITY_VALUE_COUNT: usize = FRAME_COUNT as usize * SPEAKER_COUNT as usize;
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
            && self.feature_extractor_ready
            && self.encoder_ready
            && self.modules_ready
            && self.transformer_encoder_ready
    }
}

/// Compatibility spelling for integrations that call the model contract an execution contract.
pub type ExecutionContract = PipelineContract;
/// Short error spelling retained for pipeline callers.
pub type Error = PipelineError;

/// Successful completion payload.
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

/// Bounded borrowed pipeline request.
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
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventRunFlow<'a> {
    /// Constructs a borrowed request with all result buffers owned by caller.
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
            on_done: None,
            on_error: None,
        }
    }

    /// Adds caller-owned completion callbacks.
    #[must_use]
    pub fn with_callbacks(mut self, on_done: Option<DoneCallback>, on_error: Option<ErrorCallback>) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

sml! {
    DiarizationSortformerPipeline {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<EventRunFlow> / effect_begin_run,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<EventRunFlow> [guard_model_contract_valid],
        "state_publish_error"_s <= "state_model_contract_decision"_s + completion<EventRunFlow> [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<EventRunFlow> [guard_sample_rate_valid],
        "state_publish_error"_s <= "state_sample_rate_decision"_s + completion<EventRunFlow> [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<EventRunFlow> [guard_channel_count_valid],
        "state_publish_error"_s <= "state_channel_count_decision"_s + completion<EventRunFlow> [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_probability_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<EventRunFlow> [guard_pcm_shape_valid],
        "state_publish_error"_s <= "state_pcm_shape_decision"_s + completion<EventRunFlow> [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_segment_capacity_decision"_s <= "state_probability_capacity_decision"_s + completion<EventRunFlow> [guard_probability_capacity_valid],
        "state_publish_error"_s <= "state_probability_capacity_decision"_s + completion<EventRunFlow> [guard_probability_capacity_invalid] / effect_mark_probability_capacity_invalid,
        "state_tensor_contract_decision"_s <= "state_segment_capacity_decision"_s + completion<EventRunFlow> [guard_segment_capacity_valid],
        "state_publish_error"_s <= "state_segment_capacity_decision"_s + completion<EventRunFlow> [guard_segment_capacity_invalid] / effect_mark_segment_capacity_invalid,
        "state_preparing_features"_s <= "state_tensor_contract_decision"_s + completion<EventRunFlow> [guard_tensor_contract_valid],
        "state_publish_error"_s <= "state_tensor_contract_decision"_s + completion<EventRunFlow> [guard_tensor_contract_invalid] / effect_mark_tensor_contract_invalid,
        "state_prepare_decision"_s <= "state_preparing_features"_s + completion<EventRunFlow> / effect_prepare_features,
        "state_binding_encoder"_s <= "state_prepare_decision"_s + completion<EventRunFlow> [guard_no_error],
        "state_publish_error"_s <= "state_prepare_decision"_s + completion<EventRunFlow> [guard_has_error],
        "state_computing_encoder"_s <= "state_binding_encoder"_s + completion<EventRunFlow> / effect_bind_encoder,
        "state_executing_hidden"_s <= "state_computing_encoder"_s + completion<EventRunFlow> [guard_encoder_kernel_ready] / effect_compute_encoder_frames,
        "state_publish_error"_s <= "state_computing_encoder"_s + completion<EventRunFlow> [guard_encoder_kernel_unavailable] / effect_mark_kernel_error_from_state_computing_encoder,
        "state_encoder_compute_decision"_s <= "state_executing_hidden"_s + completion<EventRunFlow> [guard_encoder_compute_succeeded],
        "state_publish_error"_s <= "state_executing_hidden"_s + completion<EventRunFlow> [guard_encoder_compute_failed],
        "state_execute_decision"_s <= "state_encoder_compute_decision"_s + completion<EventRunFlow> / effect_execute_hidden,
        "state_binding_modules"_s <= "state_execute_decision"_s + completion<EventRunFlow> [guard_no_error],
        "state_publish_error"_s <= "state_execute_decision"_s + completion<EventRunFlow> [guard_has_error],
        "state_computing_probabilities"_s <= "state_binding_modules"_s + completion<EventRunFlow> / effect_bind_modules,
        "state_probability_decision"_s <= "state_computing_probabilities"_s + completion<EventRunFlow> [guard_probability_kernel_ready] / effect_compute_probabilities,
        "state_publish_error"_s <= "state_computing_probabilities"_s + completion<EventRunFlow> [guard_probability_kernel_unavailable] / effect_mark_kernel_error_from_state_computing_probabilities,
        "state_probability_compute_decision"_s <= "state_probability_decision"_s + completion<EventRunFlow> [guard_probability_compute_succeeded],
        "state_publish_error"_s <= "state_probability_decision"_s + completion<EventRunFlow> [guard_probability_compute_failed],
        "state_decoding_segments"_s <= "state_probability_compute_decision"_s + completion<EventRunFlow>,
        "state_publish_success"_s <= "state_decoding_segments"_s + completion<EventRunFlow> / effect_decode_segments,
        "state_done"_s <= "state_publish_success"_s + completion<EventRunFlow> / effect_publish_success,
        "state_errored"_s <= "state_publish_error"_s + completion<EventRunFlow> / effect_publish_error,
        "state_ready"_s <= "state_done"_s + completion<EventRunFlow>,
        "state_ready"_s <= "state_errored"_s + completion<EventRunFlow>,
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
        "state_ready"_s <= "state_probability_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_probability_decision,
        "state_ready"_s <= "state_probability_compute_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_probability_compute_decision,
        "state_ready"_s <= "state_decoding_segments"_s + unexpected_event<_> / effect_on_unexpected_from_state_decoding_segments,
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
    fn unexpected(&mut self) -> Result<(), ()> { self.err = PipelineError::Unexpected; Ok(()) }
    fn mark(&mut self, error: PipelineError) -> Result<(), ()> { self.err = error; Ok(()) }
}

impl DiarizationSortformerPipelineStateMachineContext for DiarizationSortformerPipelineContext {
    fn effect_begin_run(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        self.err = PipelineError::None;
        self.encoder_bound = false;
        self.modules_bound = false;
        *event.frame_count_out.borrow_mut() = 0;
        *event.probability_count_out.borrow_mut() = 0;
        *event.segment_count_out.borrow_mut() = 0;
        *event.error_out.borrow_mut() = PipelineError::None;
        Ok(())
    }
    fn effect_bind_encoder(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        self.encoder_bound = event.contract.bind_encoder.is_some_and(|bind| bind(event.contract));
        Ok(())
    }
    fn effect_bind_modules(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        self.modules_bound = event.contract.bind_modules.is_some_and(|bind| bind(event.contract));
        Ok(())
    }
    fn effect_compute_encoder_frames(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        let Some(encode) = event.contract.encode_frames else { self.err = PipelineError::Kernel; return Ok(()); };
        let mut output = event.encoder_frames.borrow_mut();
        if !encode(event.features.borrow().as_ref(), event.contract, &mut output[..REQUIRED_ENCODER_VALUE_COUNT]) {
            self.err = PipelineError::Kernel;
        }
        Ok(())
    }
    fn effect_compute_probabilities(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        let Some(compute) = event.contract.compute_probabilities else { self.err = PipelineError::Kernel; return Ok(()); };
        let mut output = event.probabilities.borrow_mut();
        if !compute(event.hidden.borrow().as_ref(), event.contract, &mut output[..REQUIRED_PROBABILITY_VALUE_COUNT]) {
            self.err = PipelineError::Kernel;
        } else {
            *event.probability_count_out.borrow_mut() = REQUIRED_PROBABILITY_VALUE_COUNT as i32;
        }
        Ok(())
    }
    fn effect_decode_segments(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        let Some(decode) = event.contract.decode_segments else { self.err = PipelineError::Kernel; return Ok(()); };
        let mut segments = event.segments.borrow_mut();
        let mut count = event.segment_count_out.borrow_mut();
        if !decode(event.probabilities.borrow().as_ref(), &mut segments[..MAX_SEGMENT_COUNT], &mut count) {
            self.err = PipelineError::Kernel;
            *count = 0;
        } else if *count < 0 || *count > MAX_SEGMENT_COUNT as i32 {
            self.err = PipelineError::Kernel;
            *count = 0;
        } else {
            *event.frame_count_out.borrow_mut() = FRAME_COUNT;
        }
        Ok(())
    }
    fn effect_execute_hidden(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        let Some(execute) = event.contract.execute_hidden else { self.err = PipelineError::Executor; return Ok(()); };
        let mut output = event.hidden.borrow_mut();
        if !execute(event.encoder_frames.borrow().as_ref(), event.contract, &mut output[..REQUIRED_HIDDEN_VALUE_COUNT]) {
            self.err = PipelineError::Executor;
        }
        Ok(())
    }
    fn effect_mark_channel_count_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::ChannelCount) }
    fn effect_mark_kernel_error_from_state_computing_encoder(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::Kernel) }
    fn effect_mark_kernel_error_from_state_computing_probabilities(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::Kernel) }
    fn effect_mark_model_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::ModelInvalid) }
    fn effect_mark_pcm_shape_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::PcmShape) }
    fn effect_mark_probability_capacity_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::ProbabilityCapacity) }
    fn effect_mark_sample_rate_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::SampleRate) }
    fn effect_mark_segment_capacity_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::SegmentCapacity) }
    fn effect_mark_tensor_contract_invalid(&mut self, _: &EventRunFlow) -> Result<(), ()> { self.mark(PipelineError::TensorContract) }
    fn effect_prepare_features(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        let Some(extract) = event.contract.extract_features else { self.err = PipelineError::Request; return Ok(()); };
        let mut features = event.features.borrow_mut();
        if !extract(event.pcm, event.contract, &mut features[..REQUIRED_FEATURE_COUNT]) {
            self.err = PipelineError::Request;
        }
        Ok(())
    }
    fn effect_publish_error(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        *event.error_out.borrow_mut() = self.err;
        if let Some(callback) = event.on_error { let _ = callback(RunError { error: self.err }); }
        Ok(())
    }
    fn effect_publish_success(&mut self, event: &EventRunFlow) -> Result<(), ()> {
        *event.error_out.borrow_mut() = self.err;
        if let Some(callback) = event.on_done {
            let _ = callback(RunDone {
                frame_count: *event.frame_count_out.borrow(),
                probability_count: *event.probability_count_out.borrow(),
                segment_count: *event.segment_count_out.borrow(),
            });
        }
        Ok(())
    }

    fn guard_channel_count_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.channel_count != CHANNEL_COUNT) }
    fn guard_channel_count_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.channel_count == CHANNEL_COUNT) }
    fn guard_encoder_compute_failed(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err != PipelineError::None) }
    fn guard_encoder_compute_succeeded(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err == PipelineError::None) }
    fn guard_encoder_kernel_ready(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(self.encoder_bound && event.contract.encode_frames.is_some()) }
    fn guard_encoder_kernel_unavailable(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_encoder_kernel_ready(event)?) }
    fn guard_has_error(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err != PipelineError::None) }
    fn guard_model_contract_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!event.contract.model_contract_valid()) }
    fn guard_model_contract_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.contract.model_contract_valid()) }
    fn guard_no_error(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err == PipelineError::None) }
    fn guard_pcm_shape_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_pcm_shape_valid(event)?) }
    fn guard_pcm_shape_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.pcm.len() == REQUIRED_SAMPLE_COUNT) }
    fn guard_probability_capacity_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_probability_capacity_valid(event)?) }
    fn guard_probability_capacity_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.probabilities.borrow().len() >= REQUIRED_PROBABILITY_VALUE_COUNT) }
    fn guard_probability_compute_failed(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err != PipelineError::None) }
    fn guard_probability_compute_succeeded(&self, _: &EventRunFlow) -> Result<bool, ()> { Ok(self.err == PipelineError::None) }
    fn guard_probability_kernel_ready(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(self.modules_bound && event.contract.compute_probabilities.is_some()) }
    fn guard_probability_kernel_unavailable(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_probability_kernel_ready(event)?) }
    fn guard_sample_rate_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.sample_rate != SAMPLE_RATE) }
    fn guard_sample_rate_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.sample_rate == SAMPLE_RATE) }
    fn guard_segment_capacity_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_segment_capacity_valid(event)?) }
    fn guard_segment_capacity_valid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(event.segments.borrow().len() >= MAX_SEGMENT_COUNT) }
    fn guard_tensor_contract_invalid(&self, event: &EventRunFlow) -> Result<bool, ()> { Ok(!self.guard_tensor_contract_valid(event)?) }
    fn guard_tensor_contract_valid(&self, event: &EventRunFlow) -> Result<bool, ()> {
        Ok(event.contract.encoder_ready
            && event.contract.modules_ready
            && event.contract.transformer_encoder_ready
            && event.contract.encoder_tensors != 0
            && event.contract.modules_tensors != 0
            && event.contract.transformer_encoder_tensors != 0)
    }

    fn effect_on_unexpected_from_state_binding_encoder(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_binding_modules(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_computing_encoder(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_computing_probabilities(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decoding_segments(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encoder_compute_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_execute_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_executing_hidden(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_prepare_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_preparing_features(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_probability_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_probability_compute_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_probability_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_publish_error(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_publish_success(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_segment_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_tensor_contract_decision(&mut self) -> Result<(), ()> { self.unexpected() }
}

/// Public single-writer synchronous pipeline wrapper.
pub struct DiarizationSortformerPipeline {
    machine: DiarizationSortformerPipelineStateMachine<DiarizationSortformerPipelineContext>,
}

impl Default for DiarizationSortformerPipeline {
    fn default() -> Self { Self::new() }
}

impl DiarizationSortformerPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self { machine: DiarizationSortformerPipelineStateMachine::new(DiarizationSortformerPipelineContext::default()) }
    }

    /// Dispatches one complete bounded run synchronously.
    pub fn run(&mut self, event: EventRunFlow<'_>) -> Result<(), PipelineError> {
        if self.machine.process_event(event).is_err() {
            self.machine.context_mut().err = PipelineError::Unexpected;
            return Err(PipelineError::Unexpected);
        }
        let error = self.machine.context().err;
        if error == PipelineError::None { Ok(()) } else { Err(error) }
    }

    /// Source-compatible process spelling.
    pub fn process_event(&mut self, event: EventRunFlow<'_>) -> Result<(), PipelineError> { self.run(event) }

    #[must_use]
    pub fn state(&self) -> DiarizationSortformerPipelineStates { self.machine.state() }

    #[must_use]
    pub fn is(&self, state: &DiarizationSortformerPipelineStates) -> bool { self.machine.is(state) }
    /// Returns the current selected pipeline error.
    #[must_use]
    pub fn error(&self) -> PipelineError { self.machine.context().err }

    /// Returns whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool { self.machine.is(&DiarizationSortformerPipelineStates::StateReady) }

    #[must_use]
    pub fn context(&self) -> &DiarizationSortformerPipelineContext { self.machine.context() }
}

/// Short actor alias matching the maintained C++ `Pipeline` spelling.
pub type Pipeline = DiarizationSortformerPipeline;
