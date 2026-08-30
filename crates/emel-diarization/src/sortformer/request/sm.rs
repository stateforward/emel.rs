//! Source-aligned synchronous Sortformer request actor.
//!
//! The request is a bounded borrowed event.  The actor validates the pinned
//! model/audio contract, performs feature preparation in the caller's output
//! buffer, and publishes completion through synchronous function pointers.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions,
    clippy::return_self_not_must_use,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

pub const SAMPLE_RATE: i32 = 16_000;
pub const CHANNEL_COUNT: i32 = 1;
pub const SPEAKER_COUNT: i32 = 4;
pub const FRAME_SHIFT_MS: i32 = 80;
pub const CHUNK_LEN: i32 = 188;
pub const CHUNK_RIGHT_CONTEXT: i32 = 1;
pub const FEATURE_BIN_COUNT: i32 = 128;
pub const FEATURE_FRAME_COUNT: i32 = 1_504;
pub const REQUIRED_SAMPLE_COUNT: usize = 240_640;
pub const REQUIRED_FEATURE_COUNT: usize = FEATURE_FRAME_COUNT as usize * FEATURE_BIN_COUNT as usize;

/// Errors defined by the pinned request contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    None = 0,
    ModelInvalid = 1 << 0,
    SampleRate = 1 << 1,
    ChannelCount = 1 << 2,
    PcmShape = 1 << 3,
    Capacity = 1 << 4,
    FeatureExtractor = 1 << 5,
    Unexpected = 1 << 6,
}

impl Default for Error {
    fn default() -> Self { Self::None }
}

/// Source spelling retained for callers using the request namespace.
pub type RequestError = Error;

/// Bounded tensor-family summary used by model validation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FamilyContract {
    pub tensor_count: u32,
}

/// Caller-owned Sortformer execution contract.
///
/// The fields mirror `emel::model::sortformer::execution_contract`; pointers
/// in the C++ contract become the explicit `model_present` bit and bounded
/// family summaries at this Rust boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionContract {
    pub model_present: bool,
    pub sample_rate: i32,
    pub speaker_count: i32,
    pub frame_shift_ms: i32,
    pub chunk_len: i32,
    pub chunk_right_context: i32,
    pub feature_extractor: FamilyContract,
    pub encoder: FamilyContract,
    pub modules: FamilyContract,
    pub transformer_encoder: FamilyContract,
    pub feature_extractor_ready: bool,
    pub extract_features: Option<ExtractFeaturesFn>,
}

/// Synchronous feature extraction handoff.
///
/// The callback receives borrowed request data and caller-owned output. It must
/// not allocate, retain either borrow, or re-enter this actor.
pub type ExtractFeaturesFn = fn(&[f32], &ExecutionContract, &mut [f32]) -> bool;

impl ExecutionContract {
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            model_present: true,
            sample_rate: SAMPLE_RATE,
            speaker_count: SPEAKER_COUNT,
            frame_shift_ms: FRAME_SHIFT_MS,
            chunk_len: CHUNK_LEN,
            chunk_right_context: CHUNK_RIGHT_CONTEXT,
            feature_extractor: FamilyContract { tensor_count: 1 },
            encoder: FamilyContract { tensor_count: 1 },
            modules: FamilyContract { tensor_count: 1 },
            transformer_encoder: FamilyContract { tensor_count: 1 },
            feature_extractor_ready: true,
            extract_features: None,
        }
    }

    #[must_use]
    pub const fn model_contract_valid(self) -> bool {
        self.model_present
            && self.sample_rate == SAMPLE_RATE
            && self.speaker_count == SPEAKER_COUNT
            && self.frame_shift_ms == FRAME_SHIFT_MS
            && self.chunk_len == CHUNK_LEN
            && self.chunk_right_context == CHUNK_RIGHT_CONTEXT
            && self.feature_extractor_ready
            && self.feature_extractor.tensor_count != 0
            && self.encoder.tensor_count != 0
            && self.modules.tensor_count != 0
            && self.transformer_encoder.tensor_count != 0
    }
}

/// Successful request callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrepareDone {
    pub frame_count: i32,
    pub feature_bin_count: i32,
}

/// Failed request callback payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrepareError {
    pub error: Error,
}

/// Synchronous completion callback.
pub type DoneCallback = fn(PrepareDone) -> bool;
/// Synchronous error callback.
pub type ErrorCallback = fn(PrepareError) -> bool;

/// Runtime request corresponding to C++ `event::prepare_run`.
///
/// `RefCell` only provides interior access to caller-owned references; it does
/// not allocate. The borrowed buffers remain owned by the caller.
pub struct EventPrepareRun<'a> {
    pub contract: &'a ExecutionContract,
    pub pcm: &'a [f32],
    pub sample_rate: i32,
    pub channel_count: i32,
    pub features: RefCell<&'a mut [f32]>,
    pub frame_count_out: RefCell<&'a mut i32>,
    pub feature_bin_count_out: RefCell<&'a mut i32>,
    pub error_out: RefCell<Option<&'a mut Error>>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventPrepareRun<'a> {
    #[must_use]
    pub fn new(
        contract: &'a ExecutionContract,
        pcm: &'a [f32],
        sample_rate: i32,
        channel_count: i32,
        features: &'a mut [f32],
        frame_count_out: &'a mut i32,
        feature_bin_count_out: &'a mut i32,
    ) -> Self {
        Self {
            contract,
            pcm,
            sample_rate,
            channel_count,
            features: RefCell::new(features),
            frame_count_out: RefCell::new(frame_count_out),
            feature_bin_count_out: RefCell::new(feature_bin_count_out),
            error_out: RefCell::new(None),
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
    pub fn with_error_out(mut self, error_out: &'a mut Error) -> Self {
        *self.error_out.get_mut() = Some(error_out);
        self
    }
}

sml! {
    DiarizationSortformerRequest {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<EventPrepareRun> / effect_begin_prepare,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<EventPrepareRun> [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventPrepareRun> [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<EventPrepareRun> [guard_sample_rate_valid],
        "state_error_error_out_decision"_s <= "state_sample_rate_decision"_s + completion<EventPrepareRun> [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<EventPrepareRun> [guard_channel_count_valid],
        "state_error_error_out_decision"_s <= "state_channel_count_decision"_s + completion<EventPrepareRun> [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_output_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<EventPrepareRun> [guard_pcm_shape_valid],
        "state_error_error_out_decision"_s <= "state_pcm_shape_decision"_s + completion<EventPrepareRun> [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_preparing"_s <= "state_output_capacity_decision"_s + completion<EventPrepareRun> [guard_output_capacity_valid],
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<EventPrepareRun> [guard_output_capacity_invalid] / effect_mark_capacity_invalid,
        "state_success_error_out_decision"_s <= "state_preparing"_s + completion<EventPrepareRun> / effect_extract_features,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventPrepareRun> [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventPrepareRun> [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventPrepareRun> [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventPrepareRun> [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventPrepareRun> [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventPrepareRun> [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventPrepareRun> [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventPrepareRun> [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventPrepareRun>,
        "state_ready"_s <= "state_errored"_s + completion<EventPrepareRun>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_sample_rate_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_sample_rate_decision,
        "state_ready"_s <= "state_channel_count_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_channel_count_decision,
        "state_ready"_s <= "state_pcm_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_pcm_shape_decision,
        "state_ready"_s <= "state_output_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_output_capacity_decision,
        "state_ready"_s <= "state_preparing"_s + unexpected_event<_> / effect_on_unexpected_from_state_preparing,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

#[derive(Debug, Default)]
pub struct DiarizationSortformerRequestContext {
    pub err: Error,
}

impl DiarizationSortformerRequestStateMachineContext for DiarizationSortformerRequestContext {
    fn effect_begin_prepare(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        self.err = Error::None;
        *event.frame_count_out.borrow_mut() = 0;
        *event.feature_bin_count_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_emit_done(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        if let Some(callback) = event.on_done {
            let _ = callback(PrepareDone {
                frame_count: *event.frame_count_out.borrow(),
                feature_bin_count: *event.feature_bin_count_out.borrow(),
            });
        }
        Ok(())
    }
    fn effect_emit_error(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        if let Some(callback) = event.on_error {
            let _ = callback(PrepareError { error: self.err });
        }
        Ok(())
    }
    fn effect_extract_features(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        let mut features = event.features.borrow_mut();
        if let Some(extractor) = event.contract.extract_features {
            if !extractor(event.pcm, event.contract, &mut features[..REQUIRED_FEATURE_COUNT]) {
                self.err = Error::FeatureExtractor;
                return Ok(());
            }
        } else {
            features[..REQUIRED_FEATURE_COUNT].fill(0.0);
        }
        *event.frame_count_out.borrow_mut() = FEATURE_FRAME_COUNT;
        *event.feature_bin_count_out.borrow_mut() = FEATURE_BIN_COUNT;
        Ok(())
    }
    fn effect_mark_capacity_invalid(&mut self, _: &EventPrepareRun) -> Result<(), ()> { self.err = Error::Capacity; Ok(()) }
    fn effect_mark_channel_count_invalid(&mut self, _: &EventPrepareRun) -> Result<(), ()> { self.err = Error::ChannelCount; Ok(()) }
    fn effect_mark_model_invalid(&mut self, _: &EventPrepareRun) -> Result<(), ()> { self.err = Error::ModelInvalid; Ok(()) }
    fn effect_mark_pcm_shape_invalid(&mut self, _: &EventPrepareRun) -> Result<(), ()> { self.err = Error::PcmShape; Ok(()) }
    fn effect_mark_sample_rate_invalid(&mut self, _: &EventPrepareRun) -> Result<(), ()> { self.err = Error::SampleRate; Ok(()) }
    fn effect_store_error_error(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() { *error_out = self.err; }
        Ok(())
    }
    fn effect_store_success_error(&mut self, event: &EventPrepareRun) -> Result<(), ()> {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() { *error_out = self.err; }
        Ok(())
    }

    fn guard_channel_count_invalid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.channel_count != CHANNEL_COUNT) }
    fn guard_channel_count_valid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.channel_count == CHANNEL_COUNT) }
    fn guard_has_done_callback(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.on_done.is_some()) }
    fn guard_has_error_callback(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.on_error.is_some()) }
    fn guard_has_error_out(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.error_out.borrow().is_some()) }
    fn guard_model_contract_invalid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(!event.contract.model_contract_valid()) }
    fn guard_model_contract_valid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.contract.model_contract_valid()) }
    fn guard_no_done_callback(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.on_done.is_none()) }
    fn guard_no_error_callback(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.on_error.is_none()) }
    fn guard_no_error_out(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.error_out.borrow().is_none()) }
    fn guard_output_capacity_invalid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(!self.guard_output_capacity_valid(event)?) }
    fn guard_output_capacity_valid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.features.borrow().len() >= REQUIRED_FEATURE_COUNT) }
    fn guard_pcm_shape_invalid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(!self.guard_pcm_shape_valid(event)?) }
    fn guard_pcm_shape_valid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.pcm.len() == REQUIRED_SAMPLE_COUNT) }
    fn guard_sample_rate_invalid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.sample_rate != SAMPLE_RATE) }
    fn guard_sample_rate_valid(&self, event: &EventPrepareRun) -> Result<bool, ()> { Ok(event.sample_rate == SAMPLE_RATE) }

    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_preparing(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
}

/// Source-compatible synchronous request wrapper.
pub type Request = DiarizationSortformerRequestStateMachine;

impl Request {
    /// Dispatches one borrowed request through the complete validation and
    /// preparation graph.
    pub fn prepare<'a>(&mut self, event: EventPrepareRun<'a>) -> bool {
        self.process_event(event).is_ok()
    }
}
