//! Source-aligned synchronous Sortformer request actor.
//!
//! The request is a bounded borrowed event.  The actor validates the pinned
//! model/audio contract, performs feature preparation in the caller's output
//! buffer, and publishes completion through synchronous function pointers.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions,
    clippy::return_self_not_must_use,
    missing_debug_implementations,
    dead_code,
    missing_docs
)]

use super::super::encoder::feature_extractor::Route as NativeFeatureExtractorRoute;
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    #[default]
    None = 0,
    ModelInvalid = 1 << 0,
    SampleRate = 1 << 1,
    ChannelCount = 1 << 2,
    PcmShape = 1 << 3,
    Capacity = 1 << 4,
    FeatureExtractor = 1 << 5,
    Unexpected = 1 << 6,
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
#[allow(unpredictable_function_pointer_comparisons)]
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

/// Typed completion outcome for the bounded feature-extraction phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ExtractionOutcome {
    #[default]
    Pending,
    Succeeded,
    Failed(Error),
}
impl ExtractionOutcome {
    const fn error(self) -> Error {
        match self {
            Self::Failed(error) => error,
            Self::Pending | Self::Succeeded => Error::None,
        }
    }
}

impl From<Result<(), super::super::encoder::feature_extractor::Error>> for ExtractionOutcome {
    fn from(result: Result<(), super::super::encoder::feature_extractor::Error>) -> Self {
        match result {
            Ok(()) => Self::Succeeded,
            Err(_) => Self::Failed(Error::FeatureExtractor),
        }
    }
}

impl From<bool> for ExtractionOutcome {
    fn from(succeeded: bool) -> Self {
        if succeeded {
            Self::Succeeded
        } else {
            Self::Failed(Error::FeatureExtractor)
        }
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
/// The event is a caller-owned, single-writer synchronous dispatch envelope.
/// Every borrow is valid for this dispatch only; the caller retains ownership
/// of all buffers, output slots, and the optional native route. The request
/// actor must be the only writer while `prepare` runs, and the event and its
/// borrows must not be shared across dispatches or retained after the call.
///
/// `RefCell` provides the temporary interior access needed by the generated
/// state-machine effects. It is not a cross-dispatch synchronization primitive
/// and does not make this event thread-safe. A selected route may be taken from
/// its slot while a callback executes and is replaced into the same slot before
/// the effect returns; this does not transfer ownership to the actor.
///
/// Completion callbacks receive copied outcome data synchronously. They must
/// not retain event borrows or re-enter the request actor.
pub struct EventPrepareRun<'a> {
    /// Caller-owned model contract, immutably borrowed for this dispatch.
    pub contract: &'a ExecutionContract,
    /// Caller-owned PCM samples, immutably borrowed for this dispatch.
    pub pcm: &'a [f32],
    /// Sample rate associated with [`Self::pcm`].
    pub sample_rate: i32,
    /// Channel count associated with [`Self::pcm`].
    pub channel_count: i32,
    /// Caller-owned feature output buffer. The actor writes only during this
    /// synchronous dispatch and never retains the buffer.
    pub features: RefCell<&'a mut [f32]>,
    /// Caller-owned output slot for the produced feature-frame count.
    pub frame_count_out: RefCell<&'a mut i32>,
    /// Caller-owned output slot for the feature-bin count.
    pub feature_bin_count_out: RefCell<&'a mut i32>,
    extraction_outcome: RefCell<ExtractionOutcome>,
    /// Optional caller-owned error output slot. The actor writes the immediate
    /// dispatch result and does not retain this reference.
    pub error_out: RefCell<Option<&'a mut Error>>,
    /// Optional caller-owned native route. It is borrowed only synchronously;
    /// route dispatch temporarily takes and then replaces the slot value.
    pub native_feature_extractor_route: RefCell<Option<&'a mut NativeFeatureExtractorRoute<'a>>>,
    /// Optional synchronous completion callback. The callback must not retain
    /// event data or re-enter the request actor.
    pub on_done: Option<DoneCallback>,
    /// Optional synchronous error callback. The callback must not retain event
    /// data or re-enter the request actor.
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
            extraction_outcome: RefCell::new(ExtractionOutcome::Pending),
            error_out: RefCell::new(None),
            native_feature_extractor_route: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
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
    DiarizationSortformerRequest<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_model_contract_decision"_s <= *"state_ready"_s + EventPrepareRun(&'dispatch EventPrepareRun<'event>) / effect_begin_prepare,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_sample_rate_valid],
        "state_error_error_out_decision"_s <= "state_sample_rate_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_channel_count_valid],
        "state_error_error_out_decision"_s <= "state_channel_count_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_output_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_pcm_shape_valid],
        "state_error_error_out_decision"_s <= "state_pcm_shape_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_preparing"_s <= "state_output_capacity_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_output_capacity_valid],
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_output_capacity_invalid] / effect_mark_capacity_invalid,
        "state_native_extracting"_s <= "state_preparing"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_native_feature_extractor_route] / effect_extract_features_native,
        "state_callback_extracting"_s <= "state_preparing"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_callback_feature_extractor_route] / effect_extract_features_callback,
        "state_error_error_out_decision"_s <= "state_preparing"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_feature_extractor_route_missing] / effect_mark_feature_extractor_missing,
        "state_success_error_out_decision"_s <= "state_native_extracting"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_extraction_succeeded] / effect_mark_extraction_succeeded,
        "state_error_error_out_decision"_s <= "state_native_extracting"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_extraction_failed] / effect_mark_extraction_failed,
        "state_success_error_out_decision"_s <= "state_callback_extracting"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_extraction_succeeded] / effect_mark_extraction_succeeded,
        "state_error_error_out_decision"_s <= "state_callback_extracting"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_extraction_failed] / effect_mark_extraction_failed,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>),
        "state_ready"_s <= "state_errored"_s + completion<EventPrepareRun>(&'dispatch EventPrepareRun<'event>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_sample_rate_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_sample_rate_decision,
        "state_ready"_s <= "state_channel_count_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_channel_count_decision,
        "state_ready"_s <= "state_pcm_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_pcm_shape_decision,
        "state_ready"_s <= "state_output_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_output_capacity_decision,
        "state_ready"_s <= "state_preparing"_s + unexpected_event<_> / effect_on_unexpected_from_state_preparing,
        "state_ready"_s <= "state_native_extracting"_s + unexpected_event<_> / effect_on_unexpected_from_state_native_extracting,
        "state_ready"_s <= "state_callback_extracting"_s + unexpected_event<_> / effect_on_unexpected_from_state_callback_extracting,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

/// Actor-owned request state retained between synchronous dispatches.
///
/// `extraction_outcome` is a dispatch-local completion bridge used by later
/// generated SML guards/effects; replacing it with an event-carried outcome
/// requires a state-machine rewrite. The public error is retained because the
/// enclosing Sortformer owner reads the immediate result after dispatch.
#[derive(Debug, Default)]
pub struct DiarizationSortformerRequestContext {
    /// Most recent request error, consumed immediately by the enclosing owner.
    pub err: Error,
}

impl DiarizationSortformerRequestStateMachineContext for DiarizationSortformerRequestContext {
    fn effect_begin_prepare<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::None;
        *event.extraction_outcome.borrow_mut() = ExtractionOutcome::Pending;
        **event.frame_count_out.borrow_mut() = 0;
        **event.feature_bin_count_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_emit_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_done {
            let _ = callback(PrepareDone {
                frame_count: **event.frame_count_out.borrow(),
                feature_bin_count: **event.feature_bin_count_out.borrow(),
            });
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_error {
            let _ = callback(PrepareError { error: self.err });
        }
        Ok(())
    }
    fn effect_extract_features_native<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut route_slot = event.native_feature_extractor_route.borrow_mut();
        let route = route_slot
            .as_deref_mut()
            .expect("native extractor route selected by guard");
        let extraction = {
            let mut features = event.features.borrow_mut();
            route.extract(event.pcm, &mut features[..REQUIRED_FEATURE_COUNT])
        };
        drop(route_slot);
        *event.extraction_outcome.borrow_mut() = ExtractionOutcome::from(extraction);
        Ok(())
    }
    fn effect_extract_features_callback<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let extractor = event
            .contract
            .extract_features
            .expect("callback extractor route selected by guard");
        let extracted = {
            let mut features = event.features.borrow_mut();
            extractor(
                event.pcm,
                event.contract,
                &mut features[..REQUIRED_FEATURE_COUNT],
            )
        };
        *event.extraction_outcome.borrow_mut() = ExtractionOutcome::from(extracted);
        Ok(())
    }
    fn effect_mark_feature_extractor_missing<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::FeatureExtractor;
        Ok(())
    }
    fn effect_mark_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::Capacity;
        Ok(())
    }
    fn effect_mark_channel_count_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ChannelCount;
        Ok(())
    }
    fn effect_mark_model_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ModelInvalid;
        Ok(())
    }
    fn effect_mark_pcm_shape_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::PcmShape;
        Ok(())
    }
    fn effect_mark_sample_rate_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::SampleRate;
        Ok(())
    }
    fn effect_mark_extraction_failed<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = event.extraction_outcome.borrow().error();
        Ok(())
    }
    fn effect_mark_extraction_succeeded<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.frame_count_out.borrow_mut() = FEATURE_FRAME_COUNT;
        **event.feature_bin_count_out.borrow_mut() = FEATURE_BIN_COUNT;
        Ok(())
    }
    fn effect_store_error_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() {
            *error_out = self.err;
        }
        Ok(())
    }
    fn effect_store_success_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() {
            *error_out = self.err;
        }
        Ok(())
    }

    fn guard_channel_count_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count != CHANNEL_COUNT)
    }
    fn guard_native_feature_extractor_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_feature_extractor_route.borrow().is_some())
    }
    fn guard_callback_feature_extractor_route<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_feature_extractor_route.borrow().is_none()
            && event.contract.extract_features.is_some())
    }
    fn guard_feature_extractor_route_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.native_feature_extractor_route.borrow().is_none()
            && event.contract.extract_features.is_none())
    }
    fn guard_extraction_succeeded<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.extraction_outcome.borrow(),
            ExtractionOutcome::Succeeded
        ))
    }

    fn guard_extraction_failed<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.extraction_outcome.borrow(),
            ExtractionOutcome::Failed(_)
        ))
    }
    fn guard_channel_count_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count == CHANNEL_COUNT)
    }
    fn guard_has_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_done.is_some())
    }
    fn guard_has_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_error.is_some())
    }
    fn guard_has_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.error_out.borrow().is_some())
    }
    fn guard_model_contract_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.contract.model_contract_valid())
    }
    fn guard_model_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.contract.model_contract_valid())
    }
    fn guard_no_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_done.is_none())
    }
    fn guard_no_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_error.is_none())
    }
    fn guard_no_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.error_out.borrow().is_none())
    }
    fn guard_output_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_output_capacity_valid(event)?)
    }
    fn guard_output_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.features.borrow().len() >= REQUIRED_FEATURE_COUNT)
    }
    fn guard_pcm_shape_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_pcm_shape_valid(event)?)
    }
    fn guard_pcm_shape_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.pcm.len() == REQUIRED_SAMPLE_COUNT)
    }
    fn guard_sample_rate_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate != SAMPLE_RATE)
    }
    fn guard_sample_rate_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventPrepareRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate == SAMPLE_RATE)
    }

    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_preparing(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_native_extracting(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_callback_extracting(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
}

/// Source-compatible synchronous request wrapper.
/// Single-writer synchronous request actor.
pub struct Request {
    machine: DiarizationSortformerRequestStateMachine<DiarizationSortformerRequestContext>,
}

impl Request {
    #[must_use]
    pub fn new(context: DiarizationSortformerRequestContext) -> Self {
        Self {
            machine: DiarizationSortformerRequestStateMachine::new(context),
        }
    }

    /// Dispatches one borrowed request through the complete validation and
    /// preparation graph.
    pub fn prepare(&mut self, event: &EventPrepareRun<'_>) -> bool {
        self.machine
            .process_event(DiarizationSortformerRequestEvents::EventPrepareRun(event))
            .is_ok()
    }

    /// Returns the most recent request error without exposing machine context.
    #[must_use]
    pub fn error(&self) -> Error {
        self.machine.context().err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn callback_success(_pcm: &[f32], _contract: &ExecutionContract, features: &mut [f32]) -> bool {
        features.fill(1.25);
        true
    }

    fn callback_failure(
        _pcm: &[f32],
        _contract: &ExecutionContract,
        _features: &mut [f32],
    ) -> bool {
        false
    }

    #[test]
    fn callback_route_publishes_exact_shape_and_outputs() {
        let mut contract = ExecutionContract::pinned();
        contract.extract_features = Some(callback_success);
        let pcm = vec![0.0; REQUIRED_SAMPLE_COUNT];
        let mut features = vec![f32::NAN; REQUIRED_FEATURE_COUNT];
        let mut frame_count = -1;
        let mut feature_bin_count = -1;
        let mut error = Error::Unexpected;
        let event = EventPrepareRun::new(
            &contract,
            &pcm,
            SAMPLE_RATE,
            CHANNEL_COUNT,
            &mut features,
            &mut frame_count,
            &mut feature_bin_count,
        )
        .with_error_out(&mut error);
        let mut request = Request::new(DiarizationSortformerRequestContext::default());

        assert!(request.prepare(&event));
        assert_eq!(features, vec![1.25; REQUIRED_FEATURE_COUNT]);
        assert_eq!(frame_count, FEATURE_FRAME_COUNT);
        assert_eq!(feature_bin_count, FEATURE_BIN_COUNT);
        assert_eq!(error, Error::None);
    }

    #[test]
    fn callback_failure_and_missing_route_publish_typed_error_without_success_shape() {
        for use_callback in [true, false] {
            let mut contract = ExecutionContract::pinned();
            contract.extract_features = if use_callback {
                Some(callback_failure as ExtractFeaturesFn)
            } else {
                None
            };
            let pcm = vec![0.0; REQUIRED_SAMPLE_COUNT];
            let mut features = vec![f32::NAN; REQUIRED_FEATURE_COUNT];
            let mut frame_count = -1;
            let mut feature_bin_count = -1;
            let mut error = Error::None;
            let event = EventPrepareRun::new(
                &contract,
                &pcm,
                SAMPLE_RATE,
                CHANNEL_COUNT,
                &mut features,
                &mut frame_count,
                &mut feature_bin_count,
            )
            .with_error_out(&mut error);
            let mut request = Request::new(DiarizationSortformerRequestContext::default());

            assert!(request.prepare(&event));
            assert_eq!(frame_count, 0);
            assert_eq!(feature_bin_count, 0);
            assert_eq!(error, Error::FeatureExtractor);
            assert!(features.iter().all(|value| value.is_nan()));
        }
    }
}
