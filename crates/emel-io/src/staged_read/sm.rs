//! Explicit staged-copy orchestration and bounded data-plane copy actions.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "stateforward-sml generated state tokens intentionally derive PartialEq"
)]

use core::cell::{Cell, RefCell};

use sml::sml;

use super::event::{
    Callback, Error, StageSpan, StageWindowBatchDone, StageWindowBatchError, StageWindowDone,
    StageWindowError,
};

/// Keeps one batch assessment statically bounded within a single RTC dispatch.
pub(super) const MAX_STAGE_BATCH_TENSORS: usize = 65_536;

pub(super) struct Context {
    pub(super) platform_supported: bool,
}

#[derive(Clone, Copy)]
pub(super) struct SingleStatus {
    pub(super) result: Result<StageWindowDone, Error>,
}

impl SingleStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(Error::InternalError),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BatchStatus {
    pub(super) result: Result<StageWindowBatchDone, StageWindowBatchError>,
}

/// Dispatch-local result produced by one bounded assessment operation and
/// consumed only by explicit completion guards.
#[derive(Clone, Copy)]
pub(super) struct BatchAssessment {
    valid: bool,
    failed_index: u32,
}

impl BatchAssessment {
    pub(super) const fn pending() -> Self {
        Self {
            valid: false,
            failed_index: 0,
        }
    }
}

impl BatchStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(StageWindowBatchError::new(Error::InternalError, 0)),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SingleRuntime<'dispatch, 'data, 'callback> {
    pub(super) file_offset: u64,
    pub(super) logical_byte_length: u64,
    pub(super) stage_chunk_bytes: u64,
    pub(super) source: Option<&'data [u8]>,
    pub(super) target: &'data RefCell<&'data mut [u8]>,
    pub(super) on_done: Option<Callback<'callback, StageWindowDone>>,
    pub(super) on_error: Option<Callback<'callback, StageWindowError>>,
    pub(super) status: &'dispatch Cell<SingleStatus>,
}

#[derive(Clone, Copy)]
pub(super) struct BatchRuntime<'dispatch, 'data, 'callback> {
    pub(super) tensors: &'data [StageSpan<'data>],
    pub(super) stage_chunk_bytes: u64,
    pub(super) on_done: Option<Callback<'callback, StageWindowBatchDone>>,
    pub(super) on_error: Option<Callback<'callback, StageWindowBatchError>>,
    pub(super) assessment: &'dispatch Cell<BatchAssessment>,
    pub(super) status: &'dispatch Cell<BatchStatus>,
}

sml! {
    IoStagedRead {
        // Single-window validation and copy-path selection.
        "state_single_callbacks_decision"_s <= *"state_ready"_s + Single(SingleRuntime<'dispatch, 'data, 'callback>),
        "state_single_contract_decision"_s <= "state_single_callbacks_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_callbacks_present],
        "state_single_invalid_callbacks"_s <= "state_single_callbacks_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_callbacks_missing] / effect_mark_single_invalid_callbacks,
        "state_single_target_decision"_s <= "state_single_contract_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_contract_valid],
        "state_single_invalid_contract"_s <= "state_single_contract_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_contract_invalid] / effect_mark_single_invalid_contract,
        "state_single_platform_decision"_s <= "state_single_target_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_target_valid],
        "state_single_invalid_target"_s <= "state_single_target_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_target_invalid] / effect_mark_single_invalid_target,
        "state_single_source_presence_decision"_s <= "state_single_platform_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_platform_supported],
        "state_single_unsupported_platform"_s <= "state_single_platform_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_platform_unsupported] / effect_mark_single_unsupported_platform,
        "state_single_source_size_decision"_s <= "state_single_source_presence_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_source_present],
        "state_single_null_source"_s <= "state_single_source_presence_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_source_missing] / effect_mark_single_null_source,
        "state_single_copy_decision"_s <= "state_single_source_size_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_source_exact],
        "state_single_insufficient_source"_s <= "state_single_source_size_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_source_insufficient] / effect_mark_single_insufficient_source,
        "state_single_source_mismatch"_s <= "state_single_source_size_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_source_larger] / effect_mark_single_source_mismatch,
        "state_ready"_s <= "state_single_copy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_chunk_aligned] / effect_copy_single_aligned,
        "state_ready"_s <= "state_single_copy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_chunk_remainder] / effect_copy_single_remainder,

        // Single-window error publication.
        "state_single_error_published"_s <= "state_single_invalid_callbacks"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_invalid_callbacks"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_invalid_contract"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_invalid_contract"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_invalid_target"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_invalid_target"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_unsupported_platform"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_unsupported_platform"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_null_source"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_null_source"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_insufficient_source"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_insufficient_source"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_single_error_published"_s <= "state_single_source_mismatch"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_source_mismatch"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) [guard_single_error_callback_absent] / effect_finish_single_error,
        "state_ready"_s <= "state_single_error_published"_s + completion<Single>(SingleRuntime<'dispatch, 'data, 'callback>) / effect_finish_single_error,

        // Batch validation and one bounded data-plane action.
        "state_batch_callbacks_decision"_s <= "state_ready"_s + Batch(BatchRuntime<'dispatch, 'data, 'callback>),
        "state_batch_assessment_decision"_s <= "state_batch_callbacks_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_callbacks_present] / effect_compute_batch_assessment,
        "state_batch_invalid_callbacks"_s <= "state_batch_callbacks_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_callbacks_missing] / effect_mark_batch_invalid_callbacks,
        "state_batch_platform_decision"_s <= "state_batch_assessment_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_assessment_valid],
        "state_batch_invalid_contract"_s <= "state_batch_assessment_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_assessment_invalid] / effect_mark_batch_invalid_contract,
        "state_batch_done"_s <= "state_batch_platform_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_platform_batch_supported] / effect_copy_batch,
        "state_batch_unsupported_platform"_s <= "state_batch_platform_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_platform_batch_unsupported] / effect_mark_batch_unsupported_platform,
        "state_ready"_s <= "state_batch_done"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>),

        // Batch error publication.
        "state_batch_error_published"_s <= "state_batch_invalid_callbacks"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_present] / effect_publish_batch_error,
        "state_ready"_s <= "state_batch_invalid_callbacks"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_absent] / effect_finish_batch_error,
        "state_batch_error_published"_s <= "state_batch_invalid_contract"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_present] / effect_publish_batch_error,
        "state_ready"_s <= "state_batch_invalid_contract"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_absent] / effect_finish_batch_error,
        "state_batch_error_published"_s <= "state_batch_unsupported_platform"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_present] / effect_publish_batch_error,
        "state_ready"_s <= "state_batch_unsupported_platform"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) [guard_batch_error_callback_absent] / effect_finish_batch_error,
        "state_ready"_s <= "state_batch_error_published"_s + completion<Batch>(BatchRuntime<'dispatch, 'data, 'callback>) / effect_finish_batch_error,

        // Every externally reachable state has fail-closed unexpected behavior.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_callbacks_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_contract_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_target_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_platform_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_source_presence_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_source_size_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_copy_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_invalid_callbacks"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_invalid_contract"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_invalid_target"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_unsupported_platform"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_null_source"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_insufficient_source"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_source_mismatch"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_error_published"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_callbacks_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_assessment_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_platform_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_done"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_invalid_callbacks"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_invalid_contract"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_unsupported_platform"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_error_published"_s + unexpected_event<_> / effect_unexpected,
    }
}

impl IoStagedReadStateMachineContext for Context {
    fn guard_single_callbacks_present(
        &self,
        event: &SingleRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some() && event.on_error.is_some())
    }
    fn guard_single_callbacks_missing(
        &self,
        event: &SingleRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none() || event.on_error.is_none())
    }
    fn guard_single_contract_valid(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.logical_byte_length > 0
            && event.stage_chunk_bytes > 0
            && event.stage_chunk_bytes <= event.logical_byte_length
            && event
                .file_offset
                .checked_add(event.logical_byte_length)
                .is_some()
            && usize::try_from(event.logical_byte_length).is_ok())
    }
    fn guard_single_contract_invalid(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!self.guard_single_contract_valid(event)?)
    }
    fn guard_single_target_valid(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(u64::try_from(event.target.borrow().len())
            .is_ok_and(|len| len >= event.logical_byte_length))
    }
    fn guard_single_target_invalid(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!self.guard_single_target_valid(event)?)
    }
    fn guard_platform_supported(&self, _: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(self.platform_supported)
    }
    fn guard_platform_unsupported(&self, _: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!self.platform_supported)
    }
    fn guard_single_source_present(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.source.is_some())
    }
    fn guard_single_source_missing(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.source.is_none())
    }
    fn guard_single_source_exact(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.source.is_some_and(|source| {
            u64::try_from(source.len()).is_ok_and(|len| len == event.logical_byte_length)
        }))
    }
    fn guard_single_source_insufficient(
        &self,
        event: &SingleRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.source.is_some_and(|source| {
            u64::try_from(source.len()).is_ok_and(|len| len < event.logical_byte_length)
        }))
    }
    fn guard_single_source_larger(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.source.is_some_and(|source| {
            u64::try_from(source.len()).is_ok_and(|len| len > event.logical_byte_length)
        }))
    }
    fn guard_single_chunk_aligned(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event
            .logical_byte_length
            .is_multiple_of(event.stage_chunk_bytes))
    }
    fn guard_single_chunk_remainder(&self, event: &SingleRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!event
            .logical_byte_length
            .is_multiple_of(event.stage_chunk_bytes))
    }
    fn guard_single_error_callback_present(
        &self,
        event: &SingleRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_single_error_callback_absent(
        &self,
        event: &SingleRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn effect_mark_single_invalid_callbacks(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::InvalidCallbacks);
        Ok(())
    }
    fn effect_mark_single_invalid_contract(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::InvalidStageContract);
        Ok(())
    }
    fn effect_mark_single_invalid_target(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::InvalidTargetWindow);
        Ok(())
    }
    fn effect_mark_single_unsupported_platform(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::UnsupportedPlatform);
        Ok(())
    }
    fn effect_mark_single_null_source(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::NullSourceSpan);
        Ok(())
    }
    fn effect_mark_single_insufficient_source(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::InsufficientSourceSpan);
        Ok(())
    }
    fn effect_mark_single_source_mismatch(
        &mut self,
        event: SingleRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_single_error(event, Error::SourceSpanSizeMismatch);
        Ok(())
    }
    fn effect_publish_single_error(&mut self, event: SingleRuntime<'_, '_, '_>) -> Result<(), ()> {
        event
            .on_error
            .expect("error-callback guard selected a callback")
            .publish(StageWindowError::new(
                event
                    .status
                    .get()
                    .result
                    .expect_err("error state selected a classified result"),
            ));
        Ok(())
    }
    fn effect_finish_single_error(&mut self, _: SingleRuntime<'_, '_, '_>) -> Result<(), ()> {
        Ok(())
    }
    fn effect_copy_single_aligned(&mut self, event: SingleRuntime<'_, '_, '_>) -> Result<(), ()> {
        copy_single::<false>(event);
        Ok(())
    }
    fn effect_copy_single_remainder(&mut self, event: SingleRuntime<'_, '_, '_>) -> Result<(), ()> {
        copy_single::<true>(event);
        Ok(())
    }
    fn guard_batch_callbacks_present(&self, event: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_some() && event.on_error.is_some())
    }
    fn guard_batch_callbacks_missing(&self, event: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_none() || event.on_error.is_none())
    }
    fn guard_batch_assessment_valid(&self, event: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(event.assessment.get().valid)
    }
    fn guard_batch_assessment_invalid(&self, event: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!event.assessment.get().valid)
    }
    fn guard_platform_batch_supported(&self, _: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(self.platform_supported)
    }
    fn guard_platform_batch_unsupported(&self, _: &BatchRuntime<'_, '_, '_>) -> Result<bool, ()> {
        Ok(!self.platform_supported)
    }
    fn guard_batch_error_callback_present(
        &self,
        event: &BatchRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_batch_error_callback_absent(
        &self,
        event: &BatchRuntime<'_, '_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn effect_mark_batch_invalid_callbacks(
        &mut self,
        event: BatchRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_batch_error(event, Error::InvalidCallbacks, 0);
        Ok(())
    }
    fn effect_mark_batch_invalid_contract(
        &mut self,
        event: BatchRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.assessment.get().failed_index;
        set_batch_error(event, Error::InvalidStageContract, failed_index);
        Ok(())
    }
    fn effect_compute_batch_assessment(
        &mut self,
        event: BatchRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        event.assessment.set(compute_batch_assessment(event));
        Ok(())
    }
    fn effect_mark_batch_unsupported_platform(
        &mut self,
        event: BatchRuntime<'_, '_, '_>,
    ) -> Result<(), ()> {
        set_batch_error(event, Error::UnsupportedPlatform, 0);
        Ok(())
    }
    fn effect_publish_batch_error(&mut self, event: BatchRuntime<'_, '_, '_>) -> Result<(), ()> {
        let error = event
            .status
            .get()
            .result
            .expect_err("batch error state selected a classified result");
        event
            .on_error
            .expect("batch error-callback guard selected a callback")
            .publish(error);
        Ok(())
    }
    fn effect_finish_batch_error(&mut self, _: BatchRuntime<'_, '_, '_>) -> Result<(), ()> {
        Ok(())
    }
    fn effect_copy_batch(&mut self, event: BatchRuntime<'_, '_, '_>) -> Result<(), ()> {
        copy_batch(event);
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        std::process::abort()
    }
}

fn set_single_error(event: SingleRuntime<'_, '_, '_>, error: Error) {
    event.status.set(SingleStatus { result: Err(error) });
}

fn copy_single<const HAS_REMAINDER: bool>(event: SingleRuntime<'_, '_, '_>) {
    let logical = usize::try_from(event.logical_byte_length)
        .expect("contract guard selected a representable length");
    let chunk = usize::try_from(event.stage_chunk_bytes)
        .expect("contract guard selected a representable chunk");
    let source = event.source.expect("source guards selected an exact span");
    let mut target = event.target.borrow_mut();
    let covered = logical - (usize::from(HAS_REMAINDER) * (logical % chunk));
    let mut progressed = 0;
    while progressed < covered {
        target[progressed..progressed + chunk]
            .copy_from_slice(&source[progressed..progressed + chunk]);
        progressed += chunk;
    }
    let tail = logical - progressed;
    target[progressed..logical].copy_from_slice(&source[progressed..progressed + tail]);
    let done = StageWindowDone::new(event.logical_byte_length);
    event.status.set(SingleStatus { result: Ok(done) });
    event
        .on_done
        .expect("callback guard selected a success callback")
        .publish(done);
}

fn span_valid(span: StageSpan<'_>) -> bool {
    (span.byte_size > 0)
        & span
            .file_offset
            .checked_add(span.byte_size)
            .is_some_and(|end| {
                span.source
                    .is_some_and(|source| u64::try_from(source.len()).is_ok_and(|len| end <= len))
            })
        & u64::try_from(span.target.len()).is_ok_and(|len| len >= span.byte_size)
}

fn compute_batch_assessment(event: BatchRuntime<'_, '_, '_>) -> BatchAssessment {
    let global_valid = (event.stage_chunk_bytes > 0)
        & usize::try_from(event.stage_chunk_bytes).is_ok()
        & !event.tensors.is_empty()
        & (event.tensors.len() <= MAX_STAGE_BATCH_TENSORS)
        & u32::try_from(event.tensors.len()).is_ok();
    let scan_len = event.tensors.len().min(MAX_STAGE_BATCH_TENSORS);
    let mut first = 0_u32;
    let mut found = 0_u32;
    let mut index = 0_usize;
    while index < scan_len {
        let index_u32 = u32::try_from(index).expect("bounded batch index fits in u32");
        let failed = u32::from(!span_valid(event.tensors[index]));
        let take = (1 - found) * failed;
        first = (first * (1 - take)) + (index_u32 * take);
        found |= failed;
        index += 1;
    }
    BatchAssessment {
        valid: global_valid & (found == 0),
        failed_index: first * u32::from(global_valid),
    }
}

fn copy_batch(event: BatchRuntime<'_, '_, '_>) {
    let mut done_count = 0_u32;
    let mut bytes_committed = 0_u64;
    for tensor in event.tensors {
        let logical = usize::try_from(tensor.byte_size)
            .expect("batch guard selected a representable byte size");
        let offset = usize::try_from(tensor.file_offset)
            .expect("batch guard selected a representable offset");
        let requested = usize::try_from(event.stage_chunk_bytes)
            .expect("batch guard selected a representable chunk size");
        let chunk = requested.min(logical);
        let covered = logical - (logical % chunk);
        let source = tensor.source.expect("batch guard selected a source");
        let mut target = tensor.target.bytes.borrow_mut();
        let mut progressed = 0;
        while progressed < covered {
            target[progressed..progressed + chunk]
                .copy_from_slice(&source[offset + progressed..offset + progressed + chunk]);
            progressed += chunk;
        }
        let tail = logical - progressed;
        target[progressed..logical]
            .copy_from_slice(&source[offset + progressed..offset + progressed + tail]);
        done_count += 1;
        bytes_committed = bytes_committed.wrapping_add(tensor.byte_size);
    }
    let done = StageWindowBatchDone::new(done_count, bytes_committed);
    event.status.set(BatchStatus { result: Ok(done) });
    event
        .on_done
        .expect("callback guard selected a batch success callback")
        .publish(done);
}

fn set_batch_error(event: BatchRuntime<'_, '_, '_>, error: Error, failed_index: u32) {
    event.status.set(BatchStatus {
        result: Err(StageWindowBatchError::new(error, failed_index)),
    });
}
