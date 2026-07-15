//! Private explicit state machine for read validation, copy, and publication.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "SML-generated state enums contain completion machinery without Eq"
)]

use core::cell::{Cell, RefCell};

use sml::sml;

use super::event::{
    Callback, Error, ReadTensorBatchDone, ReadTensorBatchError, ReadTensorDone, ReadTensorError,
    SourceError, TensorRead,
};

pub(super) const MAX_FILE_INDEX: u16 = 65_534;
pub(super) const MAX_FILE_PATH_BYTES: usize = 4_095;
pub(super) const MAX_READ_BYTES: u64 = 1_u64 << 40;
pub(super) const MAX_READ_BATCH_TENSORS: usize = 65_536;
pub(super) const PLATFORM_SUPPORTED: bool = cfg!(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_family = "unix"
));

#[derive(Clone, Copy, Debug)]
pub(super) struct ReadStatus {
    pub(super) result: Result<ReadTensorDone, Error>,
}

impl ReadStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(Error::InternalError),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchStatus {
    pub(super) result: Result<ReadTensorBatchDone, ReadTensorBatchError>,
}

impl BatchStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(ReadTensorBatchError::new(Error::InternalError, 0)),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ReadRuntime<'dispatch, 'data> {
    pub(super) tensor_id: i32,
    pub(super) file_index: u16,
    pub(super) file_offset: u64,
    pub(super) byte_size: u64,
    pub(super) file_path: &'data str,
    pub(super) source: Option<&'data [u8]>,
    pub(super) source_error: Option<SourceError>,
    pub(super) target: &'data RefCell<&'data mut [u8]>,
    pub(super) on_done: Option<Callback<'data, ReadTensorDone>>,
    pub(super) on_error: Option<Callback<'data, ReadTensorError>>,
    pub(super) platform_supported: bool,
    pub(super) status: &'dispatch Cell<ReadStatus>,
}

#[derive(Clone, Debug)]
pub(super) struct TensorBatchRuntime<'dispatch, 'data> {
    pub(super) tensors: &'data [TensorRead<'data>],
    pub(super) on_done: Option<Callback<'data, ReadTensorBatchDone>>,
    pub(super) on_error: Option<Callback<'data, ReadTensorBatchError>>,
    pub(super) status: &'dispatch Cell<BatchStatus>,
    pub(super) analysis: &'dispatch Cell<BatchAnalysis>,
}

#[derive(Clone, Copy, Debug, Default)]
struct BatchFailureIndex {
    index: u32,
    found: u32,
}

impl BatchFailureIndex {
    const fn record(self, index: u32) -> Self {
        let take = 1 - self.found;
        Self {
            index: (self.index * (1 - take)) + (index * take),
            found: 1,
        }
    }
}

/// Dispatch-local first-failure evidence, grouped by pinned validation phase.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BatchAnalysis {
    invalid_request: BatchFailureIndex,
    unsupported_resource: BatchFailureIndex,
    file_open_failed: BatchFailureIndex,
    file_seek_failed: BatchFailureIndex,
    file_read_failed: BatchFailureIndex,
    short_read: BatchFailureIndex,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchSpanRuntime<'dispatch, 'data> {
    span: &'data TensorRead<'data>,
    index: u32,
    analysis: &'dispatch Cell<BatchAnalysis>,
}

sml! {
    BatchSpanClassifier {
        // Per-span request validation.
        "state_request_decision"_s <= *"state_ready"_s + EventClassifySpan(BatchSpanRuntime<'dispatch, 'data>),
        "state_resource_decision"_s <= "state_request_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_request_valid],
        "state_ready"_s <= "state_request_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_request_invalid] / effect_record_invalid_request,

        // Per-span source validation.
        "state_source_open_decision"_s <= "state_resource_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_resource_supported],
        "state_ready"_s <= "state_resource_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_resource_unsupported] / effect_record_unsupported_resource,
        "state_source_seek_decision"_s <= "state_source_open_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_open_succeeded],
        "state_ready"_s <= "state_source_open_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_open_failed] / effect_record_file_open_failed,
        "state_source_read_decision"_s <= "state_source_seek_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_seek_succeeded],
        "state_ready"_s <= "state_source_seek_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_seek_failed] / effect_record_file_seek_failed,
        "state_ready"_s <= "state_source_read_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_read_failed] / effect_record_file_read_failed,
        "state_ready"_s <= "state_source_read_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_read_short] / effect_record_short_read,
        "state_ready"_s <= "state_source_read_decision"_s + completion<EventClassifySpan>(BatchSpanRuntime<'dispatch, 'data>) [guard_source_read_succeeded],

        // Explicit unexpected-event recovery.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_resource_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_open_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_seek_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_read_decision"_s + unexpected_event<_> / effect_on_unexpected,
    }
}

impl From<BatchSpanClassifierError> for () {
    // Preserve child transition/action failure through the parent's `?` path.
    // The Reader boundary maps the resulting parent failure to InternalError.
    fn from(_error: BatchSpanClassifierError) {}
}

sml! {
    IoRead {
        // Single request validation.
        "state_request_decision"_s <= *"state_ready"_s + ReadTensor(ReadRuntime<'dispatch, 'data>),
        "state_path_decision"_s <= "state_request_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_request_valid],
        "state_errored"_s <= "state_request_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_request_invalid] / effect_mark_invalid_request,
        "state_resource_decision"_s <= "state_path_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_path_valid],
        "state_errored"_s <= "state_path_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_path_invalid] / effect_mark_invalid_request,
        "state_target_decision"_s <= "state_resource_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_resource_supported],
        "state_errored"_s <= "state_resource_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_resource_unsupported] / effect_mark_unsupported_resource,
        "state_platform_decision"_s <= "state_target_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_target_valid],
        "state_errored"_s <= "state_target_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_target_invalid] / effect_mark_invalid_request,
        "state_source_open_decision"_s <= "state_platform_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_platform_supported],
        "state_errored"_s <= "state_platform_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_platform_unsupported] / effect_mark_unsupported_platform,

        // Single source validation and copy.
        "state_source_seek_decision"_s <= "state_source_open_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_open_succeeded],
        "state_errored"_s <= "state_source_open_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_open_failed] / effect_mark_file_open_failed,
        "state_source_read_decision"_s <= "state_source_seek_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_seek_succeeded],
        "state_errored"_s <= "state_source_seek_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_seek_failed] / effect_mark_file_seek_failed,
        "state_errored"_s <= "state_source_read_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_read_failed] / effect_mark_file_read_failed,
        "state_errored"_s <= "state_source_read_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_read_short] / effect_mark_short_read,
        "state_done_callback_decision"_s <= "state_source_read_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_source_read_succeeded] / effect_copy_tensor,

        // Single callback publication.
        "state_ready"_s <= "state_done_callback_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_done_callback_present] / effect_publish_done,
        "state_ready"_s <= "state_done_callback_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_done_callback_absent],
        "state_error_callback_decision"_s <= "state_errored"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_error_callback_present] / effect_publish_error,
        "state_ready"_s <= "state_errored"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>) [guard_error_callback_absent],
        "state_ready"_s <= "state_error_callback_decision"_s + completion<ReadTensor>(ReadRuntime<'dispatch, 'data>),

        // Batch count and child classification.
        "state_batch_count_decision"_s <= "state_ready"_s + ReadTensorBatch(TensorBatchRuntime<'dispatch, 'data>),
        "state_batch_request_decision"_s <= "state_batch_count_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_count_valid] / effect_classify_batch,
        "state_batch_errored"_s <= "state_batch_count_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_count_invalid] / effect_mark_batch_count_invalid,

        // Batch phase-priority outcome selection.
        "state_batch_resource_decision"_s <= "state_batch_request_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_requests_valid],
        "state_batch_errored"_s <= "state_batch_request_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_requests_invalid] / effect_mark_batch_invalid_request,
        "state_batch_source_open_decision"_s <= "state_batch_resource_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_resources_supported],
        "state_batch_errored"_s <= "state_batch_resource_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_resources_unsupported] / effect_mark_batch_unsupported_resource,
        "state_batch_source_seek_decision"_s <= "state_batch_source_open_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_open_succeeded],
        "state_batch_errored"_s <= "state_batch_source_open_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_open_failed] / effect_mark_batch_file_open_failed,
        "state_batch_source_read_decision"_s <= "state_batch_source_seek_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_seek_succeeded],
        "state_batch_errored"_s <= "state_batch_source_seek_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_seek_failed] / effect_mark_batch_file_seek_failed,
        "state_batch_errored"_s <= "state_batch_source_read_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_read_failed] / effect_mark_batch_file_read_failed,
        "state_batch_errored"_s <= "state_batch_source_read_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_read_short] / effect_mark_batch_short_read,
        "state_batch_done_callback_decision"_s <= "state_batch_source_read_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_source_read_succeeded] / effect_copy_batch,

        // Batch callback publication.
        "state_ready"_s <= "state_batch_done_callback_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_done_callback_present] / effect_publish_batch_done,
        "state_ready"_s <= "state_batch_done_callback_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_done_callback_absent],
        "state_batch_error_callback_decision"_s <= "state_batch_errored"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_error_callback_present] / effect_publish_batch_error,
        "state_ready"_s <= "state_batch_errored"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>) [guard_batch_error_callback_absent],
        "state_ready"_s <= "state_batch_error_callback_decision"_s + completion<ReadTensorBatch>(TensorBatchRuntime<'dispatch, 'data>),

        // Explicit unexpected-event recovery.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_path_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_resource_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_target_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_platform_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_open_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_seek_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_source_read_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_done_callback_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_count_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_request_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_resource_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_source_open_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_source_seek_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_source_read_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_done_callback_decision"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_errored"_s + unexpected_event<_> / effect_on_unexpected,
        "state_ready"_s <= "state_batch_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BatchClassifierContext;

impl BatchSpanClassifierStateMachineContext for BatchClassifierContext {
    fn guard_request_valid(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.span.byte_size > 0 && target_valid(event.span))
    }

    fn guard_request_invalid(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.span.byte_size == 0 || !target_valid(event.span))
    }

    fn guard_resource_supported(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(resource_supported(event.span))
    }

    fn guard_resource_unsupported(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!resource_supported(event.span))
    }

    fn guard_source_open_succeeded(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_open_succeeded(event.span))
    }

    fn guard_source_open_failed(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!source_open_succeeded(event.span))
    }

    fn guard_source_seek_succeeded(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_seek_succeeded(event.span))
    }

    fn guard_source_seek_failed(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!source_seek_succeeded(event.span))
    }

    fn guard_source_read_failed(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_read_failed(event.span))
    }

    fn guard_source_read_short(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_read_short(event.span))
    }

    fn guard_source_read_succeeded(&self, event: &BatchSpanRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_read_succeeded(event.span))
    }

    fn effect_record_invalid_request(&mut self, event: BatchSpanRuntime<'_, '_>) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.invalid_request = analysis.invalid_request.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_record_unsupported_resource(
        &mut self,
        event: BatchSpanRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.unsupported_resource = analysis.unsupported_resource.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_record_file_open_failed(
        &mut self,
        event: BatchSpanRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.file_open_failed = analysis.file_open_failed.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_record_file_seek_failed(
        &mut self,
        event: BatchSpanRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.file_seek_failed = analysis.file_seek_failed.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_record_file_read_failed(
        &mut self,
        event: BatchSpanRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.file_read_failed = analysis.file_read_failed.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_record_short_read(&mut self, event: BatchSpanRuntime<'_, '_>) -> Result<(), ()> {
        let mut analysis = event.analysis.get();
        analysis.short_read = analysis.short_read.record(event.index);
        event.analysis.set(analysis);
        Ok(())
    }

    fn effect_on_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// Parent-owned persistent state: only the private per-span classifier actor.
pub(super) struct Context {
    batch_classifier: BatchClassifier,
}

impl Context {
    pub(super) const fn new() -> Self {
        Self {
            batch_classifier: BatchClassifier::new(),
        }
    }
}

/// Owned child actor used only for synchronous same-RTC batch classification.
struct BatchClassifier {
    machine: BatchSpanClassifierStateMachine<BatchClassifierContext>,
}

impl BatchClassifier {
    const fn new() -> Self {
        Self {
            machine: BatchSpanClassifierStateMachine::new(BatchClassifierContext),
        }
    }

    fn process_event(&mut self, event: BatchSpanRuntime<'_, '_>) -> Result<(), ()> {
        self.machine
            .process_event(BatchSpanClassifierEvents::EventClassifySpan(event))?;
        Ok(())
    }
}

impl IoReadStateMachineContext for Context {
    fn guard_request_valid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.byte_size > 0)
    }

    fn guard_request_invalid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.byte_size == 0)
    }

    fn guard_path_valid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!event.file_path.is_empty()
            && event.file_path.len() <= MAX_FILE_PATH_BYTES
            && !event.file_path.as_bytes().contains(&0))
    }

    fn guard_path_invalid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.file_path.is_empty()
            || event.file_path.len() > MAX_FILE_PATH_BYTES
            || event.file_path.as_bytes().contains(&0))
    }

    fn guard_resource_supported(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.file_index <= MAX_FILE_INDEX
            && event.byte_size <= MAX_READ_BYTES
            && usize::try_from(event.byte_size).is_ok()
            && event.file_offset.checked_add(event.byte_size).is_some())
    }

    fn guard_resource_unsupported(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.file_index > MAX_FILE_INDEX
            || event.byte_size > MAX_READ_BYTES
            || usize::try_from(event.byte_size).is_err()
            || event.file_offset.checked_add(event.byte_size).is_none())
    }

    fn guard_target_valid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(u64::try_from(event.target.borrow().len()).is_ok_and(|len| len >= event.byte_size))
    }

    fn guard_target_invalid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(u64::try_from(event.target.borrow().len()).map_or(true, |len| len < event.byte_size))
    }

    fn guard_platform_supported(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.platform_supported)
    }

    fn guard_platform_unsupported(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!event.platform_supported)
    }

    fn guard_source_open_succeeded(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error != Some(SourceError::FileOpenFailed)
            && (event.source_error.is_some() || event.source.is_some()))
    }

    fn guard_source_open_failed(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error == Some(SourceError::FileOpenFailed)
            || (event.source_error.is_none() && event.source.is_none()))
    }

    fn guard_source_seek_succeeded(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error != Some(SourceError::FileSeekFailed)
            && (event.source_error.is_some()
                || event.source.is_some_and(|source| {
                    u64::try_from(source.len()).is_ok_and(|len| event.file_offset <= len)
                })))
    }

    fn guard_source_seek_failed(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error == Some(SourceError::FileSeekFailed)
            || (event.source_error.is_none()
                && event.source.is_some_and(|source| {
                    u64::try_from(source.len()).map_or(true, |len| event.file_offset > len)
                })))
    }

    fn guard_source_read_failed(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.source_error,
            Some(SourceError::FileReadFailed | SourceError::Other)
        ))
    }

    fn guard_source_read_short(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error == Some(SourceError::ShortRead)
            || (event.source_error.is_none()
                && event.source.is_some_and(|source| {
                    let source_len = source.len() as u64;
                    event.file_offset <= source_len
                        && event.byte_size > source_len - event.file_offset
                })))
    }

    fn guard_source_read_succeeded(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.source_error.is_none()
            && event.source.is_some_and(|source| {
                let source_len = source.len() as u64;
                event.file_offset <= source_len && event.byte_size <= source_len - event.file_offset
            }))
    }

    fn guard_done_callback_present(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_done_callback_absent(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_error_callback_present(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_error_callback_absent(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn effect_mark_invalid_request(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::InvalidRequest),
        });
        Ok(())
    }

    fn effect_mark_unsupported_resource(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::UnsupportedResource),
        });
        Ok(())
    }

    fn effect_mark_unsupported_platform(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::UnsupportedPlatform),
        });
        Ok(())
    }

    fn effect_mark_file_open_failed(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::FileOpenFailed),
        });
        Ok(())
    }

    fn effect_mark_file_seek_failed(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::FileSeekFailed),
        });
        Ok(())
    }

    fn effect_mark_file_read_failed(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::FileReadFailed),
        });
        Ok(())
    }

    fn effect_mark_short_read(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.status.set(ReadStatus {
            result: Err(Error::ShortRead),
        });
        Ok(())
    }

    fn effect_copy_tensor(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        let source = event.source.expect("read-success guard requires source");
        let offset = usize::try_from(event.file_offset)
            .expect("source-seek success guarantees an addressable offset");
        let byte_size =
            usize::try_from(event.byte_size).expect("resource guard guarantees addressable size");
        event.target.borrow_mut()[..byte_size].copy_from_slice(&source[offset..offset + byte_size]);
        event.status.set(ReadStatus {
            result: Ok(ReadTensorDone::new(event.tensor_id, event.byte_size)),
        });
        Ok(())
    }

    fn effect_publish_done(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event
            .on_done
            .expect("done-callback guard requires callback")
            .publish(ReadTensorDone::new(event.tensor_id, event.byte_size));
        Ok(())
    }

    fn effect_publish_error(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .status
            .get()
            .result
            .expect_err("error state requires error");
        event
            .on_error
            .expect("error-callback guard requires callback")
            .publish(ReadTensorError::new(event.tensor_id, error));
        Ok(())
    }

    fn guard_batch_count_valid(&self, event: &TensorBatchRuntime<'_, '_>) -> Result<bool, ()> {
        let count = event.tensors.len();
        Ok(count > 0 && count <= MAX_READ_BATCH_TENSORS && u32::try_from(count).is_ok())
    }

    fn guard_batch_count_invalid(&self, event: &TensorBatchRuntime<'_, '_>) -> Result<bool, ()> {
        let count = event.tensors.len();
        Ok(count == 0 || count > MAX_READ_BATCH_TENSORS || u32::try_from(count).is_err())
    }

    fn effect_classify_batch(&mut self, event: TensorBatchRuntime<'_, '_>) -> Result<(), ()> {
        for (index, span) in (0_u32..).zip(event.tensors) {
            self.batch_classifier.process_event(BatchSpanRuntime {
                span,
                index,
                analysis: event.analysis,
            })?;
        }
        Ok(())
    }

    fn guard_batch_requests_valid(&self, event: &TensorBatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.analysis.get().invalid_request.found == 0)
    }

    fn guard_batch_requests_invalid(&self, event: &TensorBatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.analysis.get().invalid_request.found == 1)
    }

    fn guard_batch_resources_supported(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().unsupported_resource.found == 0)
    }

    fn guard_batch_resources_unsupported(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().unsupported_resource.found == 1)
    }

    fn guard_batch_source_open_succeeded(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().file_open_failed.found == 0)
    }

    fn guard_batch_source_open_failed(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().file_open_failed.found == 1)
    }

    fn guard_batch_source_seek_succeeded(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().file_seek_failed.found == 0)
    }

    fn guard_batch_source_seek_failed(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().file_seek_failed.found == 1)
    }

    fn guard_batch_source_read_failed(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().file_read_failed.found == 1)
    }

    fn guard_batch_source_read_short(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.analysis.get().short_read.found == 1)
    }

    fn guard_batch_source_read_succeeded(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        let analysis = event.analysis.get();
        Ok(analysis.file_read_failed.found == 0 && analysis.short_read.found == 0)
    }

    fn guard_batch_done_callback_present(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_batch_done_callback_absent(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_batch_error_callback_present(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_batch_error_callback_absent(
        &self,
        event: &TensorBatchRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn effect_mark_batch_count_invalid(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        event.status.set(BatchStatus {
            result: Err(ReadTensorBatchError::new(Error::InvalidRequest, 0)),
        });
        Ok(())
    }

    fn effect_mark_batch_invalid_request(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().invalid_request.index;
        event.status.set(BatchStatus {
            result: Err(ReadTensorBatchError::new(
                Error::InvalidRequest,
                failed_index,
            )),
        });
        Ok(())
    }

    fn effect_mark_batch_unsupported_resource(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().unsupported_resource.index;
        event.set_error(Error::UnsupportedResource, failed_index);
        Ok(())
    }

    fn effect_mark_batch_file_open_failed(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().file_open_failed.index;
        event.set_error(Error::FileOpenFailed, failed_index);
        Ok(())
    }

    fn effect_mark_batch_file_seek_failed(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().file_seek_failed.index;
        event.set_error(Error::FileSeekFailed, failed_index);
        Ok(())
    }

    fn effect_mark_batch_file_read_failed(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().file_read_failed.index;
        event.set_error(Error::FileReadFailed, failed_index);
        Ok(())
    }

    fn effect_mark_batch_short_read(
        &mut self,
        event: TensorBatchRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let failed_index = event.analysis.get().short_read.index;
        event.set_error(Error::ShortRead, failed_index);
        Ok(())
    }

    fn effect_copy_batch(&mut self, event: TensorBatchRuntime<'_, '_>) -> Result<(), ()> {
        let mut bytes_copied = 0_u64;
        for span in event.tensors {
            let source = span.source.expect("batch-success guard requires source");
            let source_offset = usize::try_from(span.file_offset)
                .expect("batch source-seek success guarantees an addressable offset");
            let byte_size = usize::try_from(span.byte_size)
                .expect("batch resource guard guarantees addressable size");
            span.target
                .copy_from(&source[source_offset..source_offset + byte_size]);
            bytes_copied += span.byte_size;
        }
        event.status.set(BatchStatus {
            result: Ok(ReadTensorBatchDone::new(
                u32::try_from(event.tensors.len()).expect("batch-count guard guarantees u32 count"),
                bytes_copied,
            )),
        });
        Ok(())
    }

    fn effect_publish_batch_done(&mut self, event: TensorBatchRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .status
            .get()
            .result
            .expect("done state requires result");
        event
            .on_done
            .expect("batch-done callback guard requires callback")
            .publish(done);
        Ok(())
    }

    fn effect_publish_batch_error(&mut self, event: TensorBatchRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .status
            .get()
            .result
            .expect_err("error state requires error");
        event
            .on_error
            .expect("batch-error callback guard requires callback")
            .publish(error);
        Ok(())
    }

    fn effect_on_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

impl TensorBatchRuntime<'_, '_> {
    fn set_error(&self, error: Error, failed_index: u32) {
        self.status.set(BatchStatus {
            result: Err(ReadTensorBatchError::new(error, failed_index)),
        });
    }
}

fn resource_supported(span: &TensorRead<'_>) -> bool {
    !span.file_path.is_empty()
        && span.file_path.len() <= MAX_FILE_PATH_BYTES
        && !span.file_path.as_bytes().contains(&0)
        && span.file_index <= MAX_FILE_INDEX
        && span.byte_size <= MAX_READ_BYTES
        && usize::try_from(span.byte_size).is_ok()
        && span.file_offset.checked_add(span.byte_size).is_some()
}

fn target_valid(span: &TensorRead<'_>) -> bool {
    span.target_bytes >= span.byte_size && (span.target.len() as u64) >= span.byte_size
}

fn source_open_succeeded(span: &TensorRead<'_>) -> bool {
    span.source_error != Some(SourceError::FileOpenFailed)
        && (span.source_error.is_some() || span.source.is_some())
}

fn source_seek_succeeded(span: &TensorRead<'_>) -> bool {
    span.source_error != Some(SourceError::FileSeekFailed)
        && (span.source_error.is_some()
            || span
                .source
                .is_some_and(|source| span.file_offset <= source.len() as u64))
}

const fn source_read_failed(span: &TensorRead<'_>) -> bool {
    matches!(
        span.source_error,
        Some(SourceError::FileReadFailed | SourceError::Other)
    )
}

fn source_read_short(span: &TensorRead<'_>) -> bool {
    span.source_error == Some(SourceError::ShortRead)
        || (span.source_error.is_none()
            && span.source.is_some_and(|source| {
                let source_len = source.len() as u64;
                span.file_offset <= source_len && span.byte_size > source_len - span.file_offset
            }))
}

fn source_read_succeeded(span: &TensorRead<'_>) -> bool {
    span.source_error.is_none()
        && span.source.is_some_and(|source| {
            let source_len = source.len() as u64;
            span.file_offset <= source_len && span.byte_size <= source_len - span.file_offset
        })
}
