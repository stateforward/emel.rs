//! Explicit loader strategy orchestration and synchronous child composition.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "stateforward-sml generated state tokens intentionally derive PartialEq"
)]

use core::cell::Cell;

use sml::sml;

use super::event::{
    Callback, Error, LoadTensorBatchDone, LoadTensorBatchError, LoadTensorDone, LoadTensorError,
    MAX_BATCH_TENSORS, ReadActor, StagedReadActor, StrategyError, StrategyKind, StrategyPolicy,
    TensorLoadSpan,
};
use crate::{read, staged_read};

pub(super) struct Context<R, S> {
    pub(super) reader: R,
    pub(super) stager: S,
}

#[derive(Clone, Copy)]
pub(super) struct SingleStatus {
    pub(super) result: Result<LoadTensorDone, LoadTensorError>,
}

impl SingleStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(LoadTensorError::new(
                Error::InternalError,
                StrategyError::None,
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct BatchStatus {
    pub(super) result: Result<LoadTensorBatchDone, LoadTensorBatchError>,
}

impl BatchStatus {
    pub(super) const fn new() -> Self {
        Self {
            result: Err(LoadTensorBatchError::new(
                Error::InternalError,
                StrategyError::None,
                0,
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SingleRuntime<'dispatch, 'data> {
    pub(super) tensor: TensorLoadSpan<'data>,
    pub(super) policy: StrategyPolicy,
    pub(super) on_done: Option<Callback<'data, LoadTensorDone>>,
    pub(super) on_error: Option<Callback<'data, LoadTensorError>>,
    pub(super) status: &'dispatch Cell<SingleStatus>,
    pub(super) read_result:
        &'dispatch Cell<Result<read::event::ReadTensorDone, read::event::Error>>,
    pub(super) staged_result:
        &'dispatch Cell<Result<staged_read::event::StageWindowDone, staged_read::event::Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct BatchRuntime<'dispatch, 'data> {
    pub(super) tensors: &'data [TensorLoadSpan<'data>],
    pub(super) policy: StrategyPolicy,
    pub(super) on_done: Option<Callback<'data, LoadTensorBatchDone>>,
    pub(super) on_error: Option<Callback<'data, LoadTensorBatchError>>,
    pub(super) status: &'dispatch Cell<BatchStatus>,
    pub(super) read_result: &'dispatch Cell<
        Result<read::event::ReadTensorBatchDone, read::event::ReadTensorBatchError>,
    >,
    pub(super) staged_result: &'dispatch Cell<
        Result<staged_read::event::StageWindowBatchDone, staged_read::event::StageWindowBatchError>,
    >,
}

sml! {
    IoLoader {
        // Single request validation and explicit strategy selection.
        "state_single_strategy_decision"_s <= *"state_ready"_s + Single(SingleRuntime<'dispatch, 'data>) [guard_single_span_valid],
        "state_single_error_decision"_s <= "state_ready"_s + Single(SingleRuntime<'dispatch, 'data>) [guard_single_span_invalid] / effect_mark_single_invalid,
        "state_single_read_outcome_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_read_available] / effect_dispatch_single_read,
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_read_unavailable] / effect_mark_single_unsupported,
        "state_single_staged_source_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_available],
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_unavailable] / effect_mark_single_unsupported,
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_strategy_none] / effect_mark_single_unsupported,
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_strategy_mapped] / effect_mark_single_unsupported,
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_strategy_external] / effect_mark_single_unsupported,
        "state_single_error_decision"_s <= "state_single_strategy_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_strategy_unknown] / effect_mark_single_unsupported,
        "state_single_staged_chunk_decision"_s <= "state_single_staged_source_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_source_valid],
        "state_single_error_decision"_s <= "state_single_staged_source_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_source_invalid] / effect_mark_single_invalid,
        "state_single_staged_outcome_decision"_s <= "state_single_staged_chunk_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_chunk_smaller] / effect_dispatch_single_staged_requested,
        "state_single_staged_outcome_decision"_s <= "state_single_staged_chunk_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_chunk_not_smaller] / effect_dispatch_single_staged_logical,

        // Single child outcomes and typed publication.
        "state_single_done_decision"_s <= "state_single_read_outcome_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_read_succeeded] / effect_record_single_read_done,
        "state_single_error_decision"_s <= "state_single_read_outcome_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_read_failed] / effect_record_single_read_error,
        "state_single_done_decision"_s <= "state_single_staged_outcome_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_succeeded] / effect_record_single_staged_done,
        "state_single_error_decision"_s <= "state_single_staged_outcome_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_staged_failed] / effect_record_single_staged_error,
        "state_single_done_published"_s <= "state_single_done_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_done_callback_present] / effect_publish_single_done,
        "state_ready"_s <= "state_single_done_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_done_callback_absent],
        "state_ready"_s <= "state_single_done_published"_s + completion<Single>(SingleRuntime<'dispatch, 'data>),
        "state_single_error_published"_s <= "state_single_error_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_error_callback_present] / effect_publish_single_error,
        "state_ready"_s <= "state_single_error_decision"_s + completion<Single>(SingleRuntime<'dispatch, 'data>) [guard_single_error_callback_absent],
        "state_ready"_s <= "state_single_error_published"_s + completion<Single>(SingleRuntime<'dispatch, 'data>),

        // Batch validation and exactly one selected child batch dispatch.
        "state_batch_strategy_decision"_s <= "state_ready"_s + Batch(BatchRuntime<'dispatch, 'data>) [guard_batch_spans_valid],
        "state_batch_error_decision"_s <= "state_ready"_s + Batch(BatchRuntime<'dispatch, 'data>) [guard_batch_spans_invalid] / effect_mark_batch_invalid,
        "state_batch_read_outcome_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_read_available] / effect_dispatch_batch_read,
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_read_unavailable] / effect_mark_batch_unsupported,
        "state_batch_staged_source_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_available],
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_unavailable] / effect_mark_batch_unsupported,
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_strategy_none] / effect_mark_batch_unsupported,
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_strategy_mapped] / effect_mark_batch_unsupported,
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_strategy_external] / effect_mark_batch_unsupported,
        "state_batch_error_decision"_s <= "state_batch_strategy_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_strategy_unknown] / effect_mark_batch_unsupported,
        "state_batch_staged_outcome_decision"_s <= "state_batch_staged_source_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_sources_valid] / effect_dispatch_batch_staged,
        "state_batch_error_decision"_s <= "state_batch_staged_source_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_sources_invalid] / effect_mark_batch_invalid,

        // Batch child outcomes and typed publication.
        "state_batch_done_decision"_s <= "state_batch_read_outcome_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_read_succeeded] / effect_record_batch_read_done,
        "state_batch_error_decision"_s <= "state_batch_read_outcome_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_read_failed] / effect_record_batch_read_error,
        "state_batch_done_decision"_s <= "state_batch_staged_outcome_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_succeeded] / effect_record_batch_staged_done,
        "state_batch_error_decision"_s <= "state_batch_staged_outcome_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_staged_failed] / effect_record_batch_staged_error,
        "state_batch_done_published"_s <= "state_batch_done_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_done_callback_present] / effect_publish_batch_done,
        "state_ready"_s <= "state_batch_done_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_done_callback_absent],
        "state_ready"_s <= "state_batch_done_published"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>),
        "state_batch_error_published"_s <= "state_batch_error_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_error_callback_present] / effect_publish_batch_error,
        "state_ready"_s <= "state_batch_error_decision"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>) [guard_batch_error_callback_absent],
        "state_ready"_s <= "state_batch_error_published"_s + completion<Batch>(BatchRuntime<'dispatch, 'data>),

        // Every externally reachable state fails closed on unexpected input.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_strategy_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_read_outcome_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_staged_source_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_staged_chunk_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_staged_outcome_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_done_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_done_published"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_error_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_single_error_published"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_strategy_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_read_outcome_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_staged_source_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_staged_outcome_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_done_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_done_published"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_error_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_batch_error_published"_s + unexpected_event<_> / effect_unexpected,
    }
}

impl<R: ReadActor, S: StagedReadActor> IoLoaderStateMachineContext for Context<R, S> {
    fn guard_single_span_valid(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(span_contract_valid(event.tensor))
    }

    fn guard_single_span_invalid(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!span_contract_valid(event.tensor))
    }

    fn guard_single_read_available(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::ReadCopy) && R::AVAILABLE)
    }

    fn guard_single_read_unavailable(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::ReadCopy) && !R::AVAILABLE)
    }

    fn guard_single_staged_available(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::StagedRead) && S::AVAILABLE)
    }

    fn guard_single_staged_unavailable(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::StagedRead) && !S::AVAILABLE)
    }

    fn guard_single_strategy_none(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::None))
    }

    fn guard_single_strategy_mapped(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::MappedFile))
    }

    fn guard_single_strategy_external(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.policy.strategy(),
            StrategyKind::ExternalBuffer
        ))
    }

    fn guard_single_strategy_unknown(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::Unknown(_)))
    }

    fn guard_single_staged_source_valid(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(source_span_valid(event.tensor))
    }

    fn guard_single_staged_source_invalid(
        &self,
        event: &SingleRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(!source_span_valid(event.tensor))
    }

    fn guard_single_staged_chunk_smaller(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.policy.staged_chunk_bytes() < event.tensor.byte_size)
    }

    fn guard_single_staged_chunk_not_smaller(
        &self,
        event: &SingleRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.policy.staged_chunk_bytes() >= event.tensor.byte_size)
    }

    fn guard_single_read_succeeded(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.read_result.get().is_ok())
    }

    fn guard_single_read_failed(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.read_result.get().is_err())
    }

    fn guard_single_staged_succeeded(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.staged_result.get().is_ok())
    }

    fn guard_single_staged_failed(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.staged_result.get().is_err())
    }

    fn guard_single_done_callback_present(
        &self,
        event: &SingleRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_single_done_callback_absent(&self, event: &SingleRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_single_error_callback_present(
        &self,
        event: &SingleRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_single_error_callback_absent(
        &self,
        event: &SingleRuntime<'_, '_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn effect_mark_single_invalid(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        set_single_error(event, Error::InvalidRequest, StrategyError::None);
        Ok(())
    }

    fn effect_mark_single_unsupported(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        set_single_error(event, Error::UnsupportedStrategy, StrategyError::None);
        Ok(())
    }

    fn effect_dispatch_single_read(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        event.read_result.set(
            self.reader
                .process_read(read::event::ReadSpan::new(event.tensor)),
        );
        Ok(())
    }

    fn effect_dispatch_single_staged_requested(
        &mut self,
        event: SingleRuntime<'_, '_>,
    ) -> Result<(), ()> {
        dispatch_single_staged(&mut self.stager, event, event.policy.staged_chunk_bytes());
        Ok(())
    }

    fn effect_dispatch_single_staged_logical(
        &mut self,
        event: SingleRuntime<'_, '_>,
    ) -> Result<(), ()> {
        dispatch_single_staged(&mut self.stager, event, event.tensor.byte_size);
        Ok(())
    }

    fn effect_record_single_read_done(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .read_result
            .get()
            .expect("read-success guard selected a successful result");
        set_single_done(event, done.bytes_copied());
        Ok(())
    }

    fn effect_record_single_read_error(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .read_result
            .get()
            .expect_err("read-error guard selected a failed result");
        set_single_error(event, Error::Unavailable, StrategyError::Read(error));
        Ok(())
    }

    fn effect_record_single_staged_done(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .staged_result
            .get()
            .expect("staged-success guard selected a successful result");
        set_single_done(event, done.bytes_committed());
        Ok(())
    }

    fn effect_record_single_staged_error(
        &mut self,
        event: SingleRuntime<'_, '_>,
    ) -> Result<(), ()> {
        let error = event
            .staged_result
            .get()
            .expect_err("staged-error guard selected a failed result");
        set_single_error(event, Error::Unavailable, StrategyError::StagedRead(error));
        Ok(())
    }

    fn effect_publish_single_done(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        event
            .on_done
            .expect("done-callback guard selected a callback")
            .publish(
                event
                    .status
                    .get()
                    .result
                    .expect("done state selected a successful loader result"),
            );
        Ok(())
    }

    fn effect_publish_single_error(&mut self, event: SingleRuntime<'_, '_>) -> Result<(), ()> {
        event
            .on_error
            .expect("error-callback guard selected a callback")
            .publish(
                event
                    .status
                    .get()
                    .result
                    .expect_err("error state selected a failed loader result"),
            );
        Ok(())
    }

    fn guard_batch_spans_valid(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(batch_contract_valid(event.tensors))
    }

    fn guard_batch_spans_invalid(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!batch_contract_valid(event.tensors))
    }

    fn guard_batch_read_available(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::ReadCopy) && R::AVAILABLE)
    }

    fn guard_batch_read_unavailable(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::ReadCopy) && !R::AVAILABLE)
    }

    fn guard_batch_staged_available(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::StagedRead) && S::AVAILABLE)
    }

    fn guard_batch_staged_unavailable(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::StagedRead) && !S::AVAILABLE)
    }

    fn guard_batch_strategy_none(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::None))
    }

    fn guard_batch_strategy_mapped(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::MappedFile))
    }

    fn guard_batch_strategy_external(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            event.policy.strategy(),
            StrategyKind::ExternalBuffer
        ))
    }

    fn guard_batch_strategy_unknown(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.policy.strategy(), StrategyKind::Unknown(_)))
    }

    fn guard_batch_staged_sources_valid(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(batch_sources_valid(event.tensors))
    }

    fn guard_batch_staged_sources_invalid(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!batch_sources_valid(event.tensors))
    }

    fn guard_batch_read_succeeded(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.read_result.get().is_ok())
    }

    fn guard_batch_read_failed(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.read_result.get().is_err())
    }

    fn guard_batch_staged_succeeded(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.staged_result.get().is_ok())
    }

    fn guard_batch_staged_failed(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.staged_result.get().is_err())
    }

    fn guard_batch_done_callback_present(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_batch_done_callback_absent(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_batch_error_callback_present(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_batch_error_callback_absent(&self, event: &BatchRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn effect_mark_batch_invalid(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        set_batch_error(event, Error::InvalidRequest, StrategyError::None, 0);
        Ok(())
    }

    fn effect_mark_batch_unsupported(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        set_batch_error(event, Error::UnsupportedStrategy, StrategyError::None, 0);
        Ok(())
    }

    fn effect_dispatch_batch_read(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        event.read_result.set(
            self.reader
                .process_read_batch(read::event::ReadTensorBatch::new(event.tensors)),
        );
        Ok(())
    }

    fn effect_dispatch_batch_staged(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        event
            .staged_result
            .set(
                self.stager
                    .process_staged_read_batch(staged_read::event::StageTensorBatch::new(
                        event.tensors,
                        event.policy.staged_chunk_bytes(),
                    )),
            );
        Ok(())
    }

    fn effect_record_batch_read_done(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .read_result
            .get()
            .expect("batch read-success guard selected a successful result");
        set_batch_done(event, done.done_count(), done.bytes_copied());
        Ok(())
    }

    fn effect_record_batch_read_error(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .read_result
            .get()
            .expect_err("batch read-error guard selected a failed result");
        set_batch_error(
            event,
            Error::Unavailable,
            StrategyError::Read(error.error()),
            error.failed_index(),
        );
        Ok(())
    }

    fn effect_record_batch_staged_done(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .staged_result
            .get()
            .expect("batch staged-success guard selected a successful result");
        set_batch_done(event, done.done_count(), done.bytes_committed());
        Ok(())
    }

    fn effect_record_batch_staged_error(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .staged_result
            .get()
            .expect_err("batch staged-error guard selected a failed result");
        set_batch_error(
            event,
            Error::Unavailable,
            StrategyError::StagedRead(error.error()),
            error.failed_index(),
        );
        Ok(())
    }

    fn effect_publish_batch_done(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        event
            .on_done
            .expect("batch done-callback guard selected a callback")
            .publish(
                event
                    .status
                    .get()
                    .result
                    .expect("batch done state selected a successful result"),
            );
        Ok(())
    }

    fn effect_publish_batch_error(&mut self, event: BatchRuntime<'_, '_>) -> Result<(), ()> {
        event
            .on_error
            .expect("batch error-callback guard selected a callback")
            .publish(
                event
                    .status
                    .get()
                    .result
                    .expect_err("batch error state selected a failed result"),
            );
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        std::process::abort()
    }
}

fn span_contract_valid(span: TensorLoadSpan<'_>) -> bool {
    (span.byte_size > 0)
        & (span.target_bytes >= span.byte_size)
        & u64::try_from(span.target.len()).is_ok_and(|len| len >= span.byte_size)
}

fn source_span_valid(span: TensorLoadSpan<'_>) -> bool {
    span.source.is_some_and(|source| {
        u64::try_from(source.len()).is_ok_and(|len| {
            span.file_offset
                .checked_add(span.byte_size)
                .is_some_and(|end| end <= len)
        })
    })
}

fn batch_contract_valid(tensors: &[TensorLoadSpan<'_>]) -> bool {
    let global_valid = !tensors.is_empty() && tensors.len() <= MAX_BATCH_TENSORS;
    let mut valid = true;
    let mut index = 0;
    while index < tensors.len().min(MAX_BATCH_TENSORS) {
        valid &= span_contract_valid(tensors[index]);
        index += 1;
    }
    global_valid & valid
}

fn batch_sources_valid(tensors: &[TensorLoadSpan<'_>]) -> bool {
    let mut valid = true;
    let mut index = 0;
    while index < tensors.len().min(MAX_BATCH_TENSORS) {
        valid &= source_span_valid(tensors[index]);
        index += 1;
    }
    valid
}

fn dispatch_single_staged<S: StagedReadActor>(
    stager: &mut S,
    event: SingleRuntime<'_, '_>,
    stage_chunk_bytes: u64,
) {
    let start = usize::try_from(event.tensor.file_offset)
        .expect("staged-source guard selected a representable offset");
    let end = usize::try_from(event.tensor.file_offset + event.tensor.byte_size)
        .expect("staged-source guard selected a representable end");
    let source = &event
        .tensor
        .source
        .expect("staged-source guard selected source bytes")[start..end];
    let span = TensorLoadSpan::staged(0, event.tensor.byte_size, Some(source), event.tensor.target);
    event.staged_result.set(
        stager.process_staged_read(staged_read::event::StageTensor::new(
            span,
            stage_chunk_bytes,
        )),
    );
}

fn set_single_done(event: SingleRuntime<'_, '_>, bytes_loaded: u64) {
    event.status.set(SingleStatus {
        result: Ok(LoadTensorDone::new(event.policy.strategy(), bytes_loaded)),
    });
}

fn set_single_error(event: SingleRuntime<'_, '_>, error: Error, strategy_error: StrategyError) {
    event.status.set(SingleStatus {
        result: Err(LoadTensorError::new(error, strategy_error)),
    });
}

fn set_batch_done(event: BatchRuntime<'_, '_>, done_count: u32, bytes_loaded: u64) {
    event.status.set(BatchStatus {
        result: Ok(LoadTensorBatchDone::new(
            event.policy.strategy(),
            done_count,
            bytes_loaded,
        )),
    });
}

fn set_batch_error(
    event: BatchRuntime<'_, '_>,
    error: Error,
    strategy_error: StrategyError,
    failed_index: u32,
) {
    event.status.set(BatchStatus {
        result: Err(LoadTensorBatchError::new(
            error,
            strategy_error,
            failed_index,
        )),
    });
}
