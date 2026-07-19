//! Source-exact Sortformer metadata decoding through an explicit SML phase chain.

#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use core::cell::{Cell, RefCell};

use emel_gguf::Loader;
use emel_gguf::event::{QueryError, ReadUnsigned, WithString};

use super::actor::UnexpectedRuntime;
use super::event::Parameters;
use super::sm::{
    SortformerHparamsEvents, SortformerHparamsStateMachine, SortformerHparamsStateMachineContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    Architecture,
    SourceFormat,
    TensorNameScheme,
    Outtype,
    OriginalTensorCount,
    TensorCount,
    SkippedTensorCount,
    SampleRate,
    SpeakerCount,
    ChunkLen,
    ChunkRightContext,
    FifoLen,
    SpeakerCacheUpdatePeriod,
    SpeakerCacheLen,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    Missing,
    WrongKind,
    Range,
    Contract,
    Query,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    pub field: Field,
    pub kind: ErrorKind,
}

const fn internal_error(field: Field) -> Error {
    Error {
        field,
        kind: ErrorKind::Internal,
    }
}

#[derive(Clone, Copy)]
pub(super) struct HparamLoadRuntime<'a> {
    loader: &'a RefCell<&'a mut Loader>,
    parameters: &'a Cell<Parameters>,
    string: &'a Cell<Result<Option<bool>, QueryError>>,
    unsigned: &'a Cell<Result<Option<u64>, QueryError>>,
    field: &'a Cell<Field>,
    result: &'a Cell<Result<Parameters, Error>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct HparamContext;

/// Decodes and validates the pinned Sortformer metadata contract.
///
/// # Errors
///
/// Returns the exact field and failure class for invalid metadata.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let loader = RefCell::new(loader);
    let parameters = Cell::new(Parameters::default());
    let string = Cell::new(Err(QueryError::Internal));
    let unsigned = Cell::new(Err(QueryError::Internal));
    let field = Cell::new(Field::Architecture);
    let result = Cell::new(Err(internal_error(Field::Architecture)));
    let mut machine = SortformerHparamsStateMachine::new(HparamContext);
    machine
        .process_event(SortformerHparamsEvents::Load(HparamLoadRuntime {
            loader: &loader,
            parameters: &parameters,
            string: &string,
            unsigned: &unsigned,
            field: &field,
            result: &result,
        }))
        .map_err(|_| internal_error(field.get()))?;
    result.get()
}

fn query_string(event: &HparamLoadRuntime<'_>, key: &[u8], expected: &[u8], field: Field) {
    event.field.set(field);
    event.unsigned.set(Ok(None));
    event.string.set(
        event
            .loader
            .borrow_mut()
            .process_event(WithString::new(key, |value: &[u8]| value == expected)),
    );
}

fn query_unsigned(event: &HparamLoadRuntime<'_>, key: &[u8], field: Field) {
    event.field.set(field);
    event.string.set(Ok(None));
    event.unsigned.set(
        event
            .loader
            .borrow_mut()
            .process_event(ReadUnsigned::new(key)),
    );
}

fn unsigned_i32(event: &HparamLoadRuntime<'_>) -> i32 {
    i32::try_from(
        event
            .unsigned
            .get()
            .expect("selected unsigned success")
            .expect("selected unsigned presence"),
    )
    .expect("selected i32 range")
}

fn set_error(event: &HparamLoadRuntime<'_>, kind: ErrorKind) {
    event.result.set(Err(Error {
        field: event.field.get(),
        kind,
    }));
}

impl SortformerHparamsStateMachineContext for HparamContext {
    fn guard_architecture_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(true))))
    }
    fn guard_architecture_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(false))))
    }
    fn guard_architecture_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(None)))
    }
    fn guard_string_present(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(true))))
    }
    fn guard_string_contract(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(false))))
    }
    fn guard_string_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(None)))
    }
    fn guard_unsigned_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Ok(None)))
    }
    fn guard_unsigned_i32(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_ok()))
    }
    fn guard_unsigned_positive(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_ok_and(|value| value > 0)),
        )
    }
    fn guard_unsigned_nonpositive(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_ok_and(|value| value <= 0)),
        )
    }
    fn guard_query_wrong_kind(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::TypeMismatch))
            || matches!(event.unsigned.get(), Err(QueryError::TypeMismatch)))
    }
    fn guard_query_range(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::Range))
            || matches!(event.unsigned.get(), Err(QueryError::Range)))
    }
    fn guard_query_range_or_value_range(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Err(QueryError::Range))
            || matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_err()))
    }
    fn guard_query_internal(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::Internal))
            || matches!(event.unsigned.get(), Err(QueryError::Internal)))
    }
    fn guard_query_other(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.string.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal))
                || matches!(event.unsigned.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal)),
        )
    }
    fn effect_query_architecture(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"general.architecture",
            super::ARCHITECTURE_NAME,
            Field::Architecture,
        );
        Ok(())
    }
    fn effect_query_source(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"sortformer.source.format",
            super::SOURCE_FORMAT,
            Field::SourceFormat,
        );
        Ok(())
    }
    fn effect_query_scheme(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"sortformer.tensor_name_scheme",
            super::TENSOR_NAME_SCHEME,
            Field::TensorNameScheme,
        );
        Ok(())
    }
    fn effect_query_outtype(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"sortformer.outtype",
            super::OUTTYPE,
            Field::Outtype,
        );
        Ok(())
    }
    fn effect_query_original(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.original_tensor_count",
            Field::OriginalTensorCount,
        );
        Ok(())
    }
    fn effect_assign_original_query_tensor(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.original_tensor_count = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(&event, b"sortformer.tensor_count", Field::TensorCount);
        Ok(())
    }
    fn effect_assign_tensor_query_skipped(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.tensor_count = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.skipped_tensor_count",
            Field::SkippedTensorCount,
        );
        Ok(())
    }
    fn effect_assign_skipped_query_sample_primary(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.skipped_tensor_count = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.preprocessor.sample_rate",
            Field::SampleRate,
        );
        Ok(())
    }
    fn effect_default_skipped_query_sample_primary(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.preprocessor.sample_rate",
            Field::SampleRate,
        );
        Ok(())
    }
    fn effect_query_sample_fallback(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(&event, b"sortformer.config.sample_rate", Field::SampleRate);
        Ok(())
    }
    fn effect_assign_sample_query_speaker_primary(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.sample_rate = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.num_spks",
            Field::SpeakerCount,
        );
        Ok(())
    }
    fn effect_query_speaker_primary(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.num_spks",
            Field::SpeakerCount,
        );
        Ok(())
    }
    fn effect_query_speaker_secondary(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.num_speakers",
            Field::SpeakerCount,
        );
        Ok(())
    }
    fn effect_query_speaker_fallback(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(&event, b"sortformer.config.num_spks", Field::SpeakerCount);
        Ok(())
    }
    fn effect_assign_speaker_query_chunk(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.speaker_count = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.chunk_len",
            Field::ChunkLen,
        );
        Ok(())
    }
    fn effect_query_chunk(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.chunk_len",
            Field::ChunkLen,
        );
        Ok(())
    }
    fn effect_assign_chunk_query_right(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.chunk_len = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.chunk_right_context",
            Field::ChunkRightContext,
        );
        Ok(())
    }
    fn effect_query_right(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.chunk_right_context",
            Field::ChunkRightContext,
        );
        Ok(())
    }
    fn effect_assign_right_query_fifo(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.chunk_right_context = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.fifo_len",
            Field::FifoLen,
        );
        Ok(())
    }
    fn effect_query_fifo(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.fifo_len",
            Field::FifoLen,
        );
        Ok(())
    }
    fn effect_assign_fifo_query_update(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.fifo_len = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.spkcache_update_period",
            Field::SpeakerCacheUpdatePeriod,
        );
        Ok(())
    }
    fn effect_query_update(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.spkcache_update_period",
            Field::SpeakerCacheUpdatePeriod,
        );
        Ok(())
    }
    fn effect_assign_update_query_len(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.spkcache_update_period = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.spkcache_len",
            Field::SpeakerCacheLen,
        );
        Ok(())
    }
    fn effect_query_len(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"sortformer.config.sortformer_modules.spkcache_len",
            Field::SpeakerCacheLen,
        );
        Ok(())
    }
    fn effect_assign_len_validate_sample(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.spkcache_len = unsigned_i32(&event);
        event.parameters.set(parameters);
        event.field.set(Field::SampleRate);
        Ok(())
    }
    fn effect_validate_sample(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::SampleRate);
        Ok(())
    }
    fn effect_validate_speaker(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::SpeakerCount);
        Ok(())
    }
    fn effect_validate_chunk(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::ChunkLen);
        Ok(())
    }
    fn effect_validate_right(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::ChunkRightContext);
        Ok(())
    }
    fn effect_validate_fifo(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::FifoLen);
        Ok(())
    }
    fn effect_validate_update(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::SpeakerCacheUpdatePeriod);
        Ok(())
    }
    fn effect_validate_len(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.field.set(Field::SpeakerCacheLen);
        Ok(())
    }

    fn guard_sample_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().sample_rate == super::SAMPLE_RATE)
    }
    fn guard_sample_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_sample_valid(event)?)
    }
    fn guard_speaker_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().speaker_count == super::SPEAKER_COUNT)
    }
    fn guard_speaker_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_speaker_valid(event)?)
    }
    fn guard_chunk_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().chunk_len == super::CHUNK_LEN)
    }
    fn guard_chunk_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_chunk_valid(event)?)
    }
    fn guard_right_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().chunk_right_context == super::CHUNK_RIGHT_CONTEXT)
    }
    fn guard_right_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_right_valid(event)?)
    }
    fn guard_fifo_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().fifo_len == super::FIFO_LEN)
    }
    fn guard_fifo_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_fifo_valid(event)?)
    }
    fn guard_update_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().spkcache_update_period == super::SPKCACHE_UPDATE_PERIOD)
    }
    fn guard_update_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_update_valid(event)?)
    }
    fn guard_len_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.parameters.get().spkcache_len == super::SPKCACHE_LEN)
    }
    fn guard_len_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.guard_len_valid(event)?)
    }

    fn effect_missing(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Missing);
        Ok(())
    }
    fn effect_wrong_kind(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::WrongKind);
        Ok(())
    }
    fn effect_range(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Range);
        Ok(())
    }
    fn effect_contract(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Contract);
        Ok(())
    }
    fn effect_query(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Query);
        Ok(())
    }
    fn effect_internal(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Internal);
        Ok(())
    }
    fn effect_success(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.parameters.get()));
        Ok(())
    }

    fn guard_never(&self, _: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::sm::SortformerHparamsStates;
    use super::*;

    #[test]
    fn unexpected_internal_event_fails_closed_without_mutating_hparam_state() {
        let mut machine = SortformerHparamsStateMachine::new(HparamContext);
        assert!(machine.is(&SortformerHparamsStates::StateIdle));

        let error = machine
            .process_event(SortformerHparamsEvents::UnexpectedRuntime(
                UnexpectedRuntime,
            ))
            .map_err(|_| internal_error(Field::Architecture))
            .err()
            .expect("unexpected hparam event must fail closed");

        assert_eq!(
            error,
            Error {
                field: Field::Architecture,
                kind: ErrorKind::Internal,
            }
        );
        assert!(machine.is(&SortformerHparamsStates::StateIdle));
    }
}
