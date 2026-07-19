//! Source-exact `OmniEmbed` metadata decoding through an explicit SML phase chain.

#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use core::cell::{Cell, RefCell};

use emel_gguf::Loader;
use emel_gguf::event::{
    QueryError, ReadIntegerArrayCount, ReadUnsigned, VisitSignedArray, WithString,
};

use super::actor::UnexpectedRuntime;
use super::event::{EncoderName, Parameters};
use super::sm::{
    OmniEmbedHparamsEvents, OmniEmbedHparamsStateMachine, OmniEmbedHparamsStateMachineContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    Architecture,
    EmbeddingLength,
    ImageEncoderName,
    ImageEncoderLength,
    AudioEncoderName,
    AudioEncoderLength,
    MatryoshkaDimensions,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    Missing,
    WrongKind,
    Range,
    Capacity,
    Contract,
    Query,
    Internal,
}

const MAX_METADATA_BLOB_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StringObservation {
    canonical: bool,
    empty: bool,
    length: usize,
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
    string: &'a Cell<Result<Option<StringObservation>, QueryError>>,
    metadata_bytes_used: &'a Cell<usize>,
    unsigned: &'a Cell<Result<Option<u64>, QueryError>>,
    array_count: &'a Cell<Result<Option<u64>, QueryError>>,
    array_visit: &'a Cell<Result<Option<u64>, QueryError>>,
    array_values: &'a RefCell<[i64; super::MAX_MATRYOSHKA_DIMENSIONS]>,
    field: &'a Cell<Field>,
    result: &'a Cell<Result<Parameters, Error>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct HparamContext;

/// Decodes all metadata consumed by the pinned `OmniEmbed` model contract.
///
/// Missing scalar values and encoder names preserve the source zero/empty defaults.
/// A missing or empty matryoshka array is accepted by loading and rejected only by
/// execution-contract validation, matching the source lifecycle boundary.
///
/// # Errors
///
/// Returns the exact field and failure class for malformed metadata.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let loader = RefCell::new(loader);
    let parameters = Cell::new(Parameters::default());
    let string = Cell::new(Err(QueryError::Internal));
    let metadata_bytes_used = Cell::new(0);
    let unsigned = Cell::new(Err(QueryError::Internal));
    let array_count = Cell::new(Err(QueryError::Internal));
    let array_visit = Cell::new(Err(QueryError::Internal));
    let array_values = RefCell::new([0; super::MAX_MATRYOSHKA_DIMENSIONS]);
    let field = Cell::new(Field::Architecture);
    let result = Cell::new(Err(internal_error(Field::Architecture)));
    let mut machine = OmniEmbedHparamsStateMachine::new(HparamContext);
    machine
        .process_event(OmniEmbedHparamsEvents::Load(HparamLoadRuntime {
            loader: &loader,
            parameters: &parameters,
            string: &string,
            metadata_bytes_used: &metadata_bytes_used,
            unsigned: &unsigned,
            array_count: &array_count,
            array_visit: &array_visit,
            array_values: &array_values,
            field: &field,
            result: &result,
        }))
        .map_err(|_| internal_error(field.get()))?;
    result.get()
}

fn query_string(event: &HparamLoadRuntime<'_>, key: &[u8], expected: &[u8], field: Field) {
    event.field.set(field);
    event.unsigned.set(Ok(None));
    event.array_count.set(Ok(None));
    event.array_visit.set(Ok(None));
    event
        .string
        .set(
            event
                .loader
                .borrow_mut()
                .process_event(WithString::new(key, |value: &[u8]| StringObservation {
                    canonical: value == expected,
                    empty: value.is_empty(),
                    length: value.len(),
                })),
        );
}

fn query_unsigned(event: &HparamLoadRuntime<'_>, key: &[u8], field: Field) {
    event.field.set(field);
    event.string.set(Ok(None));
    event.array_count.set(Ok(None));
    event.array_visit.set(Ok(None));
    event.unsigned.set(
        event
            .loader
            .borrow_mut()
            .process_event(ReadUnsigned::new(key)),
    );
}

fn query_array_count(event: &HparamLoadRuntime<'_>) {
    event.field.set(Field::MatryoshkaDimensions);
    event.string.set(Ok(None));
    event.unsigned.set(Ok(None));
    event.array_visit.set(Ok(None));
    event.array_count.set(
        event
            .loader
            .borrow_mut()
            .process_event(ReadIntegerArrayCount::new(b"omniembed.matryoshka_dims")),
    );
}

fn string_fits(event: &HparamLoadRuntime<'_>, observation: StringObservation) -> bool {
    event
        .metadata_bytes_used
        .get()
        .checked_add(observation.length)
        .is_some_and(|used| used <= MAX_METADATA_BLOB_BYTES)
}

fn consume_string(event: &HparamLoadRuntime<'_>) {
    let observation = event
        .string
        .get()
        .expect("selected string success")
        .expect("selected string presence");
    let used = event
        .metadata_bytes_used
        .get()
        .checked_add(observation.length)
        .expect("string-fit guard proves arithmetic");
    debug_assert!(used <= MAX_METADATA_BLOB_BYTES);
    event.metadata_bytes_used.set(used);
}

fn set_error(event: &HparamLoadRuntime<'_>, kind: ErrorKind) {
    event.result.set(Err(Error {
        field: event.field.get(),
        kind,
    }));
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

impl OmniEmbedHparamsStateMachineContext for HparamContext {
    fn guard_architecture_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(value)) if value.canonical))
    }
    fn guard_architecture_invalid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(value)) if !value.canonical))
    }
    fn guard_string_canonical_fits(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.string.get(), Ok(Some(value)) if value.canonical && !value.empty && string_fits(event, value)),
        )
    }
    fn guard_string_other_fits(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.string.get(), Ok(Some(value)) if !value.canonical && !value.empty && string_fits(event, value)),
        )
    }
    fn guard_string_empty(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(Some(value)) if value.empty))
    }
    fn guard_string_capacity(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.string.get(), Ok(Some(value)) if !value.empty && !string_fits(event, value)),
        )
    }
    fn guard_string_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Ok(None)))
    }
    fn guard_unsigned_value(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_ok()))
    }
    fn guard_unsigned_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Ok(None)))
    }
    fn guard_unsigned_range(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.unsigned.get(), Ok(Some(value)) if i32::try_from(value).is_err()))
    }
    fn guard_array_count_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.array_count.get(), Ok(Some(count)) if usize::try_from(count).is_ok_and(|count| count <= super::MAX_MATRYOSHKA_DIMENSIONS)),
        )
    }
    fn guard_array_missing(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.array_count.get(), Ok(None)))
    }
    fn guard_array_capacity(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.array_count.get(), Ok(Some(count)) if usize::try_from(count).map_or(true, |count| count > super::MAX_MATRYOSHKA_DIMENSIONS)),
        )
    }
    fn guard_array_values_valid(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        let count = usize::try_from(
            event
                .array_count
                .get()
                .expect("count-valid state has successful count")
                .expect("count-valid state has present array"),
        )
        .expect("count-valid state fits usize");
        Ok(event.array_visit.get() == Ok(Some(count as u64))
            && event.array_values.borrow()[..count]
                .iter()
                .all(|value| i32::try_from(*value).is_ok_and(|value| value > 0)))
    }
    fn guard_array_values_range(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.array_visit.get(), Ok(Some(_)))
            && !self.guard_array_values_valid(event)?)
    }
    fn guard_query_wrong_kind(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::TypeMismatch))
            || matches!(event.unsigned.get(), Err(QueryError::TypeMismatch))
            || matches!(event.array_count.get(), Err(QueryError::TypeMismatch))
            || matches!(event.array_visit.get(), Err(QueryError::TypeMismatch)))
    }
    fn guard_query_range(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::Range))
            || matches!(event.unsigned.get(), Err(QueryError::Range))
            || matches!(event.array_count.get(), Err(QueryError::Range))
            || matches!(event.array_visit.get(), Err(QueryError::Range)))
    }
    fn guard_query_internal(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.string.get(), Err(QueryError::Internal))
            || matches!(event.unsigned.get(), Err(QueryError::Internal))
            || matches!(event.array_count.get(), Err(QueryError::Internal))
            || matches!(event.array_visit.get(), Err(QueryError::Internal)))
    }
    fn guard_query_other(&self, event: &HparamLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.string.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal))
                || matches!(event.unsigned.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal))
                || matches!(event.array_count.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal))
                || matches!(event.array_visit.get(), Err(error) if !matches!(error, QueryError::TypeMismatch | QueryError::Range | QueryError::Internal)),
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
    fn effect_query_embedding(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(&event, b"omniembed.embed_dim", Field::EmbeddingLength);
        Ok(())
    }
    fn effect_assign_embedding_query_image_name(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.embedding_length = unsigned_i32(&event);
        event.parameters.set(parameters);
        self.effect_query_image_name(event)
    }
    fn effect_query_image_name(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"omniembed.image_encoder_name",
            super::IMAGE_ENCODER_NAME,
            Field::ImageEncoderName,
        );
        Ok(())
    }
    fn effect_set_image_canonical_query_length(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        consume_string(&event);
        let mut parameters = event.parameters.get();
        parameters.image_encoder = EncoderName::MobileNetV4Medium;
        event.parameters.set(parameters);
        self.effect_query_image_length(event)
    }
    fn effect_set_image_other_query_length(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        consume_string(&event);
        let mut parameters = event.parameters.get();
        parameters.image_encoder = EncoderName::Other;
        event.parameters.set(parameters);
        self.effect_query_image_length(event)
    }
    fn effect_query_image_length(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"omniembed.image_encoder_dim",
            Field::ImageEncoderLength,
        );
        Ok(())
    }
    fn effect_assign_image_query_audio_name(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.image_encoder_length = unsigned_i32(&event);
        event.parameters.set(parameters);
        self.effect_query_audio_name(event)
    }
    fn effect_query_audio_name(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_string(
            &event,
            b"omniembed.audio_encoder_name",
            super::AUDIO_ENCODER_NAME,
            Field::AudioEncoderName,
        );
        Ok(())
    }
    fn effect_set_audio_canonical_query_length(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        consume_string(&event);
        let mut parameters = event.parameters.get();
        parameters.audio_encoder = EncoderName::EfficientAtMn20As;
        event.parameters.set(parameters);
        self.effect_query_audio_length(event)
    }
    fn effect_set_audio_other_query_length(
        &mut self,
        event: HparamLoadRuntime<'_>,
    ) -> Result<(), ()> {
        consume_string(&event);
        let mut parameters = event.parameters.get();
        parameters.audio_encoder = EncoderName::Other;
        event.parameters.set(parameters);
        self.effect_query_audio_length(event)
    }
    fn effect_query_audio_length(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(
            &event,
            b"omniembed.audio_encoder_dim",
            Field::AudioEncoderLength,
        );
        Ok(())
    }
    fn effect_assign_audio_query_array(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let mut parameters = event.parameters.get();
        parameters.audio_encoder_length = unsigned_i32(&event);
        event.parameters.set(parameters);
        query_array_count(&event);
        Ok(())
    }
    fn effect_query_array(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        query_array_count(&event);
        Ok(())
    }
    fn effect_visit_array(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.array_values.borrow_mut().fill(0);
        event.array_visit.set(
            event
                .loader
                .borrow_mut()
                .process_event(VisitSignedArray::new(
                    b"omniembed.matryoshka_dims",
                    |index, raw| event.array_values.borrow_mut()[index as usize] = raw,
                )),
        );
        Ok(())
    }
    fn effect_assign_array_success(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        let count = usize::try_from(
            event
                .array_count
                .get()
                .expect("value-valid state has successful count")
                .expect("value-valid state has present array"),
        )
        .expect("value-valid state fits usize");
        let mut parameters = event.parameters.get();
        parameters.matryoshka_dimension_count =
            u32::try_from(count).expect("matryoshka capacity fits u32");
        for (destination, raw) in parameters.matryoshka_dimensions[..count]
            .iter_mut()
            .zip(event.array_values.borrow()[..count].iter())
        {
            *destination = i32::try_from(*raw).expect("value-valid guard proves positive i32");
        }
        event.result.set(Ok(parameters));
        Ok(())
    }
    fn effect_success(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.parameters.get()));
        Ok(())
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
    fn effect_capacity(&mut self, event: HparamLoadRuntime<'_>) -> Result<(), ()> {
        set_error(&event, ErrorKind::Capacity);
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
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
    fn guard_never(&self, _: &UnexpectedRuntime) -> Result<bool, ()> {
        Ok(false)
    }
}
