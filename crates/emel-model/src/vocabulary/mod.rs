//! Reusable allocation-free vocabulary materialization actor.

use core::cell::{Cell, RefCell};
use core::fmt;
use std::collections::TryReserveError;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{
    ElementKind, QueryError, ReadArrayLength, ReadBool, ReadSigned, ReadStringArrayMetrics,
    ReadUnsigned, ReadUnsignedArrayMetrics, VisitF32Array, VisitStringArray, VisitUnsignedArray,
    WithByteArray, WithString,
};
use emel_token::profile::Dependency;
use emel_token::profile::event::{Resolve, Resolved};

use crate::data::{
    MAX_MERGE_BYTES, MAX_MERGES, MAX_METADATA_BLOB_BYTES, MAX_PRECOMPILED_CHARMAP_BYTES,
    MAX_VOCAB_BYTES, MAX_VOCAB_TOKENS, Vocab,
};

pub mod event;
mod query;
mod sm;

use event::{Error, ErrorKind, Flags, Key, Loaded, Phase, SpecialIds};
use sm::{
    VocabularyLoaderStateMachine, VocabularyLoaderStateMachineContext, VocabularyLoaderStates,
};

const TOKEN_TYPE_UNDEFINED: i32 = 0;
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanOutcome {
    Pending,
    Missing,
    Done { count: u64, bytes: usize },
    Query(QueryError),
    Arithmetic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnsignedScanOutcome {
    Pending,
    Missing,
    Done { count: u64, maximum: u64 },
    Query(QueryError),
    Arithmetic,
}

#[derive(Clone, Copy)]
/// Dispatch-local vocabulary load state.
///
/// Every reference is stack-owned by one top-level `Load` dispatch. The actor
/// never retains this payload, and the next dispatch constructs fresh cells.
pub(crate) struct LoadRuntime<'a> {
    gguf: &'a RefCell<&'a mut GgufLoader>,
    string: &'a Cell<Result<Option<usize>, QueryError>>,
    model_length: &'a Cell<usize>,
    pre_length: &'a Cell<usize>,
    profile: &'a Cell<Result<Resolved, emel_token::profile::event::Error>>,
    signed: &'a Cell<Result<Option<i64>, QueryError>>,
    boolean: &'a Cell<Result<Option<bool>, QueryError>>,
    unsigned: &'a Cell<Result<Option<u64>, QueryError>>,
    array_length: &'a Cell<Result<Option<u64>, QueryError>>,
    scan: &'a Cell<ScanOutcome>,
    scan_count: &'a Cell<u64>,
    scan_bytes: &'a Cell<usize>,
    unsigned_scan: &'a Cell<UnsignedScanOutcome>,
    copy: &'a Cell<Result<Option<u64>, QueryError>>,
    result: &'a Cell<Result<Loaded, Error>>,
}

struct Context<D> {
    profile: D,
    vocab: Vocab,
    model_scratch: Box<[u8]>,
    pre_scratch: Box<[u8]>,
}

macro_rules! raw_outcome_guard {
    ($name:ident, $field:ident, $pattern:pat) => {
        fn $name(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
            Ok(matches!(event.$field.get(), $pattern))
        }
    };
}

macro_rules! fixed_error_effect {
    ($name:ident, $phase:expr, $key:expr, $kind:expr) => {
        fn $name(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
            publish_load_error(event, $phase, $key, $kind)
        }
    };
}

pub struct Loader<D: Dependency = emel_token::profile::Resolver> {
    machine: VocabularyLoaderStateMachine<Context<D>>,
}

impl Loader<emel_token::profile::Resolver> {
    /// Constructs a vocabulary loader and preallocates all bounded actor storage.
    ///
    /// # Errors
    ///
    /// Returns the allocation failure when any bounded storage cannot be reserved.
    pub fn try_new() -> Result<Self, TryReserveError> {
        Self::try_with_dependency(emel_token::profile::Resolver::new())
    }
}

impl<D: Dependency> Loader<D> {
    /// Constructs a vocabulary loader with a substitutable profile dependency.
    ///
    /// # Errors
    ///
    /// Returns the allocation failure when any bounded storage cannot be reserved.
    pub fn try_with_dependency(profile: D) -> Result<Self, TryReserveError> {
        let context = Context {
            profile,
            vocab: Vocab::try_new()?,
            model_scratch: try_zeroed(MAX_METADATA_BLOB_BYTES)?,
            pre_scratch: try_zeroed(MAX_METADATA_BLOB_BYTES)?,
        };
        Ok(Self {
            machine: VocabularyLoaderStateMachine::new(context),
        })
    }

    pub fn process_event<E: event::Event<D>>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn query<O: query::Operation>(&self, operation: &mut O) {
        query::process(
            self.machine.is(&VocabularyLoaderStates::StateLoaded),
            &self.machine.context().vocab,
            operation,
        );
    }
}

impl<D: Dependency> Loader<D> {
    pub(crate) fn load(&mut self, event: event::Load<'_>) -> Result<Loaded, Error> {
        let gguf = RefCell::new(event.gguf);
        let string = Cell::new(Err(QueryError::Internal));
        let profile = Cell::new(Err(emel_token::profile::event::Error::Internal));
        let signed = Cell::new(Err(QueryError::Internal));
        let boolean = Cell::new(Err(QueryError::Internal));
        let unsigned = Cell::new(Err(QueryError::Internal));
        let array_length = Cell::new(Err(QueryError::Internal));
        let scan = Cell::new(ScanOutcome::Pending);
        let scan_count = Cell::new(0);
        let scan_bytes = Cell::new(0);
        let unsigned_scan = Cell::new(UnsignedScanOutcome::Pending);
        let copy = Cell::new(Err(QueryError::Internal));
        let model_length = Cell::new(0);
        let pre_length = Cell::new(0);
        let result = Cell::new(Err(Error::new(
            Phase::Reset,
            Key::None,
            ErrorKind::Internal,
        )));
        let runtime = LoadRuntime {
            gguf: &gguf,
            string: &string,
            model_length: &model_length,
            pre_length: &pre_length,
            profile: &profile,
            signed: &signed,
            boolean: &boolean,
            unsigned: &unsigned,
            array_length: &array_length,
            scan: &scan,
            scan_count: &scan_count,
            scan_bytes: &scan_bytes,
            unsigned_scan: &unsigned_scan,
            copy: &copy,
            result: &result,
        };
        let dispatch = self
            .machine
            .process_event(sm::VocabularyLoaderEvents::Load(runtime));
        if dispatch.is_err() {
            return Err(Error::new(Phase::Final, Key::None, ErrorKind::Internal));
        }
        result.get()
    }
}

impl<D: Dependency> fmt::Debug for Loader<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabularyLoader").finish_non_exhaustive()
    }
}

fn try_zeroed(length: usize) -> Result<Box<[u8]>, TryReserveError> {
    let mut value = Vec::new();
    value.try_reserve_exact(length)?;
    value.resize(length, 0);
    Ok(value.into_boxed_slice())
}

impl<D: Dependency> VocabularyLoaderStateMachineContext for Context<D> {
    fn effect_reset_and_model_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        self.vocab.reset();
        event.model_length.set(0);
        event.pre_length.set(0);
        query_string(event, b"tokenizer.model", &mut self.model_scratch)
    }
    fn effect_model_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_string(event, b"tokenizer.ggml.model", &mut self.model_scratch)
    }
    fn effect_pre_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.model_length.set(string_length(event)?);
        query_string(event, b"tokenizer.pre", &mut self.pre_scratch)
    }
    fn effect_pre_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_string(event, b"tokenizer.ggml.pre", &mut self.pre_scratch)
    }
    fn guard_string_present(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .string
            .get()
            .is_ok_and(|v| v.is_some_and(|n| n != usize::MAX)))
    }
    raw_outcome_guard!(guard_string_missing, string, Ok(None));
    raw_outcome_guard!(guard_string_capacity, string, Ok(Some(usize::MAX)));
    raw_outcome_guard!(guard_string_malformed, string, Err(QueryError::Malformed));
    raw_outcome_guard!(
        guard_string_wrong_kind,
        string,
        Err(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(guard_string_range, string, Err(QueryError::Range));
    raw_outcome_guard!(
        guard_string_query,
        string,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(guard_string_internal, string, Err(QueryError::Internal));
    fn effect_model_missing(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Model,
            Key::TokenizerModel,
            ErrorKind::Missing,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_model_capacity,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::Capacity
    );
    fixed_error_effect!(
        effect_model_malformed,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_model_wrong_kind,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_model_range,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_model_query,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_model_internal,
        Phase::Model,
        Key::TokenizerModel,
        ErrorKind::Internal
    );
    fixed_error_effect!(
        effect_pre_capacity,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::Capacity
    );
    fixed_error_effect!(
        effect_pre_malformed,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_pre_wrong_kind,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_pre_range,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_pre_query,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_pre_internal,
        Phase::Pre,
        Key::TokenizerPre,
        ErrorKind::Internal
    );
    fn effect_capture_pre_length(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.pre_length.set(string_length(event)?);
        Ok(())
    }
    fn effect_capture_without_pre(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.pre_length.set(0);
        Ok(())
    }
    fn guard_model_and_pre_utf8(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.model_utf8(event).is_some() && self.pre_utf8(event).is_some())
    }
    fn guard_model_invalid_utf8(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.model_utf8(event).is_none())
    }
    fn guard_model_valid_pre_invalid_utf8(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.model_utf8(event).is_some() && self.pre_utf8(event).is_none())
    }
    fn effect_resolve_profile(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        let model = core::str::from_utf8(&self.model_scratch[..event.model_length.get()])
            .map_err(|_| ())?;
        let pre =
            core::str::from_utf8(&self.pre_scratch[..event.pre_length.get()]).map_err(|_| ())?;
        event
            .profile
            .set(self.profile.process_event(Resolve::new(model, pre)));
        Ok(())
    }
    fn effect_resolve_unknown_model(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.profile.set(
            self.profile
                .process_event(Resolve::new("__emel_invalid_utf8_model__", "default")),
        );
        Ok(())
    }
    fn effect_resolve_unknown_pre(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        let model = core::str::from_utf8(&self.model_scratch[..event.model_length.get()])
            .map_err(|_| ())?;
        event.profile.set(
            self.profile
                .process_event(Resolve::new(model, "__emel_invalid_utf8_pre__")),
        );
        Ok(())
    }
    fn guard_profile_success(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.profile.get().is_ok())
    }
    fn guard_profile_error(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.profile.get().is_err())
    }
    fn effect_profile_error(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Profile,
            Key::None,
            ErrorKind::Profile,
        )));
        Ok(())
    }
    fn effect_apply_profile(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        let resolved = event.profile.get().map_err(|_| ())?;
        apply_profile(
            &mut self.vocab,
            resolved,
            &self.model_scratch[..event.model_length.get()],
            &self.pre_scratch[..event.pre_length.get()],
        );
        Ok(())
    }
    fn effect_query_token_type_count_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(event, b"tokenizer.token_type_count")
    }
    fn effect_query_token_type_count_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_unsigned(event, b"tokenizer.ggml.token_type_count")
    }
    fn guard_unsigned_u32(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .unsigned
            .get()
            .is_ok_and(|value| value.is_some_and(|value| u32::try_from(value).is_ok())))
    }
    fn guard_unsigned_range(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .unsigned
            .get()
            .is_ok_and(|value| value.is_some_and(|value| u32::try_from(value).is_err())))
    }
    raw_outcome_guard!(guard_unsigned_missing, unsigned, Ok(None));
    raw_outcome_guard!(
        guard_unsigned_malformed,
        unsigned,
        Err(QueryError::Malformed)
    );
    raw_outcome_guard!(
        guard_unsigned_wrong_kind,
        unsigned,
        Err(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(guard_unsigned_query_range, unsigned, Err(QueryError::Range));
    raw_outcome_guard!(
        guard_unsigned_query,
        unsigned,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(guard_unsigned_internal, unsigned, Err(QueryError::Internal));
    fn effect_assign_token_type_count(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        self.vocab.n_token_types =
            u32::try_from(event.unsigned.get().ok().flatten().ok_or(())?).map_err(|_| ())?;
        Ok(())
    }
    fn effect_token_type_count_range(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::TokenTypeCount,
            Key::TokenTypeCount,
            ErrorKind::Range,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_token_type_count_malformed,
        Phase::TokenTypeCount,
        Key::TokenTypeCount,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_token_type_count_wrong_kind,
        Phase::TokenTypeCount,
        Key::TokenTypeCount,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_token_type_count_query,
        Phase::TokenTypeCount,
        Key::TokenTypeCount,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_token_type_count_internal,
        Phase::TokenTypeCount,
        Key::TokenTypeCount,
        ErrorKind::Internal
    );

    fn effect_scan_tokens_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_string_array(event, b"tokenizer.tokens")
    }
    fn effect_scan_tokens_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_string_array(event, b"tokenizer.ggml.tokens")
    }
    raw_outcome_guard!(guard_scan_missing, scan, ScanOutcome::Missing);
    raw_outcome_guard!(
        guard_scan_malformed,
        scan,
        ScanOutcome::Query(QueryError::Malformed)
    );
    raw_outcome_guard!(
        guard_scan_wrong_kind,
        scan,
        ScanOutcome::Query(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(
        guard_scan_range,
        scan,
        ScanOutcome::Query(QueryError::Range) | ScanOutcome::Arithmetic
    );
    raw_outcome_guard!(
        guard_scan_query,
        scan,
        ScanOutcome::Query(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(
        guard_scan_internal,
        scan,
        ScanOutcome::Query(QueryError::Internal)
    );
    fn guard_tokens_fit(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { count, bytes } if count <= MAX_VOCAB_TOKENS as u64 && bytes <= MAX_VOCAB_BYTES),
        )
    }
    fn guard_tokens_exceed_capacity(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { count, bytes } if count > MAX_VOCAB_TOKENS as u64 || bytes > MAX_VOCAB_BYTES),
        )
    }
    fn effect_copy_tokens_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_tokens(event, &mut self.vocab, b"tokenizer.tokens");
        Ok(())
    }
    fn effect_copy_tokens_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_tokens(event, &mut self.vocab, b"tokenizer.ggml.tokens");
        Ok(())
    }
    raw_outcome_guard!(guard_copy_success, copy, Ok(Some(_)));
    raw_outcome_guard!(guard_copy_missing, copy, Ok(None));
    raw_outcome_guard!(guard_copy_malformed, copy, Err(QueryError::Malformed));
    raw_outcome_guard!(guard_copy_wrong_kind, copy, Err(QueryError::TypeMismatch));
    raw_outcome_guard!(guard_copy_range, copy, Err(QueryError::Range));
    raw_outcome_guard!(guard_copy_count, copy, Err(QueryError::IndexOutOfBounds));
    raw_outcome_guard!(guard_copy_query, copy, Err(QueryError::NotParsed));
    raw_outcome_guard!(guard_copy_internal, copy, Err(QueryError::Internal));
    fn effect_tokens_capacity(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Tokens,
            Key::Tokens,
            ErrorKind::Capacity,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_tokens_malformed,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_tokens_wrong_kind,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_tokens_range,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_tokens_query,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_tokens_internal,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::Internal
    );
    fixed_error_effect!(
        effect_tokens_count,
        Phase::Tokens,
        Key::Tokens,
        ErrorKind::Count
    );

    fn effect_query_scores_standard_f32(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_array_length(event, b"tokenizer.scores", ElementKind::Float32)
    }
    fn effect_query_scores_standard_f64(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_array_length(event, b"tokenizer.scores", ElementKind::Float64)
    }
    fn effect_query_scores_legacy_f32(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_array_length(event, b"tokenizer.ggml.scores", ElementKind::Float32)
    }
    fn effect_query_scores_legacy_f64(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_array_length(event, b"tokenizer.ggml.scores", ElementKind::Float64)
    }
    raw_outcome_guard!(guard_array_missing, array_length, Ok(None));
    raw_outcome_guard!(
        guard_array_wrong_kind,
        array_length,
        Err(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(
        guard_array_malformed,
        array_length,
        Err(QueryError::Malformed)
    );
    raw_outcome_guard!(guard_array_range, array_length, Err(QueryError::Range));
    raw_outcome_guard!(
        guard_array_query,
        array_length,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(
        guard_array_internal,
        array_length,
        Err(QueryError::Internal)
    );
    fn guard_array_count_matches_tokens(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.array_length.get() == Ok(Some(u64::from(self.vocab.n_tokens))))
    }
    fn guard_array_count_mismatches_tokens(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .array_length
            .get()
            .is_ok_and(|value| value.is_some_and(|count| count != u64::from(self.vocab.n_tokens))))
    }
    fn effect_copy_scores_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_scores(event, &mut self.vocab, b"tokenizer.scores");
        Ok(())
    }
    fn effect_copy_scores_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_scores(event, &mut self.vocab, b"tokenizer.ggml.scores");
        Ok(())
    }
    fn effect_scores_count(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Scores,
            Key::Scores,
            ErrorKind::Count,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_scores_missing,
        Phase::Scores,
        Key::Scores,
        ErrorKind::Missing
    );
    fixed_error_effect!(
        effect_scores_wrong_kind,
        Phase::Scores,
        Key::Scores,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_scores_malformed,
        Phase::Scores,
        Key::Scores,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_scores_range,
        Phase::Scores,
        Key::Scores,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_scores_query,
        Phase::Scores,
        Key::Scores,
        ErrorKind::Query
    );
    fn effect_scores_internal(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Scores,
            Key::Scores,
            ErrorKind::Internal,
        )));
        Ok(())
    }

    fn effect_scan_token_types_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_unsigned_array(event, b"tokenizer.token_type")
    }
    fn effect_scan_token_types_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_unsigned_array(event, b"tokenizer.ggml.token_type")
    }
    raw_outcome_guard!(
        guard_unsigned_scan_missing,
        unsigned_scan,
        UnsignedScanOutcome::Missing
    );
    fn guard_unsigned_scan_valid(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.unsigned_scan.get(), UnsignedScanOutcome::Done { count, maximum } if count == u64::from(self.vocab.n_tokens) && i32::try_from(maximum).is_ok()),
        )
    }
    fn guard_unsigned_scan_count(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.unsigned_scan.get(), UnsignedScanOutcome::Done { count, .. } if count != u64::from(self.vocab.n_tokens)),
        )
    }
    fn guard_unsigned_scan_range(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.unsigned_scan.get(), UnsignedScanOutcome::Done { maximum, .. } if i32::try_from(maximum).is_err()),
        )
    }
    raw_outcome_guard!(
        guard_unsigned_scan_malformed,
        unsigned_scan,
        UnsignedScanOutcome::Query(QueryError::Malformed)
    );
    raw_outcome_guard!(
        guard_unsigned_scan_wrong_kind,
        unsigned_scan,
        UnsignedScanOutcome::Query(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(
        guard_unsigned_scan_query_range,
        unsigned_scan,
        UnsignedScanOutcome::Query(QueryError::Range) | UnsignedScanOutcome::Arithmetic
    );
    raw_outcome_guard!(
        guard_unsigned_scan_query,
        unsigned_scan,
        UnsignedScanOutcome::Query(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(
        guard_unsigned_scan_internal,
        unsigned_scan,
        UnsignedScanOutcome::Query(QueryError::Internal)
    );
    fn effect_copy_token_types_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_token_types(event, &mut self.vocab, b"tokenizer.token_type");
        Ok(())
    }
    fn effect_copy_token_types_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_token_types(event, &mut self.vocab, b"tokenizer.ggml.token_type");
        Ok(())
    }
    fn effect_token_types_count(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::TokenTypes,
            Key::TokenTypes,
            ErrorKind::Count,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_token_types_missing,
        Phase::TokenTypes,
        Key::TokenTypes,
        ErrorKind::Missing
    );
    fn effect_token_types_range(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::TokenTypes,
            Key::TokenTypes,
            ErrorKind::Range,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_token_types_malformed,
        Phase::TokenTypes,
        Key::TokenTypes,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_token_types_wrong_kind,
        Phase::TokenTypes,
        Key::TokenTypes,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_token_types_query,
        Phase::TokenTypes,
        Key::TokenTypes,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_token_types_internal,
        Phase::TokenTypes,
        Key::TokenTypes,
        ErrorKind::Internal
    );

    fn effect_scan_merges_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_string_array(event, b"tokenizer.merges")
    }
    fn effect_scan_merges_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_string_array(event, b"tokenizer.ggml.merges")
    }
    fn guard_merges_fit(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { count, bytes } if count <= MAX_MERGES as u64 && bytes <= MAX_MERGE_BYTES),
        )
    }
    fn guard_merges_exceed_capacity(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { count, bytes } if count > MAX_MERGES as u64 || bytes > MAX_MERGE_BYTES),
        )
    }
    fn effect_copy_merges_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_merges(event, &mut self.vocab, b"tokenizer.merges");
        Ok(())
    }
    fn effect_copy_merges_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_merges(event, &mut self.vocab, b"tokenizer.ggml.merges");
        Ok(())
    }
    fn effect_merges_capacity(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Merges,
            Key::Merges,
            ErrorKind::Capacity,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_merges_missing,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Missing
    );
    fixed_error_effect!(
        effect_merges_count,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Count
    );
    fixed_error_effect!(
        effect_merges_malformed,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_merges_wrong_kind,
        Phase::Merges,
        Key::Merges,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_merges_range,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_merges_query,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_merges_internal,
        Phase::Merges,
        Key::Merges,
        ErrorKind::Internal
    );

    fn effect_scan_charmap_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_byte_array(event, b"tokenizer.precompiled_charsmap")
    }
    fn effect_scan_charmap_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        scan_byte_array(event, b"tokenizer.ggml.precompiled_charsmap")
    }
    fn guard_charmap_fits(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { bytes, .. } if bytes <= MAX_PRECOMPILED_CHARMAP_BYTES),
        )
    }
    fn guard_charmap_exceeds_capacity(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(
            matches!(event.scan.get(), ScanOutcome::Done { bytes, .. } if bytes > MAX_PRECOMPILED_CHARMAP_BYTES),
        )
    }
    fn effect_copy_charmap_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_charmap(event, &mut self.vocab, b"tokenizer.precompiled_charsmap");
        Ok(())
    }
    fn effect_copy_charmap_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        copy_charmap(
            event,
            &mut self.vocab,
            b"tokenizer.ggml.precompiled_charsmap",
        );
        Ok(())
    }
    fn effect_charmap_capacity(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Charmap,
            Key::Charmap,
            ErrorKind::Capacity,
        )));
        Ok(())
    }
    fixed_error_effect!(
        effect_charmap_missing,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Missing
    );
    fixed_error_effect!(
        effect_charmap_count,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Count
    );
    fixed_error_effect!(
        effect_charmap_malformed,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_charmap_wrong_kind,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_charmap_range,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_charmap_query,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_charmap_internal,
        Phase::Charmap,
        Key::Charmap,
        ErrorKind::Internal
    );
    fn guard_signed_in_range(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .signed
            .get()
            .is_ok_and(|value| value.is_some_and(|value| i32::try_from(value).is_ok())))
    }
    fn guard_signed_out_of_range(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event
            .signed
            .get()
            .is_ok_and(|value| value.is_some_and(|value| i32::try_from(value).is_err())))
    }
    raw_outcome_guard!(guard_signed_missing, signed, Ok(None));
    raw_outcome_guard!(guard_signed_malformed, signed, Err(QueryError::Malformed));
    raw_outcome_guard!(
        guard_signed_wrong_kind,
        signed,
        Err(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(guard_signed_query_range, signed, Err(QueryError::Range));
    raw_outcome_guard!(
        guard_signed_query,
        signed,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(guard_signed_internal, signed, Err(QueryError::Internal));
    fn guard_boolean_present(&self, event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.boolean.get().is_ok_and(|value| value.is_some()))
    }
    raw_outcome_guard!(guard_boolean_missing, boolean, Ok(None));
    raw_outcome_guard!(guard_boolean_malformed, boolean, Err(QueryError::Malformed));
    raw_outcome_guard!(
        guard_boolean_wrong_kind,
        boolean,
        Err(QueryError::TypeMismatch)
    );
    raw_outcome_guard!(guard_boolean_range, boolean, Err(QueryError::Range));
    raw_outcome_guard!(
        guard_boolean_query,
        boolean,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_outcome_guard!(guard_boolean_internal, boolean, Err(QueryError::Internal));

    fn effect_query_bos_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.bos_token_id")
    }
    fn effect_query_bos_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.bos_token_id")
    }
    fn effect_assign_bos(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.bos_id)
    }
    fn effect_query_eos_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.eos_token_id")
    }
    fn effect_query_eos_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.eos_token_id")
    }
    fn effect_assign_eos(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.eos_id)
    }
    fn effect_query_eot_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.eot_token_id")
    }
    fn effect_query_eot_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.eot_token_id")
    }
    fn effect_assign_eot(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.eot_id)
    }
    fn effect_query_eom_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.eom_token_id")
    }
    fn effect_query_eom_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.eom_token_id")
    }
    fn effect_assign_eom(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.eom_id)
    }
    fn effect_query_unknown_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.unknown_token_id")
    }
    fn effect_query_unknown_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.unknown_token_id")
    }
    fn effect_assign_unknown(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.unk_id)
    }
    fn effect_query_separator_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.seperator_token_id")
    }
    fn effect_query_separator_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.seperator_token_id")
    }
    fn effect_assign_separator(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.sep_id)
    }
    fn effect_query_padding_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.padding_token_id")
    }
    fn effect_query_padding_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.padding_token_id")
    }
    fn effect_assign_padding(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.pad_id)
    }
    fn effect_query_cls_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.cls_token_id")
    }
    fn effect_query_cls_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.cls_token_id")
    }
    fn effect_assign_cls(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.cls_id)
    }
    fn effect_query_mask_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.mask_token_id")
    }
    fn effect_query_mask_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.mask_token_id")
    }
    fn effect_assign_mask(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.mask_id)
    }
    fn effect_query_prefix_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.prefix_token_id")
    }
    fn effect_query_prefix_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.prefix_token_id")
    }
    fn effect_assign_prefix(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.prefix_id)
    }
    fn effect_query_suffix_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.suffix_token_id")
    }
    fn effect_query_suffix_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.suffix_token_id")
    }
    fn effect_assign_suffix(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.suffix_id)
    }
    fn effect_query_middle_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.middle_token_id")
    }
    fn effect_query_middle_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.middle_token_id")
    }
    fn effect_assign_middle(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.middle_id)
    }
    fn effect_query_fim_pre_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_pre_token_id")
    }
    fn effect_query_fim_pre_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_pre_token_id")
    }
    fn effect_assign_fim_pre(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_pre_id)
    }
    fn effect_query_fim_suf_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_suf_token_id")
    }
    fn effect_query_fim_suf_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_suf_token_id")
    }
    fn effect_assign_fim_suf(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_suf_id)
    }
    fn effect_query_fim_mid_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_mid_token_id")
    }
    fn effect_query_fim_mid_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_mid_token_id")
    }
    fn effect_assign_fim_mid(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_mid_id)
    }
    fn effect_query_fim_pad_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_pad_token_id")
    }
    fn effect_query_fim_pad_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_pad_token_id")
    }
    fn effect_assign_fim_pad(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_pad_id)
    }
    fn effect_query_fim_rep_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_rep_token_id")
    }
    fn effect_query_fim_rep_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_rep_token_id")
    }
    fn effect_assign_fim_rep(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_rep_id)
    }
    fn effect_query_fim_sep_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.fim_sep_token_id")
    }
    fn effect_query_fim_sep_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_signed(event, b"tokenizer.ggml.fim_sep_token_id")
    }
    fn effect_assign_fim_sep(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_signed_field(event, &mut self.vocab.fim_sep_id)
    }
    fixed_error_effect!(
        effect_signed_malformed,
        Phase::SpecialIds,
        Key::SpecialId,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_signed_wrong_kind,
        Phase::SpecialIds,
        Key::SpecialId,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_signed_query,
        Phase::SpecialIds,
        Key::SpecialId,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_signed_internal,
        Phase::SpecialIds,
        Key::SpecialId,
        ErrorKind::Internal
    );
    fn effect_signed_range_error(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::SpecialIds,
            Key::SpecialId,
            ErrorKind::Range,
        )));
        Ok(())
    }

    fn effect_query_add_bos_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.add_bos_token")
    }
    fn effect_query_add_bos_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.add_bos_token")
    }
    fn effect_assign_add_bos(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.add_bos)
    }
    fn effect_query_add_eos_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.add_eos_token")
    }
    fn effect_query_add_eos_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.add_eos_token")
    }
    fn effect_assign_add_eos(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.add_eos)
    }
    fn effect_query_add_sep_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.add_sep_token")
    }
    fn effect_query_add_sep_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.add_sep_token")
    }
    fn effect_assign_add_sep(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.add_sep)
    }
    fn effect_query_add_space_prefix_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.add_space_prefix")
    }
    fn effect_query_add_space_prefix_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.add_space_prefix")
    }
    fn effect_assign_add_space_prefix(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.add_space_prefix)
    }
    fn effect_query_remove_extra_whitespaces_standard(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.remove_extra_whitespaces")
    }
    fn effect_query_remove_extra_whitespaces_legacy(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.remove_extra_whitespaces")
    }
    fn effect_assign_remove_extra_whitespaces(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.remove_extra_whitespaces)
    }
    fn effect_query_ignore_merges_standard(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ignore_merges")
    }
    fn effect_query_ignore_merges_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.ignore_merges")
    }
    fn effect_assign_ignore_merges(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.ignore_merges)
    }
    fn effect_query_escape_whitespaces_standard(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.escape_whitespaces")
    }
    fn effect_query_escape_whitespaces_legacy(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.escape_whitespaces")
    }
    fn effect_assign_escape_whitespaces(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.escape_whitespaces)
    }
    fn effect_query_treat_whitespace_as_suffix_standard(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.treat_whitespace_as_suffix")
    }
    fn effect_query_treat_whitespace_as_suffix_legacy(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        query_boolean(event, b"tokenizer.ggml.treat_whitespace_as_suffix")
    }
    fn effect_assign_treat_whitespace_as_suffix(
        &mut self,
        event: LoadRuntime<'_>,
    ) -> Result<(), ()> {
        assign_boolean_field(event, &mut self.vocab.treat_whitespace_as_suffix)
    }
    fixed_error_effect!(
        effect_boolean_malformed,
        Phase::Flags,
        Key::Flag,
        ErrorKind::Malformed
    );
    fixed_error_effect!(
        effect_boolean_wrong_kind,
        Phase::Flags,
        Key::Flag,
        ErrorKind::WrongKind
    );
    fixed_error_effect!(
        effect_boolean_range,
        Phase::Flags,
        Key::Flag,
        ErrorKind::Range
    );
    fixed_error_effect!(
        effect_boolean_query,
        Phase::Flags,
        Key::Flag,
        ErrorKind::Query
    );
    fixed_error_effect!(
        effect_boolean_internal,
        Phase::Flags,
        Key::Flag,
        ErrorKind::Internal
    );
    fn effect_tokens_missing(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Tokens,
            Key::Tokens,
            ErrorKind::Missing,
        )));
        Ok(())
    }
    fn effect_patch_special_types(&mut self, _event: LoadRuntime<'_>) -> Result<(), ()> {
        patch_special_types(&mut self.vocab);
        Ok(())
    }
    fn guard_model_unknown(&self, _event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.vocab.tokenizer_model_id.is_unknown())
    }
    fn guard_model_known(&self, _event: &LoadRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.vocab.tokenizer_model_id.is_unknown())
    }
    fn effect_unknown_model(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::new(
            Phase::Final,
            Key::TokenizerModel,
            ErrorKind::Unsupported,
        )));
        Ok(())
    }
    fn effect_publish(&mut self, event: LoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(Ok(Loaded::new(self.vocab.n_tokens, self.vocab.n_merges)));
        Ok(())
    }
    fn effect_unpublish(&mut self, _event: LoadRuntime<'_>) -> Result<(), ()> {
        self.vocab.reset();
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        self.vocab.reset();
        Ok(())
    }
}

impl<D: Dependency> Context<D> {
    fn model_utf8<'a>(&'a self, event: &LoadRuntime<'_>) -> Option<&'a str> {
        core::str::from_utf8(&self.model_scratch[..event.model_length.get()]).ok()
    }
    fn pre_utf8<'a>(&'a self, event: &LoadRuntime<'_>) -> Option<&'a str> {
        core::str::from_utf8(&self.pre_scratch[..event.pre_length.get()]).ok()
    }
}

#[allow(clippy::unnecessary_wraps)]
fn query_string(event: LoadRuntime<'_>, key: &[u8], destination: &mut [u8]) -> Result<(), ()> {
    event
        .string
        .set(
            event
                .gguf
                .borrow_mut()
                .process_event(WithString::new(key, |value: &[u8]| {
                    if value.len() > destination.len() {
                        usize::MAX
                    } else {
                        destination[..value.len()].copy_from_slice(value);
                        value.len()
                    }
                })),
        );
    Ok(())
}
fn string_length(event: LoadRuntime<'_>) -> Result<usize, ()> {
    event
        .string
        .get()
        .ok()
        .flatten()
        .filter(|n| *n != usize::MAX)
        .ok_or(())
}
#[allow(clippy::unnecessary_wraps)]
fn publish_load_error(
    event: LoadRuntime<'_>,
    phase: Phase,
    key: Key,
    kind: ErrorKind,
) -> Result<(), ()> {
    event.result.set(Err(Error::new(phase, key, kind)));
    Ok(())
}

fn apply_profile(v: &mut Vocab, r: Resolved, model: &[u8], pre: &[u8]) {
    let d = r.defaults();
    v.tokenizer_model_id = r.model();
    v.tokenizer_pre_id = Some(r.pre_id());
    copy_name(&mut v.tokenizer_model_name, model);
    copy_name(&mut v.tokenizer_pre_name, pre);
    v.bos_id = d.bos_id();
    v.eos_id = d.eos_id();
    v.eot_id = d.eot_id();
    v.eom_id = d.eom_id();
    v.unk_id = d.unk_id();
    v.sep_id = d.sep_id();
    v.pad_id = d.pad_id();
    v.cls_id = d.cls_id();
    v.mask_id = d.mask_id();
    v.prefix_id = d.prefix_id();
    v.suffix_id = d.suffix_id();
    v.middle_id = d.middle_id();
    v.fim_pre_id = d.fim_pre_id();
    v.fim_suf_id = d.fim_suf_id();
    v.fim_mid_id = d.fim_mid_id();
    v.fim_pad_id = d.fim_pad_id();
    v.fim_rep_id = d.fim_rep_id();
    v.fim_sep_id = d.fim_sep_id();
    v.add_bos = d.add_bos();
    v.add_eos = d.add_eos();
    v.add_sep = d.add_sep();
    v.add_space_prefix = d.add_space_prefix();
    v.remove_extra_whitespaces = d.remove_extra_whitespaces();
    v.escape_whitespaces = d.escape_whitespaces();
    v.treat_whitespace_as_suffix = d.treat_whitespace_as_suffix();
    v.ignore_merges = d.ignore_merges();
}
fn copy_name<const N: usize>(dst: &mut [u8; N], src: &[u8]) {
    dst.fill(0);
    let n = src.len().min(N - 1);
    dst[..n].copy_from_slice(&src[..n]);
}

#[allow(clippy::unnecessary_wraps)]
fn query_unsigned(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    event.unsigned.set(
        event
            .gguf
            .borrow_mut()
            .process_event(ReadUnsigned::new(key)),
    );
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn query_array_length(event: LoadRuntime<'_>, key: &[u8], kind: ElementKind) -> Result<(), ()> {
    event.array_length.set(
        event
            .gguf
            .borrow_mut()
            .process_event(ReadArrayLength::new(key, kind)),
    );
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn scan_string_array(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    let result = event
        .gguf
        .borrow_mut()
        .process_event(ReadStringArrayMetrics::new(key));
    event.scan.set(match result {
        Ok(None) => ScanOutcome::Missing,
        Ok(Some(metrics)) => {
            usize::try_from(metrics.total_string_bytes()).map_or(ScanOutcome::Arithmetic, |bytes| {
                event.scan_count.set(metrics.element_count());
                event.scan_bytes.set(bytes);
                ScanOutcome::Done {
                    count: metrics.element_count(),
                    bytes,
                }
            })
        }
        Err(error) => ScanOutcome::Query(error),
    });
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn scan_byte_array(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    let result = event
        .gguf
        .borrow_mut()
        .process_event(WithByteArray::new(key, |bytes: &[u8]| bytes.len()));
    event.scan.set(match result {
        Ok(None) => ScanOutcome::Missing,
        Ok(Some(bytes)) => {
            let count = u64::try_from(bytes).unwrap_or(u64::MAX);
            event.scan_count.set(count);
            event.scan_bytes.set(bytes);
            ScanOutcome::Done { count, bytes }
        }
        Err(error) => ScanOutcome::Query(error),
    });
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn scan_unsigned_array(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    let result = event
        .gguf
        .borrow_mut()
        .process_event(ReadUnsignedArrayMetrics::new(key));
    event.unsigned_scan.set(match result {
        Ok(None) => UnsignedScanOutcome::Missing,
        Ok(Some(metrics)) => UnsignedScanOutcome::Done {
            count: metrics.element_count(),
            maximum: metrics.maximum(),
        },
        Err(error) => UnsignedScanOutcome::Query(error),
    });
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn copy_tokens(event: LoadRuntime<'_>, vocab: &mut Vocab, key: &[u8]) {
    vocab.n_tokens = event.scan_count.get() as u32;
    let mut used = 0usize;
    event
        .copy
        .set(event.gguf.borrow_mut().process_event(VisitStringArray::new(
            key,
            |index: u32, text: &[u8]| {
                let index = index as usize;
                let end = used + text.len();
                vocab.token_storage[used..end].copy_from_slice(text);
                vocab.entries[index].text_offset = used as u32;
                vocab.entries[index].text_length = text.len() as u32;
                vocab.entries[index].score = 0.0;
                vocab.entries[index].r#type = TOKEN_TYPE_NORMAL;
                used = end;
            },
        )));
    vocab.token_bytes_used = used as u32;
}

#[allow(clippy::cast_possible_truncation)]
fn copy_scores(event: LoadRuntime<'_>, vocab: &mut Vocab, key: &[u8]) {
    event
        .copy
        .set(event.gguf.borrow_mut().process_event(VisitF32Array::new(
            key,
            |index: u32, value: f32| {
                vocab.entries[index as usize].score = value;
            },
        )));
}

#[allow(clippy::cast_possible_truncation)]
fn copy_token_types(event: LoadRuntime<'_>, vocab: &mut Vocab, key: &[u8]) {
    event.copy.set(
        event
            .gguf
            .borrow_mut()
            .process_event(VisitUnsignedArray::new(key, |index: u32, value: u64| {
                vocab.entries[index as usize].r#type =
                    i32::try_from(value).expect("range guard validated token type");
            })),
    );
}

#[allow(clippy::cast_possible_truncation)]
fn copy_merges(event: LoadRuntime<'_>, vocab: &mut Vocab, key: &[u8]) {
    vocab.n_merges = event.scan_count.get() as u32;
    let mut used = 0usize;
    event
        .copy
        .set(event.gguf.borrow_mut().process_event(VisitStringArray::new(
            key,
            |index: u32, text: &[u8]| {
                let index = index as usize;
                let end = used + text.len();
                vocab.merge_storage[used..end].copy_from_slice(text);
                vocab.merge_offsets[index] = used as u32;
                vocab.merge_lengths[index] = text.len() as u32;
                used = end;
            },
        )));
    vocab.merge_bytes_used = used as u32;
}

#[allow(clippy::cast_possible_truncation)]
fn copy_charmap(event: LoadRuntime<'_>, vocab: &mut Vocab, key: &[u8]) {
    let bytes = event.scan_bytes.get();
    vocab.precompiled_charsmap_size = bytes as u32;
    event
        .copy
        .set(
            event
                .gguf
                .borrow_mut()
                .process_event(WithByteArray::new(key, |bytes: &[u8]| {
                    vocab.precompiled_charsmap[..bytes.len()].copy_from_slice(bytes);
                    bytes.len() as u64
                })),
        );
}

#[allow(clippy::unnecessary_wraps)]
fn query_signed(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    event
        .signed
        .set(event.gguf.borrow_mut().process_event(ReadSigned::new(key)));
    Ok(())
}

fn assign_signed_field(event: LoadRuntime<'_>, field: &mut i32) -> Result<(), ()> {
    let value = event.signed.get().ok().flatten().ok_or(())?;
    *field = i32::try_from(value).map_err(|_| ())?;
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn query_boolean(event: LoadRuntime<'_>, key: &[u8]) -> Result<(), ()> {
    event
        .boolean
        .set(event.gguf.borrow_mut().process_event(ReadBool::new(key)));
    Ok(())
}

fn assign_boolean_field(event: LoadRuntime<'_>, field: &mut bool) -> Result<(), ()> {
    *field = event.boolean.get().ok().flatten().ok_or(())?;
    Ok(())
}
fn patch_special_types(v: &mut Vocab) {
    mark(v, v.unk_id, TOKEN_TYPE_UNKNOWN);
    for id in [
        v.bos_id,
        v.eos_id,
        v.eot_id,
        v.eom_id,
        v.sep_id,
        v.pad_id,
        v.cls_id,
        v.mask_id,
        v.prefix_id,
        v.suffix_id,
        v.middle_id,
        v.fim_pre_id,
        v.fim_suf_id,
        v.fim_mid_id,
        v.fim_pad_id,
        v.fim_rep_id,
        v.fim_sep_id,
    ] {
        mark(v, id, TOKEN_TYPE_CONTROL);
    }
}
fn mark(v: &mut Vocab, id: i32, ty: i32) {
    if let Ok(i) = usize::try_from(id)
        && i < v.n_tokens as usize
        && matches!(
            v.entries[i].r#type,
            TOKEN_TYPE_UNDEFINED | TOKEN_TYPE_NORMAL
        )
    {
        v.entries[i].r#type = ty;
    }
}
fn c_name(v: &[u8]) -> &[u8] {
    &v[..v.iter().position(|b| *b == 0).unwrap_or(v.len())]
}
fn flag_at(v: &[u8], i: usize) -> bool {
    v.get(i / 8).is_some_and(|b| b & (1 << (i % 8)) != 0)
}
const fn special_ids(v: &Vocab) -> SpecialIds {
    SpecialIds {
        bos: v.bos_id,
        eos: v.eos_id,
        eot: v.eot_id,
        eom: v.eom_id,
        unknown: v.unk_id,
        separator: v.sep_id,
        padding: v.pad_id,
        classification: v.cls_id,
        mask: v.mask_id,
        prefix: v.prefix_id,
        suffix: v.suffix_id,
        middle: v.middle_id,
        fim_pre: v.fim_pre_id,
        fim_suf: v.fim_suf_id,
        fim_mid: v.fim_mid_id,
        fim_pad: v.fim_pad_id,
        fim_rep: v.fim_rep_id,
        fim_sep: v.fim_sep_id,
    }
}
const fn flags(v: &Vocab) -> Flags {
    Flags {
        add_bos: v.add_bos,
        add_eos: v.add_eos,
        add_sep: v.add_sep,
        add_space_prefix: v.add_space_prefix,
        remove_extra_whitespaces: v.remove_extra_whitespaces,
        escape_whitespaces: v.escape_whitespaces,
        treat_whitespace_as_suffix: v.treat_whitespace_as_suffix,
        ignore_merges: v.ignore_merges,
    }
}
const fn not_loaded() -> Error {
    Error::new(Phase::Query, Key::None, ErrorKind::NotLoaded)
}

#[cfg(test)]
mod tests;
