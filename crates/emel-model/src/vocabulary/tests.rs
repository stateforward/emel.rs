use super::event::{ErrorKind, Load, WithCharmap, WithInfo, WithMerge, WithToken};
use super::*;
use crate::loader::test_gguf::{BOOL, F32, F64, Fixture, I8, I32, I64, U8, U32, U64};
use allocation_counter::measure;
use core::cell::RefCell;
use emel_gguf::event::{
    QueryError, VisitBoolArray, VisitF32Array, VisitStringArray, VisitUnsignedArray, WithByteArray,
};
use emel_token::profile::event::{Model, Resolve};
use emel_token::profile::{Dependency, Resolver};
use std::rc::Rc;

type RecordedRequest = Rc<RefCell<Option<(Vec<u8>, Vec<u8>)>>>;

struct RecordingDependency {
    resolver: Resolver,
    requests: RecordedRequest,
}

struct FailingDependency;

struct SubstitutingDependency {
    resolver: Resolver,
    requests: RecordedRequest,
}

#[test]
fn gguf_bulk_copy_validation_precedes_every_callback() {
    let integers = [1_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .strings(b"strings", &[b"one", b"two"])
            .array(b"integers", U32, &integers, 2)
            .build(),
    );
    let mut callbacks = 0_u32;

    assert_eq!(
        gguf.process_event(VisitStringArray::new(b"integers", |_: u32, _: &[u8]| {
            callbacks += 1;
        },)),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(callbacks, 0);
    assert_eq!(
        gguf.process_event(VisitF32Array::new(b"integers", |_, _| {
            callbacks += 1;
        })),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(callbacks, 0);
    assert_eq!(
        gguf.process_event(VisitUnsignedArray::new(b"strings", |_, _| {
            callbacks += 1;
        })),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(callbacks, 0);
    assert_eq!(
        gguf.process_event(VisitBoolArray::new(b"integers", |_, _| {
            callbacks += 1;
        })),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(callbacks, 0);
    assert_eq!(
        gguf.process_event(WithByteArray::new(b"strings", |_: &[u8]| {
            callbacks += 1;
        })),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(callbacks, 0);
}

impl Dependency for FailingDependency {
    fn process_event(
        &mut self,
        _event: emel_token::profile::event::Resolve<'_>,
    ) -> Result<emel_token::profile::event::Resolved, emel_token::profile::event::Error> {
        Err(emel_token::profile::event::Error::Internal)
    }
}

impl Dependency for RecordingDependency {
    fn process_event(
        &mut self,
        event: emel_token::profile::event::Resolve<'_>,
    ) -> Result<emel_token::profile::event::Resolved, emel_token::profile::event::Error> {
        self.requests.replace(Some((
            event.model().as_bytes().to_vec(),
            event.pre().as_bytes().to_vec(),
        )));
        self.resolver.process_event(event)
    }
}

impl Dependency for SubstitutingDependency {
    fn process_event(
        &mut self,
        event: emel_token::profile::event::Resolve<'_>,
    ) -> Result<emel_token::profile::event::Resolved, emel_token::profile::event::Error> {
        self.requests.replace(Some((
            event.model().as_bytes().to_vec(),
            event.pre().as_bytes().to_vec(),
        )));
        self.resolver
            .process_event(Resolve::new("gpt2", "solar-open"))
    }
}

#[test]
fn public_lifecycle_materializes_standard_gpt2_lfm2() {
    let token_types = [3u32.to_le_bytes(), 1u32.to_le_bytes()].concat();
    let scores = [0.0f32.to_le_bytes(), 1.5f32.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .string(b"tokenizer.pre", b"lfm2")
            .strings(b"tokenizer.tokens", &[b"<|pad|>", b"hello"])
            .array(b"tokenizer.token_type", U32, &token_types, 2)
            .scalar(b"tokenizer.token_type_count", U32, 4u32.to_le_bytes())
            .array(b"tokenizer.scores", F32, &scores, 2)
            .strings(b"tokenizer.merges", &[b"h ello"])
            .scalar(b"tokenizer.bos_token_id", U32, 0u32.to_le_bytes())
            .scalar(b"tokenizer.eos_token_id", U32, 0u32.to_le_bytes())
            .scalar(b"tokenizer.padding_token_id", U32, 0u32.to_le_bytes())
            .scalar(b"tokenizer.add_eos_token", BOOL, vec![0])
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();
    assert_eq!(
        loader.process_event(Load::new(&mut gguf)).unwrap(),
        Loaded::new(2, 1)
    );
    let info = loader
        .process_event(WithInfo::new(|i: event::Info<'_>| {
            (
                i.model,
                i.token_count,
                i.token_type_count,
                i.special_ids,
                i.flags,
            )
        }))
        .unwrap();
    assert_eq!(info.0, Model::Bpe);
    assert_eq!((info.1, info.2), (2, 4));
    assert_eq!(info.3.padding, 0);
    assert!(info.4.add_bos);
    assert!(!info.4.add_eos);
    assert!(info.4.ignore_merges);
    let token = loader
        .process_event(WithToken::new(1, |t: event::Token<'_>| {
            (t.text.to_vec(), t.score, t.r#type)
        }))
        .unwrap()
        .unwrap();
    assert_eq!(token, (b"hello".to_vec(), 1.5, 1));
}

#[test]
fn legacy_t5_defaults_and_failure_recovery_are_explicit() {
    let types = [
        3u32.to_le_bytes(),
        3u32.to_le_bytes(),
        2u32.to_le_bytes(),
        1u32.to_le_bytes(),
    ]
    .concat();
    let mut good = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.ggml.model", b"t5")
            .string(b"tokenizer.ggml.pre", b"default")
            .strings(
                b"tokenizer.ggml.tokens",
                &[b"<pad>", b"</s>", b"<unk>", b"x"],
            )
            .array(b"tokenizer.ggml.token_type", U32, &types, 4)
            .scalar(b"tokenizer.ggml.add_space_prefix", BOOL, vec![1])
            .build(),
    );
    let mut bad = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"future-model")
            .strings(b"tokenizer.tokens", &[b"x"])
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();
    assert_eq!(
        loader
            .process_event(Load::new(&mut bad))
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        loader
            .process_event(WithInfo::new(|i: event::Info<'_>| i.token_count))
            .unwrap_err()
            .kind(),
        ErrorKind::NotLoaded
    );
    loader.process_event(Load::new(&mut good)).unwrap();
    let got = loader
        .process_event(WithInfo::new(|i: event::Info<'_>| {
            (i.model, i.special_ids, i.flags)
        }))
        .unwrap();
    assert_eq!(got.0, Model::Unigram);
    assert_eq!((got.1.padding, got.1.eos, got.1.unknown), (0, 1, 2));
    assert!(got.2.add_space_prefix);
    assert!(got.2.escape_whitespaces);
}

#[test]
fn typed_edges_and_callback_views_match_source_semantics() {
    let scores = [0.25_f64.to_le_bytes(), (-1.5_f64).to_le_bytes()].concat();
    let types = [0_i8.to_le_bytes(), 1_i8.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .string(b"tokenizer.pre", b"future-pre-profile")
            .strings(b"tokenizer.tokens", &[b"<unk>", b"hello"])
            .array(b"tokenizer.scores", F64, &scores, 2)
            .array(b"tokenizer.token_type", I8, &types, 2)
            .strings(b"tokenizer.merges", &[b"h ello", b"he llo"])
            .array(b"tokenizer.precompiled_charsmap", U8, &[1, 2, 3, 4], 4)
            .scalar(b"tokenizer.unknown_token_id", I32, 0_i32.to_le_bytes())
            .scalar(b"tokenizer.eos_token_id", I32, (-1_i32).to_le_bytes())
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();

    loader.process_event(Load::new(&mut gguf)).unwrap();

    let info = loader
        .process_event(WithInfo::new(|info: event::Info<'_>| {
            (info.pre_name.to_vec(), info.special_ids)
        }))
        .unwrap();
    assert_eq!(info.0, b"future-pre-profile");
    assert_eq!(info.1.eos, -1);
    assert_eq!(
        loader
            .process_event(WithToken::new(0, |token: event::Token<'_>| {
                (token.score, token.r#type)
            }))
            .unwrap(),
        Some((0.25, TOKEN_TYPE_UNKNOWN))
    );
    assert_eq!(
        loader
            .process_event(WithToken::new(1, |token: event::Token<'_>| token.score))
            .unwrap(),
        Some(-1.5)
    );
    assert_eq!(
        loader
            .process_event(WithMerge::new(1, |merge: &[u8]| merge.to_vec()))
            .unwrap(),
        Some(b"he llo".to_vec())
    );
    assert_eq!(
        loader
            .process_event(WithCharmap::new(|charmap: &[u8]| charmap.to_vec()))
            .unwrap(),
        [1, 2, 3, 4]
    );
}

#[test]
fn wrong_standard_alias_does_not_fall_back_and_next_dispatch_starts_clean() {
    let mut wrong = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .scalar(b"tokenizer.tokens", U32, 1_u32.to_le_bytes())
            .strings(b"tokenizer.ggml.tokens", &[b"legacy"])
            .build(),
    );
    let mut good = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .strings(b"tokenizer.tokens", &[b"fresh"])
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();

    let error = loader.process_event(Load::new(&mut wrong)).unwrap_err();
    assert_eq!(
        (error.phase(), error.kind()),
        (Phase::Tokens, ErrorKind::WrongKind)
    );
    assert_eq!(
        loader
            .process_event(WithInfo::new(|info: event::Info<'_>| info.token_count))
            .unwrap_err()
            .kind(),
        ErrorKind::NotLoaded
    );

    loader.process_event(Load::new(&mut good)).unwrap();
    assert_eq!(
        loader
            .process_event(WithToken::new(0, |token: event::Token<'_>| token
                .text
                .to_vec()))
            .unwrap(),
        Some(b"fresh".to_vec())
    );
}

#[test]
fn prepared_load_and_callback_queries_are_allocation_free() {
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .strings(b"tokenizer.tokens", &[b"a", b"b"])
            .strings(b"tokenizer.merges", &[b"a b"])
            .array(b"tokenizer.precompiled_charsmap", U8, &[7, 8], 2)
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();
    let load_result = RefCell::new(None);
    let allocation = measure(|| {
        load_result.replace(Some(loader.process_event(Load::new(&mut gguf))));
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        load_result.into_inner().unwrap().unwrap(),
        Loaded::new(2, 1)
    );

    let query = RefCell::new(None);
    let allocation = measure(|| {
        query.replace(Some(loader.process_event(WithInfo::new(
            |info: event::Info<'_>| (info.token_count, info.merge_count),
        ))));
        assert_eq!(
            loader
                .process_event(WithToken::new(0, |token: event::Token<'_>| token
                    .text
                    .len()))
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            loader
                .process_event(WithMerge::new(0, |merge: &[u8]| merge.len()))
                .unwrap(),
            Some(3)
        );
        assert_eq!(
            loader
                .process_event(WithCharmap::new(|charmap: &[u8]| charmap.len()))
                .unwrap(),
            2
        );
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(query.into_inner().unwrap().unwrap(), (2, 1));
}

#[test]
fn dependency_receives_full_names_before_fixed_storage_truncation() {
    let pre = "x".repeat(256);
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .string(b"tokenizer.pre", pre.as_bytes())
            .strings(b"tokenizer.tokens", &[b"token"])
            .build(),
    );
    let requests = Rc::new(RefCell::new(None));
    let dependency = RecordingDependency {
        resolver: Resolver::new(),
        requests: Rc::clone(&requests),
    };
    let mut loader = Loader::try_with_dependency(dependency).unwrap();

    loader.process_event(Load::new(&mut gguf)).unwrap();

    assert_eq!(
        *requests.borrow(),
        Some((b"gpt2".to_vec(), pre.as_bytes().to_vec()))
    );
    let stored = loader
        .process_event(WithInfo::new(|info: event::Info<'_>| info.pre_name.len()))
        .unwrap();
    assert!(stored < pre.len());
}

#[test]
fn dependency_retains_substituted_pre_identity_and_rejects_other_identity() {
    let requests = Rc::new(RefCell::new(None));
    let dependency = SubstitutingDependency {
        resolver: Resolver::new(),
        requests: Rc::clone(&requests),
    };
    let mut loader = Loader::try_with_dependency(dependency).unwrap();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .string(b"tokenizer.pre", b"default")
            .strings(b"tokenizer.tokens", &[b"token"])
            .build(),
    );

    loader.process_event(Load::new(&mut gguf)).unwrap();

    assert_eq!(
        *requests.borrow(),
        Some((b"gpt2".to_vec(), b"default".to_vec()))
    );
    let published = loader
        .process_event(WithInfo::new(|info: event::Info<'_>| info.pre))
        .unwrap();
    let mut resolver = Resolver::new();
    let expected = resolver
        .process_event(Resolve::new("gpt2", "solar-open"))
        .unwrap()
        .pre_id();
    let mutated = resolver
        .process_event(Resolve::new("gpt2", "default"))
        .unwrap()
        .pre_id();
    assert_eq!(published, expected);
    assert_ne!(published, mutated);
}

#[test]
fn gemma4_accepts_the_400001_merge_regression_boundary() {
    let token_types = [3_u32.to_le_bytes(), 3_u32.to_le_bytes()].concat();
    let scores = [0.0_f64.to_le_bytes(), 0.0_f64.to_le_bytes()].concat();
    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.ggml.model", b"gemma4")
            .strings(b"tokenizer.ggml.tokens", &[b"<bos>", b"<eos>"])
            .array(b"tokenizer.ggml.token_type", U32, &token_types, 2)
            .scalar(b"tokenizer.ggml.token_type_count", U32, 4_u32.to_le_bytes())
            .array(b"tokenizer.ggml.scores", F64, &scores, 2)
            .repeated_strings(b"tokenizer.ggml.merges", b"", 400_001)
            .array(b"tokenizer.ggml.precompiled_charsmap", U8, &[1, 2, 3], 3)
            .scalar(b"tokenizer.ggml.bos_token_id", U32, 0_u32.to_le_bytes())
            .scalar(b"tokenizer.ggml.eos_token_id", U32, 1_u32.to_le_bytes())
            .scalar(b"tokenizer.ggml.add_bos_token", BOOL, [1])
            .scalar(b"tokenizer.ggml.add_space_prefix", BOOL, [1])
            .build(),
    );
    let mut loader = Loader::try_new().unwrap();

    let outcome = loader.process_event(Load::new(&mut gguf)).unwrap();

    assert_eq!(outcome.merge_count(), 400_001);
    let info = loader
        .process_event(WithInfo::new(|info: event::Info<'_>| {
            (info.model, info.special_ids, info.flags, info.charmap_bytes)
        }))
        .unwrap();
    assert_eq!(info.0, Model::SentencePiece);
    assert_eq!((info.1.bos, info.1.eos), (0, 1));
    assert!(info.2.add_bos);
    assert!(info.2.add_space_prefix);
    assert_eq!(info.3, 3);
    assert_eq!(
        loader
            .process_event(WithMerge::new(400_000, |merge: &[u8]| merge.len()))
            .unwrap(),
        Some(0)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn token_merge_and_charmap_capacity_boundaries_are_exact() {
    let mut loader = Loader::try_new().unwrap();

    {
        let mut exact = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .repeated_strings(b"tokenizer.tokens", b"", MAX_VOCAB_TOKENS)
                .build(),
        );
        assert_eq!(
            loader
                .process_event(Load::new(&mut exact))
                .unwrap()
                .token_count(),
            u32::try_from(MAX_VOCAB_TOKENS).unwrap()
        );
    }
    {
        let mut over = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .repeated_strings(b"tokenizer.tokens", b"", MAX_VOCAB_TOKENS + 1)
                .build(),
        );
        let error = loader.process_event(Load::new(&mut over)).unwrap_err();
        assert_eq!(
            (error.phase(), error.kind()),
            (Phase::Tokens, ErrorKind::Capacity)
        );
    }
    {
        let exact_text = vec![b'x'; MAX_VOCAB_BYTES];
        let mut exact = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[&exact_text])
                .build(),
        );
        assert_eq!(
            loader
                .process_event(Load::new(&mut exact))
                .unwrap()
                .token_count(),
            1
        );
    }
    {
        let over_text = vec![b'x'; MAX_VOCAB_BYTES + 1];
        let mut over = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[&over_text])
                .build(),
        );
        assert_eq!(
            loader
                .process_event(Load::new(&mut over))
                .unwrap_err()
                .kind(),
            ErrorKind::Capacity
        );
    }

    for (count, expected) in [
        (MAX_MERGES, Ok(u32::try_from(MAX_MERGES).unwrap())),
        (MAX_MERGES + 1, Err(ErrorKind::Capacity)),
    ] {
        let mut gguf = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"token"])
                .repeated_strings(b"tokenizer.merges", b"", count)
                .build(),
        );
        let result = loader.process_event(Load::new(&mut gguf));
        match expected {
            Ok(count) => assert_eq!(result.unwrap().merge_count(), count),
            Err(kind) => assert_eq!(result.unwrap_err().kind(), kind),
        }
    }

    for (count, expected) in [
        (
            MAX_PRECOMPILED_CHARMAP_BYTES,
            Ok(u32::try_from(MAX_PRECOMPILED_CHARMAP_BYTES).unwrap()),
        ),
        (MAX_PRECOMPILED_CHARMAP_BYTES + 1, Err(ErrorKind::Capacity)),
    ] {
        let bytes = vec![7; count];
        let mut gguf = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"token"])
                .array(
                    b"tokenizer.precompiled_charsmap",
                    U8,
                    &bytes,
                    u64::try_from(count).unwrap(),
                )
                .build(),
        );
        let result = loader.process_event(Load::new(&mut gguf));
        match expected {
            Ok(bytes) => {
                result.unwrap();
                assert_eq!(
                    loader
                        .process_event(WithInfo::new(|info: event::Info<'_>| {
                            info.charmap_bytes
                        }))
                        .unwrap(),
                    bytes
                );
            }
            Err(kind) => assert_eq!(result.unwrap_err().kind(), kind),
        }
    }
}

#[test]
fn score_type_merge_counts_and_byte_capacity_are_exact() {
    let mut loader = Loader::try_new().unwrap();

    for (key, kind, payload, count, phase) in [
        (
            b"tokenizer.scores" as &[u8],
            F32,
            0.0_f32.to_le_bytes().to_vec(),
            1,
            Phase::Scores,
        ),
        (
            b"tokenizer.token_type",
            U32,
            1_u32.to_le_bytes().to_vec(),
            1,
            Phase::TokenTypes,
        ),
    ] {
        let mut short = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"a", b"b"])
                .array(key, kind, &payload, count)
                .build(),
        );
        let error = loader.process_event(Load::new(&mut short)).unwrap_err();
        assert_eq!((error.phase(), error.kind()), (phase, ErrorKind::Count));

        let long_payload = [payload.as_slice(), payload.as_slice(), payload.as_slice()].concat();
        let mut long = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"a", b"b"])
                .array(key, kind, &long_payload, 3)
                .build(),
        );
        let error = loader.process_event(Load::new(&mut long)).unwrap_err();
        assert_eq!((error.phase(), error.kind()), (phase, ErrorKind::Count));
    }

    for (length, expected) in [
        (MAX_MERGE_BYTES, Ok(u32::try_from(MAX_MERGE_BYTES).unwrap())),
        (MAX_MERGE_BYTES + 1, Err(ErrorKind::Capacity)),
    ] {
        let merge = vec![b'm'; length];
        let mut gguf = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"token"])
                .strings(b"tokenizer.merges", &[&merge])
                .build(),
        );
        let result = loader.process_event(Load::new(&mut gguf));
        match expected {
            Ok(bytes) => {
                result.unwrap();
                assert_eq!(
                    loader
                        .process_event(WithInfo::new(|info: event::Info<'_>| info.merge_bytes))
                        .unwrap(),
                    bytes
                );
            }
            Err(kind) => assert_eq!(result.unwrap_err().kind(), kind),
        }
    }
}

#[test]
fn missing_and_wrong_kinds_are_typed_for_each_materialized_family() {
    let mut loader = Loader::try_new().unwrap();
    let mut missing =
        crate::loader::test_gguf::load(Fixture::new().string(b"tokenizer.model", b"gpt2").build());
    let error = loader.process_event(Load::new(&mut missing)).unwrap_err();
    assert_eq!(
        (error.phase(), error.kind()),
        (Phase::Tokens, ErrorKind::Missing)
    );

    for (fixture, phase) in [
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .string(b"tokenizer.scores", b"wrong"),
            Phase::Scores,
        ),
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .string(b"tokenizer.token_type", b"wrong"),
            Phase::TokenTypes,
        ),
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .scalar(b"tokenizer.merges", U32, 0_u32.to_le_bytes()),
            Phase::Merges,
        ),
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .string(b"tokenizer.precompiled_charsmap", b"wrong"),
            Phase::Charmap,
        ),
    ] {
        let mut gguf = crate::loader::test_gguf::load(fixture.build());
        let error = loader.process_event(Load::new(&mut gguf)).unwrap_err();
        assert_eq!((error.phase(), error.kind()), (phase, ErrorKind::WrongKind));
    }
}

#[test]
fn query_events_report_not_loaded_and_out_of_range_without_callbacks() {
    let mut loader = Loader::try_new().unwrap();
    let info_error = loader
        .process_event(WithInfo::new(|_: event::Info<'_>| ()))
        .unwrap_err();
    assert_eq!(info_error.key(), Key::None);
    assert_eq!(
        info_error.to_string(),
        "vocabulary None error at Query: NotLoaded"
    );
    assert_eq!(
        loader
            .process_event(WithToken::new(0, |_: event::Token<'_>| ()))
            .unwrap_err()
            .kind(),
        ErrorKind::NotLoaded
    );
    assert_eq!(
        loader
            .process_event(WithMerge::new(0, |_: &[u8]| ()))
            .unwrap_err()
            .kind(),
        ErrorKind::NotLoaded
    );
    assert_eq!(
        loader
            .process_event(WithCharmap::new(|_: &[u8]| ()))
            .unwrap_err()
            .kind(),
        ErrorKind::NotLoaded
    );

    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .strings(b"tokenizer.tokens", &[b"one"])
            .build(),
    );
    loader.process_event(Load::new(&mut gguf)).unwrap();
    assert!(
        loader
            .process_event(WithToken::new(u32::MAX, |_: event::Token<'_>| ()))
            .unwrap()
            .is_none()
    );
    assert!(
        loader
            .process_event(WithMerge::new(u32::MAX, |_: &[u8]| ()))
            .unwrap()
            .is_none()
    );
    assert_eq!(format!("{loader:?}"), "VocabularyLoader { .. }");
}

#[test]
fn model_profile_and_metadata_ranges_have_typed_failures() {
    let mut loader = Loader::try_new().unwrap();

    let mut missing_model = crate::loader::test_gguf::load(Fixture::new().build());
    let error = loader
        .process_event(Load::new(&mut missing_model))
        .unwrap_err();
    assert_eq!(
        (error.phase(), error.key(), error.kind()),
        (Phase::Model, Key::TokenizerModel, ErrorKind::Missing)
    );

    let oversized_model = vec![b'x'; MAX_METADATA_BLOB_BYTES + 1];
    let mut oversized = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", &oversized_model)
            .build(),
    );
    let error = loader.process_event(Load::new(&mut oversized)).unwrap_err();
    assert_eq!(
        (error.phase(), error.key(), error.kind()),
        (Phase::Model, Key::TokenizerModel, ErrorKind::Capacity)
    );

    let mut gguf = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .strings(b"tokenizer.tokens", &[b"x"])
            .build(),
    );
    let mut failing = Loader::try_with_dependency(FailingDependency).unwrap();
    assert_eq!(
        failing
            .process_event(Load::new(&mut gguf))
            .unwrap_err()
            .kind(),
        ErrorKind::Profile
    );

    for (fixture, phase, key) in [
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .scalar(b"tokenizer.token_type_count", U64, u64::MAX.to_le_bytes())
                .strings(b"tokenizer.tokens", &[b"x"]),
            Phase::TokenTypeCount,
            Key::TokenTypeCount,
        ),
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .array(b"tokenizer.token_type", U64, &u64::MAX.to_le_bytes(), 1),
            Phase::TokenTypes,
            Key::TokenTypes,
        ),
        (
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &[b"x"])
                .scalar(b"tokenizer.bos_token_id", I64, i64::MAX.to_le_bytes()),
            Phase::SpecialIds,
            Key::SpecialId,
        ),
    ] {
        let mut gguf = crate::loader::test_gguf::load(fixture.build());
        let error = loader.process_event(Load::new(&mut gguf)).unwrap_err();
        assert_eq!(
            (error.phase(), error.key(), error.kind()),
            (phase, key, ErrorKind::Range)
        );
    }

    let mut wrong_flag = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .strings(b"tokenizer.tokens", &[b"x"])
            .string(b"tokenizer.add_bos_token", b"wrong")
            .build(),
    );
    let error = loader
        .process_event(Load::new(&mut wrong_flag))
        .unwrap_err();
    assert_eq!(
        (error.phase(), error.key(), error.kind()),
        (Phase::Flags, Key::Flag, ErrorKind::WrongKind)
    );
}

#[test]
fn invalid_utf8_model_and_pre_follow_source_unknown_routes() {
    let mut loader = Loader::try_new().unwrap();
    let mut invalid_model = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", &[0xff])
            .strings(b"tokenizer.tokens", &[b"x"])
            .build(),
    );
    let error = loader
        .process_event(Load::new(&mut invalid_model))
        .unwrap_err();
    assert_eq!(
        (error.phase(), error.kind()),
        (Phase::Final, ErrorKind::Unsupported)
    );

    let mut invalid_pre = crate::loader::test_gguf::load(
        Fixture::new()
            .string(b"tokenizer.model", b"gpt2")
            .string(b"tokenizer.pre", &[0xff])
            .strings(b"tokenizer.tokens", &[b"x"])
            .build(),
    );
    loader.process_event(Load::new(&mut invalid_pre)).unwrap();
    let raw_pre = loader
        .process_event(WithInfo::new(|info: event::Info<'_>| {
            (info.pre_name.to_vec(), info.pre)
        }))
        .unwrap();
    assert_eq!(raw_pre.0, [0xff]);
    assert_eq!(
        raw_pre.1,
        Resolver::new()
            .process_event(Resolve::new("none", "future-pre"))
            .unwrap()
            .pre_id()
    );
}

fn fixture_with_all_overrides(prefix: &[u8]) -> Fixture {
    let mut fixture = Fixture::new();
    for (index, suffix) in [
        b"bos_token_id" as &[u8],
        b"eos_token_id",
        b"eot_token_id",
        b"eom_token_id",
        b"unknown_token_id",
        b"seperator_token_id",
        b"padding_token_id",
        b"cls_token_id",
        b"mask_token_id",
        b"prefix_token_id",
        b"suffix_token_id",
        b"middle_token_id",
        b"fim_pre_token_id",
        b"fim_suf_token_id",
        b"fim_mid_token_id",
        b"fim_pad_token_id",
        b"fim_rep_token_id",
        b"fim_sep_token_id",
    ]
    .into_iter()
    .enumerate()
    {
        let key = [prefix, suffix].concat();
        fixture = fixture.scalar(&key, I32, i32::try_from(index).unwrap().to_le_bytes());
    }
    for (index, suffix) in [
        b"add_bos_token" as &[u8],
        b"add_eos_token",
        b"add_sep_token",
        b"add_space_prefix",
        b"remove_extra_whitespaces",
        b"ignore_merges",
        b"escape_whitespaces",
        b"treat_whitespace_as_suffix",
    ]
    .into_iter()
    .enumerate()
    {
        let key = [prefix, suffix].concat();
        fixture = fixture.scalar(&key, BOOL, [u8::from(index % 2 == 0)]);
    }
    fixture
}

#[test]
fn every_special_id_and_flag_supports_standard_and_legacy_aliases() {
    for (model_key, tokens_key, prefix) in [
        (
            b"tokenizer.model" as &[u8],
            b"tokenizer.tokens" as &[u8],
            b"tokenizer." as &[u8],
        ),
        (
            b"tokenizer.ggml.model",
            b"tokenizer.ggml.tokens",
            b"tokenizer.ggml.",
        ),
    ] {
        let tokens = vec![b"x" as &[u8]; 18];
        let mut gguf = crate::loader::test_gguf::load(
            fixture_with_all_overrides(prefix)
                .string(model_key, b"gpt2")
                .strings(tokens_key, &tokens)
                .build(),
        );
        let mut loader = Loader::try_new().unwrap();

        loader.process_event(Load::new(&mut gguf)).unwrap();
        let (ids, flags) = loader
            .process_event(WithInfo::new(|info: event::Info<'_>| {
                (info.special_ids, info.flags)
            }))
            .unwrap();
        assert_eq!(
            [
                ids.bos,
                ids.eos,
                ids.eot,
                ids.eom,
                ids.unknown,
                ids.separator,
                ids.padding,
                ids.classification,
                ids.mask,
                ids.prefix,
                ids.suffix,
                ids.middle,
                ids.fim_pre,
                ids.fim_suf,
                ids.fim_mid,
                ids.fim_pad,
                ids.fim_rep,
                ids.fim_sep,
            ],
            core::array::from_fn::<_, 18, _>(|index| i32::try_from(index).unwrap())
        );
        assert_eq!(
            [
                flags.add_bos,
                flags.add_eos,
                flags.add_sep,
                flags.add_space_prefix,
                flags.remove_extra_whitespaces,
                flags.ignore_merges,
                flags.escape_whitespaces,
                flags.treat_whitespace_as_suffix,
            ],
            [true, false, true, false, true, false, true, false]
        );
    }
}

struct SemanticRng(u64);

impl SemanticRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self, maximum: usize) -> Vec<u8> {
        let length = usize::try_from(self.next() % u64::try_from(maximum + 1).unwrap()).unwrap();
        (0..length).map(|_| self.next().to_le_bytes()[0]).collect()
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn deterministic_semantic_fuzz_matches_every_materialized_family() {
    let mut rng = SemanticRng(0x6d6f_6465_6c76_6f63);
    let mut loader = Loader::try_new().unwrap();
    for _ in 0..96 {
        let token_count = usize::try_from((rng.next() % 8) + 1).unwrap();
        let merge_count = usize::try_from(rng.next() % 7).unwrap();
        let tokens: Vec<Vec<u8>> = (0..token_count).map(|_| rng.bytes(12)).collect();
        let merges: Vec<Vec<u8>> = (0..merge_count).map(|_| rng.bytes(10)).collect();
        let token_refs: Vec<&[u8]> = tokens.iter().map(Vec::as_slice).collect();
        let merge_refs: Vec<&[u8]> = merges.iter().map(Vec::as_slice).collect();
        let scores: Vec<f32> = (0..token_count)
            .map(|_| {
                let signed = i16::from(rng.next().to_le_bytes()[0]) - 128;
                f32::from(signed) / 4.0
            })
            .collect();
        let token_types: Vec<u32> = (0..token_count)
            .map(|_| u32::try_from(rng.next() % 4).unwrap())
            .collect();
        let mut score_bytes = Vec::with_capacity(token_count * 4);
        let mut type_bytes = Vec::with_capacity(token_count * 4);
        for value in &scores {
            score_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &token_types {
            type_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let charmap = rng.bytes(24);
        let bos_override = rng.next() & 1 != 0;
        let eos_override = rng.next() & 1 != 0;
        let token_type_count = u32::try_from((rng.next() % 8) + 1).unwrap();
        let mut gguf = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"tokenizer.model", b"gpt2")
                .strings(b"tokenizer.tokens", &token_refs)
                .array(
                    b"tokenizer.scores",
                    F32,
                    &score_bytes,
                    u64::try_from(token_count).unwrap(),
                )
                .array(
                    b"tokenizer.token_type",
                    U32,
                    &type_bytes,
                    u64::try_from(token_count).unwrap(),
                )
                .scalar(
                    b"tokenizer.token_type_count",
                    U32,
                    token_type_count.to_le_bytes(),
                )
                .strings(b"tokenizer.merges", &merge_refs)
                .array(
                    b"tokenizer.precompiled_charsmap",
                    U8,
                    &charmap,
                    u64::try_from(charmap.len()).unwrap(),
                )
                .scalar(b"tokenizer.add_bos_token", BOOL, [u8::from(bos_override)])
                .scalar(b"tokenizer.add_eos_token", BOOL, [u8::from(eos_override)])
                .build(),
        );
        loader.process_event(Load::new(&mut gguf)).unwrap();
        let info = loader
            .process_event(WithInfo::new(|info: event::Info<'_>| {
                (
                    info.token_count,
                    info.token_type_count,
                    info.merge_count,
                    info.charmap_bytes,
                    info.flags,
                )
            }))
            .unwrap();
        assert_eq!(info.0, u32::try_from(token_count).unwrap());
        assert_eq!(info.1, token_type_count);
        assert_eq!(info.2, u32::try_from(merge_count).unwrap());
        assert_eq!(info.3, u32::try_from(charmap.len()).unwrap());
        assert_eq!(
            (info.4.add_bos, info.4.add_eos),
            (bos_override, eos_override)
        );
        for index in 0..token_count {
            let observed = loader
                .process_event(WithToken::new(
                    u32::try_from(index).unwrap(),
                    |token: event::Token<'_>| {
                        (token.text.to_vec(), token.score.to_bits(), token.r#type)
                    },
                ))
                .unwrap()
                .unwrap();
            assert_eq!(observed.0, tokens[index]);
            assert_eq!(observed.1, scores[index].to_bits());
            assert_eq!(observed.2, i32::try_from(token_types[index]).unwrap());
        }
        for (index, expected) in merges.iter().enumerate() {
            assert_eq!(
                loader
                    .process_event(WithMerge::new(
                        u32::try_from(index).unwrap(),
                        <[u8]>::to_vec,
                    ))
                    .unwrap()
                    .unwrap(),
                *expected
            );
        }
        assert_eq!(
            loader
                .process_event(WithCharmap::new(<[u8]>::to_vec))
                .unwrap(),
            charmap
        );
    }
}
