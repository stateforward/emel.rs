use std::sync::Arc;

use crate::loader::test_gguf::{self, U32, U64};
use allocation_counter::measure;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage as GgufStorage, WithTensor};

use super::event::{
    ContractBegin, ContractReset, ContractVisit, Error, Family, Storage, StorageRelease,
    WithFirstName,
};
use super::{Parameters, Sortformer, load_hparams};

const NAMES: [&[u8]; 5] = [
    b"prep.feat.fb",
    b"enc.l0.conv.dw.w",
    b"mods.ep.w",
    b"te.l0.sa.q.w",
    b"enc.l1.conv.dw.w",
];

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u64(bytes, u64::try_from(value.len()).unwrap());
    bytes.extend_from_slice(value);
}

fn push_kv_string(bytes: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    push_string(bytes, key);
    push_u32(bytes, 8);
    push_string(bytes, value);
}
fn push_kv_u32(bytes: &mut Vec<u8>, key: &[u8], value: u32) {
    push_string(bytes, key);
    push_u32(bytes, U32);
    push_u32(bytes, value);
}

fn tensor_fixture_with(names: &[&[u8]], source: &[u8], speakers: u32) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, u64::try_from(names.len()).unwrap());
    push_u64(&mut bytes, 15);
    push_kv_u32(&mut bytes, b"general.alignment", 1);
    push_kv_string(&mut bytes, b"general.architecture", b"sortformer");
    push_kv_string(&mut bytes, b"sortformer.source.format", source);
    push_kv_string(&mut bytes, b"sortformer.tensor_name_scheme", b"compact_v1");
    push_kv_string(&mut bytes, b"sortformer.outtype", b"f32");
    push_kv_u32(&mut bytes, b"sortformer.original_tensor_count", 128);
    push_kv_u32(&mut bytes, b"sortformer.tensor_count", 132);
    push_kv_u32(&mut bytes, b"sortformer.skipped_tensor_count", 3);
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.preprocessor.sample_rate",
        16_000,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.num_spks",
        speakers,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_len",
        188,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_right_context",
        1,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.fifo_len",
        0,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_update_period",
        188,
    );
    push_kv_u32(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_len",
        188,
    );
    for (index, name) in names.iter().enumerate() {
        push_string(&mut bytes, name);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, u64::try_from(index * 4).unwrap());
    }
    bytes.resize(bytes.len() + names.len() * 4, 0);
    bytes
}

fn tensor_fixture(names: &[&[u8]]) -> Vec<u8> {
    tensor_fixture_with(names, b"nemo", 4)
}

fn parsed_loader(names: &[&[u8]]) -> (GgufLoader, emel_gguf::event::ParseDone) {
    parsed_loader_with(names, b"nemo", 4)
}

fn parsed_loader_with(
    names: &[&[u8]],
    source: &[u8],
    speakers: u32,
) -> (GgufLoader, emel_gguf::event::ParseDone) {
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(tensor_fixture_with(
            names, source, speakers,
        ))))
        .unwrap();
    loader
        .process_event(Bind::new(GgufStorage::exact(probe).unwrap()))
        .unwrap();
    let parsed = loader.process_event(Parse::new()).unwrap();
    (loader, parsed)
}

fn parameters() -> Parameters {
    Parameters {
        original_tensor_count: 128,
        tensor_count: 132,
        skipped_tensor_count: 3,
        ..Parameters::default()
    }
}

fn actor_with(names: &[&[u8]], capacity: usize) -> Sortformer {
    let (loader, parsed) = parsed_loader(names);
    let mut actor = Sortformer::load(
        loader,
        parsed,
        Storage::with_name_capacity(capacity).unwrap(),
    )
    .unwrap();
    actor.process_event(ContractBegin::new()).unwrap();
    actor
}

fn metadata_fixture(source: &[u8], speaker_count: u32) -> GgufLoader {
    test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"sortformer")
            .string(b"sortformer.source.format", source)
            .string(b"sortformer.tensor_name_scheme", b"compact_v1")
            .string(b"sortformer.outtype", b"f32")
            .scalar(
                b"sortformer.original_tensor_count",
                U32,
                128u32.to_le_bytes(),
            )
            .scalar(b"sortformer.tensor_count", U32, 132u32.to_le_bytes())
            .scalar(b"sortformer.skipped_tensor_count", U32, 3u32.to_le_bytes())
            .scalar(
                b"sortformer.config.preprocessor.sample_rate",
                U32,
                16_000u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.num_spks",
                U32,
                speaker_count.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.chunk_len",
                U32,
                188u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.chunk_right_context",
                U32,
                1u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.fifo_len",
                U32,
                0u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.spkcache_update_period",
                U32,
                188u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.spkcache_len",
                U32,
                188u32.to_le_bytes(),
            )
            .build(),
    )
}

fn alias_metadata_fixture(
    sample_key: Option<&[u8]>,
    speaker_key: Option<&[u8]>,
    include_optional_stream_fields: bool,
    include_skipped_count: bool,
) -> GgufLoader {
    let mut fixture = test_gguf::Fixture::new()
        .string(b"general.architecture", b"sortformer")
        .string(b"sortformer.source.format", b"nemo")
        .string(b"sortformer.tensor_name_scheme", b"compact_v1")
        .string(b"sortformer.outtype", b"f32")
        .scalar(
            b"sortformer.original_tensor_count",
            U32,
            128u32.to_le_bytes(),
        )
        .scalar(b"sortformer.tensor_count", U32, 132u32.to_le_bytes());
    if include_skipped_count {
        fixture = fixture.scalar(b"sortformer.skipped_tensor_count", U32, 3u32.to_le_bytes());
    }
    if let Some(key) = sample_key {
        fixture = fixture.scalar(key, U32, 16_000u32.to_le_bytes());
    }
    if let Some(key) = speaker_key {
        fixture = fixture.scalar(key, U32, 4u32.to_le_bytes());
    }
    if include_optional_stream_fields {
        fixture = fixture
            .scalar(
                b"sortformer.config.sortformer_modules.chunk_len",
                U32,
                188u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.chunk_right_context",
                U32,
                1u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.fifo_len",
                U32,
                0u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.spkcache_update_period",
                U32,
                188u32.to_le_bytes(),
            )
            .scalar(
                b"sortformer.config.sortformer_modules.spkcache_len",
                U32,
                188u32.to_le_bytes(),
            );
    }
    test_gguf::load(fixture.build())
}

#[test]
fn loads_source_exact_hparams_aliases_defaults_and_rejects_contract_drift() {
    let mut loader = metadata_fixture(b"nemo", 4);
    assert_eq!(load_hparams(&mut loader).unwrap(), parameters());
    let mut wrong_source = metadata_fixture(b"onnx", 4);
    assert_eq!(
        load_hparams(&mut wrong_source).unwrap_err().kind,
        super::hparams::ErrorKind::Contract
    );
    let mut wrong_speakers = metadata_fixture(b"nemo", 3);
    assert_eq!(
        load_hparams(&mut wrong_speakers).unwrap_err().field,
        super::hparams::Field::SpeakerCount
    );
}

#[test]
fn loads_all_hparam_aliases_and_optional_defaults() {
    let mut sample_fallback = alias_metadata_fixture(
        Some(b"sortformer.config.sample_rate"),
        Some(b"sortformer.config.sortformer_modules.num_speakers"),
        true,
        true,
    );
    assert_eq!(load_hparams(&mut sample_fallback).unwrap(), parameters());

    let mut speaker_fallback = alias_metadata_fixture(
        Some(b"sortformer.config.preprocessor.sample_rate"),
        Some(b"sortformer.config.num_spks"),
        true,
        true,
    );
    assert_eq!(load_hparams(&mut speaker_fallback).unwrap(), parameters());

    let mut defaults = alias_metadata_fixture(None, None, false, false);
    assert_eq!(
        load_hparams(&mut defaults).unwrap(),
        Parameters {
            skipped_tensor_count: 0,
            ..parameters()
        }
    );
}

#[test]
fn reports_public_hparam_missing_wrong_kind_range_and_query_failures() {
    let mut missing = test_gguf::load(test_gguf::Fixture::new().build());
    assert_eq!(
        load_hparams(&mut missing).unwrap_err(),
        super::hparams::Error {
            field: super::hparams::Field::Architecture,
            kind: super::hparams::ErrorKind::Missing,
        }
    );

    let mut wrong_architecture = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"llama")
            .build(),
    );
    assert_eq!(
        load_hparams(&mut wrong_architecture).unwrap_err(),
        super::hparams::Error {
            field: super::hparams::Field::Architecture,
            kind: super::hparams::ErrorKind::Contract,
        }
    );

    let mut wrong_kind = test_gguf::load(
        test_gguf::Fixture::new()
            .scalar(b"general.architecture", U32, 1u32.to_le_bytes())
            .build(),
    );
    assert_eq!(
        load_hparams(&mut wrong_kind).unwrap_err().kind,
        super::hparams::ErrorKind::WrongKind
    );

    let mut range = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"sortformer")
            .string(b"sortformer.source.format", b"nemo")
            .string(b"sortformer.tensor_name_scheme", b"compact_v1")
            .string(b"sortformer.outtype", b"f32")
            .scalar(
                b"sortformer.original_tensor_count",
                U64,
                u64::MAX.to_le_bytes(),
            )
            .build(),
    );
    assert_eq!(
        load_hparams(&mut range).unwrap_err().kind,
        super::hparams::ErrorKind::Range
    );

    let mut unparsed = GgufLoader::new();
    assert_eq!(
        load_hparams(&mut unparsed).unwrap_err().kind,
        super::hparams::ErrorKind::Query
    );
}

#[test]
fn builds_execution_contract_and_exposes_first_names_synchronously() {
    let mut actor = actor_with(&NAMES, NAMES.iter().map(|name| name.len()).sum());
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(format!("{:?}", contract.model()), "ModelIdentity { .. }");
    assert_eq!(contract.sample_rate(), 16_000);
    assert_eq!(contract.speaker_count(), 4);
    assert_eq!(contract.frame_shift_ms(), 80);
    assert_eq!(contract.chunk_len(), 188);
    assert_eq!(contract.chunk_right_context(), 1);
    assert_eq!(contract.fifo_len(), 0);
    assert_eq!(contract.spkcache_update_period(), 188);
    assert_eq!(contract.spkcache_len(), 188);
    let encoder = contract.family(Family::Encoder);
    assert_eq!(encoder.family(), Family::Encoder);
    assert_eq!(encoder.prefix(), b"enc.");
    assert_eq!(encoder.tensor_count(), 2);
    assert_eq!(encoder.first(), contract.family(Family::Encoder).first());
    let expected = [
        b"prep.feat.fb".as_slice(),
        b"enc.l0.conv.dw.w",
        b"mods.ep.w",
        b"te.l0.sa.q.w",
    ];
    for (family, expected) in Family::ALL.into_iter().zip(expected) {
        assert_eq!(actor.first_name(family), expected);
        let result = actor
            .process_event(WithFirstName::new(family, |name: &[u8]| name == expected))
            .unwrap();
        assert_eq!(result, Some(true));
    }
}

#[test]
fn rejects_missing_family_and_source_contract_drift_during_binding() {
    let (missing_loader, missing_parsed) = parsed_loader(&[NAMES[0], NAMES[1], NAMES[3]]);
    let mut missing = Sortformer::load(
        missing_loader,
        missing_parsed,
        Storage::with_name_capacity(64).unwrap(),
    )
    .unwrap();
    assert_eq!(
        missing.process_event(ContractBegin::new()),
        Err(Error::ModelInvalid)
    );
    assert_eq!(
        missing.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    let (wrong_loader, wrong_parsed) = parsed_loader_with(&NAMES, b"onnx", 4);
    assert!(matches!(
        Sortformer::load(
            wrong_loader,
            wrong_parsed,
            Storage::with_name_capacity(64).unwrap()
        ),
        Err(super::event::LoadError::Hparams(_))
    ));
}

#[test]
fn ignores_unusable_and_unowned_tensors_and_fails_closed_on_name_capacity() {
    let ignored_names = [NAMES[0], NAMES[1], NAMES[2], NAMES[3], b"other.weight"];
    let mut ignored = actor_with(
        &ignored_names,
        ignored_names.iter().map(|name| name.len()).sum(),
    );
    let contract = ignored.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.family(Family::FeatureExtractor).tensor_count(), 1);
    assert_eq!(contract.family(Family::Encoder).tensor_count(), 1);
    assert_eq!(contract.family(Family::Modules).tensor_count(), 1);
    assert_eq!(
        contract.family(Family::TransformerEncoder).tensor_count(),
        1
    );

    let names = [b"prep.feat.fb".as_slice(), b"other.weight"];
    let (loader, parsed) = parsed_loader(&names);
    let mut actor =
        Sortformer::load(loader, parsed, Storage::with_name_capacity(1).unwrap()).unwrap();
    assert_eq!(
        actor.process_event(ContractBegin::new()),
        Err(Error::Capacity)
    );
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn canonical_dispatch_is_allocation_free_and_lifecycle_releases_storage() {
    let (loader, parsed) = parsed_loader(&NAMES);
    let mut actor =
        Sortformer::load(loader, parsed, Storage::with_name_capacity(128).unwrap()).unwrap();
    let measured = measure(|| {
        actor.process_event(ContractBegin::new()).unwrap();
        actor.process_event(ContractVisit::new()).unwrap();
    });
    assert_eq!(measured.count_total, 0);
    assert!(matches!(
        actor.process_event(StorageRelease::new()),
        Err(Error::Busy)
    ));
    actor.process_event(ContractReset::new()).unwrap();
    assert_eq!(
        actor
            .process_event(StorageRelease::new())
            .unwrap()
            .name_capacity(),
        128
    );
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::StorageUnavailable)
    );
}

#[test]
fn retains_the_exact_loader_backing_until_the_actor_is_dropped() {
    let bytes: Arc<[u8]> = Arc::from(tensor_fixture(&NAMES));
    let weak = Arc::downgrade(&bytes);
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::clone(&bytes)))
        .unwrap();
    loader
        .process_event(Bind::new(GgufStorage::exact(probe).unwrap()))
        .unwrap();
    let parsed = loader.process_event(Parse::new()).unwrap();
    drop(bytes);

    let actor =
        Sortformer::load(loader, parsed, Storage::with_name_capacity(128).unwrap()).unwrap();
    assert!(weak.upgrade().is_some());
    drop(actor);
    assert!(weak.upgrade().is_none());
}

#[test]
fn protocol_errors_are_typed_and_debug_is_opaque() {
    fn ignore_name(_: &[u8]) {}
    let (loader, parsed) = parsed_loader(&NAMES);
    let mut actor =
        Sortformer::load(loader, parsed, Storage::with_name_capacity(64).unwrap()).unwrap();
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(WithFirstName::new(Family::Encoder, ignore_name)),
        Err(Error::InvalidRequest)
    );
    actor.process_event(ContractBegin::new()).unwrap();
    assert_eq!(actor.process_event(ContractBegin::new()), Err(Error::Busy));
    assert_eq!(format!("{actor:?}"), "Sortformer { .. }");
    assert_eq!(
        format!(
            "{:?}",
            WithFirstName::<_, ()>::new(Family::Encoder, ignore_name)
        ),
        "WithFirstName { family: Encoder, .. }"
    );
    let messages = [
        (Error::InvalidRequest, "invalid Sortformer request"),
        (Error::ModelInvalid, "Sortformer model contract is invalid"),
        (
            Error::Capacity,
            "Sortformer storage capacity is unavailable",
        ),
        (Error::Busy, "Sortformer contract is busy"),
        (
            Error::StorageUnavailable,
            "Sortformer storage is unavailable",
        ),
        (Error::UnexpectedEvent, "unexpected Sortformer event"),
        (Error::Internal, "internal Sortformer error"),
    ];
    for (error, expected) in messages {
        assert_eq!(error.to_string(), expected);
    }
    let load_messages = [
        (
            super::event::LoadError::Hparams(super::hparams::Error {
                field: super::hparams::Field::Architecture,
                kind: super::hparams::ErrorKind::Contract,
            }),
            "Sortformer hyperparameter loading failed",
        ),
        (
            super::event::LoadError::Gguf(emel_gguf::event::QueryError::Internal),
            "Sortformer GGUF observation failed",
        ),
        (
            super::event::LoadError::Catalog(crate::catalog::event::Error::Internal),
            "Sortformer catalog binding failed",
        ),
        (
            super::event::LoadError::Capacity,
            "Sortformer construction capacity is unavailable",
        ),
        (
            super::event::LoadError::Internal,
            "internal Sortformer construction error",
        ),
    ];
    for (error, expected) in load_messages {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn routes_all_additional_families_and_busy_protocol_paths() {
    let names = [
        b"prep.first".as_slice(),
        b"prep.second",
        b"enc.first",
        b"enc.second",
        b"mods.first",
        b"mods.second",
        b"te.first",
        b"te.second",
    ];
    let mut actor = actor_with(&names, names.iter().map(|name| name.len()).sum());
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    for family in Family::ALL {
        assert_eq!(contract.family(family).tensor_count(), 2);
    }
    assert_eq!(actor.process_event(ContractBegin::new()), Err(Error::Busy));
}

#[test]
fn distinguishes_bound_empty_from_released_storage() {
    let (loader, parsed) = parsed_loader(&[b"prep.first"]);
    let mut actor =
        Sortformer::load(loader, parsed, Storage::with_name_capacity(32).unwrap()).unwrap();
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(WithFirstName::new(Family::Encoder, |_: &[u8]| ())),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor
            .process_event(StorageRelease::new())
            .unwrap()
            .name_capacity(),
        32
    );
    actor.process_event(ContractReset::new()).unwrap();
    assert!(matches!(
        actor.process_event(StorageRelease::new()),
        Err(Error::StorageUnavailable)
    ));
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::StorageUnavailable)
    );
    assert_eq!(
        actor.process_event(WithFirstName::new(Family::Encoder, |_: &[u8]| ())),
        Err(Error::StorageUnavailable)
    );
    assert_eq!(
        Storage::with_name_capacity(usize::MAX).unwrap_err(),
        Error::Capacity
    );
}

#[test]
fn gguf_capture_callback_is_allocation_free_with_preallocated_storage() {
    let (mut loader, parsed) = parsed_loader(&[b"prep.first"]);
    let mut names = [0_u8; 10];
    let measured = measure(|| {
        loader
            .process_event(WithTensor::new(
                0,
                |name: &[u8], descriptor, payload: &[u8]| {
                    super::actor::capture_source_observation(
                        &mut names, 0, name, descriptor, payload,
                    )
                },
            ))
            .unwrap()
            .unwrap()
            .unwrap();
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(names.as_slice(), b"prep.first");
    assert_eq!(parsed.tensor_count(), 1);
}
