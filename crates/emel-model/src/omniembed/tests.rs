use std::sync::Arc;

use allocation_counter::measure;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage as GgufStorage, WithTensor};

use crate::loader::test_gguf::{self, I16, I64, STRING, U32, U64};

use super::event::{
    ContractBegin, ContractReset, ContractVisit, EncoderName, Error, Family, Storage,
    StorageRelease, WithFirstName,
};
use super::{OmniEmbed, Parameters, load_hparams};

const NAMES: [&[u8]; 7] = [
    b"text_encoder.backbone.weight",
    b"text_projection.project.weight",
    b"image_encoder.stem.weight",
    b"image_projection.project.weight",
    b"audio_encoder.stem.weight",
    b"audio_projection.project.weight",
    b"text_encoder.second.weight",
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
    push_u32(bytes, STRING);
    push_string(bytes, value);
}
fn push_kv_u32(bytes: &mut Vec<u8>, key: &[u8], value: u32) {
    push_string(bytes, key);
    push_u32(bytes, U32);
    push_u32(bytes, value);
}
fn push_kv_dims(bytes: &mut Vec<u8>, values: &[u32]) {
    push_string(bytes, b"omniembed.matryoshka_dims");
    push_u32(bytes, 9);
    push_u32(bytes, U32);
    push_u64(bytes, u64::try_from(values.len()).unwrap());
    for value in values {
        push_u32(bytes, *value);
    }
}

fn tensor_fixture(names: &[&[u8]], unusable_index: Option<usize>) -> Vec<u8> {
    tensor_fixture_with_hparams(
        names,
        unusable_index,
        super::IMAGE_ENCODER_NAME,
        super::AUDIO_ENCODER_NAME,
        &[768, 512, 256, 128],
    )
}

fn tensor_fixture_with_hparams(
    names: &[&[u8]],
    unusable_index: Option<usize>,
    image_encoder_name: &[u8],
    audio_encoder_name: &[u8],
    matryoshka_dimensions: &[u32],
) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, u64::try_from(names.len()).unwrap());
    push_u64(&mut bytes, 8);
    push_kv_u32(&mut bytes, b"general.alignment", 1);
    push_kv_string(&mut bytes, b"general.architecture", b"omniembed");
    push_kv_u32(&mut bytes, b"omniembed.embed_dim", 1280);
    push_kv_string(
        &mut bytes,
        b"omniembed.image_encoder_name",
        image_encoder_name,
    );
    push_kv_u32(&mut bytes, b"omniembed.image_encoder_dim", 640);
    push_kv_string(
        &mut bytes,
        b"omniembed.audio_encoder_name",
        audio_encoder_name,
    );
    push_kv_u32(&mut bytes, b"omniembed.audio_encoder_dim", 768);
    push_kv_dims(&mut bytes, matryoshka_dimensions);
    for (index, name) in names.iter().enumerate() {
        push_string(&mut bytes, name);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, u64::from(unusable_index != Some(index)));
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, u64::try_from(index * 4).unwrap());
    }
    bytes.resize(bytes.len() + names.len() * 4, 0);
    bytes
}

fn parsed_loader_from_bytes(bytes: Vec<u8>) -> (GgufLoader, emel_gguf::event::ParseDone) {
    let mut loader = GgufLoader::new();
    let probe = loader.process_event(Probe::new(Arc::from(bytes))).unwrap();
    loader
        .process_event(Bind::new(GgufStorage::exact(probe).unwrap()))
        .unwrap();
    let parsed = loader.process_event(Parse::new()).unwrap();
    (loader, parsed)
}

fn parsed_loader_with(
    names: &[&[u8]],
    unusable_index: Option<usize>,
) -> (GgufLoader, emel_gguf::event::ParseDone) {
    parsed_loader_from_bytes(tensor_fixture(names, unusable_index))
}

fn parsed_loader(names: &[&[u8]]) -> (GgufLoader, emel_gguf::event::ParseDone) {
    parsed_loader_with(names, None)
}

fn expected_parameters() -> Parameters {
    let mut dimensions = [0; super::MAX_MATRYOSHKA_DIMENSIONS];
    dimensions[..4].copy_from_slice(&[768, 512, 256, 128]);
    Parameters {
        embedding_length: 1280,
        image_encoder_length: 640,
        audio_encoder_length: 768,
        matryoshka_dimension_count: 4,
        matryoshka_dimensions: dimensions,
        image_encoder: EncoderName::MobileNetV4Medium,
        audio_encoder: EncoderName::EfficientAtMn20As,
    }
}

fn metadata_fixture(array_kind: u32, array_payload: &[u8], count: u64) -> GgufLoader {
    test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", super::IMAGE_ENCODER_NAME)
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", super::AUDIO_ENCODER_NAME)
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(
                b"omniembed.matryoshka_dims",
                array_kind,
                array_payload,
                count,
            )
            .build(),
    )
}

fn u32_payload(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn i16_payload(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn i64_payload(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn actor_with(names: &[&[u8]], capacity: usize) -> OmniEmbed {
    let (loader, parsed) = parsed_loader(names);
    let mut actor = OmniEmbed::load(
        loader,
        parsed,
        Storage::with_name_capacity(capacity).unwrap(),
    )
    .unwrap();
    actor.process_event(ContractBegin::new()).unwrap();
    actor
}

#[test]
fn loads_source_exact_hparams_and_preprocessing_contracts() {
    let payload = u32_payload(&[768, 512, 256, 128]);
    let mut loader = metadata_fixture(U32, &payload, 4);
    assert_eq!(load_hparams(&mut loader).unwrap(), expected_parameters());

    let mut unknown_names = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", b"other-image")
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", b"other-audio")
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(b"omniembed.matryoshka_dims", U32, &payload, 4)
            .build(),
    );
    let unknown = load_hparams(&mut unknown_names).unwrap();
    assert_eq!(unknown.image_encoder, EncoderName::Other);
    assert_eq!(unknown.audio_encoder, EncoderName::Other);
}

#[test]
fn encoder_names_preserve_empty_and_shared_metadata_capacity_semantics() {
    let payload = u32_payload(&[768, 512, 256, 128]);
    let mut empty = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", b"")
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", b"")
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(b"omniembed.matryoshka_dims", U32, &payload, 4)
            .build(),
    );
    let parameters = load_hparams(&mut empty).unwrap();
    assert_eq!(parameters.image_encoder, EncoderName::Missing);
    assert_eq!(parameters.audio_encoder, EncoderName::Missing);

    let boundary_name = vec![b'x'; 4 * 1024 * 1024];
    let mut exact_boundary = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", &boundary_name)
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", b"")
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(b"omniembed.matryoshka_dims", U32, &payload, 4)
            .build(),
    );
    let parameters = load_hparams(&mut exact_boundary).unwrap();
    assert_eq!(parameters.image_encoder, EncoderName::Other);
    assert_eq!(parameters.audio_encoder, EncoderName::Missing);

    let mut combined_overflow = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", &boundary_name)
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", b"x")
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(b"omniembed.matryoshka_dims", U32, &payload, 4)
            .build(),
    );
    assert_eq!(
        load_hparams(&mut combined_overflow).unwrap_err(),
        super::hparams::Error {
            field: super::hparams::Field::AudioEncoderName,
            kind: super::hparams::ErrorKind::Capacity,
        }
    );
}

#[test]
fn accepts_all_source_integer_array_widths_and_missing_array() {
    let i16_values = i16_payload(&[768, 512, 256, 128]);
    let mut i16_loader = metadata_fixture(I16, &i16_values, 4);
    assert_eq!(
        load_hparams(&mut i16_loader).unwrap(),
        expected_parameters()
    );

    let i64_values = i64_payload(&[1024, 512, 256]);
    let mut i64_loader = metadata_fixture(I64, &i64_values, 3);
    let i64_parameters = load_hparams(&mut i64_loader).unwrap();
    assert_eq!(i64_parameters.matryoshka_dimension_count, 3);
    assert_eq!(
        &i64_parameters.matryoshka_dimensions[..3],
        &[1024, 512, 256]
    );

    let mut missing = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .build(),
    );
    let missing = load_hparams(&mut missing).unwrap();
    assert_eq!(missing.matryoshka_dimension_count, 0);
    assert_eq!(missing.image_encoder, EncoderName::Missing);
    assert_eq!(missing.audio_encoder, EncoderName::Missing);
}

#[test]
fn reports_typed_hparam_contract_kind_range_and_query_failures() {
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
        load_hparams(&mut wrong_architecture).unwrap_err().kind,
        super::hparams::ErrorKind::Contract
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
    let mut scalar_range = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U64, u64::MAX.to_le_bytes())
            .build(),
    );
    assert_eq!(
        load_hparams(&mut scalar_range).unwrap_err().kind,
        super::hparams::ErrorKind::Range
    );
    let too_many = u32_payload(&[1; 17]);
    let mut count_range = metadata_fixture(U32, &too_many, 17);
    assert_eq!(
        load_hparams(&mut count_range).unwrap_err().field,
        super::hparams::Field::MatryoshkaDimensions
    );
    let hostile_count = 65_536_u64;
    let hostile_values = vec![1_u32; usize::try_from(hostile_count).unwrap()];
    let hostile_payload = u32_payload(&hostile_values);
    let mut hostile = metadata_fixture(U32, &hostile_payload, hostile_count);
    let measured = measure(|| {
        assert_eq!(
            load_hparams(&mut hostile).unwrap_err(),
            super::hparams::Error {
                field: super::hparams::Field::MatryoshkaDimensions,
                kind: super::hparams::ErrorKind::Range,
            }
        );
    });
    assert_eq!(measured.count_total, 0);
    let negative = i16_payload(&[768, -1]);
    let mut value_range = metadata_fixture(I16, &negative, 2);
    assert_eq!(
        load_hparams(&mut value_range).unwrap_err().kind,
        super::hparams::ErrorKind::Range
    );
    let mut unparsed = GgufLoader::new();
    assert_eq!(
        load_hparams(&mut unparsed).unwrap_err().kind,
        super::hparams::ErrorKind::Query
    );
}

#[test]
fn builds_exact_multimodal_contract_and_first_names() {
    let mut actor = actor_with(&NAMES[..6], NAMES[..6].iter().map(|name| name.len()).sum());
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(format!("{:?}", contract.model()), "ModelIdentity { .. }");
    assert_eq!(contract.embedding_length(), 1280);
    assert_eq!(contract.image_encoder_length(), 640);
    assert_eq!(contract.audio_encoder_length(), 768);
    assert_eq!(contract.matryoshka_dimension_count(), 4);
    assert_eq!(contract.matryoshka_dimensions()[1], 512);
    assert_eq!(contract.image_size(), 384);
    assert_eq!(
        contract.image_mean().map(f32::to_bits),
        [0.485_f32, 0.456, 0.406].map(f32::to_bits)
    );
    assert_eq!(
        contract.image_std().map(f32::to_bits),
        [0.229_f32, 0.224, 0.225].map(f32::to_bits)
    );
    assert_eq!(contract.audio_sample_rate(), 32_000);
    assert_eq!(contract.audio_n_fft(), 1024);
    assert_eq!(contract.audio_win_length(), 800);
    assert_eq!(contract.audio_hop_size(), 320);
    assert_eq!(contract.audio_num_mel_bins(), 128);
    assert_eq!(
        contract.audio_high_frequency().to_bits(),
        15_000.0_f32.to_bits()
    );
    assert_eq!(contract.audio_preemphasis().to_bits(), 0.97_f32.to_bits());
    assert_eq!(contract.audio_log_offset().to_bits(), 1.0e-5_f32.to_bits());
    assert_eq!(contract.audio_normalize_bias().to_bits(), 4.5_f32.to_bits());
    assert_eq!(
        contract.audio_normalize_scale().to_bits(),
        5.0_f32.to_bits()
    );
}

#[test]
fn builds_exact_family_descriptors_and_first_names() {
    let mut actor = actor_with(&NAMES[..6], NAMES[..6].iter().map(|name| name.len()).sum());
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    for (family, expected) in Family::ALL.into_iter().zip(NAMES) {
        let descriptor = contract.family(family);
        assert_eq!(descriptor.family(), family);
        assert_eq!(descriptor.prefix(), family.prefix());
        assert_eq!(descriptor.tensor_count(), 1);
        assert!(descriptor.first().data_size() > 0);
        assert_eq!(actor.first_name(family), expected);
        assert_eq!(
            actor
                .process_event(WithFirstName::new(family, |name: &[u8]| name == expected))
                .unwrap(),
            Some(true)
        );
    }
}

#[test]
fn public_errors_debug_and_name_query_lifecycle_are_stable() {
    let runtime_messages = [
        (Error::InvalidRequest, "invalid OmniEmbed request"),
        (Error::ModelInvalid, "OmniEmbed model contract is invalid"),
        (Error::Capacity, "OmniEmbed storage capacity is unavailable"),
        (Error::Busy, "OmniEmbed contract is busy"),
        (
            Error::StorageUnavailable,
            "OmniEmbed storage is unavailable",
        ),
        (Error::UnexpectedEvent, "unexpected OmniEmbed event"),
        (Error::Internal, "internal OmniEmbed error"),
    ];
    for (error, expected) in runtime_messages {
        assert_eq!(error.to_string(), expected);
    }

    let load_messages = [
        (
            super::event::LoadError::Hparams(super::hparams::Error {
                field: super::hparams::Field::Architecture,
                kind: super::hparams::ErrorKind::Missing,
            }),
            "OmniEmbed hyperparameter loading failed",
        ),
        (
            super::event::LoadError::Gguf(emel_gguf::event::QueryError::NotParsed),
            "OmniEmbed GGUF observation failed",
        ),
        (
            super::event::LoadError::Catalog(crate::catalog::event::Error::InvalidRequest),
            "OmniEmbed catalog binding failed",
        ),
        (
            super::event::LoadError::Capacity,
            "OmniEmbed construction capacity is unavailable",
        ),
        (
            super::event::LoadError::Internal,
            "internal OmniEmbed construction error",
        ),
    ];
    for (error, expected) in load_messages {
        assert_eq!(error.to_string(), expected);
    }

    let (loader, parsed) = parsed_loader(&NAMES[..6]);
    let mut actor =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    let request = WithFirstName::new(Family::TextEncoder, <[u8]>::len);
    assert_eq!(
        format!("{request:?}"),
        "WithFirstName { family: TextEncoder, .. }"
    );
    assert_eq!(actor.process_event(request), Err(Error::InvalidRequest));
    actor.process_event(StorageRelease::new()).unwrap();
    assert_eq!(
        actor.process_event(WithFirstName::new(Family::TextEncoder, <[u8]>::len)),
        Err(Error::StorageUnavailable)
    );
}

#[test]
fn rejects_missing_family_invalid_matryoshka_and_unknown_encoders() {
    let (loader, parsed) = parsed_loader(&NAMES[..5]);
    let mut missing =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert_eq!(
        missing.process_event(ContractBegin::new()),
        Err(Error::ModelInvalid)
    );
    let (loader, parsed) = parsed_loader_from_bytes(tensor_fixture_with_hparams(
        &NAMES[..6],
        None,
        super::IMAGE_ENCODER_NAME,
        super::AUDIO_ENCODER_NAME,
        &[768, 1536, 256, 128],
    ));
    let mut invalid_dimensions =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert_eq!(
        invalid_dimensions.process_event(ContractBegin::new()),
        Err(Error::ModelInvalid)
    );

    let payload = u32_payload(&[768, 512, 256, 128]);
    let mut unknown = test_gguf::load(
        test_gguf::Fixture::new()
            .string(b"general.architecture", b"omniembed")
            .scalar(b"omniembed.embed_dim", U32, 1280u32.to_le_bytes())
            .string(b"omniembed.image_encoder_name", b"unknown")
            .scalar(b"omniembed.image_encoder_dim", U32, 640u32.to_le_bytes())
            .string(b"omniembed.audio_encoder_name", b"unknown")
            .scalar(b"omniembed.audio_encoder_dim", U32, 768u32.to_le_bytes())
            .array(b"omniembed.matryoshka_dims", U32, &payload, 4)
            .build(),
    );
    assert_eq!(
        load_hparams(&mut unknown).unwrap().image_encoder,
        EncoderName::Other
    );

    let (loader, parsed) = parsed_loader_from_bytes(tensor_fixture_with_hparams(
        &NAMES[..6],
        None,
        b"unknown",
        b"unknown",
        &[768, 512, 256, 128],
    ));
    let mut unknown =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert_eq!(
        unknown.process_event(ContractBegin::new()),
        Err(Error::ModelInvalid)
    );

    let (loader, parsed) = parsed_loader_from_bytes(tensor_fixture_with_hparams(
        &NAMES[..6],
        None,
        b"",
        b"",
        &[768, 512, 256, 128],
    ));
    let mut empty_names =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert_eq!(
        empty_names.process_event(ContractBegin::new()),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn ignores_unusable_family_storage_and_routes_additional_tensors() {
    let (loader, parsed) = parsed_loader_with(&NAMES, Some(6));
    let mut actor =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    actor.process_event(ContractBegin::new()).unwrap();
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.family(Family::TextEncoder).tensor_count(), 1);

    let mut actor = actor_with(&NAMES, 256);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.family(Family::TextEncoder).tensor_count(), 2);
    for family in Family::ALL.into_iter().skip(1) {
        assert_eq!(contract.family(family).tensor_count(), 1);
    }
}

#[test]
fn dispatch_is_allocation_free_and_storage_lifecycle_is_typed() {
    let (loader, parsed) = parsed_loader(&NAMES[..6]);
    let mut actor =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    let measured = measure(|| {
        actor.process_event(ContractBegin::new()).unwrap();
        actor.process_event(ContractVisit::new()).unwrap();
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(actor.process_event(ContractBegin::new()), Err(Error::Busy));
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
        256
    );
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::StorageUnavailable)
    );
    actor.process_event(ContractReset::new()).unwrap();
    assert!(matches!(
        actor.process_event(StorageRelease::new()),
        Err(Error::StorageUnavailable)
    ));
}

#[test]
fn exact_loader_backing_is_retained_and_public_debug_is_opaque() {
    let bytes: Arc<[u8]> = Arc::from(tensor_fixture(&NAMES[..6], None));
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
    let actor = OmniEmbed::load(loader, parsed, Storage::with_name_capacity(256).unwrap()).unwrap();
    assert!(weak.upgrade().is_some());
    assert_eq!(format!("{actor:?}"), "OmniEmbed { .. }");
    drop(actor);
    assert!(weak.upgrade().is_none());
}

#[test]
fn capacity_and_capture_boundaries_fail_closed_without_dispatch_allocation() {
    let (loader, parsed) = parsed_loader(&NAMES[..6]);
    let mut actor =
        OmniEmbed::load(loader, parsed, Storage::with_name_capacity(1).unwrap()).unwrap();
    assert_eq!(
        actor.process_event(ContractBegin::new()),
        Err(Error::Capacity)
    );
    assert_eq!(
        Storage::with_name_capacity(usize::MAX).unwrap_err(),
        Error::Capacity
    );

    let (mut loader, parsed) = parsed_loader(&NAMES[..1]);
    let mut name = [0_u8; 28];
    let measured = measure(|| {
        loader
            .process_event(WithTensor::new(
                0,
                |source: &[u8], descriptor, payload: &[u8]| {
                    super::actor::capture_source_observation(
                        &mut name, 0, source, descriptor, payload,
                    )
                },
            ))
            .unwrap()
            .unwrap()
            .unwrap();
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(name.as_slice(), NAMES[0]);
    assert_eq!(parsed.tensor_count(), 1);
}
