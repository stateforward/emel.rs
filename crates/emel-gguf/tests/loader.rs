//! End-to-end coverage for the public GGUF loader lifecycle.

use emel_gguf::Loader;
use emel_gguf::event::{Bind, Error, Load, Parse, ParseDone, Probe, ProbeDone};
use sml as _;

const MAGIC: [u8; 4] = *b"GGUF";
const VERSION: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: u32 = 32;
const ALIGNMENT_KEY: &str = "general.alignment";
const TOKENS_KEY: &str = "tokenizer.tokens";
const TENSOR_NAME: &str = "weights.f32";

trait LoaderTestExt {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error>;
    fn bind(&mut self) -> Result<(), Error>;
    fn bind_with_capacity(
        &mut self,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error>;
    fn parse<'a>(&mut self, file_image: &'a [u8]) -> Result<ParseDone<'a>, Error>;
}

impl LoaderTestExt for Loader {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error> {
        self.process_event(Probe::new(file_image))
    }

    fn bind(&mut self) -> Result<(), Error> {
        self.process_event(Bind::exact())
    }

    fn bind_with_capacity(
        &mut self,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error> {
        self.process_event(Bind::with_capacity(
            metadata_bytes,
            metadata_entries,
            tensors,
        ))
    }

    fn parse<'a>(&mut self, file_image: &'a [u8]) -> Result<ParseDone<'a>, Error> {
        self.process_event(Parse::new(file_image))
    }
}

fn load(file_image: &[u8]) -> Result<ParseDone<'_>, Error> {
    Loader::new().process_event(Load::new(file_image))
}

fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_string(bytes: &mut Vec<u8>, value: &str) {
    append_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn append_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
    append_string(bytes, key);
    append_u32(bytes, TYPE_UINT32);
    append_u32(bytes, value);
}

fn append_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
    append_string(bytes, key);
    append_u32(bytes, TYPE_ARRAY);
    append_u32(bytes, TYPE_STRING);
    append_u64(bytes, values.len() as u64);
    for value in values {
        append_string(bytes, value);
    }
}

fn append_tensor(bytes: &mut Vec<u8>, name: &str, dimensions: &[u64]) {
    append_typed_tensor(bytes, name, dimensions, GGML_TYPE_F32, 0);
}

fn append_typed_tensor(
    bytes: &mut Vec<u8>,
    name: &str,
    dimensions: &[u64],
    tensor_type: u32,
    offset: u64,
) {
    append_string(bytes, name);
    append_u32(
        bytes,
        u32::try_from(dimensions.len()).expect("fixture dimension count fits u32"),
    );
    for dimension in dimensions {
        append_u64(bytes, *dimension);
    }
    append_u32(bytes, tensor_type);
    append_u64(bytes, offset);
}

fn empty_gguf(version: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, version);
    append_u64(&mut bytes, 0);
    append_u64(&mut bytes, 0);
    bytes
}

fn single_tensor_gguf(tensor_type: u32, block_size: u64, type_size: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 1);
    append_u64(&mut bytes, 0);
    append_typed_tensor(&mut bytes, "tensor", &[block_size], tensor_type, 0);
    while !bytes.len().is_multiple_of(ALIGNMENT as usize) {
        bytes.push(0);
    }
    let padded_size = type_size.next_multiple_of(ALIGNMENT as usize);
    bytes.resize(bytes.len() + padded_size, 0);
    bytes
}

fn make_gguf(tokens: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 1);
    append_u64(&mut bytes, 2);
    append_kv_u32(&mut bytes, ALIGNMENT_KEY, ALIGNMENT);
    append_kv_string_array(&mut bytes, TOKENS_KEY, tokens);
    append_tensor(&mut bytes, TENSOR_NAME, &[2, 3]);
    let alignment = usize::try_from(ALIGNMENT).expect("alignment fits usize");
    while !bytes.len().is_multiple_of(alignment) {
        bytes.push(0);
    }
    bytes.resize(bytes.len() + alignment, 0);
    bytes
}

fn valid_gguf() -> Vec<u8> {
    make_gguf(&["hi", "world"])
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte value"))
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte value"))
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn probe_bind_parse_lifecycle_populates_bound_storage() {
    let file = valid_gguf();
    let mut loader = Loader::new();

    let requirements = loader.probe(&file).expect("valid probe");
    assert_eq!(requirements.tensor_count(), 1);
    assert_eq!(requirements.metadata_count(), 2);
    assert_eq!(requirements.max_key_bytes(), 17);
    assert_eq!(requirements.max_value_bytes(), 35);
    assert_eq!(requirements.tensor_data_bytes(), 24);

    loader.bind().expect("storage binds");
    let model = loader.parse(&file).expect("valid parse");

    let entries = model.metadata().collect::<Vec<_>>();
    assert_eq!(entries[0].key(), ALIGNMENT_KEY.as_bytes());
    assert_eq!(entries[0].value_type(), TYPE_UINT32);
    assert_eq!(read_u32(entries[0].value()), ALIGNMENT);
    assert_eq!(entries[1].key(), TOKENS_KEY.as_bytes());
    assert_eq!(entries[1].value_type(), TYPE_ARRAY);
    assert_eq!(entries[1].value().len(), 35);
    let tokens = entries[1].value();
    assert_eq!(read_u32(&tokens[..4]), TYPE_STRING);
    assert_eq!(read_u64(&tokens[4..12]), 2);

    let tensor = model.tensors().next().expect("one tensor");
    assert_eq!(tensor.name(), TENSOR_NAME.as_bytes());
    assert_eq!(tensor.tensor_type(), GGML_TYPE_F32);
    assert_eq!(tensor.dimension_count(), 2);
    assert_eq!(tensor.dimensions(), [2, 3, 1, 1]);
    assert_eq!(tensor.data_offset(), 0);
    assert_eq!(tensor.data_size(), 24);
    assert!(tensor.file_offset() > tensor.data_offset());
    assert_eq!(tensor.data().len(), 24);
}

#[test]
fn probe_rejects_empty_input_and_can_recover() {
    let mut loader = Loader::new();
    assert_eq!(loader.probe(&[]), Err(Error::InvalidRequest));
    assert!(loader.probe(&valid_gguf()).is_ok());
}

#[test]
fn probe_classifies_malformed_images() {
    let mut bad_magic = valid_gguf();
    bad_magic[0] = b'F';
    assert_eq!(Loader::new().probe(&bad_magic), Err(Error::ModelInvalid));

    let mut bad_version = valid_gguf();
    bad_version[4..8].copy_from_slice(&4_u32.to_le_bytes());
    assert_eq!(Loader::new().probe(&bad_version), Err(Error::ModelInvalid));

    let mut truncated = valid_gguf();
    truncated.pop();
    assert_eq!(Loader::new().probe(&truncated), Err(Error::ParseFailed));
}

#[test]
fn bind_and_parse_report_capacity_and_format_failures() {
    let valid = valid_gguf();
    let mut loader = Loader::new();
    let requirements = loader.probe(&valid).unwrap();
    assert_eq!(
        loader.bind_with_capacity(
            requirements.required_metadata_bytes().unwrap() - 1,
            requirements.metadata_count() as usize,
            requirements.tensor_count() as usize,
        ),
        Err(Error::Capacity)
    );

    loader.probe(&valid).unwrap();
    loader.bind().unwrap();
    let larger_values = make_gguf(&["hello", "world-with-extra-bytes"]);
    assert!(matches!(loader.parse(&larger_values), Err(Error::Capacity)));

    loader.probe(&valid).unwrap();
    loader.bind().unwrap();
    let mut truncated = valid;
    truncated.pop();
    assert!(matches!(loader.parse(&truncated), Err(Error::ParseFailed)));
}

#[test]
fn parse_revalidates_an_image_after_storage_was_bound() {
    let valid = valid_gguf();
    let mut loader = Loader::new();
    loader.probe(&valid).unwrap();
    loader.bind().unwrap();

    let mut duplicate_keys = Vec::new();
    duplicate_keys.extend_from_slice(&MAGIC);
    append_u32(&mut duplicate_keys, VERSION);
    append_u64(&mut duplicate_keys, 1);
    append_u64(&mut duplicate_keys, 2);
    append_kv_u32(&mut duplicate_keys, "duplicate", 1);
    append_kv_u32(&mut duplicate_keys, "duplicate", 2);
    append_tensor(&mut duplicate_keys, TENSOR_NAME, &[2, 3]);
    while !duplicate_keys
        .len()
        .is_multiple_of(usize::try_from(ALIGNMENT).unwrap())
    {
        duplicate_keys.push(0);
    }
    duplicate_keys.resize(
        duplicate_keys.len() + usize::try_from(ALIGNMENT).unwrap(),
        0,
    );

    assert!(matches!(
        loader.parse(&duplicate_keys),
        Err(Error::ModelInvalid)
    ));
}

#[test]
fn one_shot_load_parses_a_valid_image() {
    let file = valid_gguf();
    let model = load(&file).expect("one-shot load");
    assert_eq!(model.probe().tensor_count(), 1);
    assert_eq!(model.tensors().len(), 1);
}

#[test]
fn public_diagnostics_and_event_views_are_stable() {
    let loader = Loader::default();
    assert_eq!(format!("{loader:?}"), "Loader { .. }");

    for (error, message) in [
        (Error::InvalidRequest, "invalid GGUF loader request"),
        (Error::ModelInvalid, "invalid or unsupported GGUF model"),
        (Error::Capacity, "GGUF loader capacity exceeded"),
        (Error::ParseFailed, "failed to parse GGUF image"),
        (Error::Internal, "internal GGUF loader error"),
        (Error::Untracked, "untracked GGUF loader error"),
    ] {
        assert_eq!(error.to_string(), message);
    }

    let file = valid_gguf();
    let model = load(&file).expect("valid model");
    assert_eq!(model.metadata().len(), 2);
    assert_eq!(model.tensors().len(), 1);
}

#[test]
fn accepts_llama_cpp_supported_versions_and_metadata_only_files() {
    for version in [2, 3] {
        let file = empty_gguf(version);
        let model = load(&file).expect("metadata-only GGUF loads");
        assert_eq!(model.probe().tensor_count(), 0);
        assert_eq!(model.probe().metadata_count(), 0);
        assert_eq!(model.tensors().len(), 0);
        assert_eq!(model.metadata().len(), 0);
    }
}

#[test]
fn supports_current_llama_cpp_quantized_tensor_layouts() {
    for (tensor_type, block_size, type_size) in [
        (0, 1, 4),
        (1, 1, 2),
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 36),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (16, 256, 66),
        (17, 256, 74),
        (18, 256, 98),
        (19, 256, 50),
        (20, 32, 18),
        (21, 256, 110),
        (22, 256, 82),
        (23, 256, 136),
        (24, 1, 1),
        (25, 1, 2),
        (26, 1, 4),
        (27, 1, 8),
        (28, 1, 8),
        (29, 256, 56),
        (30, 1, 2),
        (34, 256, 54),
        (35, 256, 66),
        (39, 32, 17),
    ] {
        let file = single_tensor_gguf(tensor_type, block_size, type_size);
        let model = load(&file).expect("llama.cpp tensor type loads");
        let tensor = model.tensors().next().expect("one tensor");
        assert_eq!(tensor.tensor_type(), tensor_type);
        assert_eq!(tensor.data_size(), type_size as u64);
    }
}

#[test]
fn rejects_tensor_types_after_the_pinned_llama_cpp_type_count() {
    let file = single_tensor_gguf(40, 64, 36);
    assert!(matches!(load(&file), Err(Error::ModelInvalid)));
}

#[test]
fn impossible_declared_counts_fail_without_declared_size_allocations() {
    for (tensor_count, kv_count) in [(u64::from(u32::MAX), 0), (0, u64::from(u32::MAX))] {
        let mut file = Vec::new();
        file.extend_from_slice(&MAGIC);
        append_u32(&mut file, VERSION);
        append_u64(&mut file, tensor_count);
        append_u64(&mut file, kv_count);
        assert_eq!(Loader::new().probe(&file), Err(Error::ParseFailed));
    }
}

#[test]
fn rejects_empty_and_duplicate_metadata_keys() {
    let mut empty_key = Vec::new();
    empty_key.extend_from_slice(&MAGIC);
    append_u32(&mut empty_key, VERSION);
    append_u64(&mut empty_key, 0);
    append_u64(&mut empty_key, 1);
    append_kv_u32(&mut empty_key, "", 1);
    assert!(load(&empty_key).is_err());

    let mut duplicate = Vec::new();
    duplicate.extend_from_slice(&MAGIC);
    append_u32(&mut duplicate, VERSION);
    append_u64(&mut duplicate, 0);
    append_u64(&mut duplicate, 2);
    append_kv_u32(&mut duplicate, "duplicate", 1);
    append_kv_u32(&mut duplicate, "duplicate", 2);
    assert!(load(&duplicate).is_err());
}

#[test]
fn rejects_duplicate_and_oversized_tensor_names() {
    let mut duplicate = Vec::new();
    duplicate.extend_from_slice(&MAGIC);
    append_u32(&mut duplicate, VERSION);
    append_u64(&mut duplicate, 2);
    append_u64(&mut duplicate, 0);
    append_typed_tensor(&mut duplicate, "duplicate", &[1], GGML_TYPE_F32, 0);
    append_typed_tensor(&mut duplicate, "duplicate", &[1], GGML_TYPE_F32, 32);
    while !duplicate.len().is_multiple_of(ALIGNMENT as usize) {
        duplicate.push(0);
    }
    duplicate.resize(duplicate.len() + 64, 0);
    assert!(load(&duplicate).is_err());

    let mut oversized = Vec::new();
    oversized.extend_from_slice(&MAGIC);
    append_u32(&mut oversized, VERSION);
    append_u64(&mut oversized, 1);
    append_u64(&mut oversized, 0);
    append_typed_tensor(&mut oversized, &"x".repeat(64), &[1], GGML_TYPE_F32, 0);
    while !oversized.len().is_multiple_of(ALIGNMENT as usize) {
        oversized.push(0);
    }
    oversized.resize(oversized.len() + ALIGNMENT as usize, 0);
    assert!(load(&oversized).is_err());
}

#[test]
#[ignore = "set EMEL_GGUF_FIXTURE to exercise a real model without vendoring it"]
fn probes_an_external_real_model_fixture() {
    let path = std::env::var_os("EMEL_GGUF_FIXTURE").expect("EMEL_GGUF_FIXTURE is set");
    let bytes = std::fs::read(path).expect("fixture is readable");
    let model = load(&bytes).expect("fixture loads");
    let requirements = model.probe();
    assert!(requirements.tensor_count() > 0);
    assert!(requirements.metadata_count() > 0);
    assert!(requirements.max_key_bytes() > 0);
    assert!(requirements.max_value_bytes() > 0);
    assert_eq!(
        model.tensors().len(),
        usize::try_from(requirements.tensor_count()).unwrap()
    );
    assert_eq!(
        model.metadata().len(),
        usize::try_from(requirements.metadata_count()).unwrap()
    );
}
