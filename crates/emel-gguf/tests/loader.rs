//! End-to-end proof for the public GGUF actor lifecycle and typed metadata API.

use allocation_counter::measure;
use emel_gguf::Loader;
use emel_gguf::event::{
    Bind, ElementKind, Error, MetadataDescriptor, MetadataKind, Parse, Probe, QueryError,
    ReadArrayLength, ReadBool, ReadBoolArrayElement, ReadF32, ReadF32ArrayElement, ReadF64,
    ReadF64ArrayElement, ReadSigned, ReadSignedArrayElement, ReadStringArrayMetrics, ReadUnsigned,
    ReadUnsignedArrayElement, ReadUnsignedArrayMetrics, Storage, TensorDescriptor, VisitBoolArray,
    VisitF32Array, VisitStringArray, VisitUnsignedArray, WithByteArray, WithMetadataDescriptor,
    WithString, WithStringArrayElement, WithTensor,
};
use sml as _;
use std::sync::Arc;

const MAGIC: [u8; 4] = *b"GGUF";
const VERSION: u32 = 3;
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_string(bytes: &mut Vec<u8>, value: &[u8]) {
    append_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value);
}

fn append_kv(bytes: &mut Vec<u8>, key: &[u8], kind: u32, payload: &[u8]) {
    append_string(bytes, key);
    append_u32(bytes, kind);
    bytes.extend_from_slice(payload);
}

fn array_payload(kind: u32, count: u64, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_u32(&mut bytes, kind);
    append_u64(&mut bytes, count);
    bytes.extend_from_slice(payload);
    bytes
}

fn string_payload(value: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_string(&mut bytes, value);
    bytes
}

fn string_array(values: &[&[u8]]) -> Vec<u8> {
    let mut payload = Vec::new();
    for value in values {
        append_string(&mut payload, value);
    }
    array_payload(TYPE_STRING, values.len() as u64, &payload)
}

#[allow(
    clippy::too_many_lines,
    clippy::vec_init_then_push,
    reason = "one explicit fixture row per GGUF wire kind keeps parity reviewable"
)]
fn typed_metadata_fixture() -> Vec<u8> {
    let mut entries = Vec::<(Vec<u8>, u32, Vec<u8>)>::new();
    entries.push((b"u8".to_vec(), TYPE_UINT8, vec![255]));
    entries.push((b"i8".to_vec(), TYPE_INT8, vec![254]));
    entries.push((b"u16".to_vec(), TYPE_UINT16, 513_u16.to_le_bytes().to_vec()));
    entries.push((b"i16".to_vec(), TYPE_INT16, (-3_i16).to_le_bytes().to_vec()));
    entries.push((
        b"u32".to_vec(),
        TYPE_UINT32,
        70_000_u32.to_le_bytes().to_vec(),
    ));
    entries.push((b"i32".to_vec(), TYPE_INT32, (-4_i32).to_le_bytes().to_vec()));
    entries.push((
        b"f32".to_vec(),
        TYPE_FLOAT32,
        1.5_f32.to_le_bytes().to_vec(),
    ));
    entries.push((b"bool".to_vec(), TYPE_BOOL, vec![2]));
    entries.push((b"string".to_vec(), TYPE_STRING, string_payload(&[0xff, 0])));
    entries.push((
        b"u64".to_vec(),
        TYPE_UINT64,
        u64::MAX.to_le_bytes().to_vec(),
    ));
    entries.push((b"i64".to_vec(), TYPE_INT64, (-5_i64).to_le_bytes().to_vec()));
    entries.push((
        b"f64".to_vec(),
        TYPE_FLOAT64,
        f64::from_bits(0x3ff0_0000_0000_0001).to_le_bytes().to_vec(),
    ));

    entries.push((
        b"a.u8".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_UINT8, 2, &[1, 2]),
    ));
    entries.push((
        b"a.i8".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_INT8, 2, &[255, 2]),
    ));
    entries.push((
        b"a.u16".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_UINT16, 2, &[1, 0, 2, 0]),
    ));
    entries.push((
        b"a.i16".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_INT16, 2, &[255, 255, 2, 0]),
    ));
    entries.push((
        b"a.u32".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_UINT32, 2, &[1, 0, 0, 0, 2, 0, 0, 0]),
    ));
    entries.push((
        b"a.i32".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_INT32, 2, &[255, 255, 255, 255, 2, 0, 0, 0]),
    ));
    entries.push((
        b"a.f32".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_FLOAT32, 1, &1.25_f32.to_le_bytes()),
    ));
    entries.push((
        b"a.bool".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_BOOL, 2, &[0, 3]),
    ));
    entries.push((
        b"a.string".to_vec(),
        TYPE_ARRAY,
        string_array(&[b"one", &[0xff]]),
    ));
    entries.push((
        b"a.u64".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_UINT64, 1, &u64::MAX.to_le_bytes()),
    ));
    entries.push((
        b"a.i64".to_vec(),
        TYPE_ARRAY,
        array_payload(TYPE_INT64, 1, &(-7_i64).to_le_bytes()),
    ));
    entries.push((
        b"a.f64".to_vec(),
        TYPE_ARRAY,
        array_payload(
            TYPE_FLOAT64,
            1,
            &f64::from_bits(0x4000_0000_0000_0001).to_le_bytes(),
        ),
    ));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 0);
    append_u64(&mut bytes, entries.len() as u64);
    for (key, kind, payload) in entries {
        append_kv(&mut bytes, &key, kind, &payload);
    }
    bytes
}

fn tensor_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 1);
    append_u64(&mut bytes, 0);
    append_string(&mut bytes, b"weight");
    append_u32(&mut bytes, 1);
    append_u64(&mut bytes, 4);
    append_u32(&mut bytes, 0);
    append_u64(&mut bytes, 0);
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes
}

fn large_string_array_fixture() -> Vec<u8> {
    const ELEMENT_COUNT: u64 = 4096;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 0);
    append_u64(&mut bytes, 1);
    append_string(&mut bytes, b"large.string-array");
    append_u32(&mut bytes, TYPE_ARRAY);
    append_u32(&mut bytes, TYPE_STRING);
    append_u64(&mut bytes, ELEMENT_COUNT);
    for index in 0..ELEMENT_COUNT {
        append_string(&mut bytes, &u32::try_from(index).unwrap().to_le_bytes());
    }
    bytes
}

fn large_numeric_array_fixture() -> Vec<u8> {
    const ELEMENT_COUNT: u32 = 4096;
    let mut floats = Vec::with_capacity(4096 * 4);
    let mut integers = Vec::with_capacity(4096 * 4);
    for index in 0..ELEMENT_COUNT {
        let value = f32::from(u16::try_from(index).unwrap());
        floats.extend_from_slice(&value.to_bits().to_le_bytes());
        integers.extend_from_slice(&index.to_le_bytes());
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    append_u32(&mut bytes, VERSION);
    append_u64(&mut bytes, 0);
    append_u64(&mut bytes, 2);
    append_kv(
        &mut bytes,
        b"large.f32",
        TYPE_ARRAY,
        &array_payload(TYPE_FLOAT32, u64::from(ELEMENT_COUNT), &floats),
    );
    append_kv(
        &mut bytes,
        b"large.u32",
        TYPE_ARRAY,
        &array_payload(TYPE_UINT32, u64::from(ELEMENT_COUNT), &integers),
    );
    bytes
}

fn load(file: &[u8]) -> Result<Loader, Error> {
    let source: Arc<[u8]> = Arc::from(file);
    let mut loader = Loader::new();
    let probe = loader.process_event(Probe::new(source))?;
    let storage = Storage::exact(probe)?;
    loader
        .process_event(Bind::new(storage))
        .map_err(|error| error.error())?;
    loader.process_event(Parse::new())?;
    Ok(loader)
}

fn source(bytes: &[u8]) -> Arc<[u8]> {
    Arc::from(bytes)
}

#[test]
#[allow(
    clippy::cognitive_complexity,
    reason = "one exhaustive matrix proves every scalar query and coercion contract"
)]
fn scalar_queries_cover_every_gguf_wire_kind_and_emel_coercion() {
    let file = typed_metadata_fixture();
    let mut loader = load(&file).unwrap();

    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u8")),
        Ok(Some(255))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"i8")),
        Ok(Some(u64::MAX - 1))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"i16")),
        Ok(Some(65_533))
    );
    assert_eq!(
        loader.process_event(ReadSigned::new(b"u32")),
        Ok(Some(70_000))
    );
    assert_eq!(loader.process_event(ReadSigned::new(b"i32")), Ok(Some(-4)));
    assert_eq!(loader.process_event(ReadSigned::new(b"i64")), Ok(Some(-5)));
    assert_eq!(loader.process_event(ReadF32::new(b"f32")), Ok(Some(1.5)));
    let exact_f64 = f64::from_bits(0x3ff0_0000_0000_0001);
    assert_eq!(
        loader.process_event(ReadF64::new(b"f64")),
        Ok(Some(exact_f64))
    );
    assert_eq!(
        loader.process_event(ReadF64::new(b"f32")),
        Ok(Some(1.5_f64))
    );
    assert_eq!(loader.process_event(ReadF32::new(b"f64")), Ok(Some(1.0)));
    assert_eq!(loader.process_event(ReadBool::new(b"bool")), Ok(Some(true)));
    assert_eq!(
        loader.process_event(ReadSigned::new(b"u64")),
        Err(QueryError::Range)
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u16")),
        Ok(Some(513))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u32")),
        Ok(Some(70_000))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"i32")),
        Ok(Some(4_294_967_292))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u64")),
        Ok(Some(u64::MAX))
    );
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"i64")),
        Ok(Some(u64::MAX - 4))
    );
    assert_eq!(loader.process_event(ReadSigned::new(b"u8")), Ok(Some(255)));
    assert_eq!(loader.process_event(ReadSigned::new(b"i8")), Ok(Some(-2)));
    assert_eq!(loader.process_event(ReadSigned::new(b"u16")), Ok(Some(513)));
    assert_eq!(loader.process_event(ReadSigned::new(b"i16")), Ok(Some(-3)));

    let string = loader
        .process_event(WithString::new(b"string", |value: &[u8]| {
            value == [0xff, 0]
        }))
        .unwrap();
    assert_eq!(string, Some(true));
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"missing")),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(ReadBool::new(b"u8")),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(
        loader.process_event(WithString::new(b"missing", |value: &[u8]| value.len())),
        Ok(None)
    );
}

#[test]
fn indexed_metadata_descriptors_are_typed_complete_and_value_free() {
    let file = typed_metadata_fixture();
    let mut loader = load(&file).unwrap();
    let scalar_kinds = [
        MetadataKind::Uint8,
        MetadataKind::Int8,
        MetadataKind::Uint16,
        MetadataKind::Int16,
        MetadataKind::Uint32,
        MetadataKind::Int32,
        MetadataKind::Float32,
        MetadataKind::Bool,
        MetadataKind::String,
        MetadataKind::Uint64,
        MetadataKind::Int64,
        MetadataKind::Float64,
    ];
    for (index, expected_kind) in scalar_kinds.into_iter().enumerate() {
        let descriptor = loader
            .process_event(WithMetadataDescriptor::new(
                u32::try_from(index).unwrap(),
                |_key: &[u8], descriptor: MetadataDescriptor| descriptor,
            ))
            .unwrap()
            .unwrap();
        assert_eq!(descriptor.kind(), expected_kind);
        assert_eq!(descriptor.array_element_kind(), None);
        assert_eq!(descriptor.array_length(), None);
    }

    let array_kinds = [
        ElementKind::Uint8,
        ElementKind::Int8,
        ElementKind::Uint16,
        ElementKind::Int16,
        ElementKind::Uint32,
        ElementKind::Int32,
        ElementKind::Float32,
        ElementKind::Bool,
        ElementKind::String,
        ElementKind::Uint64,
        ElementKind::Int64,
        ElementKind::Float64,
    ];
    for (array_index, expected_kind) in array_kinds.into_iter().enumerate() {
        let index = 12_u32 + u32::try_from(array_index).unwrap();
        let mut key = Vec::with_capacity(16);
        let descriptor = loader
            .process_event(WithMetadataDescriptor::new(
                index,
                |borrowed_key: &[u8], descriptor: MetadataDescriptor| {
                    key.extend_from_slice(borrowed_key);
                    descriptor
                },
            ))
            .unwrap()
            .unwrap();
        assert!(key.starts_with(b"a."));
        assert_eq!(descriptor.kind(), MetadataKind::Array);
        assert_eq!(descriptor.array_element_kind(), Some(expected_kind));
        assert!(descriptor.array_length().is_some());
    }
    assert_eq!(
        loader.process_event(WithMetadataDescriptor::new(
            24,
            |_key: &[u8], _descriptor: MetadataDescriptor| (),
        )),
        Ok(None)
    );
}

#[test]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "one exhaustive matrix proves every typed GGUF array wire-kind transition"
)]
fn array_queries_cover_lengths_elements_visitors_and_byte_views() {
    let file = typed_metadata_fixture();
    let mut loader = load(&file).unwrap();

    assert_eq!(
        loader.process_event(ReadArrayLength::new(b"a.u16", ElementKind::Uint16)),
        Ok(Some(2))
    );
    assert_eq!(
        loader.process_event(ReadArrayLength::new(b"a.u16", ElementKind::Int16)),
        Err(QueryError::TypeMismatch)
    );
    for (key, kind) in [
        (b"a.u8".as_slice(), ElementKind::Uint8),
        (b"a.i8".as_slice(), ElementKind::Int8),
        (b"a.u16".as_slice(), ElementKind::Uint16),
        (b"a.i16".as_slice(), ElementKind::Int16),
        (b"a.u32".as_slice(), ElementKind::Uint32),
        (b"a.i32".as_slice(), ElementKind::Int32),
        (b"a.f32".as_slice(), ElementKind::Float32),
        (b"a.bool".as_slice(), ElementKind::Bool),
        (b"a.string".as_slice(), ElementKind::String),
        (b"a.u64".as_slice(), ElementKind::Uint64),
        (b"a.i64".as_slice(), ElementKind::Int64),
        (b"a.f64".as_slice(), ElementKind::Float64),
    ] {
        assert!(
            loader
                .process_event(ReadArrayLength::new(key, kind))
                .is_ok()
        );
    }
    assert_eq!(
        loader.process_event(ReadUnsignedArrayElement::new(b"a.i8", 0)),
        Ok(Some(u64::MAX))
    );
    assert_eq!(
        loader.process_event(ReadSignedArrayElement::new(b"a.i16", 0)),
        Ok(Some(-1))
    );
    assert_eq!(
        loader.process_event(ReadF32ArrayElement::new(b"a.f32", 0)),
        Ok(Some(1.25))
    );
    let exact_array_f64 = f64::from_bits(0x4000_0000_0000_0001);
    assert_eq!(
        loader.process_event(ReadF64ArrayElement::new(b"a.f64", 0)),
        Ok(Some(exact_array_f64))
    );
    assert_eq!(
        loader.process_event(ReadF64ArrayElement::new(b"a.f32", 0)),
        Ok(Some(1.25_f64))
    );
    assert_eq!(
        loader.process_event(ReadF32ArrayElement::new(b"a.f64", 0)),
        Ok(Some(2.0))
    );
    assert_eq!(
        loader.process_event(ReadBoolArrayElement::new(b"a.bool", 1)),
        Ok(Some(true))
    );
    assert_eq!(
        loader.process_event(ReadUnsignedArrayElement::new(b"a.u8", 2)),
        Err(QueryError::IndexOutOfBounds)
    );
    assert_eq!(
        loader.process_event(ReadSignedArrayElement::new(b"a.u64", 0)),
        Err(QueryError::Range)
    );
    for (key, expected) in [
        (b"a.u8".as_slice(), 1),
        (b"a.i8".as_slice(), u64::MAX),
        (b"a.u16".as_slice(), 1),
        (b"a.i16".as_slice(), u64::from(u16::MAX)),
        (b"a.u32".as_slice(), 1),
        (b"a.i32".as_slice(), u64::from(u32::MAX)),
        (b"a.u64".as_slice(), u64::MAX),
        (b"a.i64".as_slice(), u64::MAX - 6),
    ] {
        assert_eq!(
            loader.process_event(ReadUnsignedArrayElement::new(key, 0)),
            Ok(Some(expected))
        );
    }
    for (key, expected) in [
        (b"a.u8".as_slice(), 1),
        (b"a.i8".as_slice(), -1),
        (b"a.u16".as_slice(), 1),
        (b"a.i16".as_slice(), -1),
        (b"a.u32".as_slice(), 1),
        (b"a.i32".as_slice(), -1),
        (b"a.i64".as_slice(), -7),
    ] {
        assert_eq!(
            loader.process_event(ReadSignedArrayElement::new(key, 0)),
            Ok(Some(expected))
        );
    }

    let element = loader
        .process_event(WithStringArrayElement::new(
            b"a.string",
            1,
            |value: &[u8]| value == [0xff],
        ))
        .unwrap();
    assert_eq!(element, Some(true));

    let mut visited = Vec::with_capacity(2);
    let mut first = Vec::with_capacity(3);
    let mut second = Vec::with_capacity(1);
    assert_eq!(
        loader.process_event(VisitStringArray::new(
            b"a.string",
            |index: u32, value: &[u8]| {
                visited.push(index);
                if index == 0 {
                    first.extend_from_slice(value);
                } else {
                    second.extend_from_slice(value);
                }
            },
        )),
        Ok(Some(2))
    );
    assert_eq!(visited, [0, 1]);
    assert_eq!(first, b"one");
    assert_eq!(second, [0xff]);
    let metrics = loader
        .process_event(ReadStringArrayMetrics::new(b"a.string"))
        .unwrap()
        .unwrap();
    assert_eq!(metrics.element_count(), 2);
    assert_eq!(metrics.total_string_bytes(), 4);

    assert_eq!(
        loader.process_event(WithByteArray::new(b"a.i8", |bytes: &[u8]| {
            bytes == [255, 2]
        })),
        Ok(Some(true))
    );
    assert_eq!(
        loader.process_event(WithByteArray::new(b"a.u8", |bytes: &[u8]| {
            bytes == [1, 2]
        })),
        Ok(Some(true))
    );
    assert_eq!(
        loader.process_event(ReadArrayLength::new(b"missing", ElementKind::Uint8)),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(ReadUnsignedArrayElement::new(b"missing", 0)),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(ReadSignedArrayElement::new(b"missing", 0)),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(ReadF32ArrayElement::new(b"missing", 0)),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(ReadBoolArrayElement::new(b"missing", 0)),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(WithStringArrayElement::new(
            b"missing",
            0,
            |value: &[u8]| value.len(),
        )),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(VisitStringArray::new(
            b"missing",
            |_index: u32, _value: &[u8]| {},
        )),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(WithByteArray::new(b"missing", |value: &[u8]| value.len())),
        Ok(None)
    );
}

#[test]
fn large_string_array_visit_is_ordered_allocation_free_and_single_pass() {
    const ELEMENT_COUNT: u32 = 4096;
    let file = large_string_array_fixture();
    let mut loader = load(&file).unwrap();
    let mut next = 0_u32;
    let mut result = None;
    let event = VisitStringArray::new(b"large.string-array", |index: u32, value: &[u8]| {
        assert_eq!(index, next);
        assert_eq!(value, next.to_le_bytes());
        next += 1;
    });

    assert_eq!(
        measure(|| result = Some(loader.process_event(event))).count_total,
        0
    );
    assert_eq!(result.unwrap(), Ok(Some(u64::from(ELEMENT_COUNT))));
    assert_eq!(next, ELEMENT_COUNT);
}

#[test]
fn string_array_metrics_are_exact_constant_work_and_allocation_free() {
    const ELEMENT_COUNT: u64 = 4096;
    const ELEMENT_BYTES: u64 = 4;
    let file = large_string_array_fixture();
    let mut loader = load(&file).unwrap();
    let mut result = None;

    assert_eq!(
        measure(|| {
            result = Some(loader.process_event(ReadStringArrayMetrics::new(b"large.string-array")));
        })
        .count_total,
        0
    );

    let metrics = result.unwrap().unwrap().unwrap();
    assert_eq!(metrics.element_count(), ELEMENT_COUNT);
    assert_eq!(metrics.total_string_bytes(), ELEMENT_COUNT * ELEMENT_BYTES);
    assert_eq!(
        loader.process_event(ReadStringArrayMetrics::new(b"missing")),
        Ok(None)
    );

    let mut typed_loader = load(&typed_metadata_fixture()).unwrap();
    assert_eq!(
        typed_loader.process_event(ReadStringArrayMetrics::new(b"a.u16")),
        Err(QueryError::TypeMismatch)
    );
}

#[test]
fn numeric_array_bulk_queries_preserve_coercions_ranges_and_zero_allocation() {
    let file = typed_metadata_fixture();
    let mut loader = load(&file).unwrap();
    let mut floats = [0.0_f32; 2];
    let mut result = None;

    assert_eq!(
        measure(|| {
            result = Some(loader.process_event(VisitF32Array::new(
                b"a.f64",
                |index: u32, value: f32| {
                    floats[index as usize] = value;
                },
            )));
        })
        .count_total,
        0
    );
    assert_eq!(result.unwrap(), Ok(Some(1)));
    assert_eq!(floats[0].to_bits(), 2.0_f32.to_bits());

    for (key, expected, count) in [
        (b"a.u8".as_slice(), [1, 2], 2),
        (b"a.i8".as_slice(), [u64::MAX, 2], 2),
        (b"a.u16".as_slice(), [1, 2], 2),
        (b"a.i16".as_slice(), [u64::from(u16::MAX), 2], 2),
        (b"a.u32".as_slice(), [1, 2], 2),
        (b"a.i32".as_slice(), [u64::from(u32::MAX), 2], 2),
        (b"a.u64".as_slice(), [u64::MAX, 0], 1),
        (b"a.i64".as_slice(), [u64::MAX - 6, 0], 1),
    ] {
        let mut integers = [0_u64; 2];
        assert_eq!(
            loader.process_event(VisitUnsignedArray::new(key, |index: u32, value: u64| {
                integers[index as usize] = value;
            },)),
            Ok(Some(count))
        );
        assert_eq!(integers, expected);
        let metrics = loader
            .process_event(ReadUnsignedArrayMetrics::new(key))
            .unwrap()
            .unwrap();
        assert_eq!(metrics.element_count(), count);
        assert_eq!(
            metrics.maximum(),
            *expected[..usize::try_from(count).unwrap()]
                .iter()
                .max()
                .unwrap()
        );
    }

    let mut f32_bits = 0_u32;
    assert_eq!(
        loader.process_event(VisitF32Array::new(b"a.f32", |_, value: f32| {
            f32_bits = value.to_bits();
        })),
        Ok(Some(1))
    );
    assert_eq!(f32_bits, 1.25_f32.to_bits());
    assert_eq!(
        loader.process_event(ReadUnsignedArrayMetrics::new(b"a.f32")),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(
        loader.process_event(ReadUnsignedArrayMetrics::new(b"missing")),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(VisitUnsignedArray::new(b"a.f32", |_, _| {})),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(
        loader.process_event(VisitUnsignedArray::new(b"missing", |_, _| {})),
        Ok(None)
    );
    assert_eq!(
        loader.process_event(VisitF32Array::new(b"a.u32", |_, _| {})),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(
        loader.process_event(VisitF32Array::new(b"missing", |_, _| {})),
        Ok(None)
    );
}

#[test]
fn bool_array_bulk_visit_is_typed_complete_and_allocation_free() {
    let file = typed_metadata_fixture();
    let mut loader = load(&file).unwrap();
    let mut values = [true, false];
    let mut next = 0_u32;
    let mut result = None;
    assert_eq!(
        measure(|| {
            result = Some(loader.process_event(VisitBoolArray::new(
                b"a.bool",
                |index: u32, value: bool| {
                    assert_eq!(index, next);
                    values[index as usize] = value;
                    next += 1;
                },
            )));
        })
        .count_total,
        0
    );
    assert_eq!(result.unwrap(), Ok(Some(2)));
    assert_eq!(next, 2);
    assert_eq!(values, [false, true]);
    assert_eq!(
        loader.process_event(VisitBoolArray::new(b"a.u8", |_, _| {})),
        Err(QueryError::TypeMismatch)
    );
    assert_eq!(
        loader.process_event(VisitBoolArray::new(b"missing", |_, _| {})),
        Ok(None)
    );
}

#[test]
fn large_numeric_arrays_use_one_allocation_free_bulk_dispatch() {
    const ELEMENT_COUNT: u32 = 4096;
    let mut loader = load(&large_numeric_array_fixture()).unwrap();
    let mut next = 0_u32;
    let mut result = None;
    assert_eq!(
        measure(|| {
            result = Some(loader.process_event(VisitF32Array::new(
                b"large.f32",
                |index: u32, value: f32| {
                    assert_eq!(index, next);
                    assert_eq!(
                        value.to_bits(),
                        f32::from(u16::try_from(index).unwrap()).to_bits()
                    );
                    next += 1;
                },
            )));
        })
        .count_total,
        0
    );
    assert_eq!(result.unwrap(), Ok(Some(u64::from(ELEMENT_COUNT))));
    assert_eq!(next, ELEMENT_COUNT);
    let metrics = loader
        .process_event(ReadUnsignedArrayMetrics::new(b"large.u32"))
        .unwrap()
        .unwrap();
    assert_eq!(metrics.element_count(), u64::from(ELEMENT_COUNT));
    assert_eq!(metrics.maximum(), u64::from(ELEMENT_COUNT - 1));
}

#[test]
fn lifecycle_requires_probe_bind_parse_and_reprobe_invalidates_queries() {
    let file = typed_metadata_fixture();
    let mut loader = Loader::new();
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u8")),
        Err(QueryError::NotParsed)
    );
    assert_eq!(
        loader.process_event(WithMetadataDescriptor::new(
            0,
            |_key: &[u8], _descriptor: MetadataDescriptor| (),
        )),
        Err(QueryError::NotParsed)
    );
    assert_eq!(
        loader.process_event(Parse::new()),
        Err(Error::InvalidRequest)
    );

    let file_source = source(&file);
    let probe = loader
        .process_event(Probe::new(file_source.clone()))
        .unwrap();
    let required_metadata = probe.required_metadata_bytes().unwrap();
    let metadata_entries = probe.metadata_count() as usize;
    let tensors = probe.tensor_count() as usize;
    let undersized =
        Storage::with_capacity(probe, required_metadata - 1, metadata_entries, tensors).unwrap();
    let mut bind_result = None;
    let bind_event = Bind::new(undersized);
    assert_eq!(
        measure(|| bind_result = Some(loader.process_event(bind_event))).count_total,
        0
    );
    let bind_error = bind_result
        .unwrap()
        .expect_err("undersized storage must be returned");
    assert_eq!(bind_error.error(), Error::Capacity);
    assert!(format!("{bind_error:?}").starts_with("BindError"));
    assert_eq!(bind_error.to_string(), "GGUF loader capacity exceeded");
    let returned_storage = bind_error.into_storage();
    assert_eq!(Arc::strong_count(&file_source), 2);
    drop(returned_storage);

    let probe = loader.process_event(Probe::new(source(&file))).unwrap();
    loader
        .process_event(Bind::new(Storage::exact(probe).unwrap()))
        .unwrap();
    loader.process_event(Parse::new()).unwrap();
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u8")),
        Ok(Some(255))
    );

    loader.process_event(Probe::new(source(&file))).unwrap();
    assert_eq!(
        loader.process_event(ReadUnsigned::new(b"u8")),
        Err(QueryError::NotParsed)
    );
    assert_eq!(
        loader.process_event(WithMetadataDescriptor::new(
            0,
            |_key: &[u8], _descriptor: MetadataDescriptor| (),
        )),
        Err(QueryError::NotParsed)
    );
}

#[test]
fn tensor_queries_expose_only_semantic_data_from_the_bound_source() {
    let file = tensor_fixture();
    let mut loader = load(&file).unwrap();
    let mut name = Vec::with_capacity(b"weight".len());
    let mut data = Vec::with_capacity(16);
    let mut observed = None;
    assert_eq!(
        loader.process_event(WithTensor::new(
            0,
            |borrowed_name: &[u8], descriptor: TensorDescriptor, borrowed_data: &[u8]| {
                name.extend_from_slice(borrowed_name);
                data.extend_from_slice(borrowed_data);
                observed = Some((
                    descriptor.tensor_type(),
                    descriptor.dimension_count(),
                    descriptor.dimensions(),
                    descriptor.data_offset(),
                    descriptor.data_size(),
                    descriptor.file_index(),
                ));
            },
        )),
        Ok(Some(()))
    );
    assert_eq!(observed, Some((0, 1, [4, 1, 1, 1], 0, 16, 0,)));
    assert_eq!(name, b"weight");
    assert_eq!(data, (1_u8..=16).collect::<Vec<_>>());
    assert_eq!(
        loader.process_event(WithTensor::new(
            1,
            |_name: &[u8], _descriptor: TensorDescriptor, _data: &[u8]| {},
        )),
        Ok(None)
    );
}

#[test]
fn probe_bind_parse_and_query_dispatches_do_not_allocate() {
    let file = typed_metadata_fixture();
    let file_source = source(&file);
    let mut loader = Loader::new();
    let mut probe = None;
    let probe_event = Probe::new(file_source);
    assert_eq!(
        measure(|| probe = Some(loader.process_event(probe_event))).count_total,
        0
    );
    let probe = probe.unwrap().unwrap();
    let storage = Storage::exact(probe).unwrap();
    let bind_event = Bind::new(storage);
    assert_eq!(
        measure(|| {
            let _ = loader.process_event(bind_event);
        })
        .count_total,
        0
    );
    assert_eq!(
        measure(|| {
            let _ = loader.process_event(Parse::new());
        })
        .count_total,
        0
    );
    assert_eq!(
        measure(|| {
            let _ = loader.process_event(ReadUnsigned::new(b"u32"));
            let _ = loader.process_event(ReadF64::new(b"f64"));
            let _ = loader.process_event(ReadF64ArrayElement::new(b"a.f64", 0));
            let _ = loader.process_event(WithMetadataDescriptor::new(
                0,
                |_key: &[u8], _descriptor: MetadataDescriptor| (),
            ));
        })
        .count_total,
        0
    );
    assert_eq!(
        measure(|| {
            let _ = loader.process_event(WithString::new(b"string", |value: &[u8]| value.len()));
        })
        .count_total,
        0
    );

    let tensor_file = tensor_fixture();
    let mut tensor_loader = load(&tensor_file).unwrap();
    assert_eq!(
        measure(|| {
            let _ = tensor_loader.process_event(WithTensor::new(
                0,
                |_name: &[u8], _descriptor: TensorDescriptor, _data: &[u8]| {},
            ));
        })
        .count_total,
        0
    );
}

#[test]
fn hostile_headers_and_payloads_are_classified_without_panics() {
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&[])))
            .unwrap_err(),
        Error::InvalidRequest
    );

    let mut bad_magic = typed_metadata_fixture();
    bad_magic[0] = b'F';
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&bad_magic)))
            .unwrap_err(),
        Error::ModelInvalid
    );

    let mut bad_version = typed_metadata_fixture();
    bad_version[4..8].copy_from_slice(&4_u32.to_le_bytes());
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&bad_version)))
            .unwrap_err(),
        Error::ModelInvalid
    );

    let mut truncated = typed_metadata_fixture();
    truncated.pop();
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&truncated)))
            .unwrap_err(),
        Error::ParseFailed
    );

    let mut truncated_string_array = large_string_array_fixture();
    truncated_string_array.pop();
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&truncated_string_array)))
            .unwrap_err(),
        Error::ParseFailed
    );

    let mut impossible = Vec::new();
    impossible.extend_from_slice(&MAGIC);
    append_u32(&mut impossible, VERSION);
    append_u64(&mut impossible, u64::from(u32::MAX) + 1);
    append_u64(&mut impossible, 0);
    assert_eq!(
        Loader::new()
            .process_event(Probe::new(source(&impossible)))
            .unwrap_err(),
        Error::Capacity
    );
}

#[test]
fn probe_rejects_duplicate_keys_and_tensor_names_without_allocating() {
    let mut duplicate_key = Vec::new();
    duplicate_key.extend_from_slice(&MAGIC);
    append_u32(&mut duplicate_key, VERSION);
    append_u64(&mut duplicate_key, 0);
    append_u64(&mut duplicate_key, 2);
    append_kv(
        &mut duplicate_key,
        b"same",
        TYPE_UINT32,
        &1_u32.to_le_bytes(),
    );
    append_kv(
        &mut duplicate_key,
        b"same",
        TYPE_UINT32,
        &2_u32.to_le_bytes(),
    );

    let mut duplicate_tensor = Vec::new();
    duplicate_tensor.extend_from_slice(&MAGIC);
    append_u32(&mut duplicate_tensor, VERSION);
    append_u64(&mut duplicate_tensor, 2);
    append_u64(&mut duplicate_tensor, 0);
    for offset in [0_u64, 32] {
        append_string(&mut duplicate_tensor, b"weight");
        append_u32(&mut duplicate_tensor, 1);
        append_u64(&mut duplicate_tensor, 4);
        append_u32(&mut duplicate_tensor, TYPE_UINT8);
        append_u64(&mut duplicate_tensor, offset);
    }
    duplicate_tensor.resize(duplicate_tensor.len().next_multiple_of(32) + 64, 0);

    let duplicate_key = source(&duplicate_key);
    let duplicate_tensor = source(&duplicate_tensor);
    let mut loader = Loader::new();
    let mut key_result = None;
    let key_probe = measure(|| {
        key_result = Some(loader.process_event(Probe::new(duplicate_key.clone())));
    });
    assert_eq!(key_probe.count_total, 0);
    assert_eq!(key_result.unwrap().unwrap_err(), Error::ModelInvalid);
    let mut tensor_result = None;
    let tensor_probe = measure(|| {
        tensor_result = Some(loader.process_event(Probe::new(duplicate_tensor.clone())));
    });
    assert_eq!(tensor_probe.count_total, 0);
    assert_eq!(tensor_result.unwrap().unwrap_err(), Error::ModelInvalid);

    let valid = typed_metadata_fixture();
    let probe = loader.process_event(Probe::new(source(&valid))).unwrap();
    loader
        .process_event(Bind::new(Storage::exact(probe).unwrap()))
        .unwrap();
    assert!(loader.process_event(Parse::new()).is_ok());
}

#[test]
fn probe_name_filter_resolves_hash_collisions_byte_exactly() {
    // These distinct byte strings share the fixed prefilter bit. Their
    // successful probes prove that collision candidates are byte-compared.
    let colliding_names = [*b"a0cdefgh", *b"a1cdefgh"];

    let mut metadata = Vec::new();
    metadata.extend_from_slice(&MAGIC);
    append_u32(&mut metadata, VERSION);
    append_u64(&mut metadata, 0);
    append_u64(&mut metadata, 2);
    for name in &colliding_names {
        append_kv(&mut metadata, name, TYPE_UINT32, &1_u32.to_le_bytes());
    }

    let mut tensors = Vec::new();
    tensors.extend_from_slice(&MAGIC);
    append_u32(&mut tensors, VERSION);
    append_u64(&mut tensors, 2);
    append_u64(&mut tensors, 0);
    for (name, offset) in colliding_names.iter().zip([0_u64, 32]) {
        append_string(&mut tensors, name);
        append_u32(&mut tensors, 1);
        append_u64(&mut tensors, 4);
        append_u32(&mut tensors, TYPE_UINT8);
        append_u64(&mut tensors, offset);
    }
    tensors.resize(tensors.len().next_multiple_of(32) + 64, 0);

    let mut loader = Loader::new();
    let metadata_probe = loader
        .process_event(Probe::new(source(&metadata)))
        .expect("distinct colliding metadata keys remain valid");
    assert_eq!(metadata_probe.metadata_count(), 2);
    let tensor_probe = loader
        .process_event(Probe::new(source(&tensors)))
        .expect("distinct colliding tensor names remain valid");
    assert_eq!(tensor_probe.tensor_count(), 2);
}

#[test]
fn diagnostics_do_not_expose_generated_state_or_storage_internals() {
    assert_eq!(format!("{:?}", Loader::default()), "Loader { .. }");
    assert_eq!(
        format!(
            "{:?}",
            Storage::with_capacity(
                Loader::new()
                    .process_event(Probe::new(source(&typed_metadata_fixture())))
                    .unwrap(),
                1,
                2,
                3,
            )
            .unwrap()
        ),
        "Storage { metadata_bytes: 1, metadata_entries: 2, tensors: 3, .. }"
    );
    assert_eq!(
        QueryError::TypeMismatch.to_string(),
        "GGUF metadata type mismatch"
    );
    assert_eq!(Error::Capacity.to_string(), "GGUF loader capacity exceeded");
    let metadata = Loader::new()
        .process_event(Probe::new(source(&typed_metadata_fixture())))
        .unwrap();
    assert_eq!(metadata.max_key_bytes(), 8);
    assert!(metadata.max_value_bytes() >= 12);
    let tensors = Loader::new()
        .process_event(Probe::new(source(&tensor_fixture())))
        .unwrap();
    assert_eq!(tensors.tensor_data_bytes(), 16);

    let string: WithString<'_, _, ()> = WithString::new(b"key", |_value: &[u8]| {});
    assert!(format!("{string:?}").starts_with("WithString"));
    let tensor: WithTensor<_, ()> = WithTensor::new(
        1,
        |_name: &[u8], _descriptor: TensorDescriptor, _data: &[u8]| {},
    );
    assert!(format!("{tensor:?}").starts_with("WithTensor"));
    let string_element: WithStringArrayElement<'_, _, ()> =
        WithStringArrayElement::new(b"key", 2, |_value: &[u8]| {});
    assert!(format!("{string_element:?}").starts_with("WithStringArrayElement"));
    assert!(
        format!(
            "{:?}",
            VisitStringArray::new(b"key", |_index: u32, _value: &[u8]| {})
        )
        .starts_with("VisitStringArray")
    );
    let bytes: WithByteArray<'_, _, ()> = WithByteArray::new(b"key", |_value: &[u8]| {});
    assert!(format!("{bytes:?}").starts_with("WithByteArray"));
    for (error, message) in [
        (QueryError::NotParsed, "GGUF metadata has not been parsed"),
        (
            QueryError::IndexOutOfBounds,
            "GGUF metadata array index is out of bounds",
        ),
        (QueryError::Range, "GGUF metadata value is out of range"),
        (QueryError::Malformed, "malformed actor-owned GGUF metadata"),
        (QueryError::Internal, "internal GGUF metadata query error"),
    ] {
        assert_eq!(error.to_string(), message);
    }
}
