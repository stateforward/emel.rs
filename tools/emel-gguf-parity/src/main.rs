//! Canonical Rust-side output for the pinned llama.cpp GGUF parity gate.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use emel_gguf::Loader;
use emel_gguf::event::{
    Bind, ElementKind, MetadataDescriptor, Parse, Probe, ReadArrayLength, ReadBool,
    ReadBoolArrayElement, ReadF32, ReadF32ArrayElement, ReadF64, ReadF64ArrayElement, ReadSigned,
    ReadSignedArrayElement, ReadUnsigned, ReadUnsignedArrayElement, Storage, TensorDescriptor,
    VisitStringArray, WithByteArray, WithMetadataDescriptor, WithString, WithTensor,
};

const MAGIC: [u8; 4] = *b"GGUF";
const VERSION: u32 = 3;
const ALIGNMENT: usize = 32;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;

const TENSOR_LAYOUTS: &[(u32, u64, usize)] = &[
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
];

const PACKED_TENSOR_LAYOUTS: &[(u32, [u64; 2], usize)] =
    &[(41, [256, 9], 2304), (42, [512, 1], 2304)];

fn main() {
    if let Err(error) = run() {
        eprintln!("emel-gguf-parity: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let first = arguments
        .next()
        .ok_or("expected a GGUF path or --write-fixtures")?;
    if first == "--write-fixtures" {
        let directory = arguments
            .next()
            .ok_or("--write-fixtures requires a directory")?;
        if arguments.next().is_some() {
            return Err("unexpected arguments after fixture directory".into());
        }
        write_fixtures(Path::new(&directory))?;
        return Ok(());
    }
    if arguments.next().is_some() {
        return Err("expected exactly one GGUF path".into());
    }
    let bytes = fs::read(&first)?;
    print!("{}", canonical_output(Path::new(&first), &bytes));
    Ok(())
}

struct ParityScratch {
    key: Vec<u8>,
    value: Vec<u8>,
    tensor_name: Vec<u8>,
}

impl ParityScratch {
    fn new(max_key_bytes: usize, max_value_bytes: usize, source_bytes: usize) -> Self {
        Self {
            key: Vec::with_capacity(max_key_bytes),
            value: Vec::with_capacity(max_value_bytes),
            tensor_name: Vec::with_capacity(source_bytes),
        }
    }
}

fn canonical_output(_path: &Path, bytes: &[u8]) -> String {
    let mut output = String::new();
    output.push_str("gguf-parity/v1\n");
    let mut loader = Loader::new();
    let Ok(probe) = loader.process_event(Probe::new(Arc::from(bytes))) else {
        output.push_str("status=error\n");
        return output;
    };
    let metadata_count = probe.metadata_count();
    let tensor_count = probe.tensor_count();
    let output_capacity = parity_output_capacity(bytes.len(), metadata_count, tensor_count);
    output.reserve(output_capacity.saturating_sub(output.capacity()));
    let max_key_bytes = usize::try_from(probe.max_key_bytes()).unwrap_or(bytes.len());
    let max_value_bytes = usize::try_from(probe.max_value_bytes()).unwrap_or(bytes.len());
    let mut scratch = ParityScratch::new(max_key_bytes, max_value_bytes, bytes.len());
    let Ok(storage) = Storage::exact(probe) else {
        output.push_str("status=error\n");
        return output;
    };
    if loader.process_event(Bind::new(storage)).is_err()
        || loader.process_event(Parse::new()).is_err()
    {
        output.push_str("status=error\n");
        return output;
    }
    output.push_str("status=ok\n");
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("validated GGUF header"));
    let alignment = loader
        .process_event(ReadUnsigned::new(b"general.alignment"))
        .ok()
        .flatten()
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(32);
    writeln!(output, "version={version}").unwrap();
    writeln!(output, "alignment={alignment}").unwrap();
    writeln!(output, "kv_count={metadata_count}").unwrap();
    writeln!(output, "tensor_count={tensor_count}").unwrap();
    if append_kv_output(&mut output, &mut loader, metadata_count, &mut scratch).is_err()
        || append_tensor_output(&mut output, &mut loader, tensor_count, &mut scratch).is_err()
    {
        return String::from("gguf-parity/v1\nstatus=error\n");
    }
    output
}

fn append_kv_output(
    output: &mut String,
    loader: &mut Loader,
    count: u32,
    scratch: &mut ParityScratch,
) -> Result<(), Box<dyn std::error::Error>> {
    for index in 0..count {
        scratch.key.clear();
        let mut descriptor = None;
        loader
            .process_event(WithMetadataDescriptor::new(
                index,
                |key: &[u8], value: MetadataDescriptor| {
                    scratch.key.extend_from_slice(key);
                    descriptor = Some(value);
                },
            ))?
            .ok_or("missing metadata descriptor")?;
        let descriptor = descriptor.ok_or("metadata callback did not run")?;
        write!(output, "kv.{index}.key=").unwrap();
        append_hex(output, &scratch.key);
        output.push('\n');
        writeln!(
            output,
            "kv.{index}.type={}",
            metadata_wire_type(descriptor.kind())
        )
        .unwrap();
        write!(output, "kv.{index}.value=").unwrap();
        read_metadata_value(loader, descriptor, scratch)?;
        append_hex(output, &scratch.value);
        output.push('\n');
    }
    Ok(())
}

fn read_metadata_value(
    loader: &mut Loader,
    descriptor: MetadataDescriptor,
    scratch: &mut ParityScratch,
) -> Result<(), Box<dyn std::error::Error>> {
    use emel_gguf::event::MetadataKind;

    scratch.value.clear();
    let key = scratch.key.as_slice();
    match descriptor.kind() {
        MetadataKind::Uint8 => scratch
            .value
            .extend_from_slice(&u8::try_from(read_unsigned(loader, key)?)?.to_le_bytes()),
        MetadataKind::Int8 => scratch
            .value
            .extend_from_slice(&i8::try_from(read_signed(loader, key)?)?.to_le_bytes()),
        MetadataKind::Uint16 => scratch
            .value
            .extend_from_slice(&u16::try_from(read_unsigned(loader, key)?)?.to_le_bytes()),
        MetadataKind::Int16 => scratch
            .value
            .extend_from_slice(&i16::try_from(read_signed(loader, key)?)?.to_le_bytes()),
        MetadataKind::Uint32 => scratch
            .value
            .extend_from_slice(&u32::try_from(read_unsigned(loader, key)?)?.to_le_bytes()),
        MetadataKind::Int32 => scratch
            .value
            .extend_from_slice(&i32::try_from(read_signed(loader, key)?)?.to_le_bytes()),
        MetadataKind::Float32 => scratch.value.extend_from_slice(
            &loader
                .process_event(ReadF32::new(key))?
                .ok_or("missing")?
                .to_le_bytes(),
        ),
        MetadataKind::Bool => scratch.value.push(u8::from(
            loader.process_event(ReadBool::new(key))?.ok_or("missing")?,
        )),
        MetadataKind::String => {
            loader
                .process_event(WithString::new(key, |value: &[u8]| {
                    append_u64(
                        &mut scratch.value,
                        u64::try_from(value.len()).expect("metadata string length fits u64"),
                    );
                    scratch.value.extend_from_slice(value);
                }))?
                .ok_or("missing")?;
        }
        MetadataKind::Array => read_array(loader, key, descriptor, &mut scratch.value)?,
        MetadataKind::Uint64 => scratch
            .value
            .extend_from_slice(&read_unsigned(loader, key)?.to_le_bytes()),
        MetadataKind::Int64 => scratch
            .value
            .extend_from_slice(&read_signed(loader, key)?.to_le_bytes()),
        MetadataKind::Float64 => scratch.value.extend_from_slice(
            &loader
                .process_event(ReadF64::new(key))?
                .ok_or("missing")?
                .to_le_bytes(),
        ),
    }
    Ok(())
}

const fn metadata_wire_type(kind: emel_gguf::event::MetadataKind) -> u32 {
    use emel_gguf::event::MetadataKind;
    match kind {
        MetadataKind::Uint8 => 0,
        MetadataKind::Int8 => 1,
        MetadataKind::Uint16 => 2,
        MetadataKind::Int16 => 3,
        MetadataKind::Uint32 => 4,
        MetadataKind::Int32 => 5,
        MetadataKind::Float32 => 6,
        MetadataKind::Bool => TYPE_BOOL,
        MetadataKind::String => TYPE_STRING,
        MetadataKind::Array => TYPE_ARRAY,
        MetadataKind::Uint64 => 10,
        MetadataKind::Int64 => 11,
        MetadataKind::Float64 => 12,
    }
}

fn read_unsigned(loader: &mut Loader, key: &[u8]) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(loader
        .process_event(ReadUnsigned::new(key))?
        .ok_or("missing")?)
}

fn read_signed(loader: &mut Loader, key: &[u8]) -> Result<i64, Box<dyn std::error::Error>> {
    Ok(loader
        .process_event(ReadSigned::new(key))?
        .ok_or("missing")?)
}

fn read_array(
    loader: &mut Loader,
    key: &[u8],
    descriptor: MetadataDescriptor,
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let kind = descriptor
        .array_element_kind()
        .ok_or("missing array kind")?;
    let expected_count = descriptor.array_length().ok_or("missing array length")?;
    match kind {
        ElementKind::Uint8 => read_byte_array(loader, key, value)?,
        ElementKind::Int8 => read_integer_array(loader, key, kind, 1, 1, true, value)?,
        ElementKind::Uint16 => read_integer_array(loader, key, kind, 2, 2, false, value)?,
        ElementKind::Int16 => read_integer_array(loader, key, kind, 3, 2, true, value)?,
        ElementKind::Uint32 => read_integer_array(loader, key, kind, 4, 4, false, value)?,
        ElementKind::Int32 => read_integer_array(loader, key, kind, 5, 4, true, value)?,
        ElementKind::Float32 => read_f32_array(loader, key, value)?,
        ElementKind::Bool => read_bool_array(loader, key, value)?,
        ElementKind::String => read_string_array(loader, key, value)?,
        ElementKind::Uint64 => read_integer_array(loader, key, kind, 10, 8, false, value)?,
        ElementKind::Int64 => read_integer_array(loader, key, kind, 11, 8, true, value)?,
        ElementKind::Float64 => read_f64_array(loader, key, value)?,
    }
    let actual_count = u64::from_le_bytes(value[4..12].try_into()?);
    if actual_count != expected_count {
        return Err("array descriptor count differs from typed query".into());
    }
    Ok(())
}

fn read_byte_array(
    loader: &mut Loader,
    key: &[u8],
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, ElementKind::Uint8))?
        .ok_or("missing")?;
    append_u32(value, 0);
    append_u64(value, count);
    loader
        .process_event(WithByteArray::new(key, |payload: &[u8]| {
            value.extend_from_slice(payload);
        }))?
        .ok_or("missing")?;
    Ok(())
}

fn read_f32_array(
    loader: &mut Loader,
    key: &[u8],
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, ElementKind::Float32))?
        .ok_or("missing")?;
    append_u32(value, 6);
    append_u64(value, count);
    for index in 0..count {
        let item = loader
            .process_event(ReadF32ArrayElement::new(key, index))?
            .ok_or("missing")?;
        value.extend_from_slice(&item.to_le_bytes());
    }
    Ok(())
}

fn read_f64_array(
    loader: &mut Loader,
    key: &[u8],
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, ElementKind::Float64))?
        .ok_or("missing")?;
    append_u32(value, 12);
    append_u64(value, count);
    for index in 0..count {
        let item = loader
            .process_event(ReadF64ArrayElement::new(key, index))?
            .ok_or("missing")?;
        value.extend_from_slice(&item.to_le_bytes());
    }
    Ok(())
}

fn read_integer_array(
    loader: &mut Loader,
    key: &[u8],
    kind: ElementKind,
    wire_type: u32,
    width: u32,
    signed: bool,
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, kind))?
        .ok_or("missing")?;
    append_u32(value, wire_type);
    append_u64(value, count);
    for index in 0..count {
        if signed {
            let item = loader
                .process_event(ReadSignedArrayElement::new(key, index))?
                .ok_or("missing")?;
            match width {
                1 => value.extend_from_slice(&i8::try_from(item)?.to_le_bytes()),
                2 => value.extend_from_slice(&i16::try_from(item)?.to_le_bytes()),
                4 => value.extend_from_slice(&i32::try_from(item)?.to_le_bytes()),
                8 => value.extend_from_slice(&item.to_le_bytes()),
                _ => return Err("invalid signed integer width".into()),
            }
        } else {
            let item = loader
                .process_event(ReadUnsignedArrayElement::new(key, index))?
                .ok_or("missing")?;
            match width {
                1 => value.extend_from_slice(&u8::try_from(item)?.to_le_bytes()),
                2 => value.extend_from_slice(&u16::try_from(item)?.to_le_bytes()),
                4 => value.extend_from_slice(&u32::try_from(item)?.to_le_bytes()),
                8 => value.extend_from_slice(&item.to_le_bytes()),
                _ => return Err("invalid unsigned integer width".into()),
            }
        }
    }
    Ok(())
}

fn read_bool_array(
    loader: &mut Loader,
    key: &[u8],
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, ElementKind::Bool))?
        .ok_or("missing")?;
    append_u32(value, TYPE_BOOL);
    append_u64(value, count);
    for index in 0..count {
        value.push(u8::from(
            loader
                .process_event(ReadBoolArrayElement::new(key, index))?
                .ok_or("missing")?,
        ));
    }
    Ok(())
}

fn read_string_array(
    loader: &mut Loader,
    key: &[u8],
    value: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = loader
        .process_event(ReadArrayLength::new(key, ElementKind::String))?
        .ok_or("missing")?;
    append_u32(value, TYPE_STRING);
    append_u64(value, count);
    loader
        .process_event(VisitStringArray::new(key, |_index: u32, item: &[u8]| {
            append_u64(
                value,
                u64::try_from(item.len()).expect("string-array item length fits u64"),
            );
            value.extend_from_slice(item);
        }))?
        .ok_or("missing")?;
    Ok(())
}

fn append_tensor_output(
    output: &mut String,
    loader: &mut Loader,
    count: u32,
    scratch: &mut ParityScratch,
) -> Result<(), Box<dyn std::error::Error>> {
    for index in 0..count {
        scratch.tensor_name.clear();
        let mut observed = None;
        loader
            .process_event(WithTensor::new(
                index,
                |name: &[u8], tensor: TensorDescriptor, data: &[u8]| {
                    // The actor owns these borrows. Copy only into preallocated caller storage,
                    // do not re-enter the loader, and retain neither borrow after this callback.
                    scratch.tensor_name.extend_from_slice(name);
                    observed = Some((tensor, fnv1a64(data)));
                },
            ))?
            .ok_or("missing tensor")?;
        let (tensor, hash) = observed.ok_or("tensor callback did not run")?;
        write!(output, "tensor.{index}.name=").unwrap();
        append_hex(output, &scratch.tensor_name);
        output.push('\n');
        writeln!(
            output,
            "tensor.{index}.type={}",
            tensor.tensor_type().wire_code()
        )
        .unwrap();
        let dimensions = tensor.dimensions();
        writeln!(
            output,
            "tensor.{index}.dims={},{},{},{}",
            dimensions[0], dimensions[1], dimensions[2], dimensions[3]
        )
        .unwrap();
        writeln!(output, "tensor.{index}.offset={}", tensor.data_offset()).unwrap();
        writeln!(output, "tensor.{index}.size={}", tensor.data_size()).unwrap();
        writeln!(output, "tensor.{index}.hash={hash:016x}").unwrap();
    }
    Ok(())
}

fn parity_output_capacity(source_bytes: usize, metadata_count: u32, tensor_count: u32) -> usize {
    let record_count =
        usize::try_from(metadata_count.saturating_add(tensor_count)).unwrap_or(usize::MAX);
    source_bytes
        .saturating_mul(3)
        .saturating_add(record_count.saturating_mul(256))
        .saturating_add(4096)
}

fn append_hex(output: &mut String, bytes: &[u8]) {
    for byte in bytes {
        write!(output, "{byte:02x}").unwrap();
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_string(bytes: &mut Vec<u8>, value: &[u8]) {
    append_u64(
        bytes,
        u64::try_from(value.len()).expect("fixture length fits u64"),
    );
    bytes.extend_from_slice(value);
}

fn append_header(bytes: &mut Vec<u8>, version: u32, tensors: u64, kv: u64) {
    bytes.extend_from_slice(&MAGIC);
    append_u32(bytes, version);
    append_u64(bytes, tensors);
    append_u64(bytes, kv);
}

fn append_scalar_kv(bytes: &mut Vec<u8>, key: &[u8], value_type: u32, value: &[u8]) {
    append_string(bytes, key);
    append_u32(bytes, value_type);
    bytes.extend_from_slice(value);
}

fn metadata_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(&mut bytes, VERSION, 0, 25);
    append_scalar_kv(&mut bytes, b"u8", 0, &[0xa5]);
    append_scalar_kv(&mut bytes, b"i8", 1, &[0x85]);
    append_scalar_kv(&mut bytes, b"u16", 2, &0xa55a_u16.to_le_bytes());
    append_scalar_kv(&mut bytes, b"i16", 3, &(-1234_i16).to_le_bytes());
    append_scalar_kv(&mut bytes, b"u32", 4, &0xa55a_1234_u32.to_le_bytes());
    append_scalar_kv(&mut bytes, b"i32", 5, &(-123_456_i32).to_le_bytes());
    append_scalar_kv(&mut bytes, b"f32", 6, &1.25_f32.to_le_bytes());
    append_scalar_kv(&mut bytes, b"bool", TYPE_BOOL, &[1]);
    let mut string = Vec::new();
    append_string(&mut string, b"hello");
    append_scalar_kv(&mut bytes, b"string", TYPE_STRING, &string);
    append_scalar_kv(
        &mut bytes,
        b"u64",
        10,
        &0xa55a_1234_5678_9abc_u64.to_le_bytes(),
    );
    append_scalar_kv(&mut bytes, b"i64", 11, &(-123_456_789_i64).to_le_bytes());
    append_scalar_kv(
        &mut bytes,
        b"f64",
        12,
        &f64::from_bits(0x3ff0_0000_0000_0001).to_le_bytes(),
    );
    append_array_kv(&mut bytes, b"array.u8", 0, &[1, 2, 3]);
    append_array_kv(&mut bytes, b"array.i8", 1, &[0xff, 0x7f]);
    append_array_kv(
        &mut bytes,
        b"array.u16",
        2,
        &[1_u16.to_le_bytes(), u16::MAX.to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.i16",
        3,
        &[(-1_i16).to_le_bytes(), i16::MIN.to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.u32",
        4,
        &[1_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.i32",
        5,
        &[(-1_i32).to_le_bytes(), i32::MIN.to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.f32",
        6,
        &[1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.f64",
        12,
        &[
            f64::from_bits(0x4000_0000_0000_0001).to_le_bytes(),
            f64::from_bits(0xc004_0000_0000_0001).to_le_bytes(),
        ]
        .concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.u64",
        10,
        &[1_u64.to_le_bytes(), u64::MAX.to_le_bytes()].concat(),
    );
    append_array_kv(
        &mut bytes,
        b"array.i64",
        11,
        &[(-1_i64).to_le_bytes(), i64::MIN.to_le_bytes()].concat(),
    );
    append_array_kv(&mut bytes, b"array.bool", TYPE_BOOL, &[0, 1]);
    let mut strings = Vec::new();
    append_string(&mut strings, b"one");
    append_string(&mut strings, b"two");
    append_array_kv(&mut bytes, b"array.string", TYPE_STRING, &strings);
    append_scalar_kv(&mut bytes, b"general.alignment", 4, &32_u32.to_le_bytes());
    bytes
}

fn bool_normalization_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(&mut bytes, VERSION, 0, 2);
    append_scalar_kv(&mut bytes, b"bool.noncanonical", TYPE_BOOL, &[2]);
    append_array_kv(
        &mut bytes,
        b"array.bool.noncanonical",
        TYPE_BOOL,
        &[0, 2, 255],
    );
    bytes
}

fn custom_alignment_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(&mut bytes, VERSION, 1, 1);
    append_scalar_kv(&mut bytes, b"general.alignment", 4, &1_u32.to_le_bytes());
    append_string(&mut bytes, b"unaligned.f32");
    append_u32(&mut bytes, 1);
    append_u64(&mut bytes, 1);
    append_u32(&mut bytes, 0);
    append_u64(&mut bytes, 0);
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    bytes
}

fn append_array_kv(bytes: &mut Vec<u8>, key: &[u8], element_type: u32, payload: &[u8]) {
    append_string(bytes, key);
    append_u32(bytes, TYPE_ARRAY);
    append_u32(bytes, element_type);
    let element_size = match element_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        TYPE_STRING => 0,
        _ => unreachable!("fixture element type"),
    };
    let count = if element_type == TYPE_STRING {
        2
    } else {
        payload.len() / element_size
    };
    append_u64(bytes, u64::try_from(count).expect("fixture count fits u64"));
    bytes.extend_from_slice(payload);
}

fn tensor_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(
        &mut bytes,
        VERSION,
        u64::try_from(TENSOR_LAYOUTS.len() + PACKED_TENSOR_LAYOUTS.len())
            .expect("fixture count fits u64"),
        0,
    );
    let mut offset = 0_u64;
    for (tensor_type, block_size, type_size) in TENSOR_LAYOUTS {
        append_string(&mut bytes, format!("tensor.{tensor_type}").as_bytes());
        append_u32(&mut bytes, 1);
        append_u64(&mut bytes, *block_size);
        append_u32(&mut bytes, *tensor_type);
        append_u64(&mut bytes, offset);
        offset += u64::try_from(type_size.next_multiple_of(ALIGNMENT)).expect("size fits u64");
    }
    for (tensor_type, dimensions, type_size) in PACKED_TENSOR_LAYOUTS {
        append_string(&mut bytes, format!("tensor.{tensor_type}").as_bytes());
        append_u32(&mut bytes, 2);
        append_u64(&mut bytes, dimensions[0]);
        append_u64(&mut bytes, dimensions[1]);
        append_u32(&mut bytes, *tensor_type);
        append_u64(&mut bytes, offset);
        offset += u64::try_from(type_size.next_multiple_of(ALIGNMENT)).expect("size fits u64");
    }
    pad(&mut bytes, ALIGNMENT);
    for (_, _, type_size) in TENSOR_LAYOUTS {
        let start = bytes.len();
        bytes.resize(start + type_size, u8::try_from(type_size % 251).unwrap());
        pad(&mut bytes, ALIGNMENT);
    }
    for (_, _, type_size) in PACKED_TENSOR_LAYOUTS {
        let start = bytes.len();
        bytes.resize(start + type_size, u8::try_from(type_size % 251).unwrap());
        pad(&mut bytes, ALIGNMENT);
    }
    bytes
}

fn invalid_fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let mut bad_version = Vec::new();
    append_header(&mut bad_version, VERSION + 1, 0, 0);

    let mut impossible_kv_count = Vec::new();
    append_header(&mut impossible_kv_count, VERSION, 0, u64::from(u32::MAX));

    let mut impossible_tensor_count = Vec::new();
    append_header(
        &mut impossible_tensor_count,
        VERSION,
        u64::from(u32::MAX),
        0,
    );

    let mut empty_key = Vec::new();
    append_header(&mut empty_key, VERSION, 0, 1);
    append_scalar_kv(&mut empty_key, b"", 4, &1_u32.to_le_bytes());

    let mut duplicate_key = Vec::new();
    append_header(&mut duplicate_key, VERSION, 0, 2);
    append_scalar_kv(&mut duplicate_key, b"key", 4, &1_u32.to_le_bytes());
    append_scalar_kv(&mut duplicate_key, b"key", 4, &2_u32.to_le_bytes());

    let mut duplicate_tensor = Vec::new();
    append_header(&mut duplicate_tensor, VERSION, 2, 0);
    for offset in [0, 32] {
        append_string(&mut duplicate_tensor, b"tensor");
        append_u32(&mut duplicate_tensor, 1);
        append_u64(&mut duplicate_tensor, 1);
        append_u32(&mut duplicate_tensor, 0);
        append_u64(&mut duplicate_tensor, offset);
    }
    pad(&mut duplicate_tensor, ALIGNMENT);
    duplicate_tensor.resize(duplicate_tensor.len() + 64, 0);

    let mut truncated = tensor_fixture();
    truncated.pop();

    let mut invalid_alignment = Vec::new();
    append_header(&mut invalid_alignment, VERSION, 0, 1);
    append_scalar_kv(
        &mut invalid_alignment,
        b"general.alignment",
        4,
        &13_u32.to_le_bytes(),
    );

    let mut invalid_alignment_type = Vec::new();
    append_header(&mut invalid_alignment_type, VERSION, 0, 1);
    let mut alignment_string = Vec::new();
    append_string(&mut alignment_string, b"32");
    append_scalar_kv(
        &mut invalid_alignment_type,
        b"general.alignment",
        TYPE_STRING,
        &alignment_string,
    );

    let mut invalid_dimension_count = Vec::new();
    append_header(&mut invalid_dimension_count, VERSION, 1, 0);
    append_string(&mut invalid_dimension_count, b"too-many-dimensions");
    append_u32(&mut invalid_dimension_count, 5);
    invalid_dimension_count.extend_from_slice(&[0; 40]);
    append_u32(&mut invalid_dimension_count, 0);
    append_u64(&mut invalid_dimension_count, 0);

    let mut invalid_block_shape = Vec::new();
    append_header(&mut invalid_block_shape, VERSION, 1, 0);
    append_string(&mut invalid_block_shape, b"bad-block-shape");
    append_u32(&mut invalid_block_shape, 1);
    append_u64(&mut invalid_block_shape, 1);
    append_u32(&mut invalid_block_shape, 2);
    append_u64(&mut invalid_block_shape, 0);

    let mut deprecated_tensor_type = Vec::new();
    append_header(&mut deprecated_tensor_type, VERSION, 1, 0);
    append_string(&mut deprecated_tensor_type, b"deprecated");
    append_u32(&mut deprecated_tensor_type, 1);
    append_u64(&mut deprecated_tensor_type, 32);
    append_u32(&mut deprecated_tensor_type, 4);
    append_u64(&mut deprecated_tensor_type, 0);

    let mut unsupported_tensor_type = Vec::new();
    append_header(&mut unsupported_tensor_type, VERSION, 1, 0);
    append_string(&mut unsupported_tensor_type, b"unsupported");
    append_u32(&mut unsupported_tensor_type, 1);
    append_u64(&mut unsupported_tensor_type, 64);
    append_u32(&mut unsupported_tensor_type, 40);
    append_u64(&mut unsupported_tensor_type, 0);
    pad(&mut unsupported_tensor_type, ALIGNMENT);
    unsupported_tensor_type.resize(unsupported_tensor_type.len() + 64, 0);
    vec![
        ("invalid-version.gguf", bad_version),
        ("impossible-kv-count.gguf", impossible_kv_count),
        ("impossible-tensor-count.gguf", impossible_tensor_count),
        ("empty-key.gguf", empty_key),
        ("duplicate-key.gguf", duplicate_key),
        ("duplicate-tensor.gguf", duplicate_tensor),
        ("truncated-data.gguf", truncated),
        ("invalid-alignment.gguf", invalid_alignment),
        ("invalid-alignment-type.gguf", invalid_alignment_type),
        ("invalid-dimension-count.gguf", invalid_dimension_count),
        ("invalid-block-shape.gguf", invalid_block_shape),
        ("deprecated-tensor-type.gguf", deprecated_tensor_type),
        ("unsupported-tensor-type.gguf", unsupported_tensor_type),
    ]
}

fn pad(bytes: &mut Vec<u8>, alignment: usize) {
    bytes.resize(bytes.len().next_multiple_of(alignment), 0);
}

fn write_fixtures(directory: &Path) -> io::Result<()> {
    let valid_directory = directory.join("valid");
    let invalid_directory = directory.join("invalid");
    if valid_directory.exists() {
        fs::remove_dir_all(&valid_directory)?;
    }
    if invalid_directory.exists() {
        fs::remove_dir_all(&invalid_directory)?;
    }
    fs::create_dir_all(&valid_directory)?;
    fs::create_dir_all(&invalid_directory)?;
    let fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("empty-v2.gguf", empty_file(2)),
        ("empty-v3.gguf", empty_file(3)),
        ("metadata.gguf", metadata_fixture()),
        ("bool-normalization.gguf", bool_normalization_fixture()),
        ("custom-alignment.gguf", custom_alignment_fixture()),
        ("tensors.gguf", tensor_fixture()),
    ];
    for (name, bytes) in fixtures {
        fs::write(PathBuf::from(&valid_directory).join(name), bytes)?;
    }
    for (name, bytes) in invalid_fixtures() {
        fs::write(PathBuf::from(&invalid_directory).join(name), bytes)?;
    }
    Ok(())
}

fn empty_file(version: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(&mut bytes, version, 0, 0);
    bytes
}

#[cfg(test)]
mod tests {
    use allocation_counter::measure;

    use super::*;

    fn prepare_walk(bytes: &[u8]) -> (Loader, u32, u32, ParityScratch, String) {
        let mut loader = Loader::new();
        let probe = loader
            .process_event(Probe::new(Arc::from(bytes)))
            .expect("fixture probe");
        let metadata_count = probe.metadata_count();
        let tensor_count = probe.tensor_count();
        let scratch = ParityScratch::new(
            usize::try_from(probe.max_key_bytes()).expect("key capacity fits usize"),
            usize::try_from(probe.max_value_bytes()).expect("value capacity fits usize"),
            bytes.len(),
        );
        let output = String::with_capacity(parity_output_capacity(
            bytes.len(),
            metadata_count,
            tensor_count,
        ));
        loader
            .process_event(Bind::new(
                Storage::exact(probe).expect("exact fixture storage"),
            ))
            .expect("bind fixture");
        loader.process_event(Parse::new()).expect("parse fixture");
        (loader, metadata_count, tensor_count, scratch, output)
    }

    #[test]
    fn complete_generic_metadata_and_tensor_walk_does_not_allocate() {
        let metadata = metadata_fixture();
        let tensors = tensor_fixture();
        let (mut metadata_loader, metadata_count, _, mut metadata_scratch, mut metadata_output) =
            prepare_walk(&metadata);
        let (mut tensor_loader, _, tensor_count, mut tensor_scratch, mut tensor_output) =
            prepare_walk(&tensors);

        let allocations = measure(|| {
            append_kv_output(
                &mut metadata_output,
                &mut metadata_loader,
                metadata_count,
                &mut metadata_scratch,
            )
            .expect("walk every metadata descriptor and typed value");
            append_tensor_output(
                &mut tensor_output,
                &mut tensor_loader,
                tensor_count,
                &mut tensor_scratch,
            )
            .expect("walk every tensor descriptor and data view");
        });

        assert_eq!(allocations.count_total, 0);
        assert!(metadata_output.contains("kv.23.type=9"));
        assert!(tensor_output.contains("tensor.0.name="));
        assert!(tensor_output.contains("tensor.31.hash="));
    }
}
