//! Canonical Rust-side output for the pinned llama.cpp GGUF parity gate.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use emel_gguf::Loader;
use emel_gguf::event::{Load, Metadata, ParseDone};

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
    print!("{}", canonical_output(&bytes));
    Ok(())
}

fn canonical_output(bytes: &[u8]) -> String {
    let mut output = String::from("gguf-parity/v1\n");
    let Ok(model) = Loader::new().process_event(Load::new(bytes)) else {
        output.push_str("status=error\n");
        return output;
    };
    output.push_str("status=ok\n");
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("validated GGUF header"));
    let alignment = model
        .metadata()
        .find(|entry| entry.key() == b"general.alignment")
        .map(Metadata::value)
        .and_then(|value| value.try_into().ok())
        .map_or(32, u32::from_le_bytes);
    writeln!(output, "version={version}").unwrap();
    writeln!(output, "alignment={alignment}").unwrap();
    writeln!(output, "kv_count={}", model.metadata().len()).unwrap();
    writeln!(output, "tensor_count={}", model.tensors().len()).unwrap();
    append_kv_output(&mut output, &model);
    append_tensor_output(&mut output, &model);
    output
}

fn append_kv_output(output: &mut String, model: &ParseDone<'_>) {
    for (index, entry) in model.metadata().enumerate() {
        write!(output, "kv.{index}.key=").unwrap();
        append_hex(output, entry.key());
        output.push('\n');
        writeln!(output, "kv.{index}.type={}", entry.value_type()).unwrap();
        write!(output, "kv.{index}.value=").unwrap();
        let value = canonical_kv_value(entry.value_type(), entry.value());
        append_hex(output, &value);
        output.push('\n');
    }
}

fn canonical_kv_value(value_type: u32, serialized: &[u8]) -> Vec<u8> {
    let mut value = serialized.to_vec();
    if value_type == TYPE_BOOL {
        value[0] = u8::from(value[0] != 0);
    }
    if value_type == TYPE_ARRAY && value.len() >= 12 {
        let element_type = u32::from_le_bytes(value[..4].try_into().expect("array type"));
        if element_type == TYPE_BOOL {
            for element in &mut value[12..] {
                *element = u8::from(*element != 0);
            }
        }
    }
    value
}

fn append_tensor_output(output: &mut String, model: &ParseDone<'_>) {
    for (index, tensor) in model.tensors().enumerate() {
        write!(output, "tensor.{index}.name=").unwrap();
        append_hex(output, tensor.name());
        output.push('\n');
        writeln!(output, "tensor.{index}.type={}", tensor.tensor_type()).unwrap();
        let dimensions = tensor.dimensions();
        writeln!(
            output,
            "tensor.{index}.dims={},{},{},{}",
            dimensions[0], dimensions[1], dimensions[2], dimensions[3]
        )
        .unwrap();
        writeln!(output, "tensor.{index}.offset={}", tensor.data_offset()).unwrap();
        writeln!(output, "tensor.{index}.size={}", tensor.data_size()).unwrap();
        writeln!(
            output,
            "tensor.{index}.hash={:016x}",
            fnv1a64(tensor.data())
        )
        .unwrap();
    }
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
    append_header(&mut bytes, VERSION, 0, 16);
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
    append_scalar_kv(&mut bytes, b"f64", 12, &2.5_f64.to_le_bytes());
    append_array_kv(
        &mut bytes,
        b"array.u32",
        4,
        &[1_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat(),
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
        u64::try_from(TENSOR_LAYOUTS.len()).expect("fixture count fits u64"),
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
    pad(&mut bytes, ALIGNMENT);
    for (_, _, type_size) in TENSOR_LAYOUTS {
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
