#![no_main]

mod common;

use emel_gguf::Loader;
use emel_gguf::event::{
    Bind, ElementKind, MetadataDescriptor, Parse, Probe, QueryError, ReadArrayLength, ReadBool,
    ReadBoolArrayElement, ReadF32, ReadF32ArrayElement, ReadF64, ReadF64ArrayElement, ReadSigned,
    ReadSignedArrayElement, ReadStringArrayMetrics, ReadUnsigned, ReadUnsignedArrayElement,
    ReadUnsignedArrayMetrics, Storage, VisitF32Array, VisitStringArray, VisitUnsignedArray,
    WithByteArray, WithMetadataDescriptor, WithString, WithStringArrayElement,
};
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;

use common::{MAX_INPUT_BYTES, hash_bytes, requirements_are_bounded};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

const ELEMENT_KINDS: [ElementKind; 12] = [
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

// The generated parity corpus contains these exact metadata keys. Exercising
// them at index zero makes every scalar, byte-array, numeric-array, bool-array,
// and string-array decoder family reachable before arbitrary mutation begins.
const SEEDED_KEYS: [&[u8]; 25] = [
    b"u8",
    b"i8",
    b"u16",
    b"i16",
    b"u32",
    b"i32",
    b"f32",
    b"bool",
    b"string",
    b"u64",
    b"i64",
    b"f64",
    b"array.u8",
    b"array.i8",
    b"array.u16",
    b"array.i16",
    b"array.u32",
    b"array.i32",
    b"array.f32",
    b"array.f64",
    b"array.u64",
    b"array.i64",
    b"array.bool",
    b"array.string",
    b"general.alignment",
];

fn exercise_queries(loader: &mut Loader, key: &[u8], index: u64) -> u64 {
    let mut hash = FNV_OFFSET;
    hash_result(&mut hash, loader.process_event(ReadUnsigned::new(key)), |hash, value| {
        hash_bytes(hash, &value.to_le_bytes());
    });
    hash_result(&mut hash, loader.process_event(ReadSigned::new(key)), |hash, value| {
        hash_bytes(hash, &value.to_le_bytes());
    });
    hash_result(&mut hash, loader.process_event(ReadF32::new(key)), |hash, value| {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    });
    hash_result(&mut hash, loader.process_event(ReadF64::new(key)), |hash, value| {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    });
    hash_result(&mut hash, loader.process_event(ReadBool::new(key)), |hash, value| {
        hash_bytes(hash, &[u8::from(value)]);
    });
    hash_result(
        &mut hash,
        loader.process_event(WithString::new(key, slice_digest)),
        |hash, value| hash_bytes(hash, &value.to_le_bytes()),
    );

    for kind in ELEMENT_KINDS {
        hash_bytes(&mut hash, &[element_kind_tag(kind)]);
        hash_result(
            &mut hash,
            loader.process_event(ReadArrayLength::new(key, kind)),
            |hash, value| hash_bytes(hash, &value.to_le_bytes()),
        );
    }
    hash_result(
        &mut hash,
        loader.process_event(ReadStringArrayMetrics::new(key)),
        |hash, value| {
            hash_bytes(hash, &value.element_count().to_le_bytes());
            hash_bytes(hash, &value.total_string_bytes().to_le_bytes());
        },
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadUnsignedArrayMetrics::new(key)),
        |hash, value| {
            hash_bytes(hash, &value.element_count().to_le_bytes());
            hash_bytes(hash, &value.maximum().to_le_bytes());
        },
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadUnsignedArrayElement::new(key, index)),
        |hash, value| hash_bytes(hash, &value.to_le_bytes()),
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadSignedArrayElement::new(key, index)),
        |hash, value| hash_bytes(hash, &value.to_le_bytes()),
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadF32ArrayElement::new(key, index)),
        |hash, value| hash_bytes(hash, &value.to_bits().to_le_bytes()),
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadF64ArrayElement::new(key, index)),
        |hash, value| hash_bytes(hash, &value.to_bits().to_le_bytes()),
    );
    hash_result(
        &mut hash,
        loader.process_event(ReadBoolArrayElement::new(key, index)),
        |hash, value| hash_bytes(hash, &[u8::from(value)]),
    );
    hash_result(
        &mut hash,
        loader.process_event(WithStringArrayElement::new(key, index, slice_digest)),
        |hash, value| hash_bytes(hash, &value.to_le_bytes()),
    );
    let mut visitor_hash = FNV_OFFSET;
    let visit = loader.process_event(VisitStringArray::new(
        key,
        |visited_index: u32, value: &[u8]| {
            hash_bytes(&mut visitor_hash, &visited_index.to_le_bytes());
            hash_bytes(&mut visitor_hash, value);
        },
    ));
    hash_result(&mut hash, visit, |hash, value| {
        hash_bytes(hash, &value.to_le_bytes());
    });
    hash_bytes(&mut hash, &visitor_hash.to_le_bytes());
    let mut float_hash = FNV_OFFSET;
    let visit = loader.process_event(VisitF32Array::new(key, |visited_index: u32, value: f32| {
        hash_bytes(&mut float_hash, &visited_index.to_le_bytes());
        hash_bytes(&mut float_hash, &value.to_bits().to_le_bytes());
    }));
    hash_result(&mut hash, visit, |hash, value| hash_bytes(hash, &value.to_le_bytes()));
    hash_bytes(&mut hash, &float_hash.to_le_bytes());
    let mut unsigned_hash = FNV_OFFSET;
    let visit = loader.process_event(VisitUnsignedArray::new(key, |visited_index: u32, value: u64| {
        hash_bytes(&mut unsigned_hash, &visited_index.to_le_bytes());
        hash_bytes(&mut unsigned_hash, &value.to_le_bytes());
    }));
    hash_result(&mut hash, visit, |hash, value| hash_bytes(hash, &value.to_le_bytes()));
    hash_bytes(&mut hash, &unsigned_hash.to_le_bytes());
    hash_result(
        &mut hash,
        loader.process_event(WithByteArray::new(key, slice_digest)),
        |hash, value| hash_bytes(hash, &value.to_le_bytes()),
    );
    hash
}

fn slice_digest(value: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash_bytes(&mut hash, value);
    hash
}

fn hash_result<T, F>(hash: &mut u64, result: Result<Option<T>, QueryError>, encode: F)
where
    F: FnOnce(&mut u64, T),
{
    match result {
        Ok(Some(value)) => {
            hash_bytes(hash, &[0]);
            encode(hash, value);
        }
        Ok(None) => hash_bytes(hash, &[1]),
        Err(error) => hash_bytes(hash, &[2, query_error_tag(error)]),
    }
}

const fn query_error_tag(error: QueryError) -> u8 {
    match error {
        QueryError::NotParsed => 0,
        QueryError::TypeMismatch => 1,
        QueryError::IndexOutOfBounds => 2,
        QueryError::Range => 3,
        QueryError::Malformed => 4,
        QueryError::Internal => 5,
        _ => u8::MAX,
    }
}

const fn element_kind_tag(kind: ElementKind) -> u8 {
    match kind {
        ElementKind::Uint8 => 0,
        ElementKind::Int8 => 1,
        ElementKind::Uint16 => 2,
        ElementKind::Int16 => 3,
        ElementKind::Uint32 => 4,
        ElementKind::Int32 => 5,
        ElementKind::Float32 => 6,
        ElementKind::Bool => 7,
        ElementKind::String => 8,
        ElementKind::Uint64 => 10,
        ElementKind::Int64 => 11,
        ElementKind::Float64 => 12,
    }
}

fn descriptor_digest(loader: &mut Loader, index: u32) -> u64 {
    let mut hash = FNV_OFFSET;
    let result = loader.process_event(WithMetadataDescriptor::new(
        index,
        |key: &[u8], descriptor: MetadataDescriptor| {
            let mut observed = FNV_OFFSET;
            hash_bytes(&mut observed, key);
            hash_bytes(&mut observed, &[metadata_kind_tag(descriptor.kind())]);
            if let Some(kind) = descriptor.array_element_kind() {
                hash_bytes(&mut observed, &[element_kind_tag(kind)]);
            }
            if let Some(length) = descriptor.array_length() {
                hash_bytes(&mut observed, &length.to_le_bytes());
            }
            observed
        },
    ));
    hash_result(&mut hash, result, |hash, value| {
        hash_bytes(hash, &value.to_le_bytes());
    });
    hash
}

const fn metadata_kind_tag(kind: emel_gguf::event::MetadataKind) -> u8 {
    use emel_gguf::event::MetadataKind;
    match kind {
        MetadataKind::Uint8 => 0,
        MetadataKind::Int8 => 1,
        MetadataKind::Uint16 => 2,
        MetadataKind::Int16 => 3,
        MetadataKind::Uint32 => 4,
        MetadataKind::Int32 => 5,
        MetadataKind::Float32 => 6,
        MetadataKind::Bool => 7,
        MetadataKind::String => 8,
        MetadataKind::Array => 9,
        MetadataKind::Uint64 => 10,
        MetadataKind::Int64 => 11,
        MetadataKind::Float64 => 12,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut loader = Loader::new();
    let Ok(requirements) = loader.process_event(Probe::new(Arc::from(data))) else {
        return;
    };
    if !requirements_are_bounded(&requirements) {
        return;
    }
    let Ok(storage) = Storage::exact(requirements) else {
        return;
    };
    if loader.process_event(Bind::new(storage)).is_err()
        || loader.process_event(Parse::new()).is_err()
    {
        return;
    }

    let key_length = data.first().map_or(0, |value| usize::from(*value) % 32);
    let key = data.get(1..1 + key_length).unwrap_or(data);
    let index = data.get(1 + key_length).map_or(0, |value| u64::from(*value));

    assert_eq!(
        exercise_queries(&mut loader, key, index),
        exercise_queries(&mut loader, key, index)
    );
    let descriptor_index = u32::try_from(index).unwrap_or(u32::MAX);
    assert_eq!(
        descriptor_digest(&mut loader, descriptor_index),
        descriptor_digest(&mut loader, descriptor_index)
    );
    let mut seeded_hash = FNV_OFFSET;
    let mut repeated_seeded_hash = FNV_OFFSET;
    for seeded_key in SEEDED_KEYS {
        hash_bytes(
            &mut seeded_hash,
            &exercise_queries(&mut loader, seeded_key, 0).to_le_bytes(),
        );
        hash_bytes(
            &mut repeated_seeded_hash,
            &exercise_queries(&mut loader, seeded_key, 0).to_le_bytes(),
        );
    }
    assert_eq!(seeded_hash, repeated_seeded_hash);
});
