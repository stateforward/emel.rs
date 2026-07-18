#![allow(
    dead_code,
    reason = "shared helpers are compiled separately for each fuzz target"
)]

use emel_gguf::Loader;
use emel_gguf::event::{
    Bind, ElementKind, Error, MetadataDescriptor, MetadataKind, Parse, ParseDone, Probe, ProbeDone,
    ReadBool, ReadBoolArrayElement, ReadF32, ReadF32ArrayElement, ReadF64, ReadF64ArrayElement,
    ReadSigned, ReadSignedArrayElement, ReadUnsigned, ReadUnsignedArrayElement, Storage,
    TensorDescriptor, VisitStringArray, WithMetadataDescriptor, WithString, WithTensor,
};
use std::sync::Arc;

pub const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_KV_ARENA_BYTES: usize = 64 * 1024;
const MAX_KV_ENTRIES: u32 = 512;
const MAX_TENSORS: u32 = 512;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticSummary {
    pub metadata_count: u32,
    pub tensor_count: u32,
    pub metadata_hash: u64,
    pub tensor_hash: u64,
}

pub fn requirements_are_bounded(requirements: &ProbeDone) -> bool {
    requirements.metadata_count() <= MAX_KV_ENTRIES
        && requirements.tensor_count() <= MAX_TENSORS
        && requirements
            .required_metadata_bytes()
            .is_ok_and(|bytes| bytes <= MAX_KV_ARENA_BYTES)
}

pub fn validate_model(model: ParseDone, requirements: &ProbeDone) {
    assert_eq!(model.metadata_count(), requirements.metadata_count());
    assert_eq!(model.tensor_count(), requirements.tensor_count());
}

pub trait LoaderExt {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error>;
    fn bind(&mut self, requirements: ProbeDone) -> Result<(), Error>;
    fn bind_with_capacity(
        &mut self,
        requirements: ProbeDone,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error>;
    fn parse(&mut self, file_image: &[u8]) -> Result<ParseDone, Error>;
}

impl LoaderExt for Loader {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error> {
        self.process_event(Probe::new(Arc::from(file_image)))
    }

    fn bind(&mut self, requirements: ProbeDone) -> Result<(), Error> {
        self.process_event(Bind::new(Storage::exact(requirements)?))
            .map_err(|error| error.error())
    }

    fn bind_with_capacity(
        &mut self,
        requirements: ProbeDone,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error> {
        self.process_event(Bind::new(Storage::with_capacity(
            requirements,
            metadata_bytes,
            metadata_entries,
            tensors,
        )?))
        .map_err(|error| error.error())
    }

    fn parse(&mut self, _file_image: &[u8]) -> Result<ParseDone, Error> {
        self.process_event(Parse::new())
    }
}

pub fn load(file_image: &[u8]) -> Result<ParseDone, Error> {
    let mut loader = Loader::new();
    let requirements = loader.probe(file_image)?;
    loader.bind(requirements)?;
    loader.parse(file_image)
}

pub fn semantic_load(file_image: &[u8]) -> Result<SemanticSummary, Error> {
    let mut loader = Loader::new();
    let requirements = loader.probe(file_image)?;
    let metadata_count = requirements.metadata_count();
    let tensor_count = requirements.tensor_count();
    let mut key = Vec::with_capacity(
        usize::try_from(requirements.max_key_bytes()).expect("bounded key length fits usize"),
    );
    loader.bind(requirements)?;
    let parsed = loader.parse(file_image)?;
    assert_eq!(parsed.metadata_count(), metadata_count);
    assert_eq!(parsed.tensor_count(), tensor_count);

    let metadata_hash = hash_metadata(&mut loader, metadata_count, &mut key);
    let tensor_hash = hash_tensors(&mut loader, tensor_count);
    Ok(SemanticSummary {
        metadata_count,
        tensor_count,
        metadata_hash,
        tensor_hash,
    })
}

fn hash_metadata(loader: &mut Loader, count: u32, key: &mut Vec<u8>) -> u64 {
    let mut hash = FNV_OFFSET;
    for index in 0..count {
        key.clear();
        let mut descriptor = None;
        loader
            .process_event(WithMetadataDescriptor::new(
                index,
                |borrowed_key: &[u8], observed: MetadataDescriptor| {
                    key.extend_from_slice(borrowed_key);
                    descriptor = Some(observed);
                },
            ))
            .expect("parsed metadata descriptor query")
            .expect("parsed metadata index exists");
        let descriptor = descriptor.expect("metadata callback ran");
        hash_bytes(&mut hash, &index.to_le_bytes());
        hash_bytes(&mut hash, key);
        hash_byte(&mut hash, metadata_kind_tag(descriptor.kind()));
        hash_metadata_value(loader, key, descriptor, &mut hash);
    }
    hash
}

fn hash_metadata_value(
    loader: &mut Loader,
    key: &[u8],
    descriptor: MetadataDescriptor,
    hash: &mut u64,
) {
    match descriptor.kind() {
        MetadataKind::Uint8
        | MetadataKind::Uint16
        | MetadataKind::Uint32
        | MetadataKind::Uint64 => hash_bytes(
            hash,
            &loader
                .process_event(ReadUnsigned::new(key))
                .expect("parsed unsigned query")
                .expect("descriptor key exists")
                .to_le_bytes(),
        ),
        MetadataKind::Int8
        | MetadataKind::Int16
        | MetadataKind::Int32
        | MetadataKind::Int64 => hash_bytes(
            hash,
            &loader
                .process_event(ReadSigned::new(key))
                .expect("parsed signed query")
                .expect("descriptor key exists")
                .to_le_bytes(),
        ),
        MetadataKind::Float32 => hash_bytes(
            hash,
            &loader
                .process_event(ReadF32::new(key))
                .expect("parsed f32 query")
                .expect("descriptor key exists")
                .to_bits()
                .to_le_bytes(),
        ),
        MetadataKind::Float64 => hash_bytes(
            hash,
            &loader
                .process_event(ReadF64::new(key))
                .expect("parsed f64 query")
                .expect("descriptor key exists")
                .to_bits()
                .to_le_bytes(),
        ),
        MetadataKind::Bool => hash_byte(
            hash,
            u8::from(
                loader
                    .process_event(ReadBool::new(key))
                    .expect("parsed bool query")
                    .expect("descriptor key exists"),
            ),
        ),
        MetadataKind::String => {
            loader
                .process_event(WithString::new(key, |value: &[u8]| {
                    hash_bytes(
                        hash,
                        &u64::try_from(value.len())
                            .expect("bounded string length fits u64")
                            .to_le_bytes(),
                    );
                    hash_bytes(hash, value);
                }))
                .expect("parsed string query")
                .expect("descriptor key exists");
        }
        MetadataKind::Array => hash_array(loader, key, descriptor, hash),
    }
}

fn hash_array(
    loader: &mut Loader,
    key: &[u8],
    descriptor: MetadataDescriptor,
    hash: &mut u64,
) {
    let kind = descriptor
        .array_element_kind()
        .expect("array descriptor has an element kind");
    let count = descriptor
        .array_length()
        .expect("array descriptor has a length");
    hash_byte(hash, element_kind_tag(kind));
    hash_bytes(hash, &count.to_le_bytes());
    if kind == ElementKind::String {
        loader
            .process_event(VisitStringArray::new(
                key,
                |index: u32, value: &[u8]| {
                    hash_bytes(hash, &index.to_le_bytes());
                    hash_bytes(
                        hash,
                        &u64::try_from(value.len())
                            .expect("bounded string length fits u64")
                            .to_le_bytes(),
                    );
                    hash_bytes(hash, value);
                },
            ))
            .expect("parsed string-array query")
            .expect("descriptor key exists");
        return;
    }

    for index in 0..count {
        match kind {
            ElementKind::Uint8
            | ElementKind::Uint16
            | ElementKind::Uint32
            | ElementKind::Uint64 => hash_bytes(
                hash,
                &loader
                    .process_event(ReadUnsignedArrayElement::new(key, index))
                    .expect("parsed unsigned-array query")
                    .expect("array element exists")
                    .to_le_bytes(),
            ),
            ElementKind::Int8
            | ElementKind::Int16
            | ElementKind::Int32
            | ElementKind::Int64 => hash_bytes(
                hash,
                &loader
                    .process_event(ReadSignedArrayElement::new(key, index))
                    .expect("parsed signed-array query")
                    .expect("array element exists")
                    .to_le_bytes(),
            ),
            ElementKind::Float32 => hash_bytes(
                hash,
                &loader
                    .process_event(ReadF32ArrayElement::new(key, index))
                    .expect("parsed f32-array query")
                    .expect("array element exists")
                    .to_bits()
                    .to_le_bytes(),
            ),
            ElementKind::Float64 => hash_bytes(
                hash,
                &loader
                    .process_event(ReadF64ArrayElement::new(key, index))
                    .expect("parsed f64-array query")
                    .expect("array element exists")
                    .to_bits()
                    .to_le_bytes(),
            ),
            ElementKind::Bool => hash_byte(
                hash,
                u8::from(
                    loader
                        .process_event(ReadBoolArrayElement::new(key, index))
                        .expect("parsed bool-array query")
                        .expect("array element exists"),
                ),
            ),
            ElementKind::String => unreachable!("string arrays use the visitor path"),
        }
    }
}

fn hash_tensors(loader: &mut Loader, count: u32) -> u64 {
    let mut hash = FNV_OFFSET;
    for index in 0..count {
        loader
            .process_event(WithTensor::new(
                index,
                |name: &[u8], descriptor: TensorDescriptor, data: &[u8]| {
                    hash_bytes(&mut hash, &index.to_le_bytes());
                    hash_bytes(&mut hash, name);
                    hash_bytes(
                        &mut hash,
                        &descriptor.tensor_type().wire_code().to_le_bytes(),
                    );
                    hash_bytes(&mut hash, &descriptor.dimension_count().to_le_bytes());
                    for dimension in descriptor.dimensions() {
                        hash_bytes(&mut hash, &dimension.to_le_bytes());
                    }
                    hash_bytes(&mut hash, &descriptor.data_offset().to_le_bytes());
                    hash_bytes(&mut hash, &descriptor.data_size().to_le_bytes());
                    hash_bytes(&mut hash, &descriptor.file_index().to_le_bytes());
                    hash_bytes(&mut hash, data);
                },
            ))
            .expect("parsed tensor query")
            .expect("parsed tensor index exists");
    }
    hash
}

fn hash_byte(hash: &mut u64, value: u8) {
    *hash = (*hash ^ u64::from(value)).wrapping_mul(FNV_PRIME);
}

pub fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        hash_byte(hash, *byte);
    }
}

const fn metadata_kind_tag(kind: MetadataKind) -> u8 {
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
