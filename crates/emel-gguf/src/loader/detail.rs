//! Bounds-checked GGUF scanner ported from `emel.cpp`.

use super::{Error, KvEntry, Requirements, TensorInfo};
use crate::{DEFAULT_ALIGNMENT, MAGIC, MAX_TENSOR_DIMS, MIN_VERSION, VERSION};

pub(super) const TYPE_UINT8: u32 = 0;
pub(super) const TYPE_INT8: u32 = 1;
pub(super) const TYPE_UINT16: u32 = 2;
pub(super) const TYPE_INT16: u32 = 3;
pub(super) const TYPE_UINT32: u32 = 4;
pub(super) const TYPE_INT32: u32 = 5;
pub(super) const TYPE_FLOAT32: u32 = 6;
pub(super) const TYPE_BOOL: u32 = 7;
pub(super) const TYPE_STRING: u32 = 8;
pub(super) const TYPE_ARRAY: u32 = 9;
pub(super) const TYPE_UINT64: u32 = 10;
pub(super) const TYPE_INT64: u32 = 11;
pub(super) const TYPE_FLOAT64: u32 = 12;
const TYPE_COUNT: u32 = 13;
const GENERAL_ALIGNMENT: &[u8] = b"general.alignment";
const MAX_STRING_LENGTH: u64 = 1024 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 1024 * 1024 * 1024;
const MAX_TENSOR_NAME_LENGTH: usize = 64;
const SEEN_NAME_WORDS: usize = 32;

/// Allocation-free prefilter for exact duplicate-name validation.
///
/// A negative answer is conclusive. A positive answer is only a collision
/// candidate and is resolved by byte-exactly rescanning prior records.
struct SeenNames {
    words: [u64; SEEN_NAME_WORDS],
}

impl SeenNames {
    const fn new() -> Self {
        Self {
            words: [0; SEEN_NAME_WORDS],
        }
    }

    fn observe(&mut self, name: &[u8]) -> bool {
        let bit = name_signature(name) & (SEEN_NAME_WORDS * 64 - 1);
        let mask = 1_u64 << (bit % 64);
        let word = &mut self.words[bit / 64];
        let seen = *word & mask != 0;
        *word |= mask;
        seen
    }
}

fn name_signature(name: &[u8]) -> usize {
    let length = name.len();
    let sample = |index: usize| usize::from(name.get(index).copied().unwrap_or_default());
    length.wrapping_mul(0x9e37)
        ^ sample(0).wrapping_mul(0x85eb)
        ^ sample(length / 4).wrapping_mul(0xc2b2)
        ^ sample(length / 2).wrapping_mul(0x27d4)
        ^ sample(length.saturating_sub(2)).wrapping_mul(0x1656)
        ^ sample(length.saturating_sub(1)).wrapping_mul(0xd3a2)
}

#[derive(Clone, Copy)]
struct GgmlLayout {
    block_size: u64,
    type_size: u64,
}

const fn ggml_layout(tensor_type: u32) -> Option<GgmlLayout> {
    let (block_size, type_size) = match tensor_type {
        0 | 26 => (1, 4),
        1 | 25 | 30 => (1, 2),
        2 | 20 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 | 21 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 | 35 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        27 | 28 => (1, 8),
        29 => (256, 56),
        34 => (256, 54),
        39 => (32, 17),
        _ => return None,
    };
    Some(GgmlLayout {
        block_size,
        type_size,
    })
}

const fn scalar_size(value_type: u32) -> Option<usize> {
    match value_type {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn tensor_data_size(
    dimensions: [u64; 4],
    dimension_count: u32,
    tensor_type: u32,
) -> Result<u64, Error> {
    let layout = ggml_layout(tensor_type).ok_or(Error::ModelInvalid)?;
    let first_dimension = if dimension_count == 0 {
        1
    } else {
        dimensions[0]
    };
    if !first_dimension.is_multiple_of(layout.block_size) {
        return Err(Error::ModelInvalid);
    }
    let count = dimensions[..usize::try_from(dimension_count).map_err(|_| Error::Capacity)?]
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count.checked_mul(*dimension).ok_or(Error::ModelInvalid)
        })?;
    if count >= i64::MAX as u64 {
        return Err(Error::ModelInvalid);
    }
    (count / layout.block_size)
        .checked_mul(layout.type_size)
        .ok_or(Error::Capacity)
}

fn align(value: usize, alignment: u32) -> Result<usize, Error> {
    let alignment = usize::try_from(alignment).map_err(|_| Error::Capacity)?;
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(Error::Capacity)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(count)?;
        let value = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(value)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<&'a [u8]> {
        let length = self.u64()?;
        if length > MAX_STRING_LENGTH {
            return None;
        }
        let length = usize::try_from(length).ok()?;
        self.take(length)
    }

    fn skip(&mut self, count: usize) -> bool {
        self.take(count).is_some()
    }

    fn align_to(&mut self, alignment: u32) -> Result<(), Error> {
        let aligned = align(self.offset, alignment)?;
        if aligned > self.bytes.len() {
            return Err(Error::ParseFailed);
        }
        self.offset = aligned;
        Ok(())
    }
}

fn read_header(reader: &mut Reader<'_>) -> Option<(u32, u64, u64)> {
    if reader.take(MAGIC.len())? != MAGIC {
        return None;
    }
    let version = reader.u32()?;
    let tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    (MIN_VERSION..=VERSION)
        .contains(&version)
        .then_some((version, tensor_count, kv_count))
}

const fn header_error(file_image: &[u8]) -> Error {
    if file_image.len() < MAGIC.len() {
        Error::ParseFailed
    } else {
        Error::ModelInvalid
    }
}

fn scan_value(
    reader: &mut Reader<'_>,
    value_type: u32,
    key: &[u8],
    alignment: &mut u32,
) -> Result<usize, Error> {
    let start = reader.offset;
    if value_type >= TYPE_COUNT {
        return Err(Error::ModelInvalid);
    }
    if key == GENERAL_ALIGNMENT && value_type != TYPE_UINT32 {
        return Err(Error::ModelInvalid);
    }
    if value_type == TYPE_STRING {
        reader.string().ok_or(Error::ParseFailed)?;
        return Ok(reader.offset - start);
    }
    if value_type == TYPE_ARRAY {
        let element_type = reader.u32().ok_or(Error::ParseFailed)?;
        let count = reader.u64().ok_or(Error::ParseFailed)?;
        if element_type >= TYPE_COUNT || element_type == TYPE_ARRAY || count > MAX_ARRAY_ELEMENTS {
            return Err(Error::ModelInvalid);
        }
        if element_type == TYPE_STRING {
            for _ in 0..count {
                reader.string().ok_or(Error::ParseFailed)?;
            }
        } else {
            let element_size = scalar_size(element_type).ok_or(Error::ModelInvalid)?;
            let count = usize::try_from(count).map_err(|_| Error::Capacity)?;
            let payload = count.checked_mul(element_size).ok_or(Error::Capacity)?;
            if !reader.skip(payload) {
                return Err(Error::ParseFailed);
            }
        }
        return Ok(reader.offset - start);
    }
    let size = scalar_size(value_type).ok_or(Error::ModelInvalid)?;
    if key == GENERAL_ALIGNMENT {
        if value_type != TYPE_UINT32 {
            return Err(Error::ModelInvalid);
        }
        let value = reader.u32().ok_or(Error::ParseFailed)?;
        if value == 0 || !value.is_power_of_two() {
            return Err(Error::ModelInvalid);
        }
        *alignment = value;
    } else if !reader.skip(size) {
        return Err(Error::ParseFailed);
    }
    Ok(reader.offset - start)
}

fn metadata_key_previously_seen(
    file_image: &[u8],
    metadata_start: usize,
    prior_count: u32,
    candidate: &[u8],
) -> Result<bool, Error> {
    let mut reader = Reader {
        bytes: file_image,
        offset: metadata_start,
    };
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..prior_count {
        let key = reader.string().ok_or(Error::ParseFailed)?;
        if key == candidate {
            return Ok(true);
        }
        let value_type = reader.u32().ok_or(Error::ParseFailed)?;
        scan_value(&mut reader, value_type, key, &mut alignment)?;
    }
    Ok(false)
}

fn tensor_name_previously_seen(
    file_image: &[u8],
    tensors_start: usize,
    prior_count: u32,
    candidate: &[u8],
) -> Result<bool, Error> {
    let mut reader = Reader {
        bytes: file_image,
        offset: tensors_start,
    };
    for _ in 0..prior_count {
        let name = reader.string().ok_or(Error::ParseFailed)?;
        if name == candidate {
            return Ok(true);
        }
        let dimension_count = reader.u32().ok_or(Error::ParseFailed)?;
        if dimension_count > MAX_TENSOR_DIMS {
            return Err(Error::ModelInvalid);
        }
        for _ in 0..dimension_count {
            reader.u64().ok_or(Error::ParseFailed)?;
        }
        reader.u32().ok_or(Error::ParseFailed)?;
        reader.u64().ok_or(Error::ParseFailed)?;
    }
    Ok(false)
}

pub(super) fn probe(file_image: &[u8]) -> Result<Requirements, Error> {
    let mut reader = Reader::new(file_image);
    let (_version, tensor_count, kv_count) =
        read_header(&mut reader).ok_or_else(|| header_error(file_image))?;
    let tensor_count = u32::try_from(tensor_count).map_err(|_| Error::Capacity)?;
    let kv_count = u32::try_from(kv_count).map_err(|_| Error::Capacity)?;
    let mut requirements = Requirements {
        tensor_count,
        kv_count,
        ..Requirements::default()
    };
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut expected_tensor_offset = 0_u64;
    let metadata_start = reader.offset;
    let mut metadata_names = SeenNames::new();
    for entry_index in 0..kv_count {
        let key = reader.string().ok_or(Error::ParseFailed)?;
        let duplicate = metadata_names.observe(key)
            && metadata_key_previously_seen(file_image, metadata_start, entry_index, key)?;
        if key.is_empty() || duplicate {
            return Err(Error::ModelInvalid);
        }
        let value_type = reader.u32().ok_or(Error::ParseFailed)?;
        let value_size = scan_value(&mut reader, value_type, key, &mut alignment)?;
        requirements.max_key_bytes = requirements
            .max_key_bytes
            .max(u32::try_from(key.len()).map_err(|_| Error::Capacity)?);
        requirements.max_value_bytes = requirements
            .max_value_bytes
            .max(u32::try_from(value_size).map_err(|_| Error::Capacity)?);
    }

    let tensors_start = reader.offset;
    let mut tensor_names = SeenNames::new();
    for tensor_index in 0..tensor_count {
        let name = reader.string().ok_or(Error::ParseFailed)?;
        let duplicate = tensor_names.observe(name)
            && tensor_name_previously_seen(file_image, tensors_start, tensor_index, name)?;
        if name.len() >= MAX_TENSOR_NAME_LENGTH || duplicate {
            return Err(Error::ModelInvalid);
        }
        let dimension_count = reader.u32().ok_or(Error::ParseFailed)?;
        if dimension_count > MAX_TENSOR_DIMS {
            return Err(Error::ModelInvalid);
        }
        let mut dimensions = [1_u64; 4];
        for dimension in
            &mut dimensions[..usize::try_from(dimension_count).map_err(|_| Error::Capacity)?]
        {
            *dimension = reader.u64().ok_or(Error::ParseFailed)?;
            if *dimension > i64::MAX as u64 {
                return Err(Error::ModelInvalid);
            }
        }
        let tensor_type = reader.u32().ok_or(Error::ParseFailed)?;
        let data_offset = reader.u64().ok_or(Error::ParseFailed)?;
        let data_size = tensor_data_size(dimensions, dimension_count, tensor_type)?;
        requirements.tensor_data_bytes = requirements
            .tensor_data_bytes
            .checked_add(data_size)
            .ok_or(Error::Capacity)?;
        if data_offset != expected_tensor_offset {
            return Err(Error::ParseFailed);
        }
        let padded = u64::try_from(align(
            usize::try_from(data_size).map_err(|_| Error::Capacity)?,
            alignment,
        )?)
        .map_err(|_| Error::Capacity)?;
        expected_tensor_offset = expected_tensor_offset
            .checked_add(padded)
            .ok_or(Error::Capacity)?;
    }

    if tensor_count != 0 {
        reader.align_to(alignment)?;
    }
    let required_size = u64::try_from(reader.offset)
        .map_err(|_| Error::Capacity)?
        .checked_add(expected_tensor_offset)
        .ok_or(Error::Capacity)?;
    if required_size > u64::try_from(file_image.len()).map_err(|_| Error::Capacity)? {
        return Err(Error::ParseFailed);
    }
    Ok(requirements)
}

#[allow(clippy::too_many_lines)]
pub(super) fn parse(
    file_image: &[u8],
    requirements: Requirements,
    kv_arena: &mut [u8],
    kv_entries: &mut [KvEntry],
    tensors: &mut [TensorInfo],
) -> Result<(), Error> {
    if kv_entries.len() < usize::try_from(requirements.kv_count).map_err(|_| Error::Capacity)?
        || tensors.len()
            < usize::try_from(requirements.tensor_count).map_err(|_| Error::Capacity)?
        || kv_arena.len() < required_kv_arena_bytes(requirements)?
    {
        return Err(Error::Capacity);
    }
    let mut reader = Reader::new(file_image);
    let (_version, tensor_count, kv_count) =
        read_header(&mut reader).ok_or_else(|| header_error(file_image))?;
    if tensor_count != u64::from(requirements.tensor_count)
        || kv_count != u64::from(requirements.kv_count)
    {
        return Err(Error::InvalidRequest);
    }
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut arena_cursor = 0_usize;
    let mut expected_tensor_offset = 0_u64;

    let kv_count = usize::try_from(kv_count).map_err(|_| Error::Capacity)?;
    for entry_index in 0..kv_count {
        let key = reader.string().ok_or(Error::ParseFailed)?;
        let duplicate_key = kv_entries[..entry_index].iter().any(|entry| {
            let start = usize::try_from(entry.key_offset).ok();
            let length = usize::try_from(entry.key_length).ok();
            start
                .zip(length)
                .and_then(|(start, length)| kv_arena.get(start..start.checked_add(length)?))
                == Some(key)
        });
        if key.is_empty() || duplicate_key {
            return Err(Error::ModelInvalid);
        }
        let value_type = reader.u32().ok_or(Error::ParseFailed)?;
        let value_start = reader.offset;
        let value_size = scan_value(&mut reader, value_type, key, &mut alignment)?;
        if key.len() > usize::try_from(requirements.max_key_bytes).map_err(|_| Error::Capacity)?
            || value_size
                > usize::try_from(requirements.max_value_bytes).map_err(|_| Error::Capacity)?
        {
            return Err(Error::Capacity);
        }
        let entry_offset = arena_cursor;
        arena_cursor = arena_cursor
            .checked_add(key.len())
            .and_then(|cursor| cursor.checked_add(value_size))
            .ok_or(Error::Capacity)?;
        if arena_cursor > kv_arena.len() {
            return Err(Error::Capacity);
        }
        kv_arena[entry_offset..entry_offset + key.len()].copy_from_slice(key);
        kv_arena[entry_offset + key.len()..arena_cursor]
            .copy_from_slice(&file_image[value_start..value_start + value_size]);
        kv_entries[entry_index] = KvEntry {
            key_offset: u32::try_from(entry_offset).map_err(|_| Error::Capacity)?,
            key_length: u32::try_from(key.len()).map_err(|_| Error::Capacity)?,
            value_offset: u32::try_from(entry_offset + key.len()).map_err(|_| Error::Capacity)?,
            value_length: u32::try_from(value_size).map_err(|_| Error::Capacity)?,
            value_type,
        };
    }

    let tensor_count = usize::try_from(tensor_count).map_err(|_| Error::Capacity)?;
    for tensor_index in 0..tensor_count {
        let name_offset = reader.offset.checked_add(8).ok_or(Error::Capacity)?;
        let name = reader.string().ok_or(Error::ParseFailed)?;
        let duplicate_name = tensors[..tensor_index].iter().any(|tensor| {
            let start = usize::try_from(tensor.name_offset).ok();
            let length = usize::try_from(tensor.name_length).ok();
            start
                .zip(length)
                .and_then(|(start, length)| file_image.get(start..start.checked_add(length)?))
                == Some(name)
        });
        if name.len() >= MAX_TENSOR_NAME_LENGTH || duplicate_name {
            return Err(Error::ModelInvalid);
        }
        let dimension_count = reader.u32().ok_or(Error::ParseFailed)?;
        if dimension_count > MAX_TENSOR_DIMS {
            return Err(Error::ModelInvalid);
        }
        let mut dimensions = [1_u64; 4];
        for dimension in
            &mut dimensions[..usize::try_from(dimension_count).map_err(|_| Error::Capacity)?]
        {
            *dimension = reader.u64().ok_or(Error::ParseFailed)?;
            if *dimension > i64::MAX as u64 {
                return Err(Error::ModelInvalid);
            }
        }
        let tensor_type = reader.u32().ok_or(Error::ParseFailed)?;
        let data_offset = reader.u64().ok_or(Error::ParseFailed)?;
        let data_size = tensor_data_size(dimensions, dimension_count, tensor_type)?;
        if data_offset != expected_tensor_offset {
            return Err(Error::ParseFailed);
        }
        let padded = u64::try_from(align(
            usize::try_from(data_size).map_err(|_| Error::Capacity)?,
            alignment,
        )?)
        .map_err(|_| Error::Capacity)?;
        expected_tensor_offset = expected_tensor_offset
            .checked_add(padded)
            .ok_or(Error::Capacity)?;
        tensors[tensor_index] = TensorInfo {
            name_offset: u32::try_from(name_offset).map_err(|_| Error::Capacity)?,
            name_length: u32::try_from(name.len()).map_err(|_| Error::Capacity)?,
            tensor_type,
            dimension_count,
            dimensions,
            data_offset,
            file_offset: 0,
            data_size,
            file_index: 0,
        };
    }

    if tensor_count != 0 {
        reader.align_to(alignment)?;
    }
    let data_section_offset = u64::try_from(reader.offset).map_err(|_| Error::Capacity)?;
    let required_size = data_section_offset
        .checked_add(expected_tensor_offset)
        .ok_or(Error::Capacity)?;
    if required_size > u64::try_from(file_image.len()).map_err(|_| Error::Capacity)? {
        return Err(Error::ParseFailed);
    }
    for tensor in tensors.iter_mut().take(tensor_count) {
        tensor.file_offset = data_section_offset
            .checked_add(tensor.data_offset)
            .ok_or(Error::Capacity)?;
        let end = tensor
            .file_offset
            .checked_add(tensor.data_size)
            .ok_or(Error::Capacity)?;
        if end > u64::try_from(file_image.len()).map_err(|_| Error::Capacity)? {
            return Err(Error::ParseFailed);
        }
    }
    Ok(())
}

pub(super) fn required_kv_arena_bytes(requirements: Requirements) -> Result<usize, Error> {
    let entry_size = usize::try_from(requirements.max_key_bytes)
        .map_err(|_| Error::Capacity)?
        .checked_add(usize::try_from(requirements.max_value_bytes).map_err(|_| Error::Capacity)?)
        .ok_or(Error::Capacity)?;
    usize::try_from(requirements.kv_count)
        .map_err(|_| Error::Capacity)?
        .checked_mul(entry_size)
        .ok_or(Error::Capacity)
}

#[cfg(test)]
mod tests {
    use super::{Error, KvEntry, TensorInfo, parse, probe, tensor_data_size};

    const TYPE_UINT8: u32 = 0;
    const TYPE_UINT16: u32 = 2;
    const TYPE_UINT32: u32 = 4;
    const TYPE_FLOAT32: u32 = 6;
    const TYPE_STRING: u32 = 8;
    const TYPE_ARRAY: u32 = 9;
    const TYPE_UINT64: u32 = 10;

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

    fn append_value(bytes: &mut Vec<u8>, key: &[u8], value_type: u32, payload: &[u8]) {
        append_string(bytes, key);
        append_u32(bytes, value_type);
        bytes.extend_from_slice(payload);
    }

    fn array(element_type: u32, count: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_u32(&mut bytes, element_type);
        append_u64(&mut bytes, count);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn parser_fixture() -> Vec<u8> {
        let entries = [
            (
                b"general.alignment".as_slice(),
                TYPE_UINT32,
                64_u32.to_le_bytes().to_vec(),
            ),
            (b"byte".as_slice(), TYPE_UINT8, vec![7]),
            (
                b"word".as_slice(),
                TYPE_UINT16,
                513_u16.to_le_bytes().to_vec(),
            ),
            (
                b"wide".as_slice(),
                TYPE_UINT64,
                9_u64.to_le_bytes().to_vec(),
            ),
            (
                b"float".as_slice(),
                TYPE_FLOAT32,
                1.25_f32.to_le_bytes().to_vec(),
            ),
            (b"text".as_slice(), TYPE_STRING, {
                let mut value = Vec::new();
                append_string(&mut value, b"value");
                value
            }),
            (
                b"bytes".as_slice(),
                TYPE_ARRAY,
                array(TYPE_UINT8, 3, &[1, 2, 3]),
            ),
            (
                b"words".as_slice(),
                TYPE_ARRAY,
                array(TYPE_UINT16, 2, &[1, 0, 2, 0]),
            ),
            (
                b"wide-array".as_slice(),
                TYPE_ARRAY,
                array(TYPE_UINT64, 1, &5_u64.to_le_bytes()),
            ),
            (b"strings".as_slice(), TYPE_ARRAY, {
                let mut payload = Vec::new();
                append_string(&mut payload, b"one");
                append_string(&mut payload, b"two");
                array(TYPE_STRING, 2, &payload)
            }),
        ];

        let mut bytes = b"GGUF".to_vec();
        append_u32(&mut bytes, 3);
        append_u64(&mut bytes, 1);
        append_u64(&mut bytes, entries.len() as u64);
        for (key, value_type, payload) in entries {
            append_value(&mut bytes, key, value_type, &payload);
        }
        append_string(&mut bytes, b"weight");
        append_u32(&mut bytes, 1);
        append_u64(&mut bytes, 4);
        append_u32(&mut bytes, 0);
        append_u64(&mut bytes, 0);
        bytes.resize(bytes.len().next_multiple_of(64), 0);
        bytes.extend_from_slice(&[0; 64]);
        bytes
    }

    #[test]
    fn types_after_pinned_ggml_count_are_rejected() {
        assert_eq!(
            tensor_data_size([64, 1, 1, 1], 1, 40),
            Err(Error::ModelInvalid)
        );
    }

    #[test]
    fn scanner_and_parser_cover_bound_storage_without_public_bypasses() {
        let file = parser_fixture();
        let requirements = probe(&file).unwrap();
        let mut arena = vec![0; requirements.required_kv_arena_bytes().unwrap()];
        let mut entries = vec![KvEntry::default(); requirements.kv_count as usize];
        let mut tensors = vec![TensorInfo::default(); requirements.tensor_count as usize];

        parse(&file, requirements, &mut arena, &mut entries, &mut tensors).unwrap();

        assert_eq!(entries.len(), 10);
        assert_eq!(tensors[0].data_size, 16);
        assert_eq!(tensors[0].file_offset % 64, 0);
        let file_offset = usize::try_from(tensors[0].file_offset).unwrap();
        assert_eq!(&file[file_offset..file_offset + 16], &[0; 16]);
    }

    #[test]
    fn scanner_rejects_truncated_headers_and_invalid_alignment_values() {
        assert_eq!(probe(&[]), Err(Error::ParseFailed));
        assert_eq!(probe(b"nope"), Err(Error::ModelInvalid));

        let mut bytes = b"GGUF".to_vec();
        append_u32(&mut bytes, 3);
        append_u64(&mut bytes, 0);
        append_u64(&mut bytes, 1);
        append_value(
            &mut bytes,
            b"general.alignment",
            TYPE_UINT32,
            &3_u32.to_le_bytes(),
        );
        assert_eq!(probe(&bytes), Err(Error::ModelInvalid));
    }
}
