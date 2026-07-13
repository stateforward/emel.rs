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
const MAX_INITIAL_RECORD_RESERVATION: usize = 4096;

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
    let mut keys = Vec::<&[u8]>::new();
    keys.try_reserve_exact(
        usize::try_from(kv_count)
            .map_err(|_| Error::Capacity)?
            .min(MAX_INITIAL_RECORD_RESERVATION),
    )
    .map_err(|_| Error::Capacity)?;

    for _ in 0..kv_count {
        let key = reader.string().ok_or(Error::ParseFailed)?;
        if key.is_empty() || keys.contains(&key) {
            return Err(Error::ModelInvalid);
        }
        keys.push(key);
        let value_type = reader.u32().ok_or(Error::ParseFailed)?;
        let value_size = scan_value(&mut reader, value_type, key, &mut alignment)?;
        requirements.max_key_bytes = requirements
            .max_key_bytes
            .max(u32::try_from(key.len()).map_err(|_| Error::Capacity)?);
        requirements.max_value_bytes = requirements
            .max_value_bytes
            .max(u32::try_from(value_size).map_err(|_| Error::Capacity)?);
    }

    let mut tensor_names = Vec::<&[u8]>::new();
    tensor_names
        .try_reserve_exact(
            usize::try_from(tensor_count)
                .map_err(|_| Error::Capacity)?
                .min(MAX_INITIAL_RECORD_RESERVATION),
        )
        .map_err(|_| Error::Capacity)?;
    for _ in 0..tensor_count {
        let name = reader.string().ok_or(Error::ParseFailed)?;
        if name.len() >= MAX_TENSOR_NAME_LENGTH || tensor_names.contains(&name) {
            return Err(Error::ModelInvalid);
        }
        tensor_names.push(name);
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
    use super::{Error, tensor_data_size};

    #[test]
    fn types_after_pinned_ggml_count_are_rejected() {
        assert_eq!(
            tensor_data_size([64, 1, 1, 1], 1, 40),
            Err(Error::ModelInvalid)
        );
    }
}
