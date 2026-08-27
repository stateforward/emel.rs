use super::event::{
    Error, NameId, TensorBindingStatus, TensorDescriptor, TensorDescriptorFacts, TensorId,
};

pub(super) const EMPTY_INDEX: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Record {
    pub(super) name_offset: u32,
    pub(super) name_length: u32,
    pub(super) wire_type: u32,
    pub(super) dimension_count: i32,
    pub(super) dimensions: [i64; 4],
    pub(super) data_size: u64,
    pub(super) binding_present: bool,
}

pub(super) fn range(bytes: &[u8], offset: u32, length: u32) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    bytes.get(start..start.checked_add(length)?)
}

pub(super) fn hash(name: &[u8]) -> u64 {
    let mut value = 0xcbf2_9ce4_8422_2325u64;
    for byte in name {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    value
}

pub(super) fn descriptor(
    owner: u64,
    generation: u64,
    ordinal: u32,
    record: Record,
    binding_status: TensorBindingStatus,
) -> Result<TensorDescriptor, Error> {
    let tensor_type = emel_tensor::dtype::SerializedType::try_from(record.wire_type)
        .map_err(|_| Error::ModelInvalid)?;
    Ok(TensorDescriptor::new(
        TensorId {
            owner,
            generation,
            ordinal,
        },
        NameId {
            owner,
            generation,
            ordinal,
        },
        TensorDescriptorFacts {
            tensor_type,
            dimension_count: record.dimension_count,
            dimensions: record.dimensions,
            data_size: record.data_size,
            binding_present: record.binding_present,
            binding_status,
        },
    ))
}
