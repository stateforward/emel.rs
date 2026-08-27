//! Pre-dispatch GGUF catalog storage construction.

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{ParseDone, QueryError, TensorDescriptor, WithTensor};

use super::event::{Error, Storage, TensorInput};

pub(super) fn from_gguf(loader: &mut GgufLoader, parsed: ParseDone) -> Result<Storage, Error> {
    let tensor_count = parsed.tensor_count();
    let count = usize::try_from(tensor_count).map_err(|_| Error::Capacity)?;
    if count > super::MAX_TENSORS {
        return Err(Error::Capacity);
    }
    validate_exact_count(loader, tensor_count)?;
    if count == 0 {
        return Err(Error::ModelInvalid);
    }

    let mut name_bytes = 0usize;
    for index in 0..tensor_count {
        let mut observed = None;
        let result = loader.process_event(WithTensor::new(
            index,
            |name: &[u8], _: TensorDescriptor, _: &[u8]| {
                observed = Some(name.len());
            },
        ));
        map_present(&result)?;
        name_bytes = name_bytes
            .checked_add(observed.ok_or(Error::ModelInvalid)?)
            .ok_or(Error::Capacity)?;
    }

    let mut storage = Storage::with_capacity(count, name_bytes, count)?;
    for index in 0..tensor_count {
        let mut pushed = None;
        let result = loader.process_event(WithTensor::new(
            index,
            |name: &[u8], descriptor: TensorDescriptor, _: &[u8]| {
                let dimensions = convert_dimensions(descriptor.dimensions());
                let dimension_count = i32::try_from(descriptor.dimension_count()).ok();
                pushed = Some(match (dimensions, dimension_count) {
                    (Some(dimensions), Some(dimension_count)) => {
                        storage.push_tensor(TensorInput::new(
                            name,
                            descriptor.tensor_type().wire_code(),
                            dimension_count,
                            dimensions,
                            descriptor.data_size(),
                            true,
                        ))
                    }
                    _ => Err(Error::ModelInvalid),
                });
            },
        ));
        map_present(&result)?;
        pushed.ok_or(Error::ModelInvalid)??;
    }
    Ok(storage)
}

fn validate_exact_count(loader: &mut GgufLoader, tensor_count: u32) -> Result<(), Error> {
    let result = loader.process_event(WithTensor::new(
        tensor_count,
        |_: &[u8], _: TensorDescriptor, _: &[u8]| (),
    ));
    match result {
        Ok(None) => Ok(()),
        Ok(Some(())) | Err(QueryError::IndexOutOfBounds | QueryError::Malformed) => {
            Err(Error::ModelInvalid)
        }
        Err(QueryError::NotParsed) => Err(Error::StorageUnavailable),
        Err(QueryError::Internal) => Err(Error::Internal),
        Err(_) => Err(Error::ModelInvalid),
    }
}

const fn map_present<R>(result: &Result<Option<R>, QueryError>) -> Result<(), Error> {
    match result {
        Ok(Some(_)) => Ok(()),
        Ok(None) | Err(QueryError::IndexOutOfBounds | QueryError::Malformed) => {
            Err(Error::ModelInvalid)
        }
        Err(QueryError::NotParsed) => Err(Error::StorageUnavailable),
        Err(QueryError::Internal) => Err(Error::Internal),
        Err(_) => Err(Error::ModelInvalid),
    }
}

fn convert_dimensions(values: [u64; 4]) -> Option<[i64; 4]> {
    Some([
        i64::try_from(values[0]).ok()?,
        i64::try_from(values[1]).ok()?,
        i64::try_from(values[2]).ok()?,
        i64::try_from(values[3]).ok()?,
    ])
}
