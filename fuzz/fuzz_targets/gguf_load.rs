#![no_main]

mod common;

use emel_gguf::Loader;
use emel_gguf::event::{Error, ParseDone, ProbeDone};
use libfuzzer_sys::fuzz_target;

use common::{LoaderExt as _, MAX_INPUT_BYTES, load, requirements_are_bounded, validate_model};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Eq, PartialEq)]
enum Outcome {
    Loaded(ModelSummary),
    Error(Error),
}

#[derive(Debug, Eq, PartialEq)]
struct ModelSummary {
    requirements: ProbeDone,
    metadata_hash: u64,
    tensor_hash: u64,
}

impl ModelSummary {
    fn new(model: &ParseDone<'_>) -> Self {
        validate_model(model);
        let mut metadata_hash = FNV_OFFSET;
        for entry in model.metadata() {
            hash_bytes(&mut metadata_hash, entry.key());
            hash_bytes(&mut metadata_hash, &entry.value_type().to_le_bytes());
            hash_bytes(&mut metadata_hash, entry.value());
        }

        let mut tensor_hash = FNV_OFFSET;
        for tensor in model.tensors() {
            hash_bytes(&mut tensor_hash, tensor.name());
            hash_bytes(&mut tensor_hash, &tensor.tensor_type().to_le_bytes());
            hash_bytes(&mut tensor_hash, &tensor.dimension_count().to_le_bytes());
            for dimension in tensor.dimensions() {
                hash_bytes(&mut tensor_hash, &dimension.to_le_bytes());
            }
            hash_bytes(&mut tensor_hash, &tensor.data_offset().to_le_bytes());
            hash_bytes(&mut tensor_hash, &tensor.data_size().to_le_bytes());
            hash_bytes(&mut tensor_hash, tensor.data());
        }

        Self {
            requirements: model.probe(),
            metadata_hash,
            tensor_hash,
        }
    }
}

fn load_outcome(data: &[u8]) -> Outcome {
    match load(data) {
        Ok(model) => Outcome::Loaded(ModelSummary::new(&model)),
        Err(error) => Outcome::Error(error),
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash = (*hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let Ok(requirements) = Loader::new().probe(data) else {
        return;
    };
    if !requirements_are_bounded(requirements) {
        return;
    }

    assert_eq!(load_outcome(data), load_outcome(data));
});
