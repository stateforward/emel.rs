#![no_main]

mod common;

use emel_gguf::Loader;
use libfuzzer_sys::fuzz_target;

use common::{LoaderExt as _, MAX_INPUT_BYTES, requirements_are_bounded, validate_model};

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut loader = Loader::new();
    let first_probe = loader.probe(data);

    let repeated_probe = loader.probe(data);
    assert_eq!(first_probe, repeated_probe);

    let Ok(requirements) = repeated_probe else {
        return;
    };
    if !requirements_are_bounded(requirements) {
        return;
    }

    if loader.bind().is_err() {
        return;
    }

    if let Ok(model) = loader.parse(data) {
        validate_model(&model);
    }
});
