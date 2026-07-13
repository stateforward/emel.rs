#![no_main]

mod common;

use emel_gguf::{Loader, LoaderState};
use libfuzzer_sys::fuzz_target;

use common::{LoaderExt as _, MAX_INPUT_BYTES, requirements_are_bounded, validate_model};

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut loader = Loader::new();
    let first_probe = loader.probe(data);
    assert_ne!(loader.state(), LoaderState::Processing);

    let repeated_probe = loader.probe(data);
    assert_eq!(first_probe, repeated_probe);
    assert_ne!(loader.state(), LoaderState::Processing);

    let Ok(requirements) = repeated_probe else {
        return;
    };
    if !requirements_are_bounded(requirements) {
        return;
    }

    if loader.bind().is_err() {
        return;
    }
    assert_eq!(loader.state(), LoaderState::Bound);

    if let Ok(model) = loader.parse(data) {
        validate_model(&model);
        assert_eq!(loader.state(), LoaderState::Parsed);
    }
    assert_ne!(loader.state(), LoaderState::Processing);
});
