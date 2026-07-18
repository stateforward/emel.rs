#![no_main]

mod common;

use emel_gguf::Loader;
use emel_gguf::event::Error;
use libfuzzer_sys::fuzz_target;

use common::{LoaderExt as _, MAX_INPUT_BYTES, SemanticSummary, requirements_are_bounded, semantic_load};

#[derive(Debug, Eq, PartialEq)]
enum Outcome {
    Loaded(SemanticSummary),
    Error(Error),
}

fn load_outcome(data: &[u8]) -> Outcome {
    match semantic_load(data) {
        Ok(model) => Outcome::Loaded(model),
        Err(error) => Outcome::Error(error),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let Ok(requirements) = Loader::new().probe(data) else {
        return;
    };
    if !requirements_are_bounded(&requirements) {
        return;
    }

    assert_eq!(load_outcome(data), load_outcome(data));
});
