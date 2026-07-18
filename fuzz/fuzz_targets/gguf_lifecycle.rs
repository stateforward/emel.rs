#![no_main]

mod common;

use emel_gguf::Loader;
use emel_gguf::event::ProbeDone;
use libfuzzer_sys::fuzz_target;

use common::{LoaderExt as _, MAX_INPUT_BYTES, requirements_are_bounded, validate_model};

const MAX_ACTIONS: usize = 32;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut loader = Loader::new();
    let mut requirements = bounded_probe(&mut loader, data);

    for (index, action) in data.iter().copied().take(MAX_ACTIONS).enumerate() {
        let shortened_length = data.len().saturating_sub(index.saturating_add(1));
        let shortened = &data[..shortened_length];

        match action % 7 {
            0 => requirements = bounded_probe(&mut loader, data),
            1 => requirements = bounded_probe(&mut loader, shortened),
            2 => bind_exact_if_bounded(&mut loader, requirements.clone()),
            3 => bind_small_capacity(&mut loader, action, requirements.clone()),
            4 => parse_and_validate(&mut loader, data, requirements.as_ref()),
            5 => parse_and_validate(&mut loader, shortened, requirements.as_ref()),
            6 => requirements = bounded_probe(&mut loader, &[]),
            _ => unreachable!(),
        }
    }
});

fn bounded_probe(loader: &mut Loader, data: &[u8]) -> Option<ProbeDone> {
    loader
        .probe(data)
        .ok()
        .filter(requirements_are_bounded)
}

fn bind_exact_if_bounded(loader: &mut Loader, requirements: Option<ProbeDone>) {
    if let Some(requirements) = requirements {
        let _ = loader.bind(requirements);
    } else {
        let _ = loader.parse(&[]);
    }
}

fn bind_small_capacity(loader: &mut Loader, action: u8, requirements: Option<ProbeDone>) {
    let units = usize::from(action >> 3);
    if let Some(requirements) = requirements {
        let _ = loader.bind_with_capacity(requirements, units * 128, units, units);
    }
}

fn parse_and_validate(loader: &mut Loader, data: &[u8], requirements: Option<&ProbeDone>) {
    if let (Ok(model), Some(requirements)) = (loader.parse(data), requirements) {
        validate_model(model, requirements);
    }
}
