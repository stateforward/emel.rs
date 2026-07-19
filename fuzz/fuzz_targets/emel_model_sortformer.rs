#![no_main]

use std::sync::Arc;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage as GgufStorage};
use emel_model::sortformer::event::{
    ContractBegin, ContractReset, ContractVisit, Family, StorageRelease, WithFirstName,
};
use emel_model::sortformer::{Sortformer, Storage, load_hparams};
use libfuzzer_sys::fuzz_target;

const U32: u32 = 4;
const STRING: u32 = 8;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u64(
        bytes,
        u64::try_from(value.len()).expect("bounded fuzz fixture"),
    );
    bytes.extend_from_slice(value);
}

fn push_string_entry(bytes: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    push_string(bytes, key);
    push_u32(bytes, STRING);
    push_string(bytes, value);
}

fn push_u32_entry(bytes: &mut Vec<u8>, key: &[u8], value: u32) {
    push_string(bytes, key);
    push_u32(bytes, U32);
    push_u32(bytes, value);
}

fn metadata_loader(selector: u8) -> GgufLoader {
    let selected = |valid: u32, invalid: u32, bit: u8| {
        if selector & bit == 0 { valid } else { invalid }
    };
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, 14);
    push_string_entry(&mut bytes, b"general.architecture", b"sortformer");
    push_string_entry(
        &mut bytes,
        b"sortformer.source.format",
        if selector & 1 == 0 { b"nemo" } else { b"onnx" },
    );
    push_string_entry(
        &mut bytes,
        b"sortformer.tensor_name_scheme",
        if selector & 2 == 0 {
            b"compact_v1"
        } else {
            b"other"
        },
    );
    push_string_entry(
        &mut bytes,
        b"sortformer.outtype",
        if selector & 4 == 0 { b"f32" } else { b"f16" },
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.original_tensor_count",
        selected(128, 0, 8),
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.tensor_count",
        selected(132, u32::MAX, 16),
    );
    push_u32_entry(&mut bytes, b"sortformer.skipped_tensor_count", 3);
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.preprocessor.sample_rate",
        selected(16_000, 8_000, 32),
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.num_spks",
        selected(4, 3, 64),
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_len",
        selected(188, 187, 128),
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_right_context",
        1,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.fifo_len",
        0,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_update_period",
        188,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_len",
        188,
    );

    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("structured metadata fixture probes");
    loader
        .process_event(Bind::new(
            GgufStorage::exact(probe).expect("bounded storage"),
        ))
        .expect("structured metadata fixture binds");
    loader
        .process_event(Parse::new())
        .expect("structured metadata fixture parses");
    loader
}

fn tensor_loader() -> (GgufLoader, emel_gguf::event::ParseDone) {
    const NAMES: [&[u8]; 5] = [
        b"prep.fuzz",
        b"enc.fuzz",
        b"mods.fuzz",
        b"te.fuzz",
        b"other.fuzz",
    ];
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, u64::try_from(NAMES.len()).expect("bounded fixture"));
    push_u64(&mut bytes, 15);
    push_u32_entry(&mut bytes, b"general.alignment", 1);
    push_string_entry(&mut bytes, b"general.architecture", b"sortformer");
    push_string_entry(&mut bytes, b"sortformer.source.format", b"nemo");
    push_string_entry(
        &mut bytes,
        b"sortformer.tensor_name_scheme",
        b"compact_v1",
    );
    push_string_entry(&mut bytes, b"sortformer.outtype", b"f32");
    push_u32_entry(&mut bytes, b"sortformer.original_tensor_count", 128);
    push_u32_entry(&mut bytes, b"sortformer.tensor_count", 132);
    push_u32_entry(&mut bytes, b"sortformer.skipped_tensor_count", 3);
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.preprocessor.sample_rate",
        16_000,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.num_spks",
        4,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_len",
        188,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.chunk_right_context",
        1,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.fifo_len",
        0,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_update_period",
        188,
    );
    push_u32_entry(
        &mut bytes,
        b"sortformer.config.sortformer_modules.spkcache_len",
        188,
    );
    for (index, name) in NAMES.iter().enumerate() {
        push_string(&mut bytes, name);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0);
        push_u64(
            &mut bytes,
            u64::try_from(index * 4).expect("bounded fixture offset"),
        );
    }
    bytes.resize(bytes.len() + NAMES.len() * 4, 0);

    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("structured tensor fixture probes");
    loader
        .process_event(Bind::new(
            GgufStorage::exact(probe).expect("bounded storage"),
        ))
        .expect("structured tensor fixture binds");
    let parsed = loader
        .process_event(Parse::new())
        .expect("structured tensor fixture parses");
    (loader, parsed)
}

fuzz_target!(|input: &[u8]| {
    let selector = input.first().copied().unwrap_or_default();
    let mut metadata = metadata_loader(selector);
    let _ = load_hparams(&mut metadata);

    let (tensors, parsed) = tensor_loader();
    let Ok(mut actor) = Sortformer::load(
        tensors,
        parsed,
        Storage::with_name_capacity(128).expect("bounded arena"),
    ) else {
        return;
    };
    for (step, event) in input.iter().copied().enumerate().take(128) {
        let argument = input
            .get(step.saturating_add(1))
            .copied()
            .unwrap_or_default();
        match event % 5 {
            0 => {
                let _ = actor.process_event(ContractBegin::new());
            }
            1 => {
                let _ = actor.process_event(ContractVisit::new());
            }
            2 => {
                let family = Family::ALL[usize::from(argument) % Family::ALL.len()];
                let _ = actor.process_event(WithFirstName::new(family, |name: &[u8]| name.len()));
            }
            3 => {
                let _ = actor.process_event(ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(StorageRelease::new());
            }
        }
    }
});
