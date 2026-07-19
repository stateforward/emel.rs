use std::sync::Arc;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage as GgufStorage};
use emel_model::omniembed::event::{
    ContractBegin, ContractReset, ContractVisit, Family, StorageRelease, WithFirstName,
};
use emel_model::omniembed::{OmniEmbed, Storage, load_hparams};

const U32: u32 = 4;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const IMAGE_ENCODER_NAME: &[u8] = b"mobilenetv4_conv_medium.e180_r384_in12k";
const NAMES: [&[u8]; 6] = [
    b"text_encoder.fuzz",
    b"text_projection.fuzz",
    b"image_encoder.fuzz",
    b"image_projection.fuzz",
    b"audio_encoder.fuzz",
    b"audio_projection.fuzz",
];

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u64(bytes, u64::try_from(value.len()).expect("bounded fixture"));
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

fn push_dimensions(bytes: &mut Vec<u8>, values: &[u32]) {
    push_string(bytes, b"omniembed.matryoshka_dims");
    push_u32(bytes, ARRAY);
    push_u32(bytes, U32);
    push_u64(
        bytes,
        u64::try_from(values.len()).expect("bounded dimensions"),
    );
    for value in values {
        push_u32(bytes, *value);
    }
}

fn encoder_name(selector: u8, canonical: &'static [u8]) -> &'static [u8] {
    match selector % 3 {
        0 => canonical,
        1 => b"",
        _ => b"other",
    }
}

pub fn metadata_loader(selector: u8) -> GgufLoader {
    let dimensions: &[u32] = if selector & 8 == 0 {
        &[768, 512, 256, 128]
    } else {
        &[1; 17]
    };
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, 7);
    push_string_entry(
        &mut bytes,
        b"general.architecture",
        if selector & 16 == 0 {
            b"omniembed"
        } else {
            b"other"
        },
    );
    push_u32_entry(&mut bytes, b"omniembed.embed_dim", 1280);
    push_string_entry(
        &mut bytes,
        b"omniembed.image_encoder_name",
        encoder_name(selector, IMAGE_ENCODER_NAME),
    );
    push_u32_entry(&mut bytes, b"omniembed.image_encoder_dim", 640);
    push_string_entry(
        &mut bytes,
        b"omniembed.audio_encoder_name",
        encoder_name(selector.rotate_right(2), b"efficientat_mn20_as"),
    );
    push_u32_entry(&mut bytes, b"omniembed.audio_encoder_dim", 768);
    push_dimensions(&mut bytes, dimensions);

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

pub fn actor() -> OmniEmbed {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(
        &mut bytes,
        u64::try_from(NAMES.len()).expect("bounded tensors"),
    );
    push_u64(&mut bytes, 8);
    push_u32_entry(&mut bytes, b"general.alignment", 1);
    push_string_entry(&mut bytes, b"general.architecture", b"omniembed");
    push_u32_entry(&mut bytes, b"omniembed.embed_dim", 1280);
    push_string_entry(
        &mut bytes,
        b"omniembed.image_encoder_name",
        IMAGE_ENCODER_NAME,
    );
    push_u32_entry(&mut bytes, b"omniembed.image_encoder_dim", 640);
    push_string_entry(
        &mut bytes,
        b"omniembed.audio_encoder_name",
        b"efficientat_mn20_as",
    );
    push_u32_entry(&mut bytes, b"omniembed.audio_encoder_dim", 768);
    push_dimensions(&mut bytes, &[768, 512, 256, 128]);
    for (index, name) in NAMES.iter().enumerate() {
        push_string(&mut bytes, name);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0);
        push_u64(
            &mut bytes,
            u64::try_from(index * 4).expect("bounded tensor offset"),
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
    OmniEmbed::load(
        loader,
        parsed,
        Storage::with_name_capacity(256).expect("bounded names"),
    )
    .expect("structured OmniEmbed actor")
}

fn assert_valid_contract_scenario() {
    let mut actor = actor();
    actor
        .process_event(ContractBegin::new())
        .expect("canonical structured fixture reaches ready");
    actor
        .process_event(ContractVisit::new())
        .expect("ready structured fixture publishes its contract");

    let mut callback_invoked = false;
    let result = actor
        .process_event(WithFirstName::new(Family::ImageEncoder, |name: &[u8]| {
            callback_invoked = true;
            name == NAMES[Family::ImageEncoder as usize]
        }))
        .expect("ready structured fixture accepts a name query");
    assert_eq!(result, Some(true));
    assert!(callback_invoked);
}

pub fn exercise(input: &[u8]) {
    assert_valid_contract_scenario();

    let selector = input.first().copied().unwrap_or_default();
    let mut metadata = metadata_loader(selector);
    let _ = load_hparams(&mut metadata);

    let mut actor = actor();
    for (step, event) in input.iter().copied().enumerate().take(128) {
        let argument = input.get(step + 1).copied().unwrap_or_default();
        match event % 5 {
            0 => {
                let _ = actor.process_event(ContractBegin::new());
            }
            1 => {
                let _ = actor.process_event(ContractVisit::new());
            }
            2 => {
                let family = Family::ALL[usize::from(argument) % Family::ALL.len()];
                let _ = actor.process_event(WithFirstName::new(family, <[u8]>::len));
            }
            3 => {
                let _ = actor.process_event(ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(StorageRelease::new());
            }
        }
    }
}
