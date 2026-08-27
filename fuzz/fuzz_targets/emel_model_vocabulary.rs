#![no_main]

mod common;

use std::sync::Arc;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage};
use emel_model::vocabulary::Loader;
use emel_model::vocabulary::event::{
    Error, Info, Load, Loaded, Token, WithCharmap, WithInfo, WithMerge, WithToken,
};
use emel_token::profile::event::PreId;
use libfuzzer_sys::fuzz_target;

use common::{MAX_INPUT_BYTES, hash_bytes, requirements_are_bounded};

const U8: u32 = 0;
const U32: u32 = 4;
const F32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const WIRE_KINDS: usize = 13;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

const MODELS: [&[u8]; 10] = [
    b"none",
    b"no_vocab",
    b"llama",
    b"gemma4",
    b"gpt2",
    b"bert",
    b"t5",
    b"rwkv",
    b"plamo2",
    b"future",
];

const PRE_ALIASES: [&[u8]; 61] = [
    b"",
    b"default",
    b"llama3",
    b"llama-v3",
    b"llama-bpe",
    b"falcon3",
    b"falcon-h1",
    b"pixtral",
    b"midm-2.0",
    b"lfm2",
    b"jina-v5-nano",
    b"jais2",
    b"dbrx",
    b"smaug",
    b"deepseek-llm",
    b"deepseek-coder",
    b"deepseek-v3",
    b"youtu",
    b"falcon",
    b"mpt",
    b"starcoder",
    b"gpt2",
    b"gpt-2",
    b"jais",
    b"refact",
    b"command-r",
    b"qwen2",
    b"qwen2.5",
    b"qwen35",
    b"stablelm2",
    b"olmo",
    b"poro",
    b"chatglm4",
    b"viking",
    b"tekken",
    b"smollm",
    b"codeshell",
    b"bloom",
    b"gpt3-finnish",
    b"exaone",
    b"exaone4",
    b"exaone-moe",
    b"chameleon",
    b"minerva",
    b"megrez",
    b"gpt4o",
    b"gpt-4o",
    b"tiny-aya",
    b"superbpe",
    b"trillion",
    b"granite-docling",
    b"bailingmoe",
    b"seed-coder",
    b"hunyuan",
    b"hunyuan-dense",
    b"joyai-llm",
    b"kimi-k2",
    b"grok-2",
    b"afmoe",
    b"minimax-m2",
    b"solar-open",
];

struct Fixture(Vec<(Vec<u8>, u32, Vec<u8>)>);

impl Fixture {
    const fn new() -> Self {
        Self(Vec::new())
    }

    fn scalar(mut self, key: &[u8], kind: u32, value: impl Into<Vec<u8>>) -> Self {
        self.0.push((key.to_vec(), kind, value.into()));
        self
    }

    fn string(self, key: &[u8], value: &[u8]) -> Self {
        let mut payload = Vec::new();
        push_string(&mut payload, value);
        self.scalar(key, STRING, payload)
    }

    fn array(self, key: &[u8], kind: u32, values: &[u8], count: u64) -> Self {
        let mut payload = Vec::new();
        push_u32(&mut payload, kind);
        push_u64(&mut payload, count);
        payload.extend_from_slice(values);
        self.scalar(key, ARRAY, payload)
    }

    fn strings(self, key: &[u8], values: &[&[u8]]) -> Self {
        let mut payload = Vec::new();
        for value in values {
            push_string(&mut payload, value);
        }
        self.array(key, STRING, &payload, u64::try_from(values.len()).unwrap())
    }

    fn build(self) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, u64::try_from(self.0.len()).unwrap());
        for (key, kind, payload) in self.0 {
            push_string(&mut bytes, &key);
            push_u32(&mut bytes, kind);
            bytes.extend_from_slice(&payload);
        }
        bytes
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u64(bytes, u64::try_from(value.len()).unwrap());
    bytes.extend_from_slice(value);
}

fn scalar_payload(kind: u32, source: &[u8]) -> Vec<u8> {
    let source_byte = source.first().copied().unwrap_or(0);
    match kind {
        0 | 1 | 7 => vec![source_byte],
        2 | 3 => u16::from(source_byte).to_le_bytes().to_vec(),
        4 => u32::from(source_byte).to_le_bytes().to_vec(),
        5 => (-1_i32).to_le_bytes().to_vec(),
        6 => f32::from(source_byte).to_le_bytes().to_vec(),
        8 => {
            let mut bytes = Vec::new();
            push_string(
                &mut bytes,
                source.get(..source.len().min(8)).unwrap_or(source),
            );
            bytes
        }
        9 => {
            let mut bytes = Vec::new();
            push_u32(&mut bytes, U8);
            push_u64(&mut bytes, 1);
            bytes.push(source_byte);
            bytes
        }
        10 | 11 => u64::from(source_byte).to_le_bytes().to_vec(),
        12 => f64::from(source_byte).to_le_bytes().to_vec(),
        _ => Vec::new(),
    }
}

fn constructed_input(data: &[u8]) -> Vec<u8> {
    let model = MODELS[usize::from(data.first().copied().unwrap_or(0)) % MODELS.len()];
    let pre = PRE_ALIASES[usize::from(data.get(1).copied().unwrap_or(0)) % PRE_ALIASES.len()];
    let wire = u32::from(data.get(2).copied().unwrap_or(0)) % u32::try_from(WIRE_KINDS).unwrap();
    let scenario = data.get(3).copied().unwrap_or(0) % 4;
    let tail = data.get(4..).unwrap_or_default();
    let split = tail.len().min(32) / 2;
    let left = tail.get(..split).unwrap_or_default();
    let right = tail.get(split..tail.len().min(32)).unwrap_or_default();
    let mut fixture = Fixture::new()
        .string(b"tokenizer.model", model)
        .string(b"tokenizer.pre", pre)
        .strings(b"tokenizer.tokens", &[left, right])
        .strings(b"tokenizer.merges", &[b"a b"])
        .array(
            b"tokenizer.precompiled_charsmap",
            U8,
            tail,
            u64::try_from(tail.len()).unwrap(),
        );
    fixture = match scenario {
        0 => fixture.array(b"tokenizer.token_type", U32, &[1, 0, 0, 0, 3, 0, 0, 0], 2),
        1 => fixture.scalar(
            b"tokenizer.padding_token_id",
            wire,
            scalar_payload(wire, tail),
        ),
        2 => fixture.array(b"tokenizer.scores", F32, &[0], 2),
        _ => Fixture::new().string(b"tokenizer.model", model).array(
            b"tokenizer.tokens",
            STRING,
            &[],
            u64::MAX,
        ),
    };
    fixture
        .scalar(
            b"tokenizer.add_bos_token",
            BOOL,
            [data.first().copied().unwrap_or(0) & 1],
        )
        .scalar(b"tokenizer.token_type_count", U32, 4_u32.to_le_bytes())
        .build()
}

fn parsed(bytes: &[u8]) -> Option<GgufLoader> {
    let mut loader = GgufLoader::new();
    let requirements = loader.process_event(Probe::new(Arc::from(bytes))).ok()?;
    if !requirements_are_bounded(&requirements) {
        return None;
    }
    let storage = Storage::exact(requirements).ok()?;
    loader.process_event(Bind::new(storage)).ok()?;
    loader.process_event(Parse::new()).ok()?;
    Some(loader)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Summary {
    outcome: Result<Loaded, Error>,
    pre: Option<PreId>,
    digest: Option<u64>,
}

fn summarize(loader: &mut Loader, gguf: &mut GgufLoader) -> Summary {
    let outcome = loader.process_event(Load::new(gguf));
    let Ok(loaded) = outcome else {
        let query = loader.process_event(WithInfo::new(|_: Info<'_>| 0));
        assert!(query.is_err());
        return Summary {
            outcome,
            pre: None,
            digest: None,
        };
    };
    let mut digest = FNV_OFFSET;
    let (pre, token_count, merge_count) = loader
        .process_event(WithInfo::new(|info: Info<'_>| {
            hash_bytes(&mut digest, info.model_name);
            hash_bytes(&mut digest, info.pre_name);
            for value in [
                info.token_count,
                info.token_type_count,
                info.token_bytes,
                info.merge_count,
                info.merge_bytes,
                info.charmap_bytes,
            ] {
                hash_bytes(&mut digest, &value.to_le_bytes());
            }
            for value in [
                info.special_ids.bos,
                info.special_ids.eos,
                info.special_ids.eot,
                info.special_ids.eom,
                info.special_ids.unknown,
                info.special_ids.separator,
                info.special_ids.padding,
                info.special_ids.classification,
                info.special_ids.mask,
                info.special_ids.prefix,
                info.special_ids.suffix,
                info.special_ids.middle,
                info.special_ids.fim_pre,
                info.special_ids.fim_suf,
                info.special_ids.fim_mid,
                info.special_ids.fim_pad,
                info.special_ids.fim_rep,
                info.special_ids.fim_sep,
            ] {
                hash_bytes(&mut digest, &value.to_le_bytes());
            }
            hash_bytes(
                &mut digest,
                &[
                    u8::from(info.flags.add_bos),
                    u8::from(info.flags.add_eos),
                    u8::from(info.flags.add_sep),
                    u8::from(info.flags.add_space_prefix),
                    u8::from(info.flags.remove_extra_whitespaces),
                    u8::from(info.flags.escape_whitespaces),
                    u8::from(info.flags.treat_whitespace_as_suffix),
                    u8::from(info.flags.ignore_merges),
                ],
            );
            (info.pre, info.token_count, info.merge_count)
        }))
        .expect("successful load publishes info");
    for index in 0..token_count {
        loader
            .process_event(WithToken::new(index, |token: Token<'_>| {
                hash_bytes(&mut digest, token.text);
                hash_bytes(&mut digest, &token.score.to_bits().to_le_bytes());
                hash_bytes(&mut digest, &token.r#type.to_le_bytes());
                hash_bytes(
                    &mut digest,
                    &[u8::from(token.lstrip), u8::from(token.rstrip)],
                );
            }))
            .expect("successful token query")
            .expect("published token index");
    }
    for index in 0..merge_count {
        loader
            .process_event(WithMerge::new(index, |merge: &[u8]| {
                hash_bytes(&mut digest, merge);
            }))
            .expect("successful merge query")
            .expect("published merge index");
    }
    loader
        .process_event(WithCharmap::new(|charmap: &[u8]| {
            hash_bytes(&mut digest, charmap);
        }))
        .expect("successful charmap query");
    Summary {
        outcome: Ok(loaded),
        pre: Some(pre),
        digest: Some(digest),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let owned = if data.first().is_some_and(|byte| byte & 0x80 != 0) {
        data.to_vec()
    } else {
        constructed_input(data)
    };
    let Some(mut gguf) = parsed(&owned) else {
        assert!(parsed(&owned).is_none());
        return;
    };
    let mut vocabulary = Loader::try_new().expect("bounded vocabulary storage");
    let first = summarize(&mut vocabulary, &mut gguf);
    let second = summarize(&mut vocabulary, &mut gguf);
    assert_eq!(first, second);
});
