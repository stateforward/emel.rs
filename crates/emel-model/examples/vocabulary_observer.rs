//! Public Rust vocabulary observation lane for pinned source comparison.

use std::fmt::Write as _;
use std::sync::Arc;

use allocation_counter::measure;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage};
use emel_io as _;
use emel_model::vocabulary::Loader;
use emel_model::vocabulary::event::{
    Flags, Info, Load, SpecialIds, Token, WithCharmap, WithInfo, WithMerge, WithToken,
};
use emel_token as _;
use emel_token::profile::event::Model;
use sml as _;

const U8: u32 = 0;
const I32: u32 = 5;
const U32: u32 = 4;
const F32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;

struct Fixture {
    entries: Vec<(Vec<u8>, u32, Vec<u8>)>,
}

impl Fixture {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn scalar(mut self, key: &[u8], kind: u32, payload: impl Into<Vec<u8>>) -> Self {
        self.entries.push((key.to_vec(), kind, payload.into()));
        self
    }

    fn string(self, key: &[u8], value: &[u8]) -> Self {
        let mut payload = Vec::new();
        push_string(&mut payload, value);
        self.scalar(key, STRING, payload)
    }

    fn boolean(self, key: &[u8], value: bool) -> Self {
        self.scalar(key, BOOL, [u8::from(value)])
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
        self.array(
            key,
            STRING,
            &payload,
            u64::try_from(values.len()).expect("fixture string count fits u64"),
        )
    }

    fn repeated_strings(self, key: &[u8], value: &[u8], count: usize) -> Self {
        let mut payload = Vec::with_capacity(count.saturating_mul(value.len().saturating_add(8)));
        for _ in 0..count {
            push_string(&mut payload, value);
        }
        self.array(
            key,
            STRING,
            &payload,
            u64::try_from(count).expect("fixture string count fits u64"),
        )
    }

    fn build(self) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(
            &mut bytes,
            u64::try_from(self.entries.len() + 1).expect("fixture entry count fits u64"),
        );
        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, U32);
        push_u32(&mut bytes, 32);
        for (key, kind, payload) in self.entries {
            push_string(&mut bytes, &key);
            push_u32(&mut bytes, kind);
            bytes.extend_from_slice(&payload);
        }
        push_string(&mut bytes, b"benchmark.tensor");
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        bytes.resize(aligned + 32, 0);
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
    push_u64(
        bytes,
        u64::try_from(value.len()).expect("fixture string length fits u64"),
    );
    bytes.extend_from_slice(value);
}

fn load_gguf(bytes: Vec<u8>) -> GgufLoader {
    let mut loader = GgufLoader::new();
    let requirements = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("fixture probe");
    loader
        .process_event(Bind::new(
            Storage::exact(requirements).expect("fixture storage"),
        ))
        .expect("fixture bind");
    loader.process_event(Parse::new()).expect("fixture parse");
    loader
}

fn packed_flags(value: Flags) -> u8 {
    u8::from(value.add_bos)
        | (u8::from(value.add_eos) << 1)
        | (u8::from(value.add_sep) << 2)
        | (u8::from(value.add_space_prefix) << 3)
        | (u8::from(value.remove_extra_whitespaces) << 4)
        | (u8::from(value.escape_whitespaces) << 5)
        | (u8::from(value.treat_whitespace_as_suffix) << 6)
        | (u8::from(value.ignore_merges) << 7)
}

fn ids(value: SpecialIds) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        value.bos,
        value.eos,
        value.eot,
        value.eom,
        value.unknown,
        value.separator,
        value.padding,
        value.classification,
        value.mask,
        value.prefix,
        value.suffix,
        value.middle,
        value.fim_pre,
        value.fim_suf,
        value.fim_mid,
        value.fim_pad,
        value.fim_rep,
        value.fim_sep
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to String");
    }
    output
}

struct SemanticDigest(u64);

impl SemanticDigest {
    const fn new() -> Self {
        Self(14_695_981_039_346_656_037)
    }

    fn byte(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(1_099_511_628_211);
    }

    fn bytes(&mut self, values: &[u8]) {
        self.u64(u64::try_from(values.len()).expect("semantic length fits u64"));
        for value in values {
            self.byte(*value);
        }
    }

    fn u32(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.byte(byte);
        }
    }

    fn i32(&mut self, value: i32) {
        self.u32(u32::from_le_bytes(value.to_le_bytes()));
    }

    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.byte(byte);
        }
    }
}

#[allow(clippy::too_many_lines)]
fn observe_gguf(case_name: &str, gguf: &mut GgufLoader) {
    let mut loader = Loader::try_new().expect("vocabulary storage");
    let outcome = loader.process_event(Load::new(gguf));
    if outcome.is_err() {
        println!("case={case_name} ok=0");
        return;
    }
    let info = loader
        .process_event(WithInfo::new(|value: Info<'_>| {
            (
                value.model,
                value.model_name.to_vec(),
                value.pre_name.to_vec(),
                value.token_count,
                value.token_type_count,
                value.token_bytes,
                value.merge_count,
                value.merge_bytes,
                value.charmap_bytes,
                value.special_ids,
                value.flags,
            )
        }))
        .expect("loaded info");
    let model = format!("{:?}", info.0);
    let mut digest = SemanticDigest::new();
    digest.bytes(model.as_bytes());
    digest.bytes(&info.1);
    digest.bytes(&info.2);
    for value in [info.3, info.4, info.5, info.6, info.7, info.8] {
        digest.u32(value);
    }
    for value in [
        info.9.bos,
        info.9.eos,
        info.9.eot,
        info.9.eom,
        info.9.unknown,
        info.9.separator,
        info.9.padding,
        info.9.classification,
        info.9.mask,
        info.9.prefix,
        info.9.suffix,
        info.9.middle,
        info.9.fim_pre,
        info.9.fim_suf,
        info.9.fim_mid,
        info.9.fim_pad,
        info.9.fim_rep,
        info.9.fim_sep,
    ] {
        digest.i32(value);
    }
    digest.byte(packed_flags(info.10));

    let mut tokens = Vec::with_capacity(usize::try_from(info.3.min(3)).unwrap_or(0));
    for index in 0..info.3 {
        let token = loader
            .process_event(WithToken::new(index, |token: Token<'_>| {
                (
                    token.text.to_vec(),
                    token.score.to_bits(),
                    token.r#type,
                    token.lstrip,
                    token.rstrip,
                )
            }))
            .expect("loaded token query")
            .expect("token index in range");
        digest.bytes(&token.0);
        digest.u32(token.1);
        digest.i32(token.2);
        digest.byte(u8::from(token.3));
        digest.byte(u8::from(token.4));
        if index < 3 {
            tokens.push(format!(
                "{}:{}:{}:{}:{}",
                hex(&token.0),
                token.1,
                token.2,
                u8::from(token.3),
                u8::from(token.4)
            ));
        }
    }
    let mut merges = Vec::new();
    for index in 0..info.6 {
        let merge = loader
            .process_event(WithMerge::new(index, <[u8]>::to_vec))
            .expect("loaded merge query")
            .expect("merge index in range");
        digest.bytes(&merge);
        if index < 3 {
            merges.push(hex(&merge));
        }
    }
    let charmap = loader
        .process_event(WithCharmap::new(<[u8]>::to_vec))
        .expect("loaded charmap query");
    digest.bytes(&charmap);
    println!(
        "case={case_name} ok=1 model={} model_name={} pre_name={} counts={},{},{},{},{},{} ids={} flags={} digest={:016x} tokens={} merges={} charmap={}",
        model,
        String::from_utf8_lossy(&info.1),
        String::from_utf8_lossy(&info.2),
        info.3,
        info.4,
        info.5,
        info.6,
        info.7,
        info.8,
        ids(info.9),
        packed_flags(info.10),
        digest.0,
        tokens.join(";"),
        merges.join(";"),
        hex(&charmap[..charmap.len().min(16)])
    );
}

fn observe(case_name: &str, fixture: Fixture) {
    let mut gguf = load_gguf(fixture.build());
    observe_gguf(case_name, &mut gguf);
}

fn standard_case() -> Fixture {
    let types = [3_u32.to_le_bytes(), 1_u32.to_le_bytes()].concat();
    let scores = [0.0_f32.to_le_bytes(), 1.5_f32.to_le_bytes()].concat();
    Fixture::new()
        .string(b"tokenizer.model", b"gpt2")
        .string(b"tokenizer.pre", b"lfm2")
        .strings(b"tokenizer.tokens", &[b"<|pad|>", b"hello"])
        .array(b"tokenizer.token_type", U32, &types, 2)
        .scalar(b"tokenizer.token_type_count", U32, 4_u32.to_le_bytes())
        .array(b"tokenizer.scores", F32, &scores, 2)
        .strings(b"tokenizer.merges", &[b"h ello"])
        .array(b"tokenizer.precompiled_charsmap", U8, &[7, 8, 9], 3)
        .scalar(b"tokenizer.bos_token_id", U32, 0_u32.to_le_bytes())
        .scalar(b"tokenizer.eos_token_id", U32, 0_u32.to_le_bytes())
        .scalar(b"tokenizer.padding_token_id", U32, 0_u32.to_le_bytes())
        .boolean(b"tokenizer.add_eos_token", false)
}

fn legacy_t5_case() -> Fixture {
    let types = [
        3_u32.to_le_bytes(),
        3_u32.to_le_bytes(),
        2_u32.to_le_bytes(),
        1_u32.to_le_bytes(),
    ]
    .concat();
    Fixture::new()
        .string(b"tokenizer.ggml.model", b"t5")
        .string(b"tokenizer.ggml.pre", b"default")
        .strings(
            b"tokenizer.ggml.tokens",
            &[b"<pad>", b"</s>", b"<unk>", b"\xe2\x96\x81"],
        )
        .array(b"tokenizer.ggml.token_type", U32, &types, 4)
        .boolean(b"tokenizer.ggml.add_space_prefix", true)
        .boolean(b"tokenizer.ggml.remove_extra_whitespaces", true)
}

fn signed_case() -> Fixture {
    let types = [3_u32.to_le_bytes(), 1_u32.to_le_bytes()].concat();
    Fixture::new()
        .string(b"tokenizer.ggml.model", b"rwkv")
        .strings(b"tokenizer.ggml.tokens", &[b"<s>", b"hello"])
        .array(b"tokenizer.ggml.token_type", U32, &types, 2)
        .scalar(b"tokenizer.ggml.bos_token_id", U32, 0_u32.to_le_bytes())
        .scalar(b"tokenizer.ggml.eos_token_id", U32, 0_u32.to_le_bytes())
        .scalar(
            b"tokenizer.ggml.padding_token_id",
            I32,
            (-1_i32).to_le_bytes(),
        )
        .scalar(
            b"tokenizer.ggml.prefix_token_id",
            I32,
            (-1_i32).to_le_bytes(),
        )
        .scalar(
            b"tokenizer.ggml.fim_pre_token_id",
            I32,
            (-1_i32).to_le_bytes(),
        )
}

fn gemma4_case() -> Fixture {
    let types = [3_u32.to_le_bytes(), 3_u32.to_le_bytes()].concat();
    let scores = [0.0_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat();
    Fixture::new()
        .string(b"tokenizer.ggml.model", b"gemma4")
        .strings(b"tokenizer.ggml.tokens", &[b"<bos>", b"<eos>"])
        .array(b"tokenizer.ggml.token_type", U32, &types, 2)
        .scalar(b"tokenizer.ggml.token_type_count", U32, 4_u32.to_le_bytes())
        .array(b"tokenizer.ggml.scores", F32, &scores, 2)
        .repeated_strings(b"tokenizer.ggml.merges", b"", 400_001)
        .scalar(b"tokenizer.ggml.bos_token_id", U32, 0_u32.to_le_bytes())
        .scalar(b"tokenizer.ggml.eos_token_id", U32, 1_u32.to_le_bytes())
        .boolean(b"tokenizer.ggml.add_bos_token", true)
        .boolean(b"tokenizer.ggml.add_space_prefix", true)
}

const fn model_name(model: Model) -> &'static [u8] {
    match model {
        Model::None => b"None",
        Model::SentencePiece => b"SentencePiece",
        Model::Bpe => b"Bpe",
        Model::WordPiece => b"WordPiece",
        Model::Unigram => b"Unigram",
        Model::Rwkv => b"Rwkv",
        Model::Plamo2 => b"Plamo2",
        _ => b"Unknown",
    }
}

fn digest_loaded(loader: &mut Loader) -> (u64, u32, u32) {
    let mut digest = SemanticDigest::new();
    let (token_count, merge_count) = loader
        .process_event(WithInfo::new(|info: Info<'_>| {
            digest.bytes(model_name(info.model));
            digest.bytes(info.model_name);
            digest.bytes(info.pre_name);
            for value in [
                info.token_count,
                info.token_type_count,
                info.token_bytes,
                info.merge_count,
                info.merge_bytes,
                info.charmap_bytes,
            ] {
                digest.u32(value);
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
                digest.i32(value);
            }
            digest.byte(packed_flags(info.flags));
            (info.token_count, info.merge_count)
        }))
        .expect("loaded vocabulary info");
    for index in 0..token_count {
        loader
            .process_event(WithToken::new(index, |token: Token<'_>| {
                digest.bytes(token.text);
                digest.u32(token.score.to_bits());
                digest.i32(token.r#type);
                digest.byte(u8::from(token.lstrip));
                digest.byte(u8::from(token.rstrip));
            }))
            .expect("loaded token query")
            .expect("token index in range");
    }
    for index in 0..merge_count {
        loader
            .process_event(WithMerge::new(index, |merge: &[u8]| {
                digest.bytes(merge);
            }))
            .expect("loaded merge query")
            .expect("merge index in range");
    }
    loader
        .process_event(WithCharmap::new(|charmap: &[u8]| {
            digest.bytes(charmap);
        }))
        .expect("loaded charmap query");
    (digest.0, token_count, merge_count)
}

fn median_ns_per_op(
    runs: usize,
    iterations: usize,
    warmup_iterations: usize,
    mut operation: impl FnMut(),
) -> f64 {
    for _ in 0..warmup_iterations {
        operation();
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let operation_count = u32::try_from(iterations).expect("benchmark iterations fit u32");
        samples.push(start.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(operation_count));
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn benchmark_case(
    name: &str,
    path: &std::path::Path,
    expected: (u32, u32),
    iterations: usize,
    runs: usize,
    warmup_iterations: usize,
) {
    let bytes = std::fs::read(path).expect("read benchmark fixture");
    let mut gguf = load_gguf(bytes);
    let mut actor = Loader::try_new().expect("vocabulary storage");
    let outcome = actor
        .process_event(Load::new(&mut gguf))
        .expect("benchmark fixture loads");
    assert_eq!((outcome.token_count(), outcome.merge_count()), expected);
    let (digest, token_count, merge_count) = digest_loaded(&mut actor);
    assert_eq!((token_count, merge_count), expected);

    let mut allocation_result = None;
    let allocation = measure(|| {
        allocation_result = Some(actor.process_event(Load::new(&mut gguf)));
    });
    let outcome = allocation_result
        .expect("allocation observation")
        .expect("allocation proof load");
    assert_eq!((outcome.token_count(), outcome.merge_count()), expected);
    assert_eq!(allocation.count_total, 0);

    let median = median_ns_per_op(runs, iterations, warmup_iterations, || {
        let outcome = actor
            .process_event(Load::new(&mut gguf))
            .expect("timed benchmark load");
        assert_eq!((outcome.token_count(), outcome.merge_count()), expected);
        core::hint::black_box(outcome);
    });
    println!(
        "case={name} rust_ns_per_op={median:.3} digest={digest:016x} token_count={token_count} merge_count={merge_count} dispatch_allocations=0"
    );
}

fn write_benchmark_fixtures(directory: &std::path::Path) {
    std::fs::create_dir_all(directory).expect("create benchmark fixture directory");
    std::fs::write(directory.join("small.gguf"), standard_case().build())
        .expect("write small fixture");
    std::fs::write(directory.join("gemma4-400001.gguf"), gemma4_case().build())
        .expect("write boundary fixture");
}

fn benchmark(arguments: &[String]) {
    assert_eq!(arguments.len(), 7, "benchmark arguments");
    let iterations = arguments[1].parse::<usize>().expect("iterations");
    let runs = arguments[2].parse::<usize>().expect("runs");
    let warmup_iterations = arguments[3].parse::<usize>().expect("warmup");
    assert!(iterations != 0 && runs != 0);
    benchmark_case(
        "small",
        std::path::Path::new(&arguments[4]),
        (2, 1),
        iterations,
        runs,
        warmup_iterations,
    );
    benchmark_case(
        "real_distilgpt2",
        std::path::Path::new(&arguments[5]),
        (50_257, 50_000),
        iterations,
        runs,
        warmup_iterations,
    );
    benchmark_case(
        "gemma4_400001",
        std::path::Path::new(&arguments[6]),
        (2, 400_001),
        iterations,
        runs,
        warmup_iterations,
    );
}

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_some_and(|value| value == "--write-benchmark-fixtures")
    {
        assert_eq!(arguments.len(), 2, "fixture writer arguments");
        write_benchmark_fixtures(std::path::Path::new(&arguments[1]));
        return;
    }
    if arguments
        .first()
        .is_some_and(|value| value == "--benchmark")
    {
        benchmark(&arguments);
        return;
    }
    println!("vocabulary-observer/v1");
    observe("standard", standard_case());
    observe("legacy_t5", legacy_t5_case());
    observe("signed", signed_case());
    observe("gemma4_400001", gemma4_case());
    for argument in arguments {
        let path = std::path::PathBuf::from(argument);
        let bytes = std::fs::read(&path).expect("read real fixture");
        let mut gguf = load_gguf(bytes);
        observe_gguf(
            &format!(
                "real/{}",
                path.file_name()
                    .expect("real fixture file name")
                    .to_string_lossy()
            ),
            &mut gguf,
        );
    }
}
