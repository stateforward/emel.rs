use std::cell::Cell;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use allocation_counter::measure;

use super::event::{Defaults, Model, Resolve};
use super::sm::KNOWN_PRE_INPUT_COUNT;
use super::{Dependency, Resolver};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const MODEL_SOURCE_BLOB: &str = "ef7ff8da51f1f281082901bf4919a4b9a63f2671";
const PRE_SOURCE_BLOB: &str = "16b2982ca16dfdfbee016d50d0eb924a7cdc18c4";

const PRE_CASES: [(&str, u8); 61] = [
    ("", 0),
    ("default", 0),
    ("llama3", 1),
    ("llama-v3", 1),
    ("llama-bpe", 1),
    ("falcon3", 1),
    ("falcon-h1", 1),
    ("pixtral", 1),
    ("midm-2.0", 1),
    ("lfm2", 1),
    ("jina-v5-nano", 1),
    ("jais2", 2),
    ("dbrx", 3),
    ("smaug", 4),
    ("deepseek-llm", 5),
    ("deepseek-coder", 6),
    ("deepseek-v3", 7),
    ("youtu", 8),
    ("falcon", 9),
    ("mpt", 10),
    ("starcoder", 11),
    ("gpt2", 12),
    ("gpt-2", 12),
    ("jais", 13),
    ("refact", 14),
    ("command-r", 15),
    ("qwen2", 16),
    ("qwen2.5", 17),
    ("qwen35", 17),
    ("stablelm2", 18),
    ("olmo", 19),
    ("poro", 20),
    ("chatglm4", 21),
    ("viking", 22),
    ("tekken", 23),
    ("smollm", 24),
    ("codeshell", 25),
    ("bloom", 26),
    ("gpt3-finnish", 27),
    ("exaone", 28),
    ("exaone4", 29),
    ("exaone-moe", 30),
    ("chameleon", 31),
    ("minerva", 32),
    ("megrez", 33),
    ("gpt4o", 34),
    ("gpt-4o", 34),
    ("tiny-aya", 35),
    ("superbpe", 36),
    ("trillion", 37),
    ("granite-docling", 38),
    ("bailingmoe", 39),
    ("seed-coder", 40),
    ("hunyuan", 41),
    ("hunyuan-dense", 42),
    ("joyai-llm", 43),
    ("kimi-k2", 44),
    ("grok-2", 45),
    ("afmoe", 46),
    ("minimax-m2", 47),
    ("solar-open", 48),
];

fn resolve(model: &str, pre: &str) -> super::event::Resolved {
    Resolver::new()
        .process_event(Resolve::new(model, pre))
        .unwrap()
}

fn assert_dependency_dispatch<D: Dependency>(dependency: &mut D) {
    let result = Cell::new(None);
    let allocation = measure(|| {
        result.set(Some(Dependency::process_event(
            dependency,
            Resolve::new("future", "future"),
        )));
    });
    assert_eq!(allocation.count_total, 0);
    assert!(result.get().unwrap().unwrap().model().is_unknown());
}

#[test]
fn all_model_spellings_and_aliases_match_source_classification() {
    let cases = [
        ("none", Model::None),
        ("no_vocab", Model::None),
        ("llama", Model::SentencePiece),
        ("gemma4", Model::SentencePiece),
        ("gpt2", Model::Bpe),
        ("bert", Model::WordPiece),
        ("t5", Model::Unigram),
        ("rwkv", Model::Rwkv),
        ("plamo2", Model::Plamo2),
        ("future", Model::Unknown),
    ];
    for (name, expected) in cases {
        assert_eq!(resolve(name, "default").model(), expected, "{name}");
    }
}

#[test]
fn all_sixty_one_pre_inputs_and_aliases_match_source_profiles() {
    assert_eq!(KNOWN_PRE_INPUT_COUNT, PRE_CASES.len());
    for (name, expected) in PRE_CASES {
        assert_eq!(resolve("none", name).pre_id().0, expected, "{name}");
    }
    assert_eq!(resolve("none", "future-pre").pre_id().0, 49);
}

#[test]
fn model_defaults_and_pre_overrides_compose_exactly() {
    let source = Defaults::SOURCE;
    assert_eq!(resolve("none", "default").defaults(), source);
    assert_eq!(resolve("gemma4", "default").defaults(), source);
    assert_eq!(resolve("rwkv", "default").defaults(), source);

    let llama = resolve("llama", "default").defaults();
    assert_eq!((llama.bos_id(), llama.eos_id(), llama.unk_id()), (1, 2, 0));
    assert!(llama.add_bos());
    assert!(llama.add_space_prefix());
    assert!(llama.escape_whitespaces());

    let bert = resolve("bert", "default").defaults();
    assert_eq!(
        (
            bert.bos_id(),
            bert.unk_id(),
            bert.sep_id(),
            bert.pad_id(),
            bert.mask_id()
        ),
        (101, 100, 102, 0, 103)
    );
    assert!(bert.add_bos());
    assert!(bert.add_sep());

    let gpt2 = resolve("gpt2", "default").defaults();
    assert_eq!((gpt2.bos_id(), gpt2.eos_id()), (11, 11));
    let t5 = resolve("t5", "default").defaults();
    assert_eq!((t5.eos_id(), t5.unk_id(), t5.pad_id()), (1, 2, 0));
    let plamo2 = resolve("plamo2", "default").defaults();
    assert_eq!(
        (
            plamo2.bos_id(),
            plamo2.eos_id(),
            plamo2.unk_id(),
            plamo2.pad_id()
        ),
        (1, 2, 0, 3)
    );

    let composed = resolve("gpt2", "llama-v3").defaults();
    assert_eq!((composed.bos_id(), composed.eos_id()), (11, 11));
    assert!(composed.add_bos());
    assert!(composed.ignore_merges());
    let youtu = resolve("bert", "youtu").defaults();
    assert!(youtu.add_bos());
    assert!(youtu.add_sep());
    assert!(youtu.ignore_merges());
}

#[test]
fn untouched_source_defaults_are_preserved() {
    let defaults = resolve("future", "future").defaults();
    assert_eq!(
        [
            defaults.eot_id(),
            defaults.eom_id(),
            defaults.cls_id(),
            defaults.prefix_id(),
            defaults.suffix_id(),
            defaults.middle_id(),
            defaults.fim_pre_id(),
            defaults.fim_suf_id(),
            defaults.fim_mid_id(),
            defaults.fim_pad_id(),
            defaults.fim_rep_id(),
            defaults.fim_sep_id(),
        ],
        [-1; 12]
    );
    assert!(!defaults.add_eos());
    assert!(!defaults.remove_extra_whitespaces());
    assert!(!defaults.treat_whitespace_as_suffix());
    assert!(defaults.escape_whitespaces());
}

#[test]
fn public_dispatch_and_static_dependency_dispatch_allocate_nothing() {
    let mut resolver = Resolver::new();
    let direct_result = Cell::new(None);
    let direct = measure(|| {
        direct_result.set(Some(
            resolver.process_event(Resolve::new("llama", "llama3")),
        ));
    });
    assert_eq!(direct.count_total, 0);
    assert!(direct_result.get().unwrap().is_ok());

    assert_dependency_dispatch(&mut resolver);
}

#[test]
fn parity_snapshot_is_source_backed() {
    let mut manifest = String::new();
    writeln!(manifest, "token-profile-parity/v1").unwrap();
    writeln!(manifest, "source_commit={SOURCE_COMMIT}").unwrap();
    writeln!(manifest, "model_source_blob={MODEL_SOURCE_BLOB}").unwrap();
    writeln!(manifest, "pre_source_blob={PRE_SOURCE_BLOB}").unwrap();
    for name in [
        "none", "no_vocab", "llama", "gemma4", "gpt2", "bert", "t5", "rwkv", "plamo2", "future",
    ] {
        let value = resolve(name, "default");
        writeln!(
            manifest,
            "model={name} class={:?} defaults={:?}",
            value.model(),
            value.defaults()
        )
        .unwrap();
    }
    for (name, _) in PRE_CASES {
        let printable = if name.is_empty() { "<empty>" } else { name };
        let value = resolve("none", name);
        writeln!(
            manifest,
            "pre={printable} profile={} defaults={:?}",
            value.pre_id().0,
            value.defaults()
        )
        .unwrap();
    }
    let value = resolve("future", "future");
    writeln!(
        manifest,
        "pre=future profile={} defaults={:?}",
        value.pre_id().0,
        value.defaults()
    )
    .unwrap();

    let output = std::env::var_os("EMEL_TOKEN_PROFILE_PARITY_OUTPUT").map(PathBuf::from);
    let path = output.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../snapshots/parity/token-profile/manifest.txt")
    });
    if output.is_some() || std::env::var_os("EMEL_TOKEN_PROFILE_PARITY_UPDATE").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &manifest).unwrap();
    }
    assert_eq!(fs::read_to_string(path).unwrap(), manifest);
}

#[test]
#[ignore = "benchmark entrypoint for scripts/bench.sh"]
fn benchmark_profile_resolver_dispatch() {
    let iterations = benchmark_env("EMEL_TOKEN_PROFILE_BENCH_ITERATIONS", 10_000_000_u32);
    let runs = benchmark_env("EMEL_TOKEN_PROFILE_BENCH_RUNS", 7_usize);
    let warmup = benchmark_env("EMEL_TOKEN_PROFILE_BENCH_WARMUP_ITERATIONS", 1_000_000_u32);
    let mut resolver = Resolver::new();
    for _ in 0..warmup {
        black_box(
            resolver
                .process_event(Resolve::new("gpt2", "solar-open"))
                .unwrap(),
        );
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(
                resolver
                    .process_event(Resolve::new("gpt2", "solar-open"))
                    .unwrap(),
            );
        }
        samples.push(start.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(iterations));
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    println!("# bench_host_arch: {}", std::env::consts::ARCH);
    println!("# bench_pointer_width: {}", usize::BITS);
    println!(
        "# benchmark_config: iterations={iterations} runs={runs} sample_policy=median warmup_iterations={warmup}"
    );
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: {SOURCE_COMMIT}");
    println!("# source_model_blob: {MODEL_SOURCE_BLOB}");
    println!("# source_pre_blob: {PRE_SOURCE_BLOB}");
    println!("# benchmark_fixture: public Resolver/Resolve model=gpt2 pre=solar-open");
    println!(
        "# benchmark_validation: typed success checked each iteration; allocation separately proven"
    );
    println!(
        "token/profile/resolve_worst_known ns_per_op={median:.3} iter={iterations} runs={runs}"
    );
}

fn benchmark_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}
