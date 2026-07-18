#![no_main]

use emel_token::profile::Resolver;
use emel_token::profile::event::Resolve;
use libfuzzer_sys::fuzz_target;

const ALIASES: [&str; 61] = [
    "", "default", "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
    "pixtral", "midm-2.0", "lfm2", "jina-v5-nano", "jais2", "dbrx", "smaug",
    "deepseek-llm", "deepseek-coder", "deepseek-v3", "youtu", "falcon", "mpt",
    "starcoder", "gpt2", "gpt-2", "jais", "refact", "command-r", "qwen2", "qwen2.5",
    "qwen35", "stablelm2", "olmo", "poro", "chatglm4", "viking", "tekken", "smollm",
    "codeshell", "bloom", "gpt3-finnish", "exaone", "exaone4", "exaone-moe", "chameleon",
    "minerva", "megrez", "gpt4o", "gpt-4o", "tiny-aya", "superbpe", "trillion",
    "granite-docling", "bailingmoe", "seed-coder", "hunyuan", "hunyuan-dense", "joyai-llm",
    "kimi-k2", "grok-2", "afmoe", "minimax-m2", "solar-open",
];

fuzz_target!(|data: &[u8]| {
    let split = data.first().map_or(0, |byte| usize::from(*byte)) % (data.len() + 1);
    let model = std::str::from_utf8(&data[..split]).unwrap_or("invalid-utf8-model");
    let pre = std::str::from_utf8(&data[split..]).unwrap_or("invalid-utf8-pre");
    let mut resolver = Resolver::new();
    assert!(resolver.process_event(Resolve::new(model, pre)).is_ok());

    let selected = data
        .first()
        .map_or(0, |byte| usize::from(*byte) % ALIASES.len());
    assert!(
        resolver
            .process_event(Resolve::new("future-model", ALIASES[selected]))
            .is_ok()
    );
});
