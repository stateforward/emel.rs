//! Public `Loader` -> source-bound `OmniEmbed` real-fixture observer and benchmark.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use allocation_counter as _;
use emel_io as _;
use emel_kernels as _;
use emel_tensor as _;
use emel_token as _;
use sml as _;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, Storage as GgufStorage};
use emel_model::omniembed::event::{ContractBegin, ContractVisit, Family, WithFirstName};
use emel_model::omniembed::{OmniEmbed, Storage};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_HEADER_BLOB: &str = "88669d66bc799b05dcd1a2b99c88516f951a48e7";
const DETAIL_IMPLEMENTATION_BLOB: &str = "03e558a90e8df7824a0880ec2aa5500d9cc6f76b";

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    match arguments.as_slice() {
        [_, option, path] if option == "--fixture" => observe(path),
        [_, option, path, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            path,
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        _ => {
            eprintln!("usage: omniembed_observer --fixture GGUF|--benchmark GGUF ITER RUNS WARMUP");
            std::process::exit(2);
        }
    }
}

fn build(path: &str) -> OmniEmbed {
    let bytes = std::fs::read(path).expect("read OmniEmbed fixture");
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("probe");
    loader
        .process_event(Bind::new(GgufStorage::exact(probe).expect("storage")))
        .expect("bind");
    let parsed = loader.process_event(Parse::new()).expect("parse");
    let mut actor = OmniEmbed::load(
        loader,
        parsed,
        Storage::with_name_capacity(4 * 1024 * 1024).expect("names"),
    )
    .expect("source-bound OmniEmbed");
    actor.process_event(ContractBegin::new()).expect("begin");
    actor
}

fn observe(path: &str) {
    let mut actor = build(path);
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    println!("model-omniembed-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_omniembed_detail_header_blob={DETAIL_HEADER_BLOB}");
    println!("source_omniembed_detail_implementation_blob={DETAIL_IMPLEMENTATION_BLOB}");
    println!(
        "contract embedding={} image_encoder={} audio_encoder={} matryoshka_count={} matryoshka_0={} matryoshka_3={} image_size={} audio_rate={} audio_fft={} audio_window={} audio_hop={} audio_mels={}",
        contract.embedding_length(),
        contract.image_encoder_length(),
        contract.audio_encoder_length(),
        contract.matryoshka_dimension_count(),
        contract.matryoshka_dimensions()[0],
        contract.matryoshka_dimensions()[3],
        contract.image_size(),
        contract.audio_sample_rate(),
        contract.audio_n_fft(),
        contract.audio_win_length(),
        contract.audio_hop_size(),
        contract.audio_num_mel_bins(),
    );
    for family in Family::ALL {
        let descriptor = contract.family(family);
        actor
            .process_event(WithFirstName::new(family, |name: &[u8]| {
                println!(
                    "family={} count={} first={}",
                    String::from_utf8_lossy(family.prefix()),
                    descriptor.tensor_count(),
                    String::from_utf8_lossy(name)
                );
            }))
            .expect("name query")
            .expect("first name");
    }
}

fn benchmark(path: &str, iterations: u64, runs: usize, warmup: u64) {
    for _ in 0..warmup {
        black_box(build(path));
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0u64;
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut actor = black_box(build(path));
            let contract = actor.process_event(ContractVisit::new()).expect("visit");
            checksum = checksum.wrapping_add(u64::from(
                contract.family(Family::TextEncoder).tensor_count(),
            ));
        }
        samples.push(start.elapsed().as_nanos() / u128::from(iterations));
    }
    samples.sort_unstable();
    println!(
        "rust_ns_per_visit={} outcome=found checksum={} iter={} runs={}",
        samples[samples.len() / 2],
        checksum,
        iterations,
        runs
    );
}
