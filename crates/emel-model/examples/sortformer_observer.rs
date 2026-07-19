//! Public Loader -> source-bound Sortformer real-fixture observer and benchmark.

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
use emel_model::sortformer::event::{ContractBegin, ContractVisit, Family, WithFirstName};
use emel_model::sortformer::{Sortformer, Storage};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const ANY_BLOB: &str = "d79cb231e0dfc75f5aa4d8da7db2ecf36c92c9b6";
const DETAIL_HEADER_BLOB: &str = "72766c7042707a8037ac49779b8e3f8e2b34c555";
const DETAIL_IMPLEMENTATION_BLOB: &str = "011a3ec4dfaab102457c3f6ca6e20a5adf9934b1";

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
            eprintln!(
                "usage: sortformer_observer --fixture GGUF|--benchmark GGUF ITER RUNS WARMUP"
            );
            std::process::exit(2);
        }
    }
}

fn build(path: &str) -> Sortformer {
    let bytes = std::fs::read(path).expect("read Sortformer fixture");
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("probe");
    loader
        .process_event(Bind::new(GgufStorage::exact(probe).expect("storage")))
        .expect("bind");
    let parsed = loader.process_event(Parse::new()).expect("parse");
    let mut actor = Sortformer::load(
        loader,
        parsed,
        Storage::with_name_capacity(4 * 1024 * 1024).expect("names"),
    )
    .expect("source-bound Sortformer");
    actor.process_event(ContractBegin::new()).expect("begin");
    actor
}

fn observe(path: &str) {
    let mut actor = build(path);
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    println!("model-sortformer-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sortformer_any_blob={ANY_BLOB}");
    println!("source_sortformer_detail_header_blob={DETAIL_HEADER_BLOB}");
    println!("source_sortformer_detail_implementation_blob={DETAIL_IMPLEMENTATION_BLOB}");
    println!(
        "contract sample_rate={} speakers={} frame_shift_ms={} chunk_len={} right_context={} fifo_len={} cache_period={} cache_len={}",
        contract.sample_rate(),
        contract.speaker_count(),
        contract.frame_shift_ms(),
        contract.chunk_len(),
        contract.chunk_right_context(),
        contract.fifo_len(),
        contract.spkcache_update_period(),
        contract.spkcache_len()
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
            checksum =
                checksum.wrapping_add(u64::from(contract.family(Family::Encoder).tensor_count()));
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
