//! Dependency-light benchmark runner for performance snapshot gates.

use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Error, Load, Parse, Probe};
use emel_io::loader::Loader as IoLoader;
use emel_io::loader::event::{LoadTensor, StrategyKind, StrategyPolicy, TensorLoadSpan};
use emel_io::mmap::Mapper;
use emel_io::mmap::event::{
    AdviseDontNeed, AdviseSequential, AdviseWillNeed, MapTensor, MappingCallback, MmapSource,
    ReleaseMapping, WithMapping,
};
use emel_io::read::Reader;
use emel_io::read::event::{ReadTensor, Target as ReadTarget};
use emel_io::staged_read::Stager;
use emel_io::staged_read::event::{
    Callback as StageCallback, StageWindow, StageWindowDone, StageWindowError,
    Target as StageTarget,
};
use std::cell::Cell;

const ALIGNMENT: usize = 32;
const MAGIC: [u8; 4] = *b"GGUF";
const VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Suite {
    Gguf,
    IoRead,
    IoMmap,
    IoStagedRead,
    IoLoader,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    iterations: u64,
    runs: usize,
    warmup_iterations: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iterations: 1_000,
            runs: 5,
            warmup_iterations: 100,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("emel-bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (suite, config) = parse_config()?;
    println!("# bench_host_arch: {}", env::consts::ARCH);
    println!("# bench_pointer_width: {}", usize::BITS);
    println!(
        "# benchmark_config: iterations={} runs={} sample_policy=median warmup_iterations={}",
        config.iterations, config.runs, config.warmup_iterations
    );

    match suite {
        Suite::Gguf => run_gguf(config)?,
        Suite::IoRead => run_io_read(config)?,
        Suite::IoMmap => run_io_mmap(config)?,
        Suite::IoStagedRead => run_io_staged_read(config)?,
        Suite::IoLoader => run_io_loader(config)?,
    }
    Ok(())
}

fn run_gguf(config: Config) -> Result<(), Error> {
    let metadata = metadata_fixture();
    let tensors = tensor_fixture();
    print_case(
        "gguf/probe/metadata_64",
        benchmark_probe(&metadata, config)?,
        config,
    );
    print_case(
        "gguf/load/metadata_64",
        benchmark_load(&metadata, config)?,
        config,
    );
    print_case(
        "gguf/probe/tensors_64",
        benchmark_probe(&tensors, config)?,
        config,
    );
    print_case(
        "gguf/parse/tensors_64",
        benchmark_parse(&tensors, config)?,
        config,
    );
    print_case(
        "gguf/load/tensors_64",
        benchmark_load(&tensors, config)?,
        config,
    );
    Ok(())
}

fn run_io_read(config: Config) -> Result<(), emel_io::read::event::Error> {
    const COPY_BYTES: usize = 1024 * 1024;
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa");
    println!(
        "# benchmark_fixture: public Reader/ReadTensor, immutable source bytes=1048576 fill=0xa5, caller target bytes=1048576"
    );
    println!(
        "# benchmark_validation: typed done checked each iteration, target equals source after measurement"
    );
    let source = vec![0xa5; COPY_BYTES];
    let mut target = vec![0_u8; COPY_BYTES];
    let target_capability = ReadTarget::new(&mut target);
    let mut reader = Reader::new();
    let timing = measure(config, || {
        let done = reader.process_event(ReadTensor::new(
            1,
            "benchmark.bin",
            Some(black_box(&source)),
            black_box(&target_capability),
        ))?;
        black_box(done);
        Ok(())
    })?;
    if !target_capability
        .try_matches(&source)
        .map_err(|_| emel_io::read::event::Error::InternalError)?
    {
        return Err(emel_io::read::event::Error::InternalError);
    }
    print_case("io/read/copy_1mib", timing, config);
    Ok(())
}

fn run_io_loader(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    const COPY_BYTES: usize = 1024 * 1024;
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa");
    println!(
        "# benchmark_fixture: public Loader/LoadTensor read_copy, immutable source bytes=1048576 fill=0xa5, caller target bytes=1048576"
    );
    println!(
        "# benchmark_validation: typed loader done checked each iteration, target equals source after measurement"
    );
    let source = vec![0xa5; COPY_BYTES];
    let mut target_bytes = vec![0_u8; COPY_BYTES];
    let target = ReadTarget::new(&mut target_bytes);
    let span = TensorLoadSpan::new(1, "benchmark.bin", Some(&source), &target);
    let policy = StrategyPolicy::new(StrategyKind::ReadCopy);
    let mut loader = IoLoader::with_reader(Reader::new());
    let timing = measure(config, || {
        let done = loader
            .process_event(LoadTensor::new(black_box(span), policy))
            .map_err(|_| "loader dispatch failed")?;
        black_box(done);
        Ok::<(), &'static str>(())
    })?;
    if !target.try_matches(&source)? {
        return Err("loader benchmark target mismatch".into());
    }
    print_case("io/loader/rust/read_copy_1mib", timing, config);
    Ok(())
}

fn run_io_mmap(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    const FIXTURE_BYTES: usize = 1_048_576;
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa");
    println!(
        "# benchmark_fixture: public file-backed mmap lifecycle, file_bytes=1048576 pattern=incrementing_u8 offset=0 cases=16384,1048576"
    );
    println!(
        "# benchmark_operations: source open, map, full immutable access checksum, sequential advice, will-need advice, don't-need advice, release"
    );
    println!(
        "# benchmark_validation: typed outcomes and handle length checked each iteration, full mapped FNV-1a equals precomputed fixture checksum"
    );
    println!("# native_semantics_complete: true");
    println!("# missing_native_semantics: none");
    println!("# contract_delta: rust_mmap_source_capability");

    let external_fixture = env::var_os("EMEL_IO_MMAP_BENCH_FIXTURE").is_some();
    let fixture = mmap_fixture_path();
    let bytes = (0_u8..=255).cycle().take(FIXTURE_BYTES).collect::<Vec<_>>();
    let checksum_16kib = fnv1a64(&bytes[..16_384]);
    let checksum_1mib = fnv1a64(&bytes);
    std::fs::write(&fixture, &bytes)?;
    let mut mapper = Mapper::new();
    let timing_16kib = benchmark_mmap_case(&mut mapper, &fixture, 16_384, checksum_16kib, config)?;
    let timing_1mib =
        benchmark_mmap_case(&mut mapper, &fixture, FIXTURE_BYTES, checksum_1mib, config)?;
    if !external_fixture {
        let _ = std::fs::remove_file(&fixture);
    }
    print_case("io/mmap/rust/lifecycle_16kib", timing_16kib, config);
    print_case("io/mmap/rust/lifecycle_1mib", timing_1mib, config);
    Ok(())
}

fn run_io_staged_read(config: Config) -> Result<(), emel_io::staged_read::event::Error> {
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("# source_tree: ff00a9978b00b4ea1e5268d2ecd483d4855b6eaa");
    println!(
        "# benchmark_fixture: public Stager/StageWindow, immutable source fill=0xa5, caller target cases=16384,1048576, chunk_bytes=4096"
    );
    println!(
        "# benchmark_validation: typed done and synchronous callback checked each iteration, target equals source after measurement"
    );
    print_case(
        "io/staged-read/rust/copy_16kib",
        benchmark_staged_read_case(16_384, config)?,
        config,
    );
    print_case(
        "io/staged-read/rust/copy_1mib",
        benchmark_staged_read_case(1_048_576, config)?,
        config,
    );
    Ok(())
}

fn benchmark_staged_read_case(
    copy_bytes: usize,
    config: Config,
) -> Result<f64, emel_io::staged_read::event::Error> {
    let source = vec![0xa5; copy_bytes];
    let mut target = vec![0_u8; copy_bytes];
    let target_capability = StageTarget::new(&mut target);
    let logical = u64::try_from(copy_bytes).expect("benchmark fixture length");
    let done = Cell::new(None::<StageWindowDone>);
    let error = Cell::new(None::<StageWindowError>);
    let mut actor = Stager::new();
    let timing = measure(config, || {
        done.set(None);
        error.set(None);
        let outcome = actor.process_event(
            StageWindow::new(
                0,
                logical,
                4_096,
                Some(black_box(&source)),
                black_box(&target_capability),
            )
            .on_done(StageCallback::store(&done))
            .on_error(StageCallback::store(&error)),
        )?;
        if outcome.bytes_committed() != logical || done.get() != Some(outcome) {
            return Err(emel_io::staged_read::event::Error::InternalError);
        }
        black_box(outcome);
        Ok(())
    })?;
    if !target_capability
        .try_matches(&source)
        .map_err(|_| emel_io::staged_read::event::Error::InternalError)?
    {
        return Err(emel_io::staged_read::event::Error::InternalError);
    }
    Ok(timing)
}

fn benchmark_mmap_case(
    mapper: &mut Mapper,
    fixture: &std::path::Path,
    mapping_bytes: usize,
    expected_checksum: u64,
    config: Config,
) -> Result<f64, emel_io::mmap::event::Error> {
    let mut final_checksum = 0_u64;
    let timing = measure(config, || {
        let source = open_benchmark_source(fixture)?;
        let done = mapper.process_event(MapTensor::new(71, source, 0, mapping_bytes as u64))?;
        if done.len() != mapping_bytes as u64 {
            return Err(emel_io::mmap::event::Error::InternalError);
        }
        let mut checksum = 0_u64;
        let mut observe = |mapped: &[u8]| checksum = fnv1a64(mapped);
        let callback = MappingCallback::new(&mut observe);
        mapper.process_event(WithMapping::new(71, done.handle(), &callback))?;
        mapper.process_event(AdviseSequential::new(
            71,
            done.handle(),
            0,
            mapping_bytes as u64,
        ))?;
        mapper.process_event(AdviseWillNeed::new(71, done.handle(), 0, 4_096))?;
        mapper.process_event(AdviseDontNeed::new(71, done.handle(), 0, 4_096))?;
        mapper.process_event(ReleaseMapping::new(71, done.handle()))?;
        if checksum != expected_checksum {
            return Err(emel_io::mmap::event::Error::InternalError);
        }
        final_checksum = black_box(checksum);
        Ok(())
    })?;
    if final_checksum != expected_checksum {
        return Err(emel_io::mmap::event::Error::InternalError);
    }
    Ok(timing)
}

#[allow(
    unsafe_code,
    reason = "the benchmark fixture is immutable from capability construction through every measured release"
)]
fn open_benchmark_source(
    fixture: &std::path::Path,
) -> Result<MmapSource, emel_io::mmap::event::Error> {
    // SAFETY: fixture construction finishes before this call, measurement only
    // reads it, and every mapping is released before the capability is dropped.
    unsafe { MmapSource::open(fixture) }
}

fn mmap_fixture_path() -> PathBuf {
    env::var_os("EMEL_IO_MMAP_BENCH_FIXTURE").map_or_else(
        || env::temp_dir().join(format!("emel-io-mmap-bench-{}.bin", std::process::id())),
        PathBuf::from,
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn parse_config() -> Result<(Suite, Config), Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut suite = None;
    for argument in env::args().skip(1) {
        if argument == "gguf" {
            suite = Some(Suite::Gguf);
            continue;
        }
        if argument == "io-read" {
            suite = Some(Suite::IoRead);
            continue;
        }
        if argument == "io-mmap" {
            suite = Some(Suite::IoMmap);
            continue;
        }
        if argument == "io-staged-read" {
            suite = Some(Suite::IoStagedRead);
            continue;
        }
        if argument == "io-loader" {
            suite = Some(Suite::IoLoader);
            continue;
        }
        if argument == "--help" || argument == "-h" {
            println!(
                "usage: emel-bench [gguf|io-read|io-mmap|io-staged-read|io-loader] [--iterations=N] [--runs=N] \
                 [--warmup-iterations=N]"
            );
            std::process::exit(0);
        }
        if let Some(value) = argument.strip_prefix("--iterations=") {
            config.iterations = value.parse()?;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--runs=") {
            config.runs = value.parse()?;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--warmup-iterations=") {
            config.warmup_iterations = value.parse()?;
            continue;
        }
        return Err(format!("unknown argument: {argument}").into());
    }
    if config.iterations == 0 || config.runs == 0 || u32::try_from(config.iterations).is_err() {
        return Err("iterations and runs must be nonzero, and iterations must fit u32".into());
    }
    Ok((suite.unwrap_or(Suite::Gguf), config))
}

fn benchmark_probe(bytes: &[u8], config: Config) -> Result<f64, Error> {
    let mut loader = GgufLoader::new();
    measure(config, || {
        let requirements = loader.process_event(Probe::new(black_box(bytes)))?;
        black_box(requirements);
        Ok(())
    })
}

fn benchmark_load(bytes: &[u8], config: Config) -> Result<f64, Error> {
    measure(config, || {
        let model = GgufLoader::new().process_event(Load::new(black_box(bytes)))?;
        black_box(model);
        Ok(())
    })
}

fn benchmark_parse(bytes: &[u8], config: Config) -> Result<f64, Error> {
    let mut loader = GgufLoader::new();
    loader.process_event(Probe::new(bytes))?;
    loader.process_event(Bind::exact())?;
    measure(config, || {
        let model = loader.process_event(Parse::new(black_box(bytes)))?;
        black_box(model);
        Ok(())
    })
}

fn measure<E>(config: Config, mut operation: impl FnMut() -> Result<(), E>) -> Result<f64, E> {
    for _ in 0..config.warmup_iterations {
        operation()?;
    }
    let mut samples = Vec::with_capacity(config.runs);
    for _ in 0..config.runs {
        let started = Instant::now();
        for _ in 0..config.iterations {
            operation()?;
        }
        let iterations = f64::from(u32::try_from(config.iterations).expect("validated iterations"));
        samples.push(started.elapsed().as_secs_f64() * 1_000_000_000.0 / iterations);
    }
    samples.sort_by(f64::total_cmp);
    Ok(median(&samples))
}

fn median(samples: &[f64]) -> f64 {
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        samples[middle - 1].midpoint(samples[middle])
    } else {
        samples[middle]
    }
}

fn print_case(name: &str, nanoseconds: f64, config: Config) {
    println!(
        "{name} ns_per_op={nanoseconds:.3} iter={} runs={}",
        config.iterations, config.runs
    );
}

fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn append_string(bytes: &mut Vec<u8>, value: &[u8]) {
    append_u64(bytes, u64::try_from(value.len()).expect("fixture length"));
    bytes.extend_from_slice(value);
}

fn append_header(bytes: &mut Vec<u8>, tensors: u64, metadata: u64) {
    bytes.extend_from_slice(&MAGIC);
    append_u32(bytes, VERSION);
    append_u64(bytes, tensors);
    append_u64(bytes, metadata);
}

fn metadata_fixture() -> Vec<u8> {
    let mut bytes = Vec::new();
    append_header(&mut bytes, 0, 64);
    for index in 0..64_u32 {
        append_string(
            &mut bytes,
            format!("benchmark.metadata.{index:02}").as_bytes(),
        );
        append_u32(&mut bytes, 8);
        append_string(
            &mut bytes,
            format!("representative metadata value {index:02}").as_bytes(),
        );
    }
    bytes
}

fn tensor_fixture() -> Vec<u8> {
    const TENSOR_COUNT: u64 = 64;
    const ELEMENTS: u64 = 256 * 4;
    const TENSOR_BYTES: u64 = ELEMENTS * 4;
    let mut bytes = Vec::new();
    append_header(&mut bytes, TENSOR_COUNT, 0);
    for index in 0..TENSOR_COUNT {
        append_string(
            &mut bytes,
            format!("benchmark.tensor.{index:02}").as_bytes(),
        );
        append_u32(&mut bytes, 2);
        append_u64(&mut bytes, 256);
        append_u64(&mut bytes, 4);
        append_u32(&mut bytes, 0);
        append_u64(&mut bytes, index * TENSOR_BYTES);
    }
    bytes.resize(bytes.len().next_multiple_of(ALIGNMENT), 0);
    bytes.resize(
        bytes.len() + usize::try_from(TENSOR_COUNT * TENSOR_BYTES).expect("fixture size"),
        0xa5,
    );
    bytes
}

#[cfg(test)]
mod tests {
    use emel_gguf::Loader as GgufLoader;
    use emel_gguf::event::{Load, ParseDone};

    use super::{median, metadata_fixture, tensor_fixture};

    fn load(file_image: &[u8]) -> ParseDone<'_> {
        GgufLoader::new()
            .process_event(Load::new(file_image))
            .expect("fixture loads")
    }

    #[test]
    fn representative_fixtures_load() {
        assert_eq!(load(&metadata_fixture()).metadata().len(), 64);
        assert_eq!(load(&tensor_fixture()).tensors().len(), 64);
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert!((median(&[1.0, 2.0, 3.0]) - 2.0).abs() < f64::EPSILON);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < f64::EPSILON);
    }
}
