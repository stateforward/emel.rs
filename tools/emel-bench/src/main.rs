//! Dependency-light benchmark runner for performance snapshot gates.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use emel_gguf::Loader;
use emel_gguf::event::{Bind, Error, Load, Parse, Probe};
use emel_io::read::Reader;
use emel_io::read::event::ReadTensor;

const ALIGNMENT: usize = 32;
const MAGIC: [u8; 4] = *b"GGUF";
const VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Suite {
    Gguf,
    IoRead,
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
    let mut reader = Reader::new();
    let timing = measure(config, || {
        let done = reader.process_event(ReadTensor::new(
            1,
            "benchmark.bin",
            Some(black_box(&source)),
            black_box(&mut target),
        ))?;
        black_box(done);
        Ok(())
    })?;
    if target != source {
        return Err(emel_io::read::event::Error::InternalError);
    }
    print_case("io/read/copy_1mib", timing, config);
    Ok(())
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
        if argument == "--help" || argument == "-h" {
            println!(
                "usage: emel-bench [gguf|io-read] [--iterations=N] [--runs=N] \
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
    let mut loader = Loader::new();
    measure(config, || {
        let requirements = loader.process_event(Probe::new(black_box(bytes)))?;
        black_box(requirements);
        Ok(())
    })
}

fn benchmark_load(bytes: &[u8], config: Config) -> Result<f64, Error> {
    measure(config, || {
        let model = Loader::new().process_event(Load::new(black_box(bytes)))?;
        black_box(model);
        Ok(())
    })
}

fn benchmark_parse(bytes: &[u8], config: Config) -> Result<f64, Error> {
    let mut loader = Loader::new();
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
    use emel_gguf::Loader;
    use emel_gguf::event::{Load, ParseDone};

    use super::{median, metadata_fixture, tensor_fixture};

    fn load(file_image: &[u8]) -> ParseDone<'_> {
        Loader::new()
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
