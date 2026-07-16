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
    AdviseDontNeed, AdviseSequential, AdviseWillNeed, MapTensor, MmapSource, ReleaseMapping,
    WithMapping,
};
use emel_io::read::Reader;
use emel_io::read::event::{ReadTensor, Target as ReadTarget};
use emel_io::staged_read::Stager;
use emel_io::staged_read::event::{
    Callback as StageCallback, StageWindow, StageWindowDone, StageWindowError,
    Target as StageTarget,
};
use emel_model::tensor::Store as TensorStore;
use emel_model::tensor::dependency::Actors as TensorActors;
use emel_model::tensor::event::{
    ApplyEffectError, BindStorage, CaptureTensorState, EffectBuffer, EffectError, EffectRequest,
    Error as TensorError, Lifecycle as TensorLifecycle, MappedLoad, PlanLoad, ReadLoad,
    ReleaseMapped, StorageBatch, StorageEntry, StrategyKind as TensorStrategy, TensorMetadata,
    WithTensor,
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
    ModelTensor,
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
    let arguments = env::args().collect::<Vec<_>>();
    if arguments
        .get(1)
        .is_some_and(|value| value == "--model-tensor-mapped-parity")
    {
        let Some(path) = arguments.get(2) else {
            eprintln!("emel-bench: mapped parity requires a fixture path");
            std::process::exit(2);
        };
        if let Err(error) = run_model_tensor_mapped_parity(&PathBuf::from(path)) {
            eprintln!("emel-bench: {error}");
            std::process::exit(2);
        }
        return;
    }
    if let Err(error) = run() {
        eprintln!("emel-bench: {error}");
        std::process::exit(2);
    }
}

fn run_model_tensor_mapped_parity(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = (0_u8..=255).cycle().take(4_096).collect::<Vec<_>>();
    std::fs::write(path, &bytes)?;
    let source = open_benchmark_source(path)?;
    let duplicate_source = source.clone();
    let dependencies = TensorActors::new(Mapper::new(), (), ());
    let mut store = TensorStore::with_dependencies(1, dependencies)?;
    store
        .process_event(BindStorage::new(StorageBatch::new(
            vec![StorageEntry::new(
                TensorMetadata::new(1_234, 5_678, 0, 0),
                None,
            )]
            .into_boxed_slice(),
        )))
        .map_err(|error| error.error())?;
    let mapped = store.process_event(MappedLoad::new(0, source, 0, 4_096))?;
    let state = store.process_event(CaptureTensorState::new(0))?;
    println!(
        "case=mapped_load outcome=done tensor_id={} bytes={} lifecycle={}",
        mapped.tensor_id(),
        mapped.mapped_bytes(),
        tensor_lifecycle(state.lifecycle())
    );
    let planned = store
        .process_event(PlanLoad::new(
            TensorStrategy::None,
            EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice()),
        ))
        .map_err(|error| error.error())?;
    let effects = planned.into_effects();
    let EffectRequest::None {
        tensor_id,
        file_index,
        offset,
        size,
    } = effects.effects()[0]
    else {
        return Err("mapped resident plan produced the wrong strategy".into());
    };
    println!(
        "case=mapped_resident_plan outcome=done first=none:{tensor_id}:{file_index}:{offset}:{size}"
    );
    if store.process_event(ApplyEffectError::new(0, EffectError::Backend))
        != Err(TensorError::BackendError)
    {
        return Err("mapped resident plan did not recover through typed backend error".into());
    }
    let duplicate = store
        .process_event(MappedLoad::new(0, duplicate_source, 0, 4_096))
        .expect_err("duplicate mapping is rejected");
    let state = store.process_event(CaptureTensorState::new(0))?;
    println!(
        "case=mapped_duplicate outcome=error error={} retained_bytes={}",
        tensor_error(duplicate),
        state.buffer_bytes()
    );
    let wrong_release = store
        .process_event(ReleaseMapped::new(0, mapped.mapping_handle() + 1))
        .expect_err("wrong release token is rejected");
    let state = store.process_event(CaptureTensorState::new(0))?;
    println!(
        "case=mapped_wrong_release outcome=error error={} lifecycle={}",
        tensor_error(wrong_release),
        tensor_lifecycle(state.lifecycle())
    );
    let released = store.process_event(ReleaseMapped::new(0, mapped.mapping_handle()))?;
    let state = store.process_event(CaptureTensorState::new(0))?;
    println!(
        "case=mapped_release outcome=done tensor_id={} lifecycle={}",
        released.tensor_id(),
        tensor_lifecycle(state.lifecycle())
    );
    drop(store);
    Ok(())
}

const fn tensor_lifecycle(lifecycle: TensorLifecycle) -> &'static str {
    match lifecycle {
        TensorLifecycle::MappedResident => "mapped_resident",
        TensorLifecycle::Evicted => "evicted",
        _ => "other",
    }
}

const fn tensor_error(error: TensorError) -> &'static str {
    match error {
        TensorError::InvalidRequest => "invalid_request",
        TensorError::TensorAlreadyResident => "tensor_already_resident",
        _ => "other",
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
        Suite::ModelTensor => run_model_tensor(config)?,
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

fn run_model_tensor(config: Config) -> Result<(), TensorError> {
    const TENSORS: usize = 64;
    println!("# source_repository: stateforward/emel.cpp");
    println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("# source_tree: 06306d4ffad3455fcf5df71dc692df52514b9865");
    println!(
        "# benchmark_fixture: production Store/process_event, direct read/copy bytes=4096 with preallocated actor-owned targets, plus 64 bound metadata records and a preallocated mapped effect buffer"
    );
    println!(
        "# benchmark_validation: direct read times public child dispatch and copy on both lanes with typed completion and byte checks outside timing; mapped planning checks typed count and lane-native recovery with returned allocation reset and reused"
    );
    println!(
        "# contract_delta: direct-read target ownership differs but both timed lanes dispatch the public tensor actor and copy 4096 bytes into preallocated target storage; each timed plan closes with its lane-native backend-error event"
    );

    print_case(
        "model/tensor/rust/direct_read_4k",
        benchmark_model_tensor_direct_read(config)?,
        config,
    );

    let entries = (0..TENSORS)
        .map(|index| {
            StorageEntry::new(
                TensorMetadata::new(
                    u64::try_from(index + 1).expect("fixture index") * 4_096,
                    32,
                    u16::try_from(index % 4).expect("fixture file index"),
                    7,
                ),
                None,
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let mut store = TensorStore::new(TENSORS)?;
    store
        .process_event(BindStorage::new(StorageBatch::new(entries)))
        .map_err(|error| error.error())?;
    let mut effects = Some(EffectBuffer::new(
        vec![EffectRequest::Empty; TENSORS].into_boxed_slice(),
    ));
    let mut plan_and_recover = || {
        let buffer = effects
            .take()
            .expect("benchmark owns one effect allocation");
        let started = Instant::now();
        let done = store
            .process_event(PlanLoad::new(TensorStrategy::MappedFile, buffer))
            .map_err(|error| error.error())?;
        let elapsed = started.elapsed();
        if done.effect_count() != TENSORS {
            return Err(TensorError::Internal);
        }
        let outcome = store.process_event(ApplyEffectError::new(0, EffectError::Backend));
        if outcome != Err(TensorError::BackendError) {
            return Err(TensorError::Internal);
        }
        let mut buffer = done.into_effects();
        buffer.reset();
        effects = Some(buffer);
        Ok(elapsed.as_secs_f64() * 1_000_000_000.0)
    };
    for _ in 0..config.warmup_iterations {
        black_box(plan_and_recover()?);
    }
    let mut samples = Vec::with_capacity(config.runs);
    for _ in 0..config.runs {
        let mut elapsed_ns = 0.0;
        for _ in 0..config.iterations {
            elapsed_ns += plan_and_recover()?;
        }
        let iterations = f64::from(u32::try_from(config.iterations).expect("validated iterations"));
        samples.push(elapsed_ns / iterations);
    }
    samples.sort_by(f64::total_cmp);
    print_case("model/tensor/rust/plan_mapped_64", median(&samples), config);
    Ok(())
}

fn benchmark_model_tensor_direct_read(config: Config) -> Result<f64, TensorError> {
    const COPY_BYTES: usize = 4_096;
    let source = vec![0xa5_u8; COPY_BYTES];
    let expected_checksum = u64::from(0xa5_u8) * u64::try_from(COPY_BYTES).expect("fixture size");
    let execute = |count: u64| -> Result<f64, TensorError> {
        let count = usize::try_from(count).map_err(|_| TensorError::Capacity)?;
        if count == 0 || count > 65_536 {
            return Err(TensorError::Capacity);
        }
        let count_f64 = f64::from(u32::try_from(count).map_err(|_| TensorError::Capacity)?);
        let entries = (0..count)
            .map(|_| {
                StorageEntry::new(
                    TensorMetadata::new(0, COPY_BYTES as u64, 0, 0),
                    Some(vec![0; COPY_BYTES].into_boxed_slice()),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let dependencies = TensorActors::new((), Reader::new(), ());
        let mut store = TensorStore::with_dependencies(count, dependencies)?;
        store
            .process_event(BindStorage::new(StorageBatch::new(entries)))
            .map_err(|error| error.error())?;

        let started = Instant::now();
        for tensor_id in 0..count {
            let done = store.process_event(ReadLoad::new(
                i32::try_from(tensor_id).expect("tensor capacity fits i32"),
                "benchmark.bin",
                Some(black_box(&source)),
                0,
                COPY_BYTES as u64,
            ))?;
            black_box(done);
        }
        let elapsed = started.elapsed().as_secs_f64() * 1_000_000_000.0;

        for tensor_id in [0, count - 1] {
            let mut checksum = |bytes: &[u8]| bytes.iter().copied().map(u64::from).sum::<u64>();
            if store.process_event(WithTensor::new(
                i32::try_from(tensor_id).expect("tensor capacity fits i32"),
                &mut checksum,
            ))? != expected_checksum
            {
                return Err(TensorError::Internal);
            }
        }
        Ok(elapsed / count_f64)
    };

    black_box(execute(config.warmup_iterations)?);
    let mut samples = Vec::with_capacity(config.runs);
    for _ in 0..config.runs {
        samples.push(execute(config.iterations)?);
    }
    samples.sort_by(f64::total_cmp);
    Ok(median(&samples))
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
        let mut observe = |mapped: &[u8]| fnv1a64(mapped);
        let checksum = mapper.process_event(WithMapping::new(71, done.handle(), &mut observe))?;
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
        if argument == "model-tensor" {
            suite = Some(Suite::ModelTensor);
            continue;
        }
        if argument == "--help" || argument == "-h" {
            println!(
                "usage: emel-bench [gguf|io-read|io-mmap|io-staged-read|io-loader|model-tensor] [--iterations=N] [--runs=N] \
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
    use allocation_counter::measure;
    use emel_gguf::Loader as GgufLoader;
    use emel_gguf::event::{Load, ParseDone};
    use emel_model::tensor::event::{
        ApplyBoundEffectResults, ApplyOwnedEffectResults, CaptureTensorState,
        Lifecycle as TensorLifecycle, MappedLoad, OwnedEffectResult, ReleaseMapped,
    };

    use super::{
        BindStorage, EffectBuffer, EffectRequest, Mapper, PlanLoad, StorageBatch, StorageEntry,
        TensorActors, TensorMetadata, TensorStore, TensorStrategy, WithTensor, median,
        metadata_fixture, open_benchmark_source, tensor_fixture,
    };

    fn load(file_image: &[u8]) -> ParseDone<'_> {
        GgufLoader::new()
            .process_event(Load::new(file_image))
            .expect("fixture loads")
    }

    fn assert_tensor_lifecycle<Dependencies>(
        store: &mut TensorStore<Dependencies>,
        expected: TensorLifecycle,
    ) where
        Dependencies: emel_model::tensor::dependency::TensorDependencies,
    {
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            expected
        );
    }

    fn assert_quarantine_rejects_bulk_plan<Dependencies>(
        store: &mut TensorStore<Dependencies>,
        live: &std::cell::Cell<bool>,
    ) where
        Dependencies: emel_model::tensor::dependency::TensorDependencies,
    {
        let plan_result = std::cell::RefCell::new(None);
        let plan = PlanLoad::new(
            TensorStrategy::ReadCopy,
            EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice()),
        );
        let allocation = measure(|| {
            plan_result.replace(Some(store.process_event(plan)));
        });
        assert_eq!(allocation.count_total, 0);
        let plan_error = plan_result
            .borrow_mut()
            .take()
            .unwrap()
            .expect_err("cleanup-pending mapping rejects bulk planning");
        assert_eq!(
            plan_error.error(),
            super::TensorError::MappedTensorRequiresRelease
        );
        assert_eq!(plan_error.into_effects().effects(), &[EffectRequest::Empty]);

        let apply_result = std::cell::RefCell::new(None);
        let results = ApplyOwnedEffectResults::new(
            vec![OwnedEffectResult::new(0, vec![9; 4_096].into_boxed_slice())].into_boxed_slice(),
        );
        let allocation = measure(|| {
            apply_result.replace(Some(store.process_event(results)));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            apply_result
                .borrow()
                .as_ref()
                .unwrap()
                .as_ref()
                .expect_err("rejected plan leaves no result phase")
                .error(),
            super::TensorError::InvalidRequest
        );
        assert_tensor_lifecycle(store, TensorLifecycle::MappedCleanupPending);
        assert!(live.get());
    }

    struct MalformedMapper<'state> {
        live: &'state std::cell::Cell<bool>,
        releases: &'state std::cell::Cell<u32>,
        release_owner: &'state std::cell::Cell<Option<(i32, u32)>>,
        fail_release: &'state std::cell::Cell<bool>,
    }

    impl emel_model::tensor::dependency::Mapper for MalformedMapper<'_> {
        const AVAILABLE: bool = true;

        fn map_tensor(
            &mut self,
            event: emel_io::mmap::event::MapTensor,
        ) -> Result<emel_io::mmap::event::MapDone, emel_io::mmap::event::Error> {
            assert_eq!(event.tensor_id(), 0);
            assert_eq!(event.file_index(), 0);
            assert_eq!(event.offset(), 0);
            assert_eq!(event.len(), 4_096);
            self.live.set(true);
            Ok(emel_io::mmap::event::MapDone::new(7, 99, 3))
        }

        fn release_mapping(
            &mut self,
            event: emel_io::mmap::event::ReleaseMapping,
        ) -> Result<(), emel_io::mmap::event::Error> {
            self.release_owner
                .set(Some((event.tensor_id(), event.handle())));
            self.releases.set(self.releases.get() + 1);
            if self.fail_release.get() {
                Err(emel_io::mmap::event::Error::UnmapFailed)
            } else {
                self.live.set(false);
                Ok(())
            }
        }

        fn with_mapping<Operation>(
            &mut self,
            event: emel_io::mmap::event::WithMapping<'_, Operation>,
        ) -> Result<Operation::Output, emel_io::mmap::event::Error>
        where
            Operation: emel_io::mmap::event::MappingOperation,
        {
            Ok(event.apply(&[1, 2, 3]))
        }
    }

    impl emel_model::tensor::dependency::Reader for MalformedMapper<'_> {
        const AVAILABLE: bool = false;

        fn read_tensor(
            &mut self,
            _: emel_io::read::event::ReadTensor<'_>,
        ) -> Result<emel_io::read::event::ReadTensorDone, emel_io::read::event::Error> {
            Err(emel_io::read::event::Error::UnsupportedPlatform)
        }
    }

    impl emel_model::tensor::dependency::Stager for MalformedMapper<'_> {
        const AVAILABLE: bool = false;

        fn stage_tensor(
            &mut self,
            _: emel_io::staged_read::event::StageWindow<'_>,
        ) -> Result<emel_io::staged_read::event::StageWindowDone, emel_io::staged_read::event::Error>
        {
            Err(emel_io::staged_read::event::Error::UnsupportedPlatform)
        }
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

    #[test]
    fn malformed_mapped_success_releases_or_retains_ownership_for_retry() {
        let path = std::env::temp_dir().join(format!(
            "emel-model-tensor-malformed-mapped-test-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [0_u8; 4_096]).unwrap();
        let source = open_benchmark_source(&path).unwrap();
        let failure_source = source.clone();
        let live = std::cell::Cell::new(false);
        let release_calls = std::cell::Cell::new(0);
        let release_owner = std::cell::Cell::new(None);
        let fail_release = std::cell::Cell::new(false);
        let malformed_mapper = MalformedMapper {
            live: &live,
            releases: &release_calls,
            release_owner: &release_owner,
            fail_release: &fail_release,
        };
        let mut store = TensorStore::with_dependencies(1, malformed_mapper).unwrap();
        store
            .process_event(BindStorage::new(StorageBatch::new(
                vec![StorageEntry::new(TensorMetadata::new(0, 4_096, 0, 0), None)]
                    .into_boxed_slice(),
            )))
            .unwrap();

        let malformed_result = std::cell::Cell::new(None);
        let allocation = measure(|| {
            malformed_result.set(Some(
                store.process_event(MappedLoad::new(0, source, 0, 4_096)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            malformed_result.get(),
            Some(Err(super::TensorError::DependencyContract))
        );
        assert_eq!(release_calls.get(), 1);
        assert_eq!(release_owner.get(), Some((99, 7)));
        assert!(!live.get());
        assert_tensor_lifecycle(&mut store, TensorLifecycle::Unbound);

        fail_release.set(true);
        let malformed_failure = std::cell::Cell::new(None);
        let allocation = measure(|| {
            malformed_failure.set(Some(store.process_event(MappedLoad::new(
                0,
                failure_source,
                0,
                4_096,
            ))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            malformed_failure.get(),
            Some(Err(super::TensorError::DependencyContractCleanup {
                mapping_handle: 7,
                error: emel_io::mmap::event::Error::UnmapFailed,
            }))
        );
        assert_eq!(release_calls.get(), 2);
        assert_eq!(release_owner.get(), Some((99, 7)));
        assert!(live.get());
        assert_tensor_lifecycle(&mut store, TensorLifecycle::MappedCleanupPending);

        let operation_called = std::cell::Cell::new(false);
        let mut inspect = |_: &[u8]| operation_called.set(true);
        let quarantined_access = std::cell::Cell::new(None);
        let allocation = measure(|| {
            quarantined_access.set(Some(store.process_event(WithTensor::new(0, &mut inspect))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            quarantined_access.get(),
            Some(Err(super::TensorError::MappedTensorRequiresRelease))
        );
        assert!(!operation_called.get());

        assert_quarantine_rejects_bulk_plan(&mut store, &live);

        fail_release.set(false);
        let recovered = std::cell::Cell::new(None);
        let allocation = measure(|| {
            recovered.set(Some(store.process_event(ReleaseMapped::new(0, 7))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(recovered.get().unwrap().unwrap().tensor_id(), 0);
        assert_eq!(release_calls.get(), 3);
        assert_eq!(release_owner.get(), Some((99, 7)));
        assert!(!live.get());
        assert_tensor_lifecycle(&mut store, TensorLifecycle::Evicted);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[allow(
        clippy::cognitive_complexity,
        clippy::too_many_lines,
        reason = "the mapped capability invariant is proven by one linear end-to-end ownership lifecycle"
    )]
    fn model_tensor_mapped_route_owns_access_and_release_through_public_actors() {
        let path = std::env::temp_dir().join(format!(
            "emel-model-tensor-mapped-test-{}.bin",
            std::process::id()
        ));
        let bytes = (0_u8..=255).cycle().take(4_096).collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let source = open_benchmark_source(&path).unwrap();
        let absent_source = source.clone();
        let resident_source = source.clone();
        let unbound_source = source.clone();
        let inactive_source = source.clone();

        let dependencies = TensorActors::new(Mapper::new(), (), ());
        let mut unbound = TensorStore::with_dependencies(2, dependencies).unwrap();
        let invalid = std::cell::Cell::new(None);
        let allocation = measure(|| {
            invalid.set(Some(unbound.process_event(MappedLoad::new(
                0,
                unbound_source,
                0,
                4_096,
            ))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(invalid.get(), Some(Err(super::TensorError::InvalidRequest)));
        unbound
            .process_event(BindStorage::new(StorageBatch::new(
                vec![StorageEntry::new(TensorMetadata::new(0, 4_096, 0, 0), None)]
                    .into_boxed_slice(),
            )))
            .unwrap();
        assert_eq!(
            unbound.process_event(MappedLoad::new(1, inactive_source, 0, 4_096)),
            Err(super::TensorError::InvalidRequest)
        );
        assert_eq!(
            unbound.process_event(ReleaseMapped::new(1, 0)),
            Err(super::TensorError::InvalidRequest)
        );

        let mut absent = TensorStore::new(1).unwrap();
        absent
            .process_event(BindStorage::new(StorageBatch::new(
                vec![StorageEntry::new(TensorMetadata::new(0, 4_096, 0, 0), None)]
                    .into_boxed_slice(),
            )))
            .unwrap();
        let unavailable = std::cell::Cell::new(None);
        let allocation = measure(|| {
            unavailable.set(Some(absent.process_event(MappedLoad::new(
                0,
                absent_source,
                0,
                4_096,
            ))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            unavailable.get(),
            Some(Err(super::TensorError::MmapUnavailable))
        );

        let dependencies = TensorActors::new(Mapper::new(), (), ());
        let mut store = TensorStore::with_dependencies(1, dependencies).unwrap();
        store
            .process_event(BindStorage::new(StorageBatch::new(
                vec![StorageEntry::new(
                    TensorMetadata::new(1_234, 5_678, 0, 0),
                    None,
                )]
                .into_boxed_slice(),
            )))
            .unwrap();

        let mapped = std::cell::Cell::new(None);
        let allocation = measure(|| {
            mapped.set(Some(
                store.process_event(MappedLoad::new(0, source, 0, 4_096)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        let mapped = mapped.get().unwrap().unwrap();
        assert_eq!(mapped.mapped_bytes(), 4_096);
        assert_eq!(mapped.tensor_id(), 0);
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            TensorLifecycle::MappedResident
        );

        let planned = std::cell::RefCell::new(None);
        let effects = EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice());
        let allocation = measure(|| {
            planned.replace(Some(
                store.process_event(PlanLoad::new(TensorStrategy::None, effects)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        let planned = planned
            .borrow_mut()
            .take()
            .unwrap()
            .expect("ordinary mapped residency preserves pinned planning");
        assert_eq!(
            planned.into_effects().effects(),
            &[EffectRequest::None {
                tensor_id: 0,
                file_index: 0,
                offset: 1_234,
                size: 5_678,
            }]
        );

        let bound_result = std::cell::RefCell::new(None);
        let tensor_ids = vec![0].into_boxed_slice();
        let allocation = measure(|| {
            bound_result.replace(Some(
                store.process_event(ApplyBoundEffectResults::new(tensor_ids)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            bound_result
                .borrow()
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap_err()
                .error(),
            super::TensorError::MappedTensorRequiresRelease
        );
        assert_tensor_lifecycle(&mut store, TensorLifecycle::MappedResident);

        let planned = std::cell::RefCell::new(None);
        let effects = EffectBuffer::new(vec![EffectRequest::Empty].into_boxed_slice());
        let allocation = measure(|| {
            planned.replace(Some(
                store.process_event(PlanLoad::new(TensorStrategy::ReadCopy, effects)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        planned
            .borrow_mut()
            .take()
            .unwrap()
            .expect("ordinary mapped residency preserves strategy planning");
        let owned_result = std::cell::RefCell::new(None);
        let results = ApplyOwnedEffectResults::new(
            vec![OwnedEffectResult::new(0, vec![9; 5_678].into_boxed_slice())].into_boxed_slice(),
        );
        let allocation = measure(|| {
            owned_result.replace(Some(store.process_event(results)));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            owned_result
                .borrow()
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap_err()
                .error(),
            super::TensorError::MappedTensorRequiresRelease
        );
        assert_tensor_lifecycle(&mut store, TensorLifecycle::MappedResident);

        let checksum = std::cell::Cell::new(None);
        let mut inspect = |mapped: &[u8]| super::fnv1a64(mapped);
        let allocation = measure(|| {
            checksum.set(Some(store.process_event(WithTensor::new(0, &mut inspect))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(checksum.get(), Some(Ok(super::fnv1a64(&bytes))));

        let wrong_release = std::cell::Cell::new(None);
        let allocation = measure(|| {
            wrong_release.set(Some(
                store.process_event(ReleaseMapped::new(0, mapped.mapping_handle() + 1)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            wrong_release.get(),
            Some(Err(super::TensorError::InvalidRequest))
        );

        let already_resident = std::cell::Cell::new(None);
        let allocation = measure(|| {
            already_resident.set(Some(store.process_event(MappedLoad::new(
                0,
                resident_source,
                0,
                4_096,
            ))));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            already_resident.get(),
            Some(Err(super::TensorError::TensorAlreadyResident))
        );
        let mut length = |mapped: &[u8]| mapped.len();
        assert_eq!(
            store.process_event(WithTensor::new(0, &mut length)),
            Ok(4_096)
        );

        let released = std::cell::Cell::new(None);
        let allocation = measure(|| {
            released.set(Some(
                store.process_event(ReleaseMapped::new(0, mapped.mapping_handle())),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(released.get().unwrap().unwrap().tensor_id(), 0);
        assert_eq!(
            store
                .process_event(CaptureTensorState::new(0))
                .unwrap()
                .lifecycle(),
            TensorLifecycle::Evicted
        );

        let mut after_release = |mapped: &[u8]| mapped.len();
        let missing = std::cell::Cell::new(None);
        let allocation = measure(|| {
            missing.set(Some(
                store.process_event(WithTensor::new(0, &mut after_release)),
            ));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(missing.get(), Some(Err(super::TensorError::TensorUnbound)));

        drop(store);
        drop(absent);
        drop(unbound);
        std::fs::remove_file(path).unwrap();
    }
}
