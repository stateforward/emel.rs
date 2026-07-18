//! Source-parity and performance observer for the public model catalog actor.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use allocation_counter as _;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe, WithTensor};
use emel_io as _;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, DescribeModel, Error, FindTensor, ReleaseStorage, Reset, SealModel, Storage,
    TensorBindingStatus, TensorDescriptor, TensorInput, WithTensorName,
};
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DATA_HEADER_BLOB: &str = "78a25b987423d8cbef17965a8ca92596ffc0ecef";
const DATA_IMPLEMENTATION_BLOB: &str = "b33ace170b569d076844a36146c7ca86d4ffa7fc";
const GENERATION_HEADER_BLOB: &str = "d521cf68e1bf52a2a193bbdb460741772199b318";
const GENERATION_IMPLEMENTATION_BLOB: &str = "099058ccd441d1dc6bebbb0c4994070d2f533c47";

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    match arguments.as_slice() {
        [_] => parity(),
        [_, option, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        [_, option, llama, lfm] if option == "--fixtures" => fixtures(llama, lfm),
        _ => {
            eprintln!(
                "usage: catalog_observer [--benchmark ITER RUNS WARMUP|--fixtures LLAMA LFM]"
            );
            std::process::exit(2);
        }
    }
}

fn fixtures(llama: &str, lfm: &str) {
    observe_fixture("llama", llama, b"output_norm.weight");
    observe_fixture("lfm", lfm, b"token_embd.weight");
}

fn observe_fixture(label: &str, path: &str, required_name: &[u8]) {
    let bytes = std::fs::read(path).expect("read real fixture");
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("probe real fixture");
    loader
        .process_event(Bind::new(
            emel_gguf::event::Storage::exact(probe).expect("GGUF storage"),
        ))
        .expect("bind real fixture");
    let parsed = loader
        .process_event(Parse::new())
        .expect("parse real fixture");

    let tensor_count = parsed.tensor_count();
    let mut name_bytes = 0usize;
    for index in 0..tensor_count {
        let result = loader.process_event(WithTensor::new(index, |name: &[u8], _, _: &[u8]| {
            name_bytes += name.len();
        }));
        assert!(result.expect("name sizing query").is_some());
    }
    let mut arena = vec![0u8; name_bytes];
    let mut ranges = vec![(0usize, 0usize); usize::try_from(tensor_count).expect("tensor count")];
    let mut cursor = 0usize;
    for index in 0..tensor_count {
        let result = loader.process_event(WithTensor::new(index, |name: &[u8], _, _: &[u8]| {
            let end = cursor + name.len();
            arena[cursor..end].copy_from_slice(name);
            ranges[usize::try_from(index).expect("tensor index")] = (cursor, end);
            cursor = end;
        }));
        assert!(result.expect("name copy query").is_some());
    }

    let storage = Storage::from_gguf(&mut loader, parsed).expect("catalog GGUF storage");
    let mut catalog = Catalog::try_new().expect("real fixture catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind real catalog");
    let model = catalog
        .process_event(SealModel::new())
        .expect("seal real catalog");
    let mut all = SemanticDigest::new();
    let mut selected = None;
    for &(start, end) in &ranges {
        let name = &arena[start..end];
        let descriptor = catalog
            .process_event(FindTensor::new(model, name))
            .expect("find real tensor")
            .expect("real tensor present");
        digest_descriptor(&mut all, name, descriptor);
        if name == required_name {
            let mut digest = SemanticDigest::new();
            digest_descriptor(&mut digest, name, descriptor);
            selected = Some(digest.value);
        }
    }
    println!(
        "fixture_case={label} tensor_count={tensor_count} digest={:016x} required_name={} required_digest={:016x}",
        all.value,
        std::str::from_utf8(required_name).expect("required name is UTF-8"),
        selected.expect("required tensor")
    );
}

struct SemanticDigest {
    value: u64,
}

impl SemanticDigest {
    const fn new() -> Self {
        Self {
            value: 14_695_981_039_346_656_037,
        }
    }

    fn byte(&mut self, input: u8) {
        self.value ^= u64::from(input);
        self.value = self.value.wrapping_mul(1_099_511_628_211);
    }

    fn u32(&mut self, input: u32) {
        for byte in input.to_le_bytes() {
            self.byte(byte);
        }
    }

    fn u64(&mut self, input: u64) {
        for byte in input.to_le_bytes() {
            self.byte(byte);
        }
    }

    fn bytes(&mut self, input: &[u8]) {
        self.u64(u64::try_from(input.len()).expect("semantic byte length"));
        for byte in input {
            self.byte(*byte);
        }
    }
}

fn digest_descriptor(digest: &mut SemanticDigest, name: &[u8], descriptor: TensorDescriptor) {
    digest.bytes(name);
    digest.u32(descriptor.tensor_type().wire_code());
    digest.u32(u32::try_from(descriptor.dimension_count()).expect("semantic dimension count"));
    let count = usize::min(
        usize::try_from(descriptor.dimension_count()).unwrap_or(0),
        descriptor.dimensions().len(),
    );
    for dimension in &descriptor.dimensions()[..count] {
        digest.u64(u64::try_from(*dimension).expect("semantic dimension"));
    }
    digest.u64(descriptor.data_size());
    digest.byte(u8::from(
        descriptor.binding_status() == TensorBindingStatus::Bound,
    ));
}

fn parity() {
    println!("model-catalog-parity-snapshot/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_data_header_blob={DATA_HEADER_BLOB}");
    println!("source_data_implementation_blob={DATA_IMPLEMENTATION_BLOB}");
    println!("source_generation_header_blob={GENERATION_HEADER_BLOB}");
    println!("source_generation_implementation_blob={GENERATION_IMPLEMENTATION_BLOB}");
    println!(
        "fixture_llama_sha256=8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67"
    );
    println!("fixture_lfm_sha256=855be85429300602eda72958547614703541b7d6dd965a8f8f6052b85a7aa935");
    println!("operand_sha256=4c0c663efcac31d2c927c729d2f88aeb3fc52dc07a55d8f47c6fce9a42fda4b7");
    println!(
        "fixture_config=seven_records,exact_arbitrary_bytes,first_duplicate,empty_name,binding_edges,safe_malformed_range_prevention,reference_raw_range"
    );

    let records = [
        input(b"same"),
        TensorInput::new(b"same", 1, 1, [99, 1, 1, 1], 42, true),
        input(&[0xff, 0, b'x']),
        input(b""),
        TensorInput::new(b"unbound", 0, 1, [1; 4], 1, false),
        TensorInput::new(b"zero-size", 0, 1, [1; 4], 0, true),
        TensorInput::new(b"five-good", 0, 5, [1, 2, 3, 4], 1, true),
    ];
    let mut catalog = sealed_catalog(&records);
    let model = catalog
        .process_event(SealModel::new())
        .expect("seal fixture");
    print_case(&mut catalog, model, "first_duplicate", b"same");
    print_case(&mut catalog, model, "arbitrary_nul", &[0xff, 0, b'x']);
    print_case(&mut catalog, model, "empty", b"");
    print_case(&mut catalog, model, "unbound", b"unbound");
    print_case(&mut catalog, model, "zero_size", b"zero-size");
    print_case(&mut catalog, model, "five_dims", b"five-good");
    print_case(&mut catalog, model, "missing", b"missing");
    let mut guarded = Storage::with_capacity(1, 0, 1).expect("guarded storage");
    println!(
        "actor_case=malformed_range_prevented outcome=error error={}",
        error(guarded.push_tensor(input(b"x")).unwrap_err())
    );

    let description = catalog
        .process_event(DescribeModel::new(model))
        .expect("describe model");
    println!(
        "actor_case=describe outcome=done tensor_count={}",
        description.tensor_count()
    );

    let mut other = sealed_catalog(&[input(b"other")]);
    let other_model = other.process_event(SealModel::new()).expect("other seal");
    println!(
        "actor_case=wrong_identity outcome=error error={}",
        error(
            catalog
                .process_event(DescribeModel::new(other_model))
                .unwrap_err()
        )
    );
    catalog.process_event(Reset::new()).expect("reset");
    catalog
        .process_event(SealModel::new())
        .expect("reseal after reset");
    println!(
        "actor_case=reset outcome=done stale_error={}",
        error(
            catalog
                .process_event(DescribeModel::new(model))
                .unwrap_err()
        )
    );
    catalog
        .process_event(Reset::new())
        .expect("reset before release");
    let released = catalog
        .process_event(ReleaseStorage::new())
        .expect("release storage");
    println!(
        "actor_case=release outcome=done tensor_capacity={} name_capacity={}",
        released.tensor_capacity(),
        released.name_capacity()
    );

    let mut unknown = Catalog::try_new().expect("unknown actor");
    unknown
        .process_event(BindStorage::new(storage(&[TensorInput::new(
            b"x",
            u32::MAX,
            1,
            [1; 4],
            1,
            true,
        )])))
        .expect("unknown bind");
    println!(
        "actor_case=unknown_wire outcome=error error={}",
        error(unknown.process_event(SealModel::new()).unwrap_err())
    );
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    assert!(iterations > 0 && runs > 0);
    let mut names = Vec::with_capacity(256);
    for index in 0..256 {
        names.push(format!("tensor.{index:03}"));
    }
    let inputs: Vec<_> = names.iter().map(|name| input(name.as_bytes())).collect();
    let setup_start = Instant::now();
    let mut catalog = sealed_catalog(&inputs);
    let model = catalog
        .process_event(SealModel::new())
        .expect("benchmark seal");
    let setup_nanoseconds = setup_start.elapsed().as_nanos();
    for _ in 0..warmup {
        black_box(
            catalog
                .process_event(FindTensor::new(model, b"tensor.255"))
                .expect("warmup lookup")
                .expect("warmup tensor"),
        );
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0u64;
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..iterations {
            let descriptor = catalog
                .process_event(FindTensor::new(model, b"tensor.255"))
                .expect("benchmark lookup")
                .expect("benchmark tensor");
            checksum = checksum.wrapping_add(black_box(descriptor.data_size()));
        }
        samples.push(start.elapsed().as_nanos() / u128::from(iterations));
    }
    samples.sort_unstable();
    println!(
        "rust_ns_per_lookup={} setup_ns={} outcome=found checksum={} iter={} runs={}",
        samples[samples.len() / 2],
        setup_nanoseconds,
        checksum,
        iterations,
        runs
    );
}

const fn input(name: &[u8]) -> TensorInput<'_> {
    TensorInput::new(name, 0, 2, [4, 8, 1, 1], 128, true)
}

fn storage(records: &[TensorInput<'_>]) -> Storage {
    let name_bytes = records.iter().map(|record| record.name_len()).sum();
    let mut storage =
        Storage::with_capacity(records.len(), name_bytes, records.len()).expect("fixture storage");
    for record in records {
        storage.push_tensor(*record).expect("fixture record");
    }
    storage
}

fn sealed_catalog(records: &[TensorInput<'_>]) -> Catalog {
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage(records)))
        .expect("bind storage");
    catalog
}

fn print_case(
    catalog: &mut Catalog,
    model: emel_model::catalog::event::ModelIdentity,
    label: &str,
    name: &[u8],
) {
    let descriptor = catalog
        .process_event(FindTensor::new(model, name))
        .expect("find tensor");
    print!(
        "source_case={label} present={} bindable={}",
        u8::from(descriptor.is_some()),
        u8::from(descriptor.is_some_and(|descriptor| {
            descriptor.binding_status() == TensorBindingStatus::Bound
        }))
    );
    if let Some(descriptor) = descriptor {
        let mut canonical = [0u8; 18];
        let canonical_length = catalog
            .process_event(WithTensorName::new(
                descriptor.name_id(),
                |name: &[u8]| hex_into(name, &mut canonical),
            ))
            .expect("canonical name")
            .expect("present name");
        let canonical =
            core::str::from_utf8(&canonical[..canonical_length]).expect("hex output is ASCII");
        print!(
            " type={} dims={} data_size={} name_hex={}",
            descriptor.tensor_type().wire_code(),
            descriptor.dimension_count(),
            descriptor.data_size(),
            canonical
        );
    }
    println!();
}

fn hex_into(bytes: &[u8], output: &mut [u8]) -> usize {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let length = bytes.len().checked_mul(2).expect("bounded fixture name");
    assert!(length <= output.len(), "bounded fixture hex output");
    for (index, byte) in bytes.iter().enumerate() {
        output[index * 2] = DIGITS[usize::from(byte >> 4)];
        output[index * 2 + 1] = DIGITS[usize::from(byte & 0x0f)];
    }
    length
}

const fn error(error: Error) -> &'static str {
    match error {
        Error::ModelInvalid => "model_invalid",
        Error::Capacity => "capacity",
        Error::WrongModelIdentity => "wrong_model_identity",
        Error::StaleModelIdentity => "stale_model_identity",
        _ => "other",
    }
}
