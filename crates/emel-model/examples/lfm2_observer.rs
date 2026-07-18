//! Pinned-source public LFM2 façade parity and equivalent-operand benchmark.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use allocation_counter as _;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe};
use emel_io as _;
use emel_kernels::capability::Resolver;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};
use emel_model::generation::QuantizedStageFamily;
use emel_model::lfm2::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractVisit, PlanBuild,
    StageAudit, TopologyBuild,
};
use emel_model::lfm2::{Lfm2, Parameters, Variant, load_hparams, tensor_type_name};
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_HEADER_BLOB: &str = "af0e7d509a3854a847bdde2dad831c5ed60de379";
const DETAIL_IMPLEMENTATION_BLOB: &str = "a6f215e74ed793ea8f606048c00faa80c37056be";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, option, path] if option == "--fixture" => fixture(path),
        [_, option] if option == "--validation" => validation(),
        [_, option, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        _ => {
            eprintln!(
                "usage: lfm2_observer --fixture LFM2|--validation|--benchmark ITER RUNS WARMUP"
            );
            std::process::exit(2);
        }
    }
}

fn parameters_230m() -> Parameters {
    let mut parameters = Parameters {
        context_length: 128_000,
        embedding_length: 1024,
        embedding_length_out: 1024,
        feed_forward_length: 4096,
        attention_head_count: 16,
        attention_head_count_kv: 8,
        block_count: 14,
        vocab_size: 65_536,
        shortconv_l_cache: 3,
        attention_layer_norm_rms_epsilon: 1e-6,
        rope_freq_base: 1_000_000.0,
        attention_layer_pattern_count: 14,
        ..Parameters::default()
    };
    for index in 2..14 {
        parameters.attention_layer_pattern_flags[index] = u8::from(index % 2 == 0);
    }
    parameters
}

fn parameters_validation() -> Parameters {
    let mut parameters = Parameters {
        context_length: 17,
        embedding_length: 64,
        embedding_length_out: 64,
        feed_forward_length: 128,
        attention_head_count: 4,
        attention_head_count_kv: 2,
        block_count: 3,
        vocab_size: 99,
        shortconv_l_cache: 1,
        attention_layer_norm_rms_epsilon: 1e-5,
        rope_freq_base: 10.0,
        attention_layer_pattern_count: 3,
        ..Parameters::default()
    };
    parameters.attention_layer_pattern_flags[2] = 1;
    parameters
}

fn synthetic_names(parameters: &Parameters) -> Vec<Vec<u8>> {
    let mut names = vec![
        b"token_embd.weight".to_vec(),
        b"token_embd_norm.weight".to_vec(),
    ];
    for index in 0..parameters.block_count {
        for suffix in [
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
            "ffn_up.weight",
        ] {
            names.push(format!("blk.{index}.{suffix}").into_bytes());
        }
        let attention = parameters.attention_layer_pattern_flags
            [usize::try_from(index).expect("bounded block")]
            != 0;
        let selected: &[&str] = if attention {
            &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_output.weight",
            ]
        } else {
            &[
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
            ]
        };
        for suffix in selected {
            names.push(format!("blk.{index}.{suffix}").into_bytes());
        }
    }
    names
}

fn synthetic() -> (
    Catalog,
    emel_model::catalog::event::ModelIdentity,
    Parameters,
) {
    let parameters = parameters_230m();
    let names = synthetic_names(&parameters);
    let name_bytes = names.iter().map(Vec::len).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len())
        .expect("catalog storage");
    for name in &names {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .expect("tensor");
    }
    let (catalog, model) = seal(storage);
    (catalog, model, parameters)
}

fn seal(storage: CatalogStorage) -> (Catalog, emel_model::catalog::event::ModelIdentity) {
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("catalog bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    (catalog, model)
}

fn build(
    catalog: Catalog,
    model: emel_model::catalog::event::ModelIdentity,
    parameters: &Parameters,
) -> Lfm2 {
    let mut actor = Lfm2::new(
        catalog,
        Resolver::new(),
        emel_model::lfm2::event::Storage::with_block_capacity(
            usize::try_from(parameters.block_count).expect("block count"),
        )
        .expect("generation storage"),
    )
    .expect("LFM2 actor");
    actor
        .process_event(ContractBegin::new(b"lfm2", model, parameters))
        .expect("begin");
    for index in 0..parameters.block_count {
        actor.process_event(BlockBuild::new(index)).expect("block");
    }
    actor.process_event(TopologyBuild::new()).expect("topology");
    actor.process_event(PlanBuild::new()).expect("plan");
    for index in 0..parameters.block_count {
        actor
            .process_event(BlockValidation::new(index))
            .expect("validate");
    }
    for index in 0..parameters.block_count {
        actor.process_event(BlockAudit::new(index)).expect("audit");
    }
    for family in QuantizedStageFamily::ALL {
        actor.process_event(StageAudit::new(family)).expect("stage");
    }
    actor
}

const fn variant_name(parameters: &Parameters) -> &'static str {
    match parameters.variant().expect("maintained LFM2 variant") {
        Variant::OnePointTwoB => "1.2b",
        Variant::TwoHundredThirtyM => "230m",
    }
}

fn print_contract(actor: &mut Lfm2, parameters: &Parameters, label: &str) {
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    let block0 = actor.process_event(BlockVisit::new(0)).expect("block 0");
    let block2 = actor.process_event(BlockVisit::new(2)).expect("block 2");
    let mut attention_blocks = 0u32;
    for index in 0..parameters.block_count {
        attention_blocks += u32::from(
            actor
                .process_event(BlockVisit::new(index))
                .expect("block")
                .uses_attention(),
        );
    }
    println!(
        "lfm2_case={label} variant={} block_count={} tensor_count={} workspace={} prefill={} decode={} attention_blocks={} shortconv_blocks={} block0_attention={} block2_attention={}",
        variant_name(parameters),
        contract.block_count(),
        contract.topology().tensor_count(),
        contract.topology().workspace_capacity_bytes(),
        contract.prefill_plan().max_step_tokens(),
        contract.decode_plan().max_step_tokens(),
        attention_blocks,
        u32::try_from(parameters.block_count).expect("block count") - attention_blocks,
        u8::from(block0.uses_attention()),
        u8::from(block2.uses_attention()),
    );
    for stage in contract.audit() {
        let tensor_type = stage.tensor_type().map_or("unknown", tensor_type_name);
        println!(
            "stage={} type={} contract={} consistent={}",
            stage.family().name(),
            tensor_type,
            stage.contract().name(),
            u8::from(stage.consistent_across_layers())
        );
    }
}

fn fixture(path: &str) {
    let bytes = std::fs::read(path).expect("read fixture");
    let mut loader = GgufLoader::new();
    let probe = loader
        .process_event(Probe::new(Arc::from(bytes)))
        .expect("probe");
    loader
        .process_event(Bind::new(
            emel_gguf::event::Storage::exact(probe).expect("storage"),
        ))
        .expect("bind");
    let parsed = loader.process_event(Parse::new()).expect("parse");
    let parameters = load_hparams(&mut loader).expect("LFM2 hparams");
    let storage = CatalogStorage::from_gguf(&mut loader, parsed).expect("catalog storage");
    let (catalog, model) = seal(storage);
    let mut actor = build(catalog, model, &parameters);
    println!("model-lfm2-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_lfm2_detail_header_blob={DETAIL_HEADER_BLOB}");
    println!("source_lfm2_detail_implementation_blob={DETAIL_IMPLEMENTATION_BLOB}");
    let label = match parameters.variant().expect("maintained LFM2 fixture") {
        Variant::OnePointTwoB => "real_fixture_1_2b",
        Variant::TwoHundredThirtyM => "real_fixture_230m",
    };
    print_contract(&mut actor, &parameters, label);
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    let (catalog, model, parameters) = synthetic();
    let mut actor = build(catalog, model, &parameters);
    for iteration in 0..warmup {
        let index = if iteration & 1 == 0 { 0 } else { 2 };
        black_box(actor.process_event(BlockVisit::new(index)).expect("warmup"));
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0i64;
    for _ in 0..runs {
        let start = Instant::now();
        for iteration in 0..iterations {
            let index = if iteration & 1 == 0 { 0 } else { 2 };
            checksum += i64::from(
                black_box(actor.process_event(BlockVisit::new(index)).expect("visit")).index(),
            );
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

fn validation_result(execution: bool) -> bool {
    let parameters = parameters_validation();
    let names = synthetic_names(&parameters);
    let name_bytes = names.iter().map(Vec::len).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len())
        .expect("catalog storage");
    for name in &names {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .expect("tensor");
    }
    let (catalog, model) = seal(storage);
    let mut actor = Lfm2::new(
        catalog,
        Resolver::new(),
        emel_model::lfm2::event::Storage::with_block_capacity(3).expect("generation storage"),
    )
    .expect("LFM2 actor");
    let begin = if execution {
        actor.process_event(ContractBegin::new(b"lfm2", model, &parameters))
    } else {
        actor.process_event(ContractBegin::validation(
            b"lfm2",
            model,
            &parameters,
            parameters.block_count,
        ))
    };
    if begin.is_err() {
        return false;
    }
    for index in 0..parameters.block_count {
        if actor.process_event(BlockBuild::new(index)).is_err() {
            return false;
        }
    }
    true
}

fn validation() {
    let builder_ok = validation_result(false);
    let data_ok = validation_result(false);
    let execution_invalid = !validation_result(true);
    assert!(builder_ok && data_ok && execution_invalid);
    println!("lfm2_validation=non_strict builder=ok data=ok execution=model_invalid block_count=3");
}
