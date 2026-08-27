//! Pinned-source public Qwen3 façade parity and equivalent-work benchmark.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use allocation_counter as _;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe};
use emel_io as _;
use emel_kernels as _;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use emel_model::generation::QuantizedStageFamily;
use emel_model::generation::quantized_path::Resolver;
use emel_model::qwen3::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractVisit, PlanBuild,
    StageAudit, TopologyBuild,
};
use emel_model::qwen3::{Parameters, Qwen3, load_hparams, tensor_type_name};
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_HEADER_BLOB: &str = "6e31b097184042bfe1ec9486af060214f1d01b2b";
const DETAIL_IMPLEMENTATION_BLOB: &str = "a43d8fee3ed962d84447360e7601c999468bf6f8";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, option, path] if option == "--fixture" => fixture(path),
        [_, option, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        _ => {
            eprintln!("usage: qwen3_observer --fixture QWEN3|--benchmark ITER RUNS WARMUP");
            std::process::exit(2);
        }
    }
}

const fn parameters(block_count: i32) -> Parameters {
    Parameters {
        context_length: 4096,
        embedding_length: 128,
        embedding_length_out: 128,
        feed_forward_length: 512,
        attention_head_count: 8,
        attention_head_count_kv: 4,
        attention_key_length: 32,
        attention_value_length: 40,
        rope_dimension_count: 32,
        block_count,
        attention_layer_norm_rms_epsilon: 1e-6,
        rope_freq_base: 10_000.0,
        tie_word_embeddings: true,
        rope_pair_x0_stride: 1,
        rope_pair_x1_stride: 1,
        rope_pair_x1_offset: 0,
        rope_pair_x1_half_rot_offset: 1,
    }
}

fn synthetic_names(block_count: i32) -> Vec<Vec<u8>> {
    let mut names = vec![
        b"token_embd.weight".to_vec(),
        b"output_norm.weight".to_vec(),
        b"output.weight".to_vec(),
    ];
    for index in 0..block_count {
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
            "ffn_up.weight",
        ] {
            names.push(format!("blk.{index}.{suffix}").into_bytes());
        }
    }
    names
}

fn synthetic(block_count: i32) -> (Catalog, ModelIdentity) {
    let names = synthetic_names(block_count);
    let name_bytes = names.iter().map(Vec::len).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len())
        .expect("catalog storage");
    for name in &names {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .expect("tensor");
    }
    seal(storage)
}

fn seal(storage: CatalogStorage) -> (Catalog, ModelIdentity) {
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("catalog bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    (catalog, model)
}

fn build(catalog: Catalog, model: ModelIdentity, parameters: Parameters) -> Qwen3 {
    let mut actor = Qwen3::new(
        catalog,
        Resolver::new(),
        emel_model::qwen3::event::Storage::with_block_capacity(
            usize::try_from(parameters.block_count).expect("block count"),
        )
        .expect("generation storage"),
    )
    .expect("Qwen3 actor");
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters))
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

fn print_contract(actor: &mut Qwen3, label: &str) {
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    let block = actor.process_event(BlockVisit::new(0)).expect("block");
    println!(
        "qwen3_case={label} block_count={} tensor_count={} workspace={} prefill={} decode={} uses_attention={} qk_norm_headwise_rms={} key_length={} value_length={} rope_dim={}",
        contract.block_count(),
        contract.topology().tensor_count(),
        contract.topology().workspace_capacity_bytes(),
        contract.prefill_plan().max_step_tokens(),
        contract.decode_plan().max_step_tokens(),
        u8::from(block.uses_attention()),
        u8::from(
            block.layer().qk_norm_route() == emel_model::qwen3::AttentionQkNormRoute::HeadwiseRms
        ),
        block.layer().attention_key_length(),
        block.layer().attention_value_length(),
        block.layer().attention_rope_dim(),
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
    let parameters = load_hparams(&mut loader).expect("Qwen3 hparams");
    let storage = CatalogStorage::from_gguf(&mut loader, parsed).expect("catalog storage");
    let (catalog, model) = seal(storage);
    let mut actor = build(catalog, model, parameters);
    println!("model-qwen3-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_qwen3_detail_header_blob={DETAIL_HEADER_BLOB}");
    println!("source_qwen3_detail_implementation_blob={DETAIL_IMPLEMENTATION_BLOB}");
    print_contract(&mut actor, "real_fixture_0_6b");
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    let (catalog, model) = synthetic(2);
    let mut actor = build(catalog, model, parameters(2));
    for iteration in 0..warmup {
        let index = i32::try_from(iteration & 1).expect("benchmark index");
        black_box(actor.process_event(BlockVisit::new(index)).expect("warmup"));
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0i64;
    for _ in 0..runs {
        let start = Instant::now();
        for iteration in 0..iterations {
            let index = i32::try_from(iteration & 1).expect("benchmark index");
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
