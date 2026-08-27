//! Pinned-source public Llama façade parity and equivalent-operand benchmark.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use allocation_counter as _;
use emel_gguf::Loader as GgufLoader;
use emel_gguf::event::{Bind, Parse, Probe};
use emel_io as _;
use emel_kernels as _;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};
use emel_model::generation::QuantizedStageFamily;
use emel_model::generation::quantized_path::Resolver;
use emel_model::llama::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractVisit, PlanBuild,
    StageAudit, TopologyBuild,
};
use emel_model::llama::{Llama, Parameters, load_hparams, tensor_type_name};
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const TENSORS: [&[u8]; 23] = [
    b"token_embd.weight",
    b"output_norm.weight",
    b"output.weight",
    b"blk.0.attn_norm.weight",
    b"blk.0.attn_q.weight",
    b"blk.0.attn_k.weight",
    b"blk.0.attn_v.weight",
    b"blk.0.attn_q_norm.weight",
    b"blk.0.attn_k_norm.weight",
    b"blk.0.attn_output.weight",
    b"blk.0.ffn_norm.weight",
    b"blk.0.ffn_gate.weight",
    b"blk.0.ffn_down.weight",
    b"blk.0.ffn_up.weight",
    b"blk.1.attn_norm.weight",
    b"blk.1.attn_q.weight",
    b"blk.1.attn_k.weight",
    b"blk.1.attn_v.weight",
    b"blk.1.attn_output.weight",
    b"blk.1.ffn_norm.weight",
    b"blk.1.ffn_gate.weight",
    b"blk.1.ffn_down.weight",
    b"blk.1.ffn_up.weight",
];

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
            eprintln!("usage: llama_observer --fixture LLAMA|--benchmark ITER RUNS WARMUP");
            std::process::exit(2);
        }
    }
}

const fn parameters(block_count: i32) -> Parameters {
    Parameters {
        context_length: 4096,
        embedding_length: 128,
        embedding_length_out: 128,
        feed_forward_length: 0,
        attention_head_count: 8,
        attention_head_count_kv: 4,
        rope_dimension_count: 32,
        block_count,
        vocab_size: 0,
        attention_layer_norm_epsilon: 0.0,
        attention_layer_norm_rms_epsilon: 0.0,
        attention_clamp_kqv: 0.0,
        attn_logit_softcapping: 0.0,
        final_logit_softcapping: 0.0,
        residual_scale: 0.0,
        embedding_scale: 0.0,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 0.0,
        attention_key_length: 32,
        attention_value_length: 32,
    }
}

fn synthetic() -> (Catalog, emel_model::catalog::event::ModelIdentity) {
    let name_bytes = TENSORS.iter().map(|name| name.len()).sum();
    let mut storage = CatalogStorage::with_capacity(TENSORS.len(), name_bytes, TENSORS.len())
        .expect("catalog storage");
    for (index, name) in TENSORS.iter().enumerate() {
        let wire_type = if matches!(index, 1 | 3 | 7 | 8 | 10) {
            0
        } else {
            14
        };
        storage
            .push_tensor(TensorInput::new(
                name,
                wire_type,
                1,
                [256, 1, 1, 1],
                210,
                true,
            ))
            .expect("tensor");
    }
    seal(storage)
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
    parameters: Parameters,
) -> Llama {
    let mut actor = Llama::new(
        catalog,
        Resolver::new(),
        emel_model::llama::event::Storage::with_block_capacity(
            usize::try_from(parameters.block_count).expect("block count"),
        )
        .expect("generation storage"),
    )
    .expect("Llama actor");
    actor
        .process_event(ContractBegin::new(b"llama", model, parameters))
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

fn print_contract(actor: &mut Llama, label: &str) {
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    let block = actor.process_event(BlockVisit::new(0)).expect("block");
    println!(
        "llama_case={label} block_count={} tensor_count={} workspace={} prefill={} decode={} uses_attention={}",
        contract.block_count(),
        contract.topology().tensor_count(),
        contract.topology().workspace_capacity_bytes(),
        contract.prefill_plan().max_step_tokens(),
        contract.decode_plan().max_step_tokens(),
        u8::from(block.uses_attention())
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
    let parameters = load_hparams(&mut loader).expect("Llama hparams");
    let storage = CatalogStorage::from_gguf(&mut loader, parsed).expect("catalog storage");
    let (catalog, model) = seal(storage);
    let mut actor = build(catalog, model, parameters);
    println!("model-llama-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_llama_any_blob=430ba6a8897854b5f58660d3f4d51d006727559e");
    println!("source_llama_detail_header_blob=91f96245d29a22aa83c570cb4fe8bc8cf9e1d081");
    println!("source_llama_detail_implementation_blob=c78e3656d41772dcc4dba5c46623d9f2fec784ed");
    print_contract(&mut actor, "real_fixture");
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    let (catalog, model) = synthetic();
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
