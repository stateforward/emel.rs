//! Pinned-source public Gemma4 façade parity and equivalent-work benchmark.

use std::hint::black_box;
use std::time::Instant;

use allocation_counter as _;
use emel_gguf as _;
use emel_io as _;
use emel_kernels::capability::Resolver;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use emel_model::gemma4::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractVisit, PlanBuild,
    StageAudit, TopologyBuild,
};
use emel_model::gemma4::{Gemma4, Parameters, tensor_type_name};
use emel_model::generation::QuantizedStageFamily;
use emel_model::generation::event::BlockTensorSlot;
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_HEADER_BLOB: &str = "5ee94dcc8554ef8e9de0a51007acae6bb58565c9";
const DETAIL_IMPLEMENTATION_BLOB: &str = "ac9cc9757a5d6e97ebfbd1ef905d01f4af34dee4";

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    match arguments.as_slice() {
        [_, option] if option == "--source-built" => observe(),
        [_, option, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        _ => {
            eprintln!("usage: gemma4_observer --source-built|--benchmark ITER RUNS WARMUP");
            std::process::exit(2);
        }
    }
}

const fn parameters() -> Parameters {
    Parameters::canonical()
}

fn inputs() -> Vec<(Vec<u8>, u32)> {
    let mut tensors = vec![
        (b"token_embd.weight".to_vec(), 10),
        (b"output_norm.weight".to_vec(), 0),
    ];
    for index in 0..35 {
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
            if index >= 15 && suffix == "attn_v.weight" {
                continue;
            }
            let wire_type = if matches!(suffix, "attn_k.weight" | "attn_v.weight") {
                14
            } else {
                0
            };
            tensors.push((format!("blk.{index}.{suffix}").into_bytes(), wire_type));
        }
    }
    tensors
}

fn catalog() -> (Catalog, ModelIdentity) {
    let tensors = inputs();
    let name_bytes = tensors.iter().map(|(name, _)| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(tensors.len(), name_bytes, tensors.len()).expect("catalog");
    for (name, wire_type) in &tensors {
        let (dimensions, data_size) = match wire_type {
            0 => ([1, 1, 1, 1], 4),
            10 => ([256, 1, 1, 1], 84),
            14 => ([256, 1, 1, 1], 210),
            _ => unreachable!("source fixture type"),
        };
        storage
            .push_tensor(TensorInput::new(
                name, *wire_type, 1, dimensions, data_size, true,
            ))
            .expect("tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog actor");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind catalog");
    let model = catalog.process_event(SealModel::new()).expect("seal model");
    (catalog, model)
}

fn build() -> Gemma4 {
    let parameters = parameters();
    let (catalog, model) = catalog();
    let mut actor = Gemma4::new(
        catalog,
        Resolver::new(),
        emel_model::gemma4::event::Storage::with_block_capacity(35).expect("generation storage"),
    )
    .expect("Gemma4 actor");
    actor
        .process_event(ContractBegin::new(b"gemma4", model, &parameters))
        .expect("begin");
    for index in 0..35 {
        actor.process_event(BlockBuild::new(index)).expect("block");
    }
    actor.process_event(TopologyBuild::new()).expect("topology");
    actor.process_event(PlanBuild::new()).expect("plan");
    for index in 0..35 {
        actor
            .process_event(BlockValidation::new(index))
            .expect("validation");
    }
    for index in 0..35 {
        actor.process_event(BlockAudit::new(index)).expect("audit");
    }
    for family in QuantizedStageFamily::ALL {
        actor.process_event(StageAudit::new(family)).expect("stage");
    }
    actor
}

fn print_layer(actor: &mut Gemma4, index: i32) {
    let block = actor
        .process_event(BlockVisit::new(index))
        .expect("block view");
    let layer = block.layer();
    println!(
        "block={index} window={} value={} v_norm={} qk_norm={} key={} val={} rope={} freq={:.0} shared_alias={}",
        layer.window_route() as u8,
        layer.value_route() as u8,
        layer.v_norm_route() as u8,
        layer.qk_norm_route() as u8,
        layer.attention_key_length(),
        layer.attention_value_length(),
        layer.attention_rope_dim(),
        layer.attention_rope_freq_base(),
        u8::from(
            block.tensor(BlockTensorSlot::AttentionV) == block.tensor(BlockTensorSlot::AttentionK)
        )
    );
}

fn observe() {
    let mut actor = build();
    let contract = actor.process_event(ContractVisit::new()).expect("contract");
    println!("model-gemma4-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_gemma4_detail_header_blob={DETAIL_HEADER_BLOB}");
    println!("source_gemma4_detail_implementation_blob={DETAIL_IMPLEMENTATION_BLOB}");
    println!("scope=pinned_source_built_model_no_real_fixture");
    println!(
        "contract blocks={} tensors={} nodes={} workspace={} prefill={} decode={} tied={}",
        contract.block_count(),
        contract.topology().tensor_count(),
        contract.topology().node_count(),
        contract.topology().workspace_capacity_bytes(),
        contract.prefill_plan().max_step_tokens(),
        contract.decode_plan().max_step_tokens(),
        u8::from(contract.output() == contract.token_embedding())
    );
    for index in [0, 4, 14, 15] {
        print_layer(&mut actor, index);
    }
    for stage in contract.audit() {
        println!(
            "stage={} type={} contract={} consistent={}",
            stage.family().name(),
            stage.tensor_type().map_or("unknown", tensor_type_name),
            stage.contract().name(),
            u8::from(stage.consistent_across_layers())
        );
    }
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    let mut actor = build();
    for iteration in 0..warmup {
        let index = 14 + i32::try_from(iteration & 1).expect("index");
        black_box(actor.process_event(BlockVisit::new(index)).expect("warmup"));
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0_i64;
    for _ in 0..runs {
        let start = Instant::now();
        for iteration in 0..iterations {
            let index = 14 + i32::try_from(iteration & 1).expect("index");
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
