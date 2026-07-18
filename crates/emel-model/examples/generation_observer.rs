//! Pinned-source parity and equivalent-operand benchmark for the common generation actor.

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
use emel_model::generation::event::{
    AttentionBlock, BlockAudit, BlockValidation, BlockVisit, ContractBegin, ContractVisit,
    GlobalBindings, Plan, StageAudit, Storage, StorageBind, Topology,
};
use emel_model::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute, Builder,
    LayerExecution, QuantizedStageFamily, ResidualRoute, tensor_type_name,
};
use emel_tensor as _;
use emel_token as _;
use sml as _;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const HEADER_BLOB: &str = "d521cf68e1bf52a2a193bbdb460741772199b318";
const IMPLEMENTATION_BLOB: &str = "099058ccd441d1dc6bebbb0c4994070d2f533c47";
const TENSORS: [&[u8]; 14] = [
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
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_] => parity(),
        [_, option, path] if option == "--fixture" => fixture(path),
        [_, option, iterations, runs, warmup] if option == "--benchmark" => benchmark(
            iterations.parse().expect("iterations"),
            runs.parse().expect("runs"),
            warmup.parse().expect("warmup"),
        ),
        _ => {
            eprintln!("usage: generation_observer [--fixture LLAMA|--benchmark ITER RUNS WARMUP]");
            std::process::exit(2);
        }
    }
}

const fn layer(qk_norm: AttentionQkNormRoute) -> LayerExecution {
    LayerExecution::new(
        ResidualRoute::Attention,
        qk_norm,
        AttentionValueRoute::DedicatedValue,
        AttentionVNormRoute::None,
        AttentionWindowRoute::FullContext,
        128,
        128,
        128,
        10_000.0,
    )
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
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    (catalog, model)
}

fn build(
    mut builder: Builder,
    model: emel_model::catalog::event::ModelIdentity,
    block_count: i32,
    qk_norm: AttentionQkNormRoute,
) -> Builder {
    builder
        .process_event(StorageBind::new(
            Storage::with_block_capacity(usize::try_from(block_count).expect("block count"))
                .expect("generation storage"),
        ))
        .expect("bind");
    builder
        .process_event(ContractBegin::new(model, block_count, 4096))
        .expect("begin");
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .expect("globals");
    for index in 0..block_count {
        builder
            .process_event(AttentionBlock::new(index, layer(qk_norm)))
            .expect("block");
    }
    builder
        .process_event(Topology::new(13, 13, 4, 13 * 128 * 4))
        .expect("topology");
    builder.process_event(Plan::new()).expect("plan");
    for index in 0..block_count {
        builder
            .process_event(BlockValidation::new(index))
            .expect("validate");
    }
    for index in 0..block_count {
        builder
            .process_event(BlockAudit::new(index))
            .expect("audit");
    }
    for family in QuantizedStageFamily::ALL {
        builder
            .process_event(StageAudit::new(family))
            .expect("stage");
    }
    builder
}

fn print_contract(builder: &mut Builder, label: &str) {
    let contract = builder
        .process_event(ContractVisit::new())
        .expect("contract");
    let block = builder.process_event(BlockVisit::new(0)).expect("block");
    println!(
        "generation_case={label} block_count={} prefill={} decode={} uses_attention={}",
        contract.block_count(),
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

fn parity() {
    println!("model-generation-parity-snapshot/v1");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_generation_header_blob={HEADER_BLOB}");
    println!("source_generation_implementation_blob={IMPLEMENTATION_BLOB}");
    let (catalog, model) = synthetic();
    let mut builder = build(
        Builder::new(catalog, Resolver::new()),
        model,
        1,
        AttentionQkNormRoute::HeadwiseRms,
    );
    print_contract(&mut builder, "attention_qk");
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
    let storage = CatalogStorage::from_gguf(&mut loader, parsed).expect("catalog storage");
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("catalog bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    let mut builder = build(
        Builder::new(catalog, Resolver::new()),
        model,
        2,
        AttentionQkNormRoute::None,
    );
    print_contract(&mut builder, "llama_real_fixture");
}

fn benchmark(iterations: u64, runs: usize, warmup: u64) {
    let (catalog, model) = synthetic();
    let mut builder = build(
        Builder::new(catalog, Resolver::new()),
        model,
        1,
        AttentionQkNormRoute::HeadwiseRms,
    );
    for _ in 0..warmup {
        black_box(builder.process_event(BlockVisit::new(0)).expect("warmup"));
    }
    let mut samples = Vec::with_capacity(runs);
    let mut checksum = 0i64;
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..iterations {
            checksum += i64::from(
                black_box(builder.process_event(BlockVisit::new(0)).expect("visit")).index(),
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
