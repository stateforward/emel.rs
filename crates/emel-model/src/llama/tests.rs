use allocation_counter::measure;
use emel_kernels::capability::Resolver;

use crate::catalog::Catalog;
use crate::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};
use crate::generation::{AttentionQkNormRoute, AttentionValueRoute, QuantizedStageFamily};

use super::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractReset,
    ContractVisit, Error, HPARAM_KEYS, PlanBuild, StageAudit, StorageRelease, TopologyBuild,
};
use super::{Llama, Parameters};

const TENSORS: [&[u8]; 12] = [
    b"token_embd.weight",
    b"output_norm.weight",
    b"output.weight",
    b"blk.0.attn_norm.weight",
    b"blk.0.attn_q.weight",
    b"blk.0.attn_k.weight",
    b"blk.0.attn_v.weight",
    b"blk.0.attn_output.weight",
    b"blk.0.ffn_norm.weight",
    b"blk.0.ffn_gate.weight",
    b"blk.0.ffn_down.weight",
    b"blk.0.ffn_up.weight",
];

fn parameters() -> Parameters {
    Parameters {
        context_length: 128,
        embedding_length: 64,
        embedding_length_out: 64,
        feed_forward_length: 256,
        attention_head_count: 8,
        attention_head_count_kv: 4,
        rope_dimension_count: 16,
        block_count: 1,
        vocab_size: 32000,
        attention_layer_norm_epsilon: 0.0,
        attention_layer_norm_rms_epsilon: 0.00001,
        attention_clamp_kqv: 0.0,
        attn_logit_softcapping: 0.0,
        final_logit_softcapping: 0.0,
        residual_scale: 0.0,
        embedding_scale: 0.0,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 0.0,
        attention_key_length: 16,
        attention_value_length: 16,
    }
}

fn actor_with(names: &[&[u8]]) -> (Llama, crate::catalog::event::ModelIdentity) {
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut catalog_storage =
        CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
    for (index, name) in names.iter().enumerate() {
        let wire_type = if index == 0 {
            10
        } else if matches!(index, 1 | 3 | 8) {
            0
        } else {
            14
        };
        let (dimensions, data_size) = match wire_type {
            0 => ([1, 1, 1, 1], 4),
            10 => ([256, 1, 1, 1], 84),
            _ => ([256, 1, 1, 1], 210),
        };
        catalog_storage
            .push_tensor(TensorInput::new(
                name, wire_type, 1, dimensions, data_size, true,
            ))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog
        .process_event(BindStorage::new(catalog_storage))
        .unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    let actor = Llama::new(
        catalog,
        Resolver::new(),
        super::event::Storage::with_block_capacity(1).unwrap(),
    )
    .unwrap();
    (actor, model)
}

fn actor() -> (Llama, crate::catalog::event::ModelIdentity) {
    actor_with(&TENSORS)
}

fn complete(actor: &mut Llama, model: crate::catalog::event::ModelIdentity) {
    actor
        .process_event(ContractBegin::new(b"llama", model, parameters()))
        .unwrap();
    actor.process_event(BlockBuild::new(0)).unwrap();
    actor.process_event(TopologyBuild::new()).unwrap();
    actor.process_event(PlanBuild::new()).unwrap();
    actor.process_event(BlockValidation::new(0)).unwrap();
    actor.process_event(BlockAudit::new(0)).unwrap();
    for family in QuantizedStageFamily::ALL {
        actor.process_event(StageAudit::new(family)).unwrap();
    }
}

#[test]
fn builds_source_exact_llama_contract_and_facade_views() {
    let (mut actor, model) = actor();
    complete(&mut actor, model);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.model(), model);
    assert_eq!(contract.block_count(), 1);
    assert_eq!(contract.topology().tensor_count(), 11);
    assert_eq!(contract.topology().node_count(), 11);
    assert_eq!(contract.topology().bytes_per_tensor(), 4);
    assert_eq!(contract.topology().workspace_capacity_bytes(), 11 * 64 * 4);
    let block = actor.process_event(BlockVisit::new(0)).unwrap();
    assert!(block.uses_attention());
    assert_eq!(block.layer().qk_norm_route(), AttentionQkNormRoute::None);
    assert_eq!(
        block.layer().value_route(),
        AttentionValueRoute::DedicatedValue
    );
    assert_eq!(block.layer().attention_key_length(), 16);
    assert_eq!(block.layer().attention_value_length(), 16);
    assert_eq!(block.layer().attention_rope_dim(), 16);
    assert_eq!(
        block.layer().attention_rope_freq_base().to_bits(),
        10_000.0f32.to_bits()
    );
}

#[test]
fn rejects_wrong_architecture_invalid_parameters_and_missing_required_tensor() {
    let (mut actor, model) = actor();
    assert_eq!(
        actor.process_event(ContractBegin::new(b"lfm2", model, parameters())),
        Err(Error::InvalidRequest)
    );
    let mut invalid = parameters();
    invalid.embedding_length = 0;
    assert_eq!(
        actor.process_event(ContractBegin::new(b"llama", model, invalid)),
        Err(Error::ModelInvalid)
    );

    let names = &TENSORS[..11];
    let (mut missing, missing_model) = actor_with(names);
    missing
        .process_event(ContractBegin::new(b"llama", missing_model, parameters()))
        .unwrap();
    assert_eq!(
        missing.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
    missing.process_event(ContractReset::new()).unwrap();
    missing
        .process_event(ContractBegin::new(b"llama", missing_model, parameters()))
        .unwrap();
    assert_eq!(
        missing.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn one_block_per_dispatch_and_lifecycle_are_explicit() {
    let (mut actor, model) = actor();
    actor
        .process_event(ContractBegin::new(b"llama", model, parameters()))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(-1)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(BlockBuild::new(1)),
        Err(Error::InvalidRequest)
    );
    actor.process_event(BlockBuild::new(0)).unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::InvalidRequest)
    );
    actor.process_event(ContractReset::new()).unwrap();
    let storage = actor.process_event(StorageRelease::new()).unwrap();
    assert_eq!(storage.block_capacity(), 1);
}

#[test]
fn complete_dispatch_path_is_allocation_free() {
    let (mut actor, model) = actor();
    let measured = measure(|| complete(&mut actor, model));
    assert_eq!(measured.count_total, 0);
}

#[test]
fn hparam_inventory_is_source_exact() {
    assert_eq!(HPARAM_KEYS.len(), 18);
    assert_eq!(HPARAM_KEYS[0], "llama.context_length");
    assert_eq!(HPARAM_KEYS[17], "llama.rope.freq_base_swa");
    assert_eq!(
        parameters().to_string(),
        "Llama 1 layers, embedding 64, context 128"
    );
}

#[test]
fn loads_source_exact_hparam_values_from_typed_gguf_queries() {
    use crate::loader::test_gguf::{F32, F64, Fixture, U32};

    let mut loader = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"llama.context_length", U32, 4096_u32.to_le_bytes())
            .scalar(b"llama.embedding_length", U32, 256_u32.to_le_bytes())
            .scalar(b"llama.embedding_length_out", U32, 320_u32.to_le_bytes())
            .scalar(b"llama.feed_forward_length", U32, 768_u32.to_le_bytes())
            .scalar(b"llama.attention.head_count", U32, 8_u32.to_le_bytes())
            .scalar(b"llama.attention.head_count_kv", U32, 2_u32.to_le_bytes())
            .scalar(b"llama.rope.dimension_count", U32, 32_u32.to_le_bytes())
            .scalar(b"llama.block_count", U32, 12_u32.to_le_bytes())
            .scalar(b"llama.vocab_size", U32, 32_000_u32.to_le_bytes())
            .scalar(
                b"llama.attention.layer_norm_epsilon",
                F32,
                1e-5_f32.to_le_bytes(),
            )
            .scalar(
                b"llama.attention.layer_norm_rms_epsilon",
                F64,
                1e-6_f64.to_le_bytes(),
            )
            .scalar(b"llama.attention.clamp_kqv", F32, 4.0_f32.to_le_bytes())
            .scalar(b"llama.attn_logit_softcapping", F32, 12.0_f32.to_le_bytes())
            .scalar(b"llama.final_logit_softcapping", F32, 8.0_f32.to_le_bytes())
            .scalar(b"llama.residual_scale", F32, 0.5_f32.to_le_bytes())
            .scalar(b"llama.embedding_scale", F32, 1.5_f32.to_le_bytes())
            .scalar(b"llama.rope.freq_base", F32, 10_000.0_f32.to_le_bytes())
            .scalar(
                b"llama.rope.freq_base_swa",
                F64,
                500_000.0_f64.to_le_bytes(),
            )
            .build(),
    );
    let decoded = super::load_hparams(&mut loader).unwrap();
    let expected = Parameters {
        context_length: 4096,
        embedding_length: 256,
        embedding_length_out: 320,
        feed_forward_length: 768,
        attention_head_count: 8,
        attention_head_count_kv: 2,
        rope_dimension_count: 32,
        block_count: 12,
        vocab_size: 32_000,
        attention_layer_norm_epsilon: 1e-5,
        attention_layer_norm_rms_epsilon: 1e-6,
        attention_clamp_kqv: 4.0,
        attn_logit_softcapping: 12.0,
        final_logit_softcapping: 8.0,
        residual_scale: 0.5,
        embedding_scale: 1.5,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 500_000.0,
        attention_key_length: 0,
        attention_value_length: 0,
    };
    assert_eq!(decoded, expected);

    let mut wrong = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"llama.context_length", F32, 1.0_f32.to_le_bytes())
            .build(),
    );
    assert!(super::load_hparams(&mut wrong).is_err());
}
