use crate::generation::quantized_path::Resolver;
use allocation_counter::measure;

use crate::catalog::Catalog;
use crate::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    QuantizedStageFamily,
};

use super::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractReset,
    ContractVisit, Error, HPARAM_KEYS, PlanBuild, StageAudit, StorageRelease, TopologyBuild,
};
use super::{Parameters, Qwen3};

const TENSORS: [&[u8]; 13] = [
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
];

const TENSORS_WITH_UP: [&[u8]; 14] = [
    TENSORS[0],
    TENSORS[1],
    TENSORS[2],
    TENSORS[3],
    TENSORS[4],
    TENSORS[5],
    TENSORS[6],
    TENSORS[7],
    TENSORS[8],
    TENSORS[9],
    TENSORS[10],
    TENSORS[11],
    TENSORS[12],
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
        attention_key_length: 16,
        attention_value_length: 20,
        rope_dimension_count: 16,
        block_count: 1,
        attention_layer_norm_rms_epsilon: 0.00001,
        rope_freq_base: 10_000.0,
        tie_word_embeddings: true,
        rope_pair_x0_stride: 1,
        rope_pair_x1_stride: 1,
        rope_pair_x1_offset: 0,
        rope_pair_x1_half_rot_offset: 1,
    }
}

fn actor_with(names: &[&[u8]]) -> (Qwen3, ModelIdentity) {
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
    for name in names {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    let actor = Qwen3::new(
        catalog,
        Resolver::new(),
        super::event::Storage::with_block_capacity(1).unwrap(),
    )
    .unwrap();
    (actor, model)
}

fn actor() -> (Qwen3, ModelIdentity) {
    actor_with(&TENSORS_WITH_UP)
}

fn complete(actor: &mut Qwen3, model: ModelIdentity) {
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
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
fn builds_source_exact_qwen3_contract_and_facade_views() {
    let (mut actor, model) = actor();
    complete(&mut actor, model);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.model(), model);
    assert_eq!(contract.block_count(), 1);
    assert_eq!(contract.topology().tensor_count(), 13);
    assert_eq!(contract.topology().node_count(), 13);
    assert_eq!(contract.topology().bytes_per_tensor(), 4);
    assert_eq!(contract.topology().workspace_capacity_bytes(), 13 * 64 * 4);
    assert_eq!(contract.prefill_plan().max_step_tokens(), 128);
    assert_eq!(contract.decode_plan().max_step_tokens(), 1);
    let block = actor.process_event(BlockVisit::new(0)).unwrap();
    assert!(block.uses_attention());
    assert_eq!(
        block.layer().qk_norm_route(),
        AttentionQkNormRoute::HeadwiseRms
    );
    assert_eq!(
        block.layer().value_route(),
        AttentionValueRoute::DedicatedValue
    );
    assert_eq!(block.layer().v_norm_route(), AttentionVNormRoute::None);
    assert_eq!(
        block.layer().window_route(),
        AttentionWindowRoute::FullContext
    );
    assert_eq!(block.layer().attention_key_length(), 16);
    assert_eq!(block.layer().attention_value_length(), 20);
    assert_eq!(block.layer().attention_rope_dim(), 16);
    assert_eq!(
        block.layer().attention_rope_freq_base().to_bits(),
        10_000.0f32.to_bits()
    );
}

#[test]
fn tied_output_falls_back_to_token_embedding() {
    let names: Vec<&[u8]> = TENSORS_WITH_UP
        .iter()
        .copied()
        .filter(|name| *name != b"output.weight")
        .collect();
    let (mut actor, model) = actor_with(&names);
    complete(&mut actor, model);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.output(), contract.token_embedding());
}

#[test]
fn missing_attention_q_norm_is_model_invalid_and_rolls_back() {
    let names: Vec<&[u8]> = TENSORS_WITH_UP
        .iter()
        .copied()
        .filter(|name| *name != b"blk.0.attn_q_norm.weight")
        .collect();
    let (mut actor, model) = actor_with(&names);
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
    actor.process_event(ContractReset::new()).unwrap();
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn missing_attention_k_norm_is_model_invalid_and_rolls_back() {
    let names: Vec<&[u8]> = TENSORS_WITH_UP
        .iter()
        .copied()
        .filter(|name| *name != b"blk.0.attn_k_norm.weight")
        .collect();
    let (mut actor, model) = actor_with(&names);
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn rejects_wrong_architecture_invalid_geometry_and_missing_global_tensor() {
    let (mut actor, model) = actor();
    assert_eq!(
        actor.process_event(ContractBegin::new(b"llama", model, parameters())),
        Err(Error::InvalidRequest)
    );
    let mut invalid = parameters();
    invalid.embedding_length = 0;
    assert_eq!(
        actor.process_event(ContractBegin::new(b"qwen3", model, invalid)),
        Err(Error::ModelInvalid)
    );

    let names: Vec<&[u8]> = TENSORS_WITH_UP
        .iter()
        .copied()
        .filter(|name| *name != b"output_norm.weight")
        .collect();
    let (mut missing, missing_model) = actor_with(&names);
    assert_eq!(
        missing.process_event(ContractBegin::new(b"qwen3", missing_model, parameters())),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn one_block_per_dispatch_and_lifecycle_are_explicit() {
    let (mut actor, model) = actor();
    actor
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
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
    assert!(matches!(
        actor.process_event(StorageRelease::new()),
        Err(Error::Busy)
    ));
    actor.process_event(ContractReset::new()).unwrap();
    let storage = actor.process_event(StorageRelease::new()).unwrap();
    assert_eq!(storage.block_capacity(), 1);
}

#[test]
fn incomplete_and_duplicate_lifecycle_routes_are_typed_and_recoverable() {
    let (mut bound, _) = actor();
    assert_eq!(
        bound.process_event(BlockBuild::new(0)),
        Err(Error::UnexpectedEvent)
    );
    assert_eq!(format!("{bound:?}"), "Qwen3 { .. }");
    bound.process_event(ContractReset::new()).unwrap();

    let (mut begin_capacity, model) = actor();
    let mut too_many_blocks = parameters();
    too_many_blocks.block_count = 2;
    assert_eq!(
        begin_capacity.process_event(ContractBegin::new(b"qwen3", model, too_many_blocks)),
        Err(Error::InvalidRequest)
    );

    let (mut busy, model) = actor();
    busy.process_event(ContractBegin::new(b"qwen3", model, parameters()))
        .unwrap();
    assert_eq!(
        busy.process_event(ContractBegin::new(b"qwen3", model, parameters())),
        Err(Error::Busy)
    );
    assert_eq!(
        busy.process_event(PlanBuild::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        busy.process_event(BlockValidation::new(0)),
        Err(Error::ModelInvalid)
    );
    assert_eq!(
        busy.process_event(BlockAudit::new(0)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        busy.process_event(StageAudit::new(QuantizedStageFamily::Output)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        busy.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    assert!(matches!(
        busy.process_event(BlockVisit::new(-1)),
        Err(Error::InvalidRequest)
    ));

    let (mut duplicate, model) = actor();
    duplicate
        .process_event(ContractBegin::new(b"qwen3", model, parameters()))
        .unwrap();
    duplicate.process_event(BlockBuild::new(0)).unwrap();
    duplicate.process_event(TopologyBuild::new()).unwrap();
    assert_eq!(
        duplicate.process_event(TopologyBuild::new()),
        Err(Error::InvalidRequest)
    );
    duplicate.process_event(PlanBuild::new()).unwrap();
    assert_eq!(
        duplicate.process_event(PlanBuild::new()),
        Err(Error::InvalidRequest)
    );
    duplicate.process_event(BlockValidation::new(0)).unwrap();
    assert_eq!(
        duplicate.process_event(BlockValidation::new(0)),
        Err(Error::ModelInvalid)
    );
    duplicate.process_event(BlockAudit::new(0)).unwrap();
    assert_eq!(
        duplicate.process_event(BlockAudit::new(0)),
        Err(Error::InvalidRequest)
    );
    duplicate
        .process_event(StageAudit::new(QuantizedStageFamily::Output))
        .unwrap();
    assert_eq!(
        duplicate.process_event(StageAudit::new(QuantizedStageFamily::Output)),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn caller_storage_helpers_preserve_capacity_and_typed_errors() {
    let mut storage = super::event::Storage::with_block_capacity(1).unwrap();
    storage.clear();
    assert_eq!(storage.block_capacity(), 1);
    let rejected = super::event::StorageBindError::new(Error::Capacity, storage);
    assert_eq!(rejected.error(), Error::Capacity);
    assert_eq!(rejected.to_string(), Error::Capacity.to_string());
    assert_eq!(rejected.into_storage().block_capacity(), 1);
}

#[test]
fn complete_dispatch_path_is_allocation_free() {
    let (mut actor, model) = actor();
    let measured = measure(|| complete(&mut actor, model));
    assert_eq!(measured.count_total, 0);
}

#[test]
fn hparam_inventory_and_display_are_source_exact() {
    assert_eq!(HPARAM_KEYS.len(), 10);
    assert_eq!(HPARAM_KEYS[0], "qwen3.context_length");
    assert_eq!(HPARAM_KEYS[9], "qwen3.rope.freq_base");
    assert_eq!(
        parameters().to_string(),
        "Qwen3 1 layers, embedding 64, context 128"
    );
}

#[test]
fn loads_source_exact_hparams_and_derived_layout_from_typed_gguf_queries() {
    use crate::loader::test_gguf::{F32, F64, Fixture, U32};

    let mut loader = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"qwen3.context_length", U32, 32_768_u32.to_le_bytes())
            .scalar(b"qwen3.embedding_length", U32, 1024_u32.to_le_bytes())
            .scalar(b"qwen3.feed_forward_length", U32, 4096_u32.to_le_bytes())
            .scalar(b"qwen3.attention.head_count", U32, 16_u32.to_le_bytes())
            .scalar(b"qwen3.attention.head_count_kv", U32, 2_u32.to_le_bytes())
            .scalar(b"qwen3.attention.key_length", U32, 64_u32.to_le_bytes())
            .scalar(b"qwen3.attention.value_length", U32, 80_u32.to_le_bytes())
            .scalar(b"qwen3.block_count", U32, 24_u32.to_le_bytes())
            .scalar(
                b"qwen3.attention.layer_norm_rms_epsilon",
                F64,
                1e-6_f64.to_le_bytes(),
            )
            .scalar(b"qwen3.rope.freq_base", F32, 1_000_000.0_f32.to_le_bytes())
            .build(),
    );
    let decoded = super::load_hparams(&mut loader).unwrap();
    assert_eq!(
        decoded,
        Parameters {
            context_length: 32_768,
            embedding_length: 1024,
            embedding_length_out: 1024,
            feed_forward_length: 4096,
            attention_head_count: 16,
            attention_head_count_kv: 2,
            attention_key_length: 64,
            attention_value_length: 80,
            rope_dimension_count: 64,
            block_count: 24,
            attention_layer_norm_rms_epsilon: 1e-6,
            rope_freq_base: 1_000_000.0,
            tie_word_embeddings: true,
            rope_pair_x0_stride: 1,
            rope_pair_x1_stride: 1,
            rope_pair_x1_offset: 0,
            rope_pair_x1_half_rot_offset: 1,
        }
    );
}

#[test]
fn hparam_decode_rejects_wrong_kind_and_nonpositive_attention_lengths() {
    use crate::loader::test_gguf::{F32, Fixture, U32};

    let mut wrong = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"qwen3.context_length", F32, 1.0_f32.to_le_bytes())
            .build(),
    );
    assert!(super::load_hparams(&mut wrong).is_err());

    let mut zero = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"qwen3.attention.key_length", U32, 0_u32.to_le_bytes())
            .scalar(b"qwen3.attention.value_length", U32, 80_u32.to_le_bytes())
            .build(),
    );
    assert!(super::load_hparams(&mut zero).is_err());
}

#[test]
fn every_required_hparam_rejects_a_missing_key_without_dispatching() {
    for key in HPARAM_KEYS {
        let key_bytes = key.as_bytes();
        let mut loader =
            crate::loader::test_gguf::load(crate::loader::test_gguf::Fixture::new().build());
        assert!(
            super::load_hparams(&mut loader).is_err(),
            "missing {key_bytes:?}"
        );
    }
}
