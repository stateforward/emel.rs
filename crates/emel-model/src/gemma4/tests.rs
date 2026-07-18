use allocation_counter::measure;
use emel_kernels::capability::Resolver;

use crate::catalog::Catalog;
use crate::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use crate::generation::event::BlockTensorSlot;
use crate::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute,
    QuantizedContractKind, QuantizedStageFamily, ResidualRoute,
};

use super::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractReset,
    ContractVisit, Error, HPARAM_KEYS, PlanBuild, StageAudit, StorageRelease, TopologyBuild,
};
use super::{Gemma4, Parameters, SLIDING_WINDOW_PATTERN};

fn canonical_parameters() -> Parameters {
    Parameters::canonical()
}

fn inputs(
    block_count: i32,
    first_shared: i32,
    include_output: bool,
    missing_dedicated_v: bool,
) -> Vec<(Vec<u8>, u32)> {
    let mut tensors = vec![
        (b"token_embd.weight".to_vec(), 10),
        (b"output_norm.weight".to_vec(), 0),
    ];
    if include_output {
        tensors.push((b"output.weight".to_vec(), 10));
    }
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
            if suffix == "attn_v.weight"
                && (index >= first_shared || (missing_dedicated_v && index == 0))
            {
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

fn actor_with(tensors: &[(Vec<u8>, u32)], capacity: usize) -> (Gemma4, ModelIdentity) {
    let name_bytes = tensors.iter().map(|(name, _)| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(tensors.len(), name_bytes, tensors.len()).unwrap();
    for (name, wire_type) in tensors {
        let (dimensions, data_size) = match wire_type {
            0 => ([1, 1, 1, 1], 4),
            10 => ([256, 1, 1, 1], 84),
            14 => ([256, 1, 1, 1], 210),
            _ => unreachable!("focused Gemma4 tensor type"),
        };
        storage
            .push_tensor(TensorInput::new(
                name, *wire_type, 1, dimensions, data_size, true,
            ))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    let actor = Gemma4::new(
        catalog,
        Resolver::new(),
        super::event::Storage::with_block_capacity(capacity).unwrap(),
    )
    .unwrap();
    (actor, model)
}

fn canonical_actor() -> (Gemma4, ModelIdentity) {
    actor_with(
        &inputs(super::BLOCK_COUNT, 15, false, false),
        super::BLOCK_COUNT as usize,
    )
}

fn complete(actor: &mut Gemma4, model: ModelIdentity, parameters: &Parameters, strict: bool) {
    if strict {
        actor
            .process_event(ContractBegin::new(b"gemma4", model, parameters))
            .unwrap();
    } else {
        actor
            .process_event(ContractBegin::validation(
                b"gemma4",
                model,
                parameters,
                parameters.block_count,
            ))
            .unwrap();
    }
    for index in 0..parameters.block_count {
        actor.process_event(BlockBuild::new(index)).unwrap();
    }
    actor.process_event(TopologyBuild::new()).unwrap();
    actor.process_event(PlanBuild::new()).unwrap();
    for index in 0..parameters.block_count {
        actor.process_event(BlockValidation::new(index)).unwrap();
    }
    for index in 0..parameters.block_count {
        actor.process_event(BlockAudit::new(index)).unwrap();
    }
    for family in QuantizedStageFamily::ALL {
        actor.process_event(StageAudit::new(family)).unwrap();
    }
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn builds_canonical_shared_kv_contract_routes_topology_views_and_audit() {
    let parameters = canonical_parameters();
    let (mut actor, model) = canonical_actor();
    complete(&mut actor, model, &parameters, true);

    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.model(), model);
    assert_eq!(contract.block_count(), 35);
    assert_eq!(contract.output(), contract.token_embedding());
    assert_eq!(contract.topology().tensor_count(), 333);
    assert_eq!(contract.topology().node_count(), 333);
    assert_eq!(
        contract.topology().workspace_capacity_bytes(),
        333 * 1536 * 4
    );
    assert_eq!(contract.prefill_plan().max_step_tokens(), 131_072);
    assert_eq!(contract.decode_plan().max_step_tokens(), 1);

    let sliding_dedicated = actor.process_event(BlockVisit::new(0)).unwrap();
    assert_eq!(
        sliding_dedicated.layer().residual_route(),
        ResidualRoute::Attention
    );
    assert_eq!(
        sliding_dedicated.layer().qk_norm_route(),
        AttentionQkNormRoute::HeadwiseRms
    );
    assert_eq!(
        sliding_dedicated.layer().value_route(),
        AttentionValueRoute::DedicatedValue
    );
    assert_eq!(
        sliding_dedicated.layer().v_norm_route(),
        AttentionVNormRoute::None
    );
    assert_eq!(
        sliding_dedicated.layer().window_route(),
        AttentionWindowRoute::SlidingWindow
    );
    assert_eq!(sliding_dedicated.layer().attention_key_length(), 256);
    assert_eq!(sliding_dedicated.layer().attention_value_length(), 256);
    assert_eq!(sliding_dedicated.layer().attention_rope_dim(), 256);
    assert_eq!(
        sliding_dedicated
            .layer()
            .attention_rope_freq_base()
            .to_bits(),
        10_000.0f32.to_bits()
    );

    let full_dedicated = actor.process_event(BlockVisit::new(4)).unwrap();
    assert_eq!(
        full_dedicated.layer().window_route(),
        AttentionWindowRoute::FullContext
    );
    assert_eq!(full_dedicated.layer().attention_key_length(), 512);
    assert_eq!(full_dedicated.layer().attention_value_length(), 512);
    assert_eq!(full_dedicated.layer().attention_rope_dim(), 512);
    assert_eq!(
        full_dedicated.layer().attention_rope_freq_base().to_bits(),
        1_000_000.0f32.to_bits()
    );

    let dedicated = actor.process_event(BlockVisit::new(14)).unwrap();
    assert_eq!(
        dedicated.layer().value_route(),
        AttentionValueRoute::DedicatedValue
    );
    assert_eq!(dedicated.layer().v_norm_route(), AttentionVNormRoute::None);

    let shared = actor.process_event(BlockVisit::new(15)).unwrap();
    assert_eq!(
        shared.layer().value_route(),
        AttentionValueRoute::SharedKeyValue
    );
    assert_eq!(shared.layer().v_norm_route(), AttentionVNormRoute::Rms);
    assert_eq!(
        shared.tensor(BlockTensorSlot::AttentionV),
        shared.tensor(BlockTensorSlot::AttentionK)
    );

    let audit = contract.audit();
    assert_eq!(
        audit[QuantizedStageFamily::TokenEmbedding as usize].contract(),
        QuantizedContractKind::ApprovedDenseF32ByContract
    );
    assert_eq!(
        audit[QuantizedStageFamily::Output as usize].contract(),
        QuantizedContractKind::NativeQuantized
    );
    assert_eq!(
        audit[QuantizedStageFamily::AttentionV as usize].contract(),
        QuantizedContractKind::NativeQuantized
    );
    assert!(audit[QuantizedStageFamily::AttentionV as usize].consistent_across_layers());
    assert_eq!(
        audit[QuantizedStageFamily::AttentionQNorm as usize].contract(),
        QuantizedContractKind::ApprovedDenseF32ByContract
    );
    assert_eq!(
        audit[QuantizedStageFamily::AttentionKNorm as usize].contract(),
        QuantizedContractKind::ApprovedDenseF32ByContract
    );
}

#[test]
fn strict_contract_rejects_pattern_drift_and_missing_dedicated_value() {
    let (mut pattern_actor, model) = canonical_actor();
    let mut bad_pattern = canonical_parameters();
    bad_pattern.sliding_window_pattern_flags[4] = 1;
    assert_eq!(
        pattern_actor.process_event(ContractBegin::new(b"gemma4", model, &bad_pattern)),
        Err(Error::ModelInvalid)
    );

    let tensors = inputs(super::BLOCK_COUNT, 15, false, true);
    let (mut missing_v, model) = actor_with(&tensors, super::BLOCK_COUNT as usize);
    let parameters = canonical_parameters();
    missing_v
        .process_event(ContractBegin::new(b"gemma4", model, &parameters))
        .unwrap();
    assert_eq!(
        missing_v.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn non_strict_sliding_fields_fall_back_independently_and_all_dedicated_topology() {
    let tensors = inputs(1, 1, false, false);
    let (mut actor, model) = actor_with(&tensors, 1);
    let mut parameters = canonical_parameters();
    parameters.block_count = 1;
    parameters.context_length = 128;
    parameters.embedding_length = 64;
    parameters.embedding_length_out = 64;
    parameters.feed_forward_length = 256;
    parameters.attention_head_count = 8;
    parameters.attention_head_count_kv = 1;
    parameters.attention_key_length = 32;
    parameters.attention_key_length_swa = 0;
    parameters.attention_value_length = 40;
    parameters.attention_value_length_swa = 0;
    parameters.vocab_size = 1024;
    parameters.attention_shared_kv_layers = 0;
    parameters.rope_dimension_count = 32;
    parameters.rope_dimension_count_swa = 0;
    parameters.rope_freq_base = 20_000.0;
    parameters.rope_freq_base_swa = 0.0;
    parameters.sliding_window_pattern_count = 1;
    parameters.sliding_window_pattern_flags[0] = 1;

    complete(&mut actor, model, &parameters, false);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.topology().tensor_count(), 13);
    let block = actor.process_event(BlockVisit::new(0)).unwrap();
    assert_eq!(
        block.layer().window_route(),
        AttentionWindowRoute::SlidingWindow
    );
    assert_eq!(block.layer().attention_key_length(), 32);
    assert_eq!(block.layer().attention_value_length(), 40);
    assert_eq!(block.layer().attention_rope_dim(), 32);
    assert_eq!(
        block.layer().attention_rope_freq_base().to_bits(),
        20_000.0f32.to_bits()
    );
}

#[test]
fn non_strict_validation_uses_the_fixed_source_shared_boundary_for_required_value() {
    let tensors = inputs(1, 0, false, false);
    let (mut actor, model) = actor_with(&tensors, 1);
    let mut parameters = canonical_parameters();
    parameters.block_count = 1;
    parameters.context_length = 128;
    parameters.embedding_length = 64;
    parameters.embedding_length_out = 64;
    parameters.feed_forward_length = 256;
    parameters.attention_head_count = 8;
    parameters.attention_head_count_kv = 1;
    parameters.attention_key_length = 32;
    parameters.attention_value_length = 40;
    parameters.vocab_size = 1024;
    parameters.attention_shared_kv_layers = 1;
    parameters.rope_freq_base = 20_000.0;

    actor
        .process_event(ContractBegin::validation(
            b"gemma4",
            model,
            &parameters,
            parameters.block_count,
        ))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::ModelInvalid)
    );

    let tensors = inputs(1, 1, false, false);
    let (mut actor, model) = actor_with(&tensors, 1);
    complete(&mut actor, model, &parameters, false);
    let block = actor.process_event(BlockVisit::new(0)).unwrap();
    assert_eq!(
        block.layer().value_route(),
        AttentionValueRoute::SharedKeyValue
    );
    assert_eq!(
        block.tensor(BlockTensorSlot::AttentionV),
        block.tensor(BlockTensorSlot::AttentionK)
    );
}

#[test]
fn non_strict_validation_rejects_block_counts_above_the_fixed_pattern_bound() {
    let tensors = inputs(0, 0, false, false);
    let (mut actor, model) = actor_with(&tensors, super::MAX_BLOCKS);
    let mut parameters = canonical_parameters();
    parameters.block_count = i32::try_from(super::MAX_BLOCKS + 1).unwrap();
    parameters.sliding_window_pattern_count = u32::try_from(super::MAX_BLOCKS + 1).unwrap();

    assert_eq!(
        actor.process_event(ContractBegin::validation(
            b"gemma4",
            model,
            &parameters,
            parameters.block_count,
        )),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn canonical_dispatch_path_is_allocation_free_and_lifecycle_releases_storage() {
    let parameters = canonical_parameters();
    let (mut actor, model) = canonical_actor();
    let measured = measure(|| complete(&mut actor, model, &parameters, true));
    assert_eq!(measured.count_total, 0);
    assert!(matches!(
        actor.process_event(StorageRelease::new()),
        Err(Error::Busy)
    ));
    actor.process_event(ContractReset::new()).unwrap();
    let storage = actor.process_event(StorageRelease::new()).unwrap();
    assert_eq!(storage.block_capacity(), super::BLOCK_COUNT as usize);
}

#[test]
fn begin_and_block_fail_closed_with_typed_outcomes() {
    let parameters = canonical_parameters();
    let (mut actor, model) = canonical_actor();
    assert_eq!(
        actor.process_event(ContractBegin::new(b"qwen3", model, &parameters)),
        Err(Error::InvalidRequest)
    );
    let mut invalid = parameters;
    invalid.embedding_length = 0;
    assert_eq!(
        actor.process_event(ContractBegin::new(b"gemma4", model, &invalid)),
        Err(Error::ModelInvalid)
    );
    actor
        .process_event(ContractBegin::new(b"gemma4", model, &parameters))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(-1)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(BlockBuild::new(super::BLOCK_COUNT)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(ContractBegin::new(b"gemma4", model, &parameters)),
        Err(Error::Busy)
    );
    assert_eq!(format!("{actor:?}"), "Gemma4 { .. }");
}

#[test]
fn hparam_inventory_display_and_canonical_pattern_are_source_exact() {
    let parameters = canonical_parameters();
    assert_eq!(HPARAM_KEYS.len(), 21);
    assert_eq!(HPARAM_KEYS[0], "gemma4.context_length");
    assert_eq!(HPARAM_KEYS[20], "gemma4.attention.sliding_window_pattern");
    assert_eq!(
        parameters.to_string(),
        "Gemma4 35 layers, embedding 1536, context 131072"
    );
    assert_eq!(
        &parameters.sliding_window_pattern_flags[..35],
        &SLIDING_WINDOW_PATTERN
    );
}

#[test]
fn loads_all_source_hparams_array_feed_forward_and_derived_facts() {
    use crate::loader::test_gguf::{BOOL, F32, F64, Fixture, U32};

    let feed_forward = 6_144_u32.to_le_bytes();
    let pattern = SLIDING_WINDOW_PATTERN;
    let mut loader = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"gemma4.context_length", U32, 131_072_u32.to_le_bytes())
            .scalar(b"gemma4.embedding_length", U32, 1_536_u32.to_le_bytes())
            .scalar(
                b"gemma4.embedding_length_per_layer_input",
                U32,
                256_u32.to_le_bytes(),
            )
            .array(b"gemma4.feed_forward_length", U32, &feed_forward, 1)
            .scalar(b"gemma4.attention.head_count", U32, 8_u32.to_le_bytes())
            .scalar(b"gemma4.attention.head_count_kv", U32, 1_u32.to_le_bytes())
            .scalar(b"gemma4.attention.key_length", U32, 512_u32.to_le_bytes())
            .scalar(
                b"gemma4.attention.key_length_swa",
                U32,
                256_u32.to_le_bytes(),
            )
            .scalar(b"gemma4.attention.value_length", U32, 512_u32.to_le_bytes())
            .scalar(
                b"gemma4.attention.value_length_swa",
                U32,
                256_u32.to_le_bytes(),
            )
            .scalar(b"gemma4.block_count", U32, 35_u32.to_le_bytes())
            .scalar(b"gemma4.vocab_size", U32, 262_144_u32.to_le_bytes())
            .scalar(
                b"gemma4.attention.sliding_window",
                U32,
                512_u32.to_le_bytes(),
            )
            .scalar(
                b"gemma4.attention.shared_kv_layers",
                U32,
                20_u32.to_le_bytes(),
            )
            .scalar(b"gemma4.rope.dimension_count", U32, 512_u32.to_le_bytes())
            .scalar(
                b"gemma4.rope.dimension_count_swa",
                U32,
                256_u32.to_le_bytes(),
            )
            .scalar(
                b"gemma4.attention.layer_norm_rms_epsilon",
                F64,
                1e-6_f64.to_le_bytes(),
            )
            .scalar(
                b"gemma4.final_logit_softcapping",
                F32,
                30.0_f32.to_le_bytes(),
            )
            .scalar(b"gemma4.rope.freq_base", F32, 1_000_000.0_f32.to_le_bytes())
            .scalar(
                b"gemma4.rope.freq_base_swa",
                F32,
                10_000.0_f32.to_le_bytes(),
            )
            .array(
                b"gemma4.attention.sliding_window_pattern",
                BOOL,
                &pattern,
                35,
            )
            .build(),
    );
    let decoded = super::load_hparams(&mut loader).unwrap();
    assert_eq!(decoded, canonical_parameters());
}

#[test]
fn hparam_loader_requires_a_bounded_typed_pattern() {
    use crate::loader::hparams::{ErrorKind, Operation};
    use crate::loader::test_gguf::{Fixture, STRING};

    let mut missing = crate::loader::test_gguf::load(Fixture::new().build());
    let error = super::load_hparams(&mut missing).unwrap_err();
    assert_eq!(error.operation, Operation::RequiredFlagArray);
    assert_eq!(error.kind, ErrorKind::Missing);

    let mut wrong_kind = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(
                b"gemma4.attention.sliding_window_pattern",
                STRING,
                0_u64.to_le_bytes(),
            )
            .build(),
    );
    let error = super::load_hparams(&mut wrong_kind).unwrap_err();
    assert_eq!(error.operation, Operation::RequiredFlagArray);
    assert_eq!(error.kind, ErrorKind::WrongKind);
}

#[test]
fn absent_scalar_hparams_preserve_zero_initialized_source_values() {
    use crate::loader::test_gguf::{BOOL, Fixture};

    let pattern = [1_u8];
    let mut loader = crate::loader::test_gguf::load(
        Fixture::new()
            .array(
                b"gemma4.attention.sliding_window_pattern",
                BOOL,
                &pattern,
                1,
            )
            .build(),
    );
    let decoded = super::load_hparams(&mut loader).unwrap();

    assert_eq!(decoded.context_length, 0);
    assert_eq!(decoded.embedding_length, 0);
    assert_eq!(decoded.embedding_length_out, 0);
    assert_eq!(decoded.feed_forward_length, 0);
    assert_eq!(decoded.attention_head_count, 0);
    assert_eq!(decoded.attention_key_length, 0);
    assert_eq!(decoded.attention_value_length, 0);
    assert_eq!(decoded.block_count, 0);
    assert_eq!(decoded.vocab_size, 0);
    assert_eq!(decoded.rope_freq_base.to_bits(), 0.0_f32.to_bits());
    assert_eq!(decoded.sliding_window_pattern_count, 1);
    assert_eq!(
        decoded.full_attention_interval,
        super::FULL_ATTENTION_INTERVAL
    );
    assert!(decoded.tie_word_embeddings);
}
