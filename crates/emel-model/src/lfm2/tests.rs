use crate::generation::quantized_path::Resolver;
use allocation_counter::measure;

use crate::catalog::Catalog;
use crate::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};
use crate::generation::{AttentionQkNormRoute, QuantizedStageFamily, ResidualRoute};

use super::event::{
    BlockAudit, BlockBuild, BlockValidation, BlockVisit, ContractBegin, ContractReset,
    ContractVisit, Error, HPARAM_KEYS, PlanBuild, StageAudit, StorageRelease, TopologyBuild,
};
use super::{Lfm2, Parameters, Variant};

const FALLBACK_ATTENTION: [i32; 6] = [2, 5, 8, 10, 12, 14];

fn parameters_1_2b() -> Parameters {
    Parameters {
        context_length: 128_000,
        embedding_length: 2048,
        embedding_length_out: 2048,
        feed_forward_length: 8192,
        attention_head_count: 32,
        attention_head_count_kv: 8,
        block_count: 16,
        vocab_size: 65_536,
        shortconv_l_cache: 3,
        attention_layer_norm_rms_epsilon: 1e-6,
        rope_freq_base: 1_000_000.0,
        ..Parameters::default()
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

fn is_attention(parameters: &Parameters, index: i32) -> bool {
    if parameters.attention_layer_pattern_count == 0 {
        FALLBACK_ATTENTION.contains(&index)
    } else {
        usize::try_from(index)
            .is_ok_and(|value| parameters.attention_layer_pattern_flags[value] != 0)
    }
}

fn names(parameters: &Parameters) -> Vec<Vec<u8>> {
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
        let selected: &[&str] = if is_attention(parameters, index) {
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

fn actor_with_names(
    parameters: &Parameters,
    names: &[Vec<u8>],
) -> (Lfm2, crate::catalog::event::ModelIdentity) {
    let name_bytes = names.iter().map(Vec::len).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
    for name in names {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    let actor = Lfm2::new(
        catalog,
        Resolver::new(),
        super::event::Storage::with_block_capacity(
            usize::try_from(parameters.block_count).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    (actor, model)
}

fn actor(parameters: &Parameters) -> (Lfm2, crate::catalog::event::ModelIdentity) {
    actor_with_names(parameters, &names(parameters))
}

fn complete(
    actor: &mut Lfm2,
    model: crate::catalog::event::ModelIdentity,
    parameters: &Parameters,
) {
    actor
        .process_event(ContractBegin::new(b"lfm2", model, parameters))
        .unwrap();
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
fn builds_source_exact_1_2b_fallback_hybrid_contract() {
    let parameters = parameters_1_2b();
    let (mut actor, model) = actor(&parameters);
    complete(&mut actor, model, &parameters);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(parameters.variant(), Some(Variant::OnePointTwoB));
    assert_eq!(contract.block_count(), 16);
    assert_eq!(contract.topology().tensor_count(), 149);
    assert_eq!(contract.topology().node_count(), 149);
    assert_eq!(
        contract.topology().workspace_capacity_bytes(),
        149 * 2048 * 4
    );
    let shortconv = actor.process_event(BlockVisit::new(0)).unwrap();
    assert!(!shortconv.uses_attention());
    assert_eq!(shortconv.layer().residual_route(), ResidualRoute::Shortconv);
    let attention = actor.process_event(BlockVisit::new(2)).unwrap();
    assert!(attention.uses_attention());
    assert_eq!(attention.layer().residual_route(), ResidualRoute::Attention);
    assert_eq!(
        attention.layer().qk_norm_route(),
        AttentionQkNormRoute::HeadwiseRms
    );
}

#[test]
fn builds_source_exact_230m_pattern_hybrid_contract() {
    let parameters = parameters_230m();
    let (mut actor, model) = actor(&parameters);
    complete(&mut actor, model, &parameters);
    let contract = actor.process_event(ContractVisit::new()).unwrap();
    assert_eq!(parameters.variant(), Some(Variant::TwoHundredThirtyM));
    assert_eq!(contract.block_count(), 14);
    assert_eq!(contract.topology().tensor_count(), 133);
    assert_eq!(
        contract.topology().workspace_capacity_bytes(),
        133 * 1024 * 4
    );
    assert!(
        !actor
            .process_event(BlockVisit::new(1))
            .unwrap()
            .uses_attention()
    );
    assert!(
        actor
            .process_event(BlockVisit::new(2))
            .unwrap()
            .uses_attention()
    );
}

#[test]
fn rejects_opposite_family_tensors_and_rolls_back_partial_contract() {
    for (parameters, index, rejected_suffix) in [
        (parameters_1_2b(), 0, "attn_q.weight"),
        (parameters_1_2b(), 2, "shortconv.conv.weight"),
    ] {
        let mut tensor_names = names(&parameters);
        tensor_names.push(format!("blk.{index}.{rejected_suffix}").into_bytes());
        let (mut actor, model) = actor_with_names(&parameters, &tensor_names);
        actor
            .process_event(ContractBegin::new(b"lfm2", model, &parameters))
            .unwrap();
        assert_eq!(
            actor.process_event(BlockBuild::new(index)),
            Err(Error::ModelInvalid)
        );
        actor
            .process_event(ContractBegin::new(b"lfm2", model, &parameters))
            .unwrap();
    }
}

#[test]
fn pattern_classification_fails_closed_when_it_contradicts_tensors() {
    let tensor_parameters = parameters_230m();
    let tensor_names = names(&tensor_parameters);
    let mut contradictory = tensor_parameters;
    contradictory.attention_layer_pattern_flags[2] = 0;
    let (mut actor, model) = actor_with_names(&contradictory, &tensor_names);
    actor
        .process_event(ContractBegin::new(b"lfm2", model, &contradictory))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(2)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn rejects_wrong_architecture_geometry_and_missing_global_tensor() {
    let parameters = parameters_1_2b();
    let (mut actor, model) = actor(&parameters);
    assert_eq!(
        actor.process_event(ContractBegin::new(b"llama", model, &parameters)),
        Err(Error::InvalidRequest)
    );
    let mut invalid = parameters;
    invalid.embedding_length = 1024;
    assert_eq!(invalid.variant(), None);
    assert_eq!(
        actor.process_event(ContractBegin::new(b"lfm2", model, &invalid)),
        Err(Error::ModelInvalid)
    );

    let mut missing_names = names(&parameters);
    missing_names.remove(1);
    let (mut missing, missing_model) = actor_with_names(&parameters, &missing_names);
    assert_eq!(
        missing.process_event(ContractBegin::new(b"lfm2", missing_model, &parameters)),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn non_strict_builder_and_data_validation_accept_source_positive_geometry() {
    let parameters = parameters_validation();
    assert_eq!(parameters.variant(), None);
    let (mut actor, model) = actor(&parameters);
    assert_eq!(
        actor.process_event(ContractBegin::new(b"lfm2", model, &parameters)),
        Err(Error::ModelInvalid)
    );
    assert_eq!(
        actor.process_event(ContractBegin::validation(
            b"lfm2",
            model,
            &parameters,
            parameters.block_count + 1,
        )),
        Err(Error::ModelInvalid)
    );
    actor
        .process_event(ContractBegin::validation(
            b"lfm2",
            model,
            &parameters,
            parameters.block_count,
        ))
        .unwrap();
    for index in 0..parameters.block_count {
        actor.process_event(BlockBuild::new(index)).unwrap();
    }
    actor.process_event(ContractReset::new()).unwrap();
}

#[test]
fn one_block_per_dispatch_and_lifecycle_are_explicit() {
    let parameters = parameters_230m();
    let (mut actor, model) = actor(&parameters);
    actor
        .process_event(ContractBegin::new(b"lfm2", model, &parameters))
        .unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(-1)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(BlockBuild::new(14)),
        Err(Error::InvalidRequest)
    );
    actor.process_event(BlockBuild::new(0)).unwrap();
    assert_eq!(
        actor.process_event(BlockBuild::new(0)),
        Err(Error::InvalidRequest)
    );
    actor.process_event(ContractReset::new()).unwrap();
    let storage = actor.process_event(StorageRelease::new()).unwrap();
    assert_eq!(storage.block_capacity(), 14);
}

#[test]
fn incomplete_lifecycle_routes_are_typed_and_recoverable() {
    let parameters = parameters_230m();
    let (mut actor, model) = actor(&parameters);
    assert_eq!(format!("{actor:?}"), "Lfm2 { .. }");
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::UnexpectedEvent)
    );

    actor
        .process_event(ContractBegin::new(b"lfm2", model, &parameters))
        .unwrap();
    assert_eq!(
        actor.process_event(ContractBegin::new(b"lfm2", model, &parameters)),
        Err(Error::Busy)
    );
    assert_eq!(
        actor.process_event(StorageRelease::new()).unwrap_err(),
        Error::Busy
    );
    assert_eq!(
        actor.process_event(TopologyBuild::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(PlanBuild::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(BlockValidation::new(0)),
        Err(Error::ModelInvalid)
    );
    assert_eq!(
        actor.process_event(BlockAudit::new(0)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(StageAudit::new(QuantizedStageFamily::TokenEmbedding)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(ContractVisit::new()),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        actor.process_event(BlockVisit::new(0)).unwrap_err(),
        Error::InvalidRequest
    );

    actor.process_event(ContractReset::new()).unwrap();
    let mut storage = actor.process_event(StorageRelease::new()).unwrap();
    storage.clear();
    assert_eq!(storage.block_capacity(), 14);

    let bind_error = super::event::StorageBindError::new(Error::Capacity, storage);
    assert_eq!(bind_error.error(), Error::Capacity);
    assert_eq!(bind_error.to_string(), Error::Capacity.to_string());
    let storage = bind_error.into_storage();
    assert_eq!(storage.block_capacity(), 14);
}

#[test]
fn complete_dispatch_path_is_allocation_free() {
    let parameters = parameters_230m();
    let (mut actor, model) = actor(&parameters);
    let measured = measure(|| complete(&mut actor, model, &parameters));
    assert_eq!(measured.count_total, 0);
}

#[test]
fn hparam_inventory_and_display_are_source_exact() {
    assert_eq!(HPARAM_KEYS.len(), 10);
    assert_eq!(HPARAM_KEYS[0], "lfm2.context_length");
    assert_eq!(HPARAM_KEYS[9], "lfm2.rope.freq_base");
    assert_eq!(
        parameters_230m().to_string(),
        "Lfm2 14 layers, embedding 1024, context 128000"
    );
}

#[test]
fn loads_source_exact_hparams_and_per_layer_flags_from_typed_gguf_queries() {
    use crate::loader::test_gguf::{F32, F64, Fixture, U32};

    let pattern = [0_u32, 8, 0, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut loader = crate::loader::test_gguf::load(
        Fixture::new()
            .scalar(b"lfm2.context_length", U32, 128_000_u32.to_le_bytes())
            .scalar(b"lfm2.embedding_length", U32, 2048_u32.to_le_bytes())
            .scalar(b"lfm2.feed_forward_length", U32, 8192_u32.to_le_bytes())
            .scalar(b"lfm2.attention.head_count", U32, 32_u32.to_le_bytes())
            .scalar(b"lfm2.block_count", U32, 16_u32.to_le_bytes())
            .scalar(b"lfm2.vocab_size", U32, 65_536_u32.to_le_bytes())
            .scalar(b"lfm2.shortconv.l_cache", U32, 3_u32.to_le_bytes())
            .scalar(
                b"lfm2.attention.layer_norm_rms_epsilon",
                F64,
                1e-6_f64.to_le_bytes(),
            )
            .scalar(b"lfm2.rope.freq_base", F32, 1_000_000.0_f32.to_le_bytes())
            .array(b"lfm2.attention.head_count_kv", U32, &pattern, 4)
            .build(),
    );
    let decoded = super::load_hparams(&mut loader).unwrap();
    assert_eq!(decoded.context_length, 128_000);
    assert_eq!(decoded.embedding_length, 2048);
    assert_eq!(decoded.embedding_length_out, 2048);
    assert_eq!(decoded.feed_forward_length, 8192);
    assert_eq!(decoded.attention_head_count, 32);
    assert_eq!(decoded.attention_head_count_kv, 8);
    assert_eq!(decoded.block_count, 16);
    assert_eq!(decoded.vocab_size, 65_536);
    assert_eq!(decoded.shortconv_l_cache, 3);
    assert_eq!(decoded.attention_layer_pattern_count, 4);
    assert_eq!(&decoded.attention_layer_pattern_flags[..4], &[0, 1, 0, 1]);
    assert_eq!(
        decoded.attention_layer_norm_rms_epsilon.to_bits(),
        1e-6_f32.to_bits()
    );
    assert_eq!(decoded.rope_freq_base.to_bits(), 1_000_000.0_f32.to_bits());

    let scalar_prefix = |count: usize| {
        let mut fixture = Fixture::new();
        if count >= 1 {
            fixture = fixture.scalar(b"lfm2.context_length", U32, 128_000_u32.to_le_bytes());
        }
        if count >= 2 {
            fixture = fixture.scalar(b"lfm2.embedding_length", U32, 2048_u32.to_le_bytes());
        }
        if count >= 3 {
            fixture = fixture.scalar(b"lfm2.feed_forward_length", U32, 8192_u32.to_le_bytes());
        }
        if count >= 4 {
            fixture = fixture.scalar(b"lfm2.attention.head_count", U32, 32_u32.to_le_bytes());
        }
        if count >= 5 {
            fixture = fixture.scalar(b"lfm2.block_count", U32, 16_u32.to_le_bytes());
        }
        if count >= 6 {
            fixture = fixture.scalar(b"lfm2.vocab_size", U32, 65_536_u32.to_le_bytes());
        }
        if count >= 7 {
            fixture = fixture.scalar(b"lfm2.shortconv.l_cache", U32, 3_u32.to_le_bytes());
        }
        if count >= 8 {
            fixture = fixture.scalar(
                b"lfm2.attention.layer_norm_rms_epsilon",
                F64,
                1e-6_f64.to_le_bytes(),
            );
        }
        if count >= 9 {
            fixture = fixture.scalar(b"lfm2.rope.freq_base", F32, 1_000_000.0_f32.to_le_bytes());
        }
        fixture
    };
    for count in 0..=9 {
        let mut missing = crate::loader::test_gguf::load(scalar_prefix(count).build());
        assert!(super::load_hparams(&mut missing).is_err());
    }

    let over_capacity_pattern = vec![8_u32; 513]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut over_capacity = crate::loader::test_gguf::load(
        scalar_prefix(9)
            .array(
                b"lfm2.attention.head_count_kv",
                U32,
                &over_capacity_pattern,
                513,
            )
            .build(),
    );
    assert!(super::load_hparams(&mut over_capacity).is_err());
}

#[test]
fn kv_head_flag_array_rejects_a_string_payload() {
    let mut wrong_kind = crate::loader::test_gguf::load(
        crate::loader::test_gguf::Fixture::new()
            .repeated_strings(b"lfm2.attention.head_count_kv", b"wrong", 4)
            .build(),
    );
    assert!(super::load_hparams(&mut wrong_kind).is_err());
}
