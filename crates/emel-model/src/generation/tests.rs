#![allow(clippy::manual_range_contains)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::single_element_loop)]

use crate::generation::quantized_path::Resolver;
use allocation_counter::measure;

use crate::catalog::Catalog;
use crate::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};

use super::event::{
    AttentionBlock, BlockAudit, BlockFamily, BlockTensorSlot, BlockValidation, BlockVisit,
    ContractBegin, ContractReset, ContractVisit, Error, GlobalBindings, Plan, RejectBlockTensors,
    ShortconvBlock, StageAudit, Storage, StorageBind, StorageRelease, Topology,
};
use super::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute, Builder,
    LayerExecution, QuantizedContractKind, QuantizedStageFamily, ResidualRoute, StepKind,
};

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

fn actor() -> (Builder, crate::catalog::event::ModelIdentity) {
    actor_with(&TENSORS)
}

fn actor_with(names: &[&[u8]]) -> (Builder, crate::catalog::event::ModelIdentity) {
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage = CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
    for (index, name) in names.iter().enumerate() {
        let wire_type = if index == 0 {
            10
        } else if matches!(index, 1 | 3 | 7 | 8 | 10) {
            0
        } else {
            14
        };
        let data_size = if wire_type == 0 {
            4
        } else if wire_type == 10 {
            84
        } else {
            210
        };
        let dimensions = if wire_type == 0 {
            [1, 1, 1, 1]
        } else {
            [256, 1, 1, 1]
        };
        storage
            .push_tensor(TensorInput::new(
                name, wire_type, 1, dimensions, data_size, true,
            ))
            .unwrap();
    }
    let mut catalog = Catalog::try_new().unwrap();
    catalog.process_event(BindStorage::new(storage)).unwrap();
    let model = catalog.process_event(SealModel::new()).unwrap();
    (Builder::new(catalog, Resolver::new()), model)
}

fn actor_with_typed(tensors: &[(&[u8], u32)]) -> (Builder, crate::catalog::event::ModelIdentity) {
    let name_bytes = tensors.iter().map(|(name, _)| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(tensors.len(), name_bytes, tensors.len()).unwrap();
    for (name, wire_type) in tensors {
        let (dimensions, data_size) = match wire_type {
            0 => ([1, 1, 1, 1], 4),
            10 => ([256, 1, 1, 1], 84),
            14 => ([256, 1, 1, 1], 210),
            _ => unreachable!("focused fixture type"),
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
    (Builder::new(catalog, Resolver::new()), model)
}

fn shortconv_layer() -> LayerExecution {
    LayerExecution::new(
        ResidualRoute::Shortconv,
        AttentionQkNormRoute::None,
        AttentionValueRoute::DedicatedValue,
        AttentionVNormRoute::None,
        AttentionWindowRoute::FullContext,
        0,
        0,
        0,
        0.0,
    )
}

fn layer() -> LayerExecution {
    LayerExecution::new(
        ResidualRoute::Attention,
        AttentionQkNormRoute::HeadwiseRms,
        AttentionValueRoute::DedicatedValue,
        AttentionVNormRoute::None,
        AttentionWindowRoute::FullContext,
        128,
        128,
        128,
        10_000.0,
    )
}

fn complete(builder: &mut Builder, model: crate::catalog::event::ModelIdentity) {
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 4096))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    builder
        .process_event(AttentionBlock::new(0, layer()))
        .unwrap();
    builder
        .process_event(Topology::new(13, 13, 4, 13 * 128 * 4))
        .unwrap();
    builder.process_event(Plan::new()).unwrap();
    builder.process_event(BlockValidation::new(0)).unwrap();
    builder.process_event(BlockAudit::new(0)).unwrap();
    for family in QuantizedStageFamily::ALL {
        builder.process_event(StageAudit::new(family)).unwrap();
    }
}

#[test]
fn public_actor_builds_source_exact_attention_contract_and_fourteen_stage_audit() {
    let (mut builder, model) = actor();
    complete(&mut builder, model);
    let contract = builder.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.model(), model);
    assert_ne!(contract.token_embedding(), contract.output());
    assert_ne!(contract.output_norm(), contract.output());
    assert_eq!(contract.block_count(), 1);
    assert_eq!(contract.prefill_plan().kind(), StepKind::Prefill);
    assert_eq!(contract.prefill_plan().max_step_tokens(), 4096);
    assert_eq!(contract.decode_plan().kind(), StepKind::Decode);
    assert_eq!(contract.decode_plan().max_step_tokens(), 1);
    assert_eq!(contract.audit().len(), 14);
    assert_eq!(
        contract.audit()[0].contract(),
        QuantizedContractKind::ApprovedDenseF32ByContract
    );
    assert_eq!(
        contract.audit()[2].contract(),
        QuantizedContractKind::NativeQuantized
    );
    assert_eq!(
        contract.audit()[6].contract(),
        QuantizedContractKind::NativeQuantized
    );
    assert!(
        contract
            .audit()
            .iter()
            .all(|stage| stage.consistent_across_layers())
    );
    let block = builder.process_event(BlockVisit::new(0)).unwrap();
    assert!(block.uses_attention());
    assert_eq!(block.layer(), layer());
}

#[test]
fn reset_and_release_preserve_preallocated_storage_ownership() {
    let (mut builder, model) = actor();
    complete(&mut builder, model);
    builder.process_event(ContractReset::new()).unwrap();
    let storage = builder.process_event(StorageRelease::new()).unwrap();
    assert_eq!(storage.block_capacity(), 1);
}

#[test]
fn dispatch_path_is_allocation_free() {
    let (mut builder, model) = actor();
    let storage = Storage::with_block_capacity(1).unwrap();
    let measured = measure(|| {
        builder.process_event(StorageBind::new(storage)).unwrap();
        builder
            .process_event(ContractBegin::new(model, 1, 4096))
            .unwrap();
        builder
            .process_event(GlobalBindings::new(
                b"token_embd.weight",
                b"output_norm.weight",
                false,
            ))
            .unwrap();
        builder
            .process_event(AttentionBlock::new(0, layer()))
            .unwrap();
        builder
            .process_event(Topology::new(13, 13, 4, 13 * 128 * 4))
            .unwrap();
        builder.process_event(Plan::new()).unwrap();
        builder.process_event(BlockValidation::new(0)).unwrap();
        builder.process_event(BlockAudit::new(0)).unwrap();
        for family in QuantizedStageFamily::ALL {
            builder.process_event(StageAudit::new(family)).unwrap();
        }
        builder.process_event(ContractVisit::new()).unwrap();
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn exact_labels_cover_source_enums() {
    assert_eq!(QuantizedStageFamily::AttentionV.name(), "attention_v");
    assert_eq!(
        QuantizedContractKind::NotApplicable.name(),
        "not_applicable"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::Q4K),
        "q4_k"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::F16),
        "unknown"
    );
}

#[test]
fn shared_value_and_tied_output_preserve_source_bindings() {
    let names = [
        b"token_embd.weight".as_slice(),
        b"output_norm.weight".as_slice(),
        b"blk.0.attn_norm.weight".as_slice(),
        b"blk.0.attn_q.weight".as_slice(),
        b"blk.0.attn_k.weight".as_slice(),
        b"blk.0.attn_output.weight".as_slice(),
        b"blk.0.ffn_norm.weight".as_slice(),
        b"blk.0.ffn_gate.weight".as_slice(),
        b"blk.0.ffn_down.weight".as_slice(),
        b"blk.0.ffn_up.weight".as_slice(),
    ];
    let (mut builder, model) = actor_with(&names);
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            true,
        ))
        .unwrap();
    let shared = LayerExecution::new(
        ResidualRoute::Attention,
        AttentionQkNormRoute::None,
        AttentionValueRoute::SharedKeyValue,
        AttentionVNormRoute::None,
        AttentionWindowRoute::SlidingWindow,
        32,
        32,
        32,
        10_000.0,
    );
    builder
        .process_event(AttentionBlock::new(0, shared))
        .unwrap();
    builder.process_event(Topology::new(1, 1, 4, 4)).unwrap();
    builder.process_event(Plan::new()).unwrap();
    builder.process_event(BlockValidation::new(0)).unwrap();
    let block = builder.process_event(BlockVisit::new(0)).unwrap();
    assert_eq!(
        block.tensor(BlockTensorSlot::AttentionK),
        block.tensor(BlockTensorSlot::AttentionV)
    );
    builder.process_event(BlockAudit::new(0)).unwrap();
    for family in QuantizedStageFamily::ALL {
        builder.process_event(StageAudit::new(family)).unwrap();
    }
    let contract = builder.process_event(ContractVisit::new()).unwrap();
    assert_eq!(contract.token_embedding(), contract.output());
    assert_eq!(
        contract.audit()[0].tensor_type(),
        contract.audit()[2].tensor_type()
    );
}

#[test]
fn shortconv_marks_attention_stages_not_applicable() {
    let names = [
        b"token_embd.weight".as_slice(),
        b"output_norm.weight".as_slice(),
        b"output.weight".as_slice(),
        b"blk.0.attn_norm.weight".as_slice(),
        b"blk.0.shortconv.conv.weight".as_slice(),
        b"blk.0.shortconv.in_proj.weight".as_slice(),
        b"blk.0.shortconv.out_proj.weight".as_slice(),
        b"blk.0.ffn_norm.weight".as_slice(),
        b"blk.0.ffn_gate.weight".as_slice(),
        b"blk.0.ffn_down.weight".as_slice(),
        b"blk.0.ffn_up.weight".as_slice(),
    ];
    let (mut builder, model) = actor_with(&names);
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    builder
        .process_event(ShortconvBlock::new(0, shortconv_layer()))
        .unwrap();
    builder.process_event(Topology::new(1, 1, 4, 4)).unwrap();
    builder.process_event(Plan::new()).unwrap();
    builder.process_event(BlockValidation::new(0)).unwrap();
    builder.process_event(BlockAudit::new(0)).unwrap();
    for family in QuantizedStageFamily::ALL {
        builder.process_event(StageAudit::new(family)).unwrap();
    }
    let audit = builder.process_event(ContractVisit::new()).unwrap().audit();
    for index in [4, 5, 6, 7, 8, 9] {
        assert_eq!(
            audit[index].contract(),
            QuantizedContractKind::NotApplicable
        );
        assert!(!audit[index].consistent_across_layers());
    }
}

#[test]
fn block_stage_consistency_is_an_explicit_transition_outcome() {
    let tensors = [
        (b"token_embd.weight".as_slice(), 10),
        (b"output_norm.weight".as_slice(), 0),
        (b"output.weight".as_slice(), 14),
        (b"blk.0.attn_norm.weight".as_slice(), 0),
        (b"blk.0.attn_q.weight".as_slice(), 14),
        (b"blk.0.attn_k.weight".as_slice(), 14),
        (b"blk.0.attn_v.weight".as_slice(), 14),
        (b"blk.0.attn_q_norm.weight".as_slice(), 0),
        (b"blk.0.attn_k_norm.weight".as_slice(), 0),
        (b"blk.0.attn_output.weight".as_slice(), 14),
        (b"blk.0.ffn_norm.weight".as_slice(), 0),
        (b"blk.0.ffn_gate.weight".as_slice(), 14),
        (b"blk.0.ffn_down.weight".as_slice(), 14),
        (b"blk.0.ffn_up.weight".as_slice(), 14),
        (b"blk.1.attn_norm.weight".as_slice(), 0),
        (b"blk.1.attn_q.weight".as_slice(), 14),
        (b"blk.1.attn_k.weight".as_slice(), 14),
        (b"blk.1.attn_v.weight".as_slice(), 10),
        (b"blk.1.attn_q_norm.weight".as_slice(), 0),
        (b"blk.1.attn_k_norm.weight".as_slice(), 0),
        (b"blk.1.attn_output.weight".as_slice(), 14),
        (b"blk.1.ffn_norm.weight".as_slice(), 0),
        (b"blk.1.ffn_gate.weight".as_slice(), 14),
        (b"blk.1.ffn_down.weight".as_slice(), 14),
        (b"blk.1.ffn_up.weight".as_slice(), 14),
    ];
    let (mut builder, model) = actor_with_typed(&tensors);
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(2).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 2, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    for index in 0..2 {
        builder
            .process_event(AttentionBlock::new(index, layer()))
            .unwrap();
    }
    builder.process_event(Topology::new(1, 1, 4, 4)).unwrap();
    builder.process_event(Plan::new()).unwrap();
    for index in 0..2 {
        builder.process_event(BlockValidation::new(index)).unwrap();
    }
    for index in 0..2 {
        builder.process_event(BlockAudit::new(index)).unwrap();
    }
    for family in QuantizedStageFamily::ALL {
        builder.process_event(StageAudit::new(family)).unwrap();
    }
    let stage = builder.process_event(ContractVisit::new()).unwrap().audit()
        [QuantizedStageFamily::AttentionV.index()];
    assert_eq!(stage.contract(), QuantizedContractKind::NativeQuantized);
    assert!(!stage.consistent_across_layers());
}

#[test]
fn lifecycle_and_validation_failures_are_typed_and_recoverable() {
    assert_eq!(
        Storage::with_block_capacity(0).unwrap_err(),
        Error::Capacity
    );
    let (mut builder, model) = actor();
    assert_eq!(
        builder.process_event(ContractBegin::new(model, 1, 1)),
        Err(Error::StorageUnavailable)
    );
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    let rejected = builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap_err();
    assert_eq!(rejected.error(), Error::Busy);
    assert_eq!(rejected.into_storage().block_capacity(), 1);
    assert_eq!(
        builder.process_event(ContractBegin::new(model, 2, 1)),
        Err(Error::InvalidRequest)
    );
    builder
        .process_event(ContractBegin::new(model, 1, 1))
        .unwrap();
    assert_eq!(
        builder.process_event(Plan::new()),
        Err(Error::InvalidRequest)
    );
    assert!(matches!(
        builder.process_event(BlockVisit::new(0)),
        Err(Error::InvalidRequest)
    ));
    assert!(matches!(
        builder.process_event(StorageRelease::new()),
        Err(Error::Busy)
    ));
}

#[test]
fn missing_required_tensor_is_model_invalid() {
    let names = &TENSORS[..TENSORS.len() - 1];
    let (mut builder, model) = actor_with(names);
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    assert_eq!(
        builder.process_event(AttentionBlock::new(0, layer())),
        Err(Error::ModelInvalid)
    );
}

#[test]
fn reject_block_tensor_classifies_both_opposite_families() {
    let (mut attention, model) = actor();
    attention
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    attention
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    attention
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    attention
        .process_event(AttentionBlock::new(0, layer()))
        .unwrap();
    assert_eq!(
        attention.process_event(RejectBlockTensors::new(0, BlockFamily::Shortconv)),
        Ok(())
    );
    let (mut invalid_attention, model) = actor();
    invalid_attention
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    invalid_attention
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    invalid_attention
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    invalid_attention
        .process_event(AttentionBlock::new(0, layer()))
        .unwrap();
    assert_eq!(
        invalid_attention.process_event(RejectBlockTensors::new(0, BlockFamily::Attention)),
        Err(Error::InvalidRequest)
    );

    let shortconv_names = [
        b"token_embd.weight".as_slice(),
        b"output_norm.weight",
        b"blk.0.attn_norm.weight",
        b"blk.0.shortconv.conv.weight",
        b"blk.0.shortconv.in_proj.weight",
        b"blk.0.shortconv.out_proj.weight",
        b"blk.0.ffn_norm.weight",
        b"blk.0.ffn_gate.weight",
        b"blk.0.ffn_down.weight",
        b"blk.0.ffn_up.weight",
    ];
    let (mut shortconv, model) = actor_with(&shortconv_names);
    shortconv
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    shortconv
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    shortconv
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            true,
        ))
        .unwrap();
    shortconv
        .process_event(ShortconvBlock::new(0, shortconv_layer()))
        .unwrap();
    assert_eq!(
        shortconv.process_event(RejectBlockTensors::new(0, BlockFamily::Attention)),
        Ok(())
    );
    let (mut invalid_shortconv, model) = actor_with(&shortconv_names);
    invalid_shortconv
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    invalid_shortconv
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    invalid_shortconv
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            true,
        ))
        .unwrap();
    invalid_shortconv
        .process_event(ShortconvBlock::new(0, shortconv_layer()))
        .unwrap();
    assert_eq!(
        invalid_shortconv.process_event(RejectBlockTensors::new(0, BlockFamily::Shortconv)),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn reject_block_tensor_dispatch_is_allocation_free() {
    let (mut builder, model) = actor();
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    builder
        .process_event(AttentionBlock::new(0, layer()))
        .unwrap();
    let measured = measure(|| {
        assert_eq!(
            builder.process_event(RejectBlockTensors::new(0, BlockFamily::Shortconv)),
            Ok(())
        );
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn every_public_error_has_a_stable_message() {
    let errors = [
        Error::InvalidRequest,
        Error::ModelInvalid,
        Error::Capacity,
        Error::StorageUnavailable,
        Error::Busy,
        Error::WrongModelIdentity,
        Error::StaleModelIdentity,
        Error::Dependency,
        Error::UnexpectedEvent,
        Error::Internal,
    ];
    assert!(
        errors
            .into_iter()
            .all(|error| !error.to_string().is_empty())
    );
}

#[test]
fn every_source_label_and_descriptor_accessor_is_observed() {
    let (mut builder, model) = actor();
    complete(&mut builder, model);
    let contract = builder.process_event(ContractVisit::new()).unwrap();
    let topology = contract.topology();
    assert_eq!((topology.node_count(), topology.tensor_count()), (13, 13));
    assert_eq!(
        (
            topology.bytes_per_tensor(),
            topology.workspace_capacity_bytes()
        ),
        (4, 6656)
    );
    let prefill = contract.prefill_plan();
    assert_eq!(
        (
            prefill.node_count(),
            prefill.tensor_count(),
            prefill.expected_outputs()
        ),
        (13, 13, 1)
    );
    let layer = builder.process_event(BlockVisit::new(0)).unwrap().layer();
    assert_eq!(layer.v_norm_route(), AttentionVNormRoute::None);
    assert_eq!(layer.window_route(), AttentionWindowRoute::FullContext);
    assert_eq!(
        (layer.attention_key_length(), layer.attention_value_length()),
        (128, 128)
    );
    assert_eq!(layer.attention_rope_dim(), 128);
    assert!((layer.attention_rope_freq_base() - 10_000.0).abs() < f32::EPSILON);
    let stage_names: Vec<_> = QuantizedStageFamily::ALL
        .into_iter()
        .map(QuantizedStageFamily::name)
        .collect();
    assert_eq!(stage_names.len(), 14);
    let contract_names = [
        QuantizedContractKind::NativeQuantized,
        QuantizedContractKind::ApprovedDenseF32ByContract,
        QuantizedContractKind::DisallowedFallback,
        QuantizedContractKind::ExplicitNoClaim,
        QuantizedContractKind::NotApplicable,
    ]
    .map(QuantizedContractKind::name);
    assert_eq!(
        contract_names,
        [
            "native_quantized",
            "approved_dense_f32_by_contract",
            "disallowed_fallback",
            "explicit_no_claim",
            "not_applicable"
        ]
    );
    for stage in contract.audit() {
        assert_eq!(stage.family().name(), stage_names[stage.family() as usize]);
        assert!(stage.tensor_type().is_some());
    }
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::F32),
        "f32"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::Q2K),
        "q2_k"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::Q3K),
        "q3_k"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::Q6K),
        "q6_k"
    );
    assert_eq!(
        super::tensor_type_name(emel_tensor::dtype::SerializedType::Q4_0),
        "q4_0"
    );
    let (name, length) = super::actor::block_name(123, b"attn_q.weight");
    assert_eq!(&name[..length], b"blk.123.attn_q.weight");
}

fn bind_variant(qk: AttentionQkNormRoute, value: AttentionValueRoute) -> Builder {
    let (mut builder, model) = actor();
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    let selected = LayerExecution::new(
        ResidualRoute::Attention,
        qk,
        value,
        AttentionVNormRoute::Rms,
        AttentionWindowRoute::SlidingWindow,
        32,
        48,
        16,
        500_000.0,
    );
    builder
        .process_event(AttentionBlock::new(0, selected))
        .unwrap();
    builder.process_event(Topology::new(1, 1, 4, 4)).unwrap();
    builder.process_event(Plan::new()).unwrap();
    builder.process_event(BlockValidation::new(0)).unwrap();
    builder
}

#[test]
fn remaining_attention_variants_use_explicit_transition_rows() {
    let mut plain = bind_variant(
        AttentionQkNormRoute::None,
        AttentionValueRoute::DedicatedValue,
    );
    let plain_block = plain.process_event(BlockVisit::new(0)).unwrap();
    assert_ne!(
        plain_block.tensor(BlockTensorSlot::AttentionK),
        plain_block.tensor(BlockTensorSlot::AttentionV)
    );
    assert_eq!(plain_block.tensor(BlockTensorSlot::AttentionQNorm), None);
    let mut shared_qk = bind_variant(
        AttentionQkNormRoute::HeadwiseRms,
        AttentionValueRoute::SharedKeyValue,
    );
    let shared_block = shared_qk.process_event(BlockVisit::new(0)).unwrap();
    assert_eq!(
        shared_block.tensor(BlockTensorSlot::AttentionK),
        shared_block.tensor(BlockTensorSlot::AttentionV)
    );
    assert!(
        shared_block
            .tensor(BlockTensorSlot::AttentionQNorm)
            .is_some()
    );
    assert!(format!("{shared_qk:?}").starts_with("Builder"));
}

#[test]
#[allow(clippy::too_many_lines)]
fn explicit_invalid_transition_rows_recover_for_followup_dispatch() {
    let (mut builder, model) = actor();
    assert_eq!(
        builder.process_event(ContractReset::new()),
        Err(Error::StorageUnavailable)
    );
    assert!(matches!(
        builder.process_event(StorageRelease::new()),
        Err(Error::StorageUnavailable)
    ));
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    assert_eq!(
        builder.process_event(ContractVisit::new()).unwrap_err(),
        Error::UnexpectedEvent
    );
    assert_eq!(
        builder.process_event(ContractBegin::new(model, -1, 0)),
        Err(Error::InvalidRequest)
    );
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .unwrap();
    assert_eq!(
        builder.process_event(ContractBegin::new(model, 1, 64)),
        Err(Error::Busy)
    );
    assert_eq!(
        builder.process_event(GlobalBindings::new(b"", b"", false)),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        builder.process_event(GlobalBindings::new(
            b"missing",
            b"output_norm.weight",
            false
        )),
        Err(Error::ModelInvalid)
    );
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .unwrap();
    assert_eq!(
        builder.process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false
        )),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        builder.process_event(AttentionBlock::new(-1, layer())),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        builder.process_event(ShortconvBlock::new(0, layer())),
        Err(Error::InvalidRequest)
    );
    builder
        .process_event(AttentionBlock::new(0, layer()))
        .unwrap();
    assert_eq!(
        builder.process_event(AttentionBlock::new(0, layer())),
        Err(Error::InvalidRequest)
    );
    assert_eq!(
        builder.process_event(Topology::new(0, 0, 0, 0)),
        Err(Error::InvalidRequest)
    );
    builder.process_event(Topology::new(1, 1, 4, 4)).unwrap();
    assert_eq!(
        builder.process_event(Topology::new(1, 1, 4, 4)),
        Err(Error::InvalidRequest)
    );
    builder.process_event(Plan::new()).unwrap();
    assert_eq!(
        builder.process_event(Plan::new()),
        Err(Error::InvalidRequest)
    );
    builder.process_event(BlockValidation::new(0)).unwrap();
    assert_eq!(
        builder.process_event(BlockValidation::new(0)),
        Err(Error::ModelInvalid)
    );
    assert_eq!(
        builder.process_event(StageAudit::new(QuantizedStageFamily::Output)),
        Err(Error::InvalidRequest)
    );
    builder.process_event(BlockAudit::new(0)).unwrap();
    assert_eq!(
        builder.process_event(BlockAudit::new(0)),
        Err(Error::InvalidRequest)
    );
    builder
        .process_event(StageAudit::new(QuantizedStageFamily::Output))
        .unwrap();
    assert_eq!(
        builder.process_event(StageAudit::new(QuantizedStageFamily::Output)),
        Err(Error::InvalidRequest)
    );
}

#[test]
fn wrong_catalog_identity_is_preserved_as_typed_dependency_error() {
    let (mut builder, _) = actor();
    let (_, other_model) = actor();
    builder
        .process_event(StorageBind::new(Storage::with_block_capacity(1).unwrap()))
        .unwrap();
    assert_eq!(
        builder.process_event(ContractBegin::new(other_model, 1, 64)),
        Err(Error::WrongModelIdentity)
    );
}
