#![no_main]

use emel_kernels::capability::Resolver;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use emel_model::generation::event::{
    AttentionBlock, BlockAudit, BlockFamily, BlockValidation, BlockVisit, ContractBegin,
    ContractReset, ContractVisit, GlobalBindings, Plan, RejectBlockTensors, ShortconvBlock,
    StageAudit, Storage, StorageBind, StorageRelease, Topology,
};
use emel_model::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute, Builder,
    LayerExecution, QuantizedStageFamily, ResidualRoute,
};
use emel_model::gemma4::event as gemma4_event;
use emel_model::gemma4::{Gemma4, Parameters as Gemma4Parameters};
use emel_model::llama::event as llama_event;
use emel_model::llama::{Llama, Parameters as LlamaParameters};
use emel_model::lfm2::event as lfm2_event;
use emel_model::lfm2::{Lfm2, Parameters as Lfm2Parameters};
use emel_model::qwen3::event as qwen3_event;
use emel_model::qwen3::{Parameters as Qwen3Parameters, Qwen3};
use libfuzzer_sys::fuzz_target;

const ATTENTION_NAMES: [&[u8]; 14] = [
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
const SHORTCONV_NAMES: [&[u8]; 13] = [
    b"token_embd.weight",
    b"output_norm.weight",
    b"output.weight",
    b"blk.0.attn_norm.weight",
    b"blk.0.shortconv.conv.weight",
    b"blk.0.shortconv.in_proj.weight",
    b"blk.0.shortconv.out_proj.weight",
    b"blk.0.ffn_norm.weight",
    b"blk.0.ffn_gate.weight",
    b"blk.0.ffn_down.weight",
    b"blk.0.ffn_up.weight",
    b"unused.0.weight",
    b"unused.1.weight",
];
const LFM2_NAMES: [&[u8]; 21] = [
    b"token_embd.weight",
    b"token_embd_norm.weight",
    b"blk.0.attn_norm.weight",
    b"blk.0.shortconv.conv.weight",
    b"blk.0.shortconv.in_proj.weight",
    b"blk.0.shortconv.out_proj.weight",
    b"blk.0.ffn_norm.weight",
    b"blk.0.ffn_gate.weight",
    b"blk.0.ffn_down.weight",
    b"blk.0.ffn_up.weight",
    b"blk.2.attn_norm.weight",
    b"blk.2.attn_q.weight",
    b"blk.2.attn_k.weight",
    b"blk.2.attn_v.weight",
    b"blk.2.attn_q_norm.weight",
    b"blk.2.attn_k_norm.weight",
    b"blk.2.attn_output.weight",
    b"blk.2.ffn_norm.weight",
    b"blk.2.ffn_gate.weight",
    b"blk.2.ffn_down.weight",
    b"blk.2.ffn_up.weight",
];
const VALID_WIRE_TYPES: [u32; 8] = [0, 1, 2, 3, 10, 11, 12, 14];

fn actor(input: &[u8], shortconv: bool) -> (Builder, ModelIdentity) {
    let names: &[&[u8]] = if shortconv {
        &SHORTCONV_NAMES
    } else {
        &ATTENTION_NAMES
    };
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).expect("bounded");
    for (index, name) in names.iter().enumerate() {
        let selector = input.get(index).copied().unwrap_or(index as u8) as usize;
        let wire_type = VALID_WIRE_TYPES[selector % VALID_WIRE_TYPES.len()];
        storage
            .push_tensor(TensorInput::new(
                name,
                wire_type,
                1,
                [32, 1, 1, 1],
                u64::try_from(index).expect("bounded") * 64,
                true,
            ))
            .expect("valid tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    (Builder::new(catalog, Resolver::new()), model)
}

fn llama_actor(input: &[u8]) -> (Llama, ModelIdentity) {
    let names = &ATTENTION_NAMES;
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).expect("bounded");
    for (index, name) in names.iter().enumerate() {
        let selector = input.get(index).copied().unwrap_or(index as u8) as usize;
        storage
            .push_tensor(TensorInput::new(
                name,
                VALID_WIRE_TYPES[selector % VALID_WIRE_TYPES.len()],
                1,
                [32, 1, 1, 1],
                u64::try_from(index).expect("bounded") * 64,
                true,
            ))
            .expect("valid tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    let actor = Llama::new(
        catalog,
        Resolver::new(),
        llama_event::Storage::with_block_capacity(2).expect("storage"),
    )
    .expect("Llama actor");
    (actor, model)
}

fn llama_parameters(selector: u8) -> LlamaParameters {
    LlamaParameters {
        context_length: i32::from(selector).wrapping_sub(64),
        embedding_length: i32::from(selector).wrapping_sub(32),
        embedding_length_out: i32::from(selector),
        feed_forward_length: i32::from(selector.rotate_left(1)),
        attention_head_count: i32::from(selector & 31),
        attention_head_count_kv: i32::from(selector & 15),
        rope_dimension_count: i32::from(selector.rotate_left(2)),
        block_count: i32::from(selector % 5) - 1,
        vocab_size: i32::from(selector) * 128,
        attention_layer_norm_epsilon: f32::from(selector),
        attention_layer_norm_rms_epsilon: f32::from(selector.rotate_left(1)),
        attention_clamp_kqv: f32::from(selector.rotate_left(2)),
        attn_logit_softcapping: f32::from(selector.rotate_left(3)),
        final_logit_softcapping: f32::from(selector.rotate_left(4)),
        residual_scale: f32::from(selector.rotate_left(5)),
        embedding_scale: f32::from(selector.rotate_left(6)),
        rope_freq_base: f32::from(selector) * 100.0,
        rope_freq_base_swa: f32::from(selector) * 1000.0,
        attention_key_length: i32::from(selector.rotate_left(1)),
        attention_value_length: i32::from(selector.rotate_right(1)),
    }
}

fn fuzz_llama_protocol(input: &[u8]) {
    let (mut actor, model) = llama_actor(input);
    for (step, byte) in input.iter().copied().enumerate().take(128) {
        let parameter = input.get(step + 1).copied().unwrap_or_default();
        let index = i32::from(parameter % 5) - 2;
        match byte % 11 {
            0 => {
                let architecture = if parameter & 1 == 0 {
                    b"llama".as_slice()
                } else {
                    b"other".as_slice()
                };
                let _ = actor.process_event(llama_event::ContractBegin::new(
                    architecture,
                    model,
                    llama_parameters(parameter),
                ));
            }
            1 => {
                let _ = actor.process_event(llama_event::BlockBuild::new(index));
            }
            2 => {
                let _ = actor.process_event(llama_event::TopologyBuild::new());
            }
            3 => {
                let _ = actor.process_event(llama_event::PlanBuild::new());
            }
            4 => {
                let _ = actor.process_event(llama_event::BlockValidation::new(index));
            }
            5 => {
                let _ = actor.process_event(llama_event::BlockAudit::new(index));
            }
            6 => {
                let family = QuantizedStageFamily::ALL
                    [usize::from(parameter) % QuantizedStageFamily::ALL.len()];
                let _ = actor.process_event(llama_event::StageAudit::new(family));
            }
            7 => {
                let _ = actor.process_event(llama_event::ContractVisit::new());
            }
            8 => {
                let _ = actor.process_event(llama_event::BlockVisit::new(index));
            }
            9 => {
                let _ = actor.process_event(llama_event::ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(llama_event::StorageRelease::new());
            }
        }
    }
}

fn lfm2_actor(input: &[u8]) -> (Lfm2, ModelIdentity) {
    let extra = input.first().and_then(|selector| match selector & 6 {
        2 => Some(b"blk.0.attn_q.weight".as_slice()),
        4 => Some(b"blk.2.shortconv.conv.weight".as_slice()),
        _ => None,
    });
    let tensor_count = LFM2_NAMES.len() + usize::from(extra.is_some());
    let name_bytes = LFM2_NAMES.iter().map(|name| name.len()).sum::<usize>()
        + extra.map_or(0, <[u8]>::len);
    let mut storage =
        CatalogStorage::with_capacity(tensor_count, name_bytes, tensor_count).expect("bounded");
    for name in LFM2_NAMES.into_iter().chain(extra) {
        storage
            .push_tensor(TensorInput::new(name, 0, 1, [1, 1, 1, 1], 4, true))
            .expect("valid tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    let actor = Lfm2::new(
        catalog,
        Resolver::new(),
        lfm2_event::Storage::with_block_capacity(14).expect("storage"),
    )
    .expect("LFM2 actor");
    (actor, model)
}

fn lfm2_parameters(input: &[u8], selector: u8) -> Lfm2Parameters {
    let mut parameters = Lfm2Parameters {
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
        attention_layer_pattern_count: u32::from(selector % 16),
        ..Lfm2Parameters::default()
    };
    for index in 0..16 {
        parameters.attention_layer_pattern_flags[index] =
            input.get(index + 2).copied().unwrap_or(u8::from(index == 2)) & 1;
    }
    if selector & 0x80 != 0 {
        parameters.embedding_length = i32::from(selector);
    }
    parameters
}

fn fuzz_lfm2_protocol(input: &[u8]) {
    let (mut actor, model) = lfm2_actor(input);
    for (step, byte) in input.iter().copied().enumerate().take(128) {
        let parameter = input.get(step + 1).copied().unwrap_or_default();
        let index = i32::from(parameter % 18) - 2;
        match byte % 11 {
            0 => {
                let architecture = if parameter & 1 == 0 {
                    b"lfm2".as_slice()
                } else {
                    b"other".as_slice()
                };
                let parameters = lfm2_parameters(input, parameter);
                let _ = actor.process_event(lfm2_event::ContractBegin::new(
                    architecture,
                    model,
                    &parameters,
                ));
            }
            1 => {
                let _ = actor.process_event(lfm2_event::BlockBuild::new(index));
            }
            2 => {
                let _ = actor.process_event(lfm2_event::TopologyBuild::new());
            }
            3 => {
                let _ = actor.process_event(lfm2_event::PlanBuild::new());
            }
            4 => {
                let _ = actor.process_event(lfm2_event::BlockValidation::new(index));
            }
            5 => {
                let _ = actor.process_event(lfm2_event::BlockAudit::new(index));
            }
            6 => {
                let family = QuantizedStageFamily::ALL
                    [usize::from(parameter) % QuantizedStageFamily::ALL.len()];
                let _ = actor.process_event(lfm2_event::StageAudit::new(family));
            }
            7 => {
                let _ = actor.process_event(lfm2_event::ContractVisit::new());
            }
            8 => {
                let _ = actor.process_event(lfm2_event::BlockVisit::new(index));
            }
            9 => {
                let _ = actor.process_event(lfm2_event::ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(lfm2_event::StorageRelease::new());
            }
        }
    }
}

fn qwen3_actor(input: &[u8]) -> (Qwen3, ModelIdentity) {
    let names = &ATTENTION_NAMES;
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).expect("bounded");
    for (index, name) in names.iter().enumerate() {
        let selector = input.get(index).copied().unwrap_or(index as u8) as usize;
        storage
            .push_tensor(TensorInput::new(
                name,
                VALID_WIRE_TYPES[selector % VALID_WIRE_TYPES.len()],
                1,
                [32, 1, 1, 1],
                u64::try_from(index).expect("bounded") * 64,
                true,
            ))
            .expect("valid tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    let actor = Qwen3::new(
        catalog,
        Resolver::new(),
        qwen3_event::Storage::with_block_capacity(2).expect("storage"),
    )
    .expect("Qwen3 actor");
    (actor, model)
}

fn qwen3_parameters(selector: u8) -> Qwen3Parameters {
    Qwen3Parameters {
        context_length: i32::from(selector).wrapping_sub(64),
        embedding_length: i32::from(selector).wrapping_sub(32),
        embedding_length_out: i32::from(selector),
        feed_forward_length: i32::from(selector.rotate_left(1)),
        attention_head_count: i32::from(selector & 31),
        attention_head_count_kv: i32::from(selector & 15),
        attention_key_length: i32::from(selector.rotate_left(1)),
        attention_value_length: i32::from(selector.rotate_right(1)),
        rope_dimension_count: i32::from(selector.rotate_left(2)),
        block_count: i32::from(selector % 5) - 1,
        attention_layer_norm_rms_epsilon: f32::from(selector.rotate_left(1)),
        rope_freq_base: f32::from(selector) * 100.0,
        tie_word_embeddings: selector & 1 == 0,
        rope_pair_x0_stride: i32::from(selector & 3),
        rope_pair_x1_stride: i32::from(selector.rotate_left(1) & 3),
        rope_pair_x1_offset: i32::from(selector.rotate_left(2) & 3),
        rope_pair_x1_half_rot_offset: i32::from(selector.rotate_left(3) & 3),
    }
}

fn fuzz_qwen3_protocol(input: &[u8]) {
    let (mut actor, model) = qwen3_actor(input);
    for (step, byte) in input.iter().copied().enumerate().take(128) {
        let parameter = input.get(step + 1).copied().unwrap_or_default();
        let index = i32::from(parameter % 5) - 2;
        match byte % 11 {
            0 => {
                let architecture = if parameter & 1 == 0 {
                    b"qwen3".as_slice()
                } else {
                    b"other".as_slice()
                };
                let _ = actor.process_event(qwen3_event::ContractBegin::new(
                    architecture,
                    model,
                    qwen3_parameters(parameter),
                ));
            }
            1 => {
                let _ = actor.process_event(qwen3_event::BlockBuild::new(index));
            }
            2 => {
                let _ = actor.process_event(qwen3_event::TopologyBuild::new());
            }
            3 => {
                let _ = actor.process_event(qwen3_event::PlanBuild::new());
            }
            4 => {
                let _ = actor.process_event(qwen3_event::BlockValidation::new(index));
            }
            5 => {
                let _ = actor.process_event(qwen3_event::BlockAudit::new(index));
            }
            6 => {
                let family = QuantizedStageFamily::ALL
                    [usize::from(parameter) % QuantizedStageFamily::ALL.len()];
                let _ = actor.process_event(qwen3_event::StageAudit::new(family));
            }
            7 => {
                let _ = actor.process_event(qwen3_event::ContractVisit::new());
            }
            8 => {
                let _ = actor.process_event(qwen3_event::BlockVisit::new(index));
            }
            9 => {
                let _ = actor.process_event(qwen3_event::ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(qwen3_event::StorageRelease::new());
            }
        }
    }
}

fn gemma4_actor(input: &[u8]) -> (Gemma4, ModelIdentity) {
    let names = &ATTENTION_NAMES;
    let name_bytes = names.iter().map(|name| name.len()).sum();
    let mut storage =
        CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).expect("bounded");
    for (index, name) in names.iter().enumerate() {
        let selector = input.get(index).copied().unwrap_or(index as u8) as usize;
        storage
            .push_tensor(TensorInput::new(
                name,
                VALID_WIRE_TYPES[selector % VALID_WIRE_TYPES.len()],
                1,
                [32, 1, 1, 1],
                u64::try_from(index).expect("bounded") * 64,
                true,
            ))
            .expect("valid tensor");
    }
    let mut catalog = Catalog::try_new().expect("catalog");
    catalog
        .process_event(BindStorage::new(storage))
        .expect("bind");
    let model = catalog.process_event(SealModel::new()).expect("seal");
    let actor = Gemma4::new(
        catalog,
        Resolver::new(),
        gemma4_event::Storage::with_block_capacity(2).expect("storage"),
    )
    .expect("Gemma4 actor");
    (actor, model)
}

fn gemma4_parameters(input: &[u8], selector: u8) -> Gemma4Parameters {
    let mut parameters = Gemma4Parameters {
        block_count: 1,
        context_length: 128,
        embedding_length: 64,
        embedding_length_out: 64,
        feed_forward_length: 256,
        attention_head_count: 8,
        attention_head_count_kv: 1,
        attention_key_length: 32,
        attention_value_length: 40,
        vocab_size: 1024,
        rope_dimension_count: 32,
        rope_freq_base: 20_000.0,
        attention_shared_kv_layers: i32::from(selector % 4) - 1,
        attention_key_length_swa: i32::from(selector & 32),
        attention_value_length_swa: i32::from(selector.rotate_left(1) & 32),
        rope_dimension_count_swa: i32::from(selector.rotate_left(2) & 32),
        rope_freq_base_swa: f32::from(selector.rotate_left(3) & 32) * 100.0,
        tie_word_embeddings: selector & 1 == 0,
        sliding_window_pattern_count: u32::from(selector % 3),
        ..Gemma4Parameters::default()
    };
    parameters.sliding_window_pattern_flags[0] =
        input.get(2).copied().unwrap_or(selector) & 1;
    parameters
}

fn fuzz_gemma4_protocol(input: &[u8]) {
    let (mut actor, model) = gemma4_actor(input);
    for (step, byte) in input.iter().copied().enumerate().take(128) {
        let parameter = input.get(step + 1).copied().unwrap_or_default();
        let index = i32::from(parameter % 5) - 2;
        match byte % 11 {
            0 => {
                let architecture = if parameter & 2 == 0 {
                    b"gemma4".as_slice()
                } else {
                    b"other".as_slice()
                };
                let parameters = gemma4_parameters(input, parameter);
                if parameter & 0x80 == 0 {
                    let _ = actor.process_event(gemma4_event::ContractBegin::validation(
                        architecture,
                        model,
                        &parameters,
                        i32::from(parameter % 4) - 1,
                    ));
                } else {
                    let _ = actor.process_event(gemma4_event::ContractBegin::new(
                        architecture,
                        model,
                        &parameters,
                    ));
                }
            }
            1 => {
                let _ = actor.process_event(gemma4_event::BlockBuild::new(index));
            }
            2 => {
                let _ = actor.process_event(gemma4_event::TopologyBuild::new());
            }
            3 => {
                let _ = actor.process_event(gemma4_event::PlanBuild::new());
            }
            4 => {
                let _ = actor.process_event(gemma4_event::BlockValidation::new(index));
            }
            5 => {
                let _ = actor.process_event(gemma4_event::BlockAudit::new(index));
            }
            6 => {
                let family = QuantizedStageFamily::ALL
                    [usize::from(parameter) % QuantizedStageFamily::ALL.len()];
                let _ = actor.process_event(gemma4_event::StageAudit::new(family));
            }
            7 => {
                let _ = actor.process_event(gemma4_event::ContractVisit::new());
            }
            8 => {
                let _ = actor.process_event(gemma4_event::BlockVisit::new(index));
            }
            9 => {
                let _ = actor.process_event(gemma4_event::ContractReset::new());
            }
            _ => {
                let _ = actor.process_event(gemma4_event::StorageRelease::new());
            }
        }
    }
}

fn layer(selector: u8, shortconv: bool) -> LayerExecution {
    LayerExecution::new(
        if shortconv {
            ResidualRoute::Shortconv
        } else {
            ResidualRoute::Attention
        },
        if selector & 1 == 0 {
            AttentionQkNormRoute::None
        } else {
            AttentionQkNormRoute::HeadwiseRms
        },
        if selector & 2 == 0 {
            AttentionValueRoute::DedicatedValue
        } else {
            AttentionValueRoute::SharedKeyValue
        },
        if selector & 4 == 0 {
            AttentionVNormRoute::None
        } else {
            AttentionVNormRoute::Rms
        },
        if selector & 8 == 0 {
            AttentionWindowRoute::FullContext
        } else {
            AttentionWindowRoute::SlidingWindow
        },
        i32::from(selector),
        i32::from(selector.rotate_left(1)),
        i32::from(selector.rotate_left(2)),
        f32::from(selector),
    )
}

fn complete_selected(input: &[u8], shortconv: bool) {
    let (mut builder, model) = actor(input, shortconv);
    builder
        .process_event(StorageBind::new(
            Storage::with_block_capacity(1).expect("storage"),
        ))
        .expect("bind");
    builder
        .process_event(ContractBegin::new(model, 1, 64))
        .expect("begin");
    builder
        .process_event(GlobalBindings::new(
            b"token_embd.weight",
            b"output_norm.weight",
            false,
        ))
        .expect("globals");
    if shortconv {
        builder
            .process_event(ShortconvBlock::new(0, layer(0, true)))
            .expect("shortconv");
    } else {
        let selector = input.get(15).copied().unwrap_or_default() & 3;
        builder
            .process_event(AttentionBlock::new(0, layer(selector, false)))
            .expect("attention");
    }
    builder
        .process_event(Topology::new(1, 1, 4, 4))
        .expect("topology");
    builder.process_event(Plan::new()).expect("plan");
    builder
        .process_event(BlockValidation::new(0))
        .expect("validate");
    builder.process_event(BlockAudit::new(0)).expect("audit");
    for family in QuantizedStageFamily::ALL {
        builder
            .process_event(StageAudit::new(family))
            .expect("stage");
    }
    builder
        .process_event(ContractVisit::new())
        .expect("complete");
}

fn fuzz_protocol(input: &[u8], shortconv: bool) {
    let (mut builder, model) = actor(input, shortconv);
    let mut storage = Some(Storage::with_block_capacity(2).expect("storage"));
    for (step, byte) in input.iter().copied().enumerate().take(128) {
        let parameter = input.get(step + 1).copied().unwrap_or_default();
        let index = i32::from(parameter % 5) - 2;
        match byte % 15 {
            0 => {
                if let Some(candidate) = storage.take()
                    && let Err(error) = builder.process_event(StorageBind::new(candidate))
                {
                    storage = Some(error.into_storage());
                }
            }
            1 => {
                let blocks = i32::from(parameter % 5) - 1;
                let _ = builder.process_event(ContractBegin::new(
                    model,
                    blocks,
                    i32::from(parameter) - 64,
                ));
            }
            2 => {
                let token = if parameter & 1 == 0 {
                    b"token_embd.weight".as_slice()
                } else {
                    b"missing.weight".as_slice()
                };
                let _ = builder.process_event(GlobalBindings::new(
                    token,
                    b"output_norm.weight",
                    parameter & 2 != 0,
                ));
            }
            3 => {
                let _ = builder.process_event(AttentionBlock::new(index, layer(parameter, false)));
            }
            4 => {
                let _ = builder.process_event(ShortconvBlock::new(index, layer(parameter, true)));
            }
            5 => {
                let value = u32::from(parameter).wrapping_sub(64);
                let _ = builder.process_event(Topology::new(
                    value,
                    value.rotate_left(1),
                    u64::from(parameter),
                    u64::from(parameter).saturating_mul(16),
                ));
            }
            6 => {
                let _ = builder.process_event(Plan::new());
            }
            7 => {
                let _ = builder.process_event(BlockValidation::new(index));
            }
            8 => {
                let _ = builder.process_event(BlockAudit::new(index));
            }
            9 => {
                let family = QuantizedStageFamily::ALL
                    [usize::from(parameter) % QuantizedStageFamily::ALL.len()];
                let _ = builder.process_event(StageAudit::new(family));
            }
            10 => {
                let _ = builder.process_event(ContractVisit::new());
            }
            11 => {
                let _ = builder.process_event(BlockVisit::new(index));
            }
            12 => {
                let _ = builder.process_event(ContractReset::new());
            }
            13 => {
                if let Ok(released) = builder.process_event(StorageRelease::new()) {
                    storage = Some(released);
                }
            }
            _ => {
                let family = if parameter & 1 == 0 {
                    BlockFamily::Shortconv
                } else {
                    BlockFamily::Attention
                };
                let _ = builder.process_event(RejectBlockTensors::new(index, family));
            }
        }
    }
}

fuzz_target!(|input: &[u8]| {
    let shortconv = input.first().is_some_and(|byte| byte & 1 != 0);
    complete_selected(input, shortconv);
    fuzz_protocol(input, shortconv);
    fuzz_llama_protocol(input);
    fuzz_lfm2_protocol(input);
    fuzz_qwen3_protocol(input);
    fuzz_gemma4_protocol(input);
});
