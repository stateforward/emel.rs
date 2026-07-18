#![no_main]

use emel_kernels::capability::Resolver;
use emel_model::catalog::Catalog;
use emel_model::catalog::event::{
    BindStorage, ModelIdentity, SealModel, Storage as CatalogStorage, TensorInput,
};
use emel_model::generation::event::{
    AttentionBlock, BlockAudit, BlockValidation, BlockVisit, ContractBegin, ContractReset,
    ContractVisit, GlobalBindings, Plan, ShortconvBlock, StageAudit, Storage, StorageBind,
    StorageRelease, Topology,
};
use emel_model::generation::{
    AttentionQkNormRoute, AttentionVNormRoute, AttentionValueRoute, AttentionWindowRoute, Builder,
    LayerExecution, QuantizedStageFamily, ResidualRoute,
};
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
                let _ = builder.process_event(ContractBegin::new(model, 1, 64));
            }
        }
    }
}

fuzz_target!(|input: &[u8]| {
    let shortconv = input.first().is_some_and(|byte| byte & 1 != 0);
    complete_selected(input, shortconv);
    fuzz_protocol(input, shortconv);
});
