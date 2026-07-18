//! Deterministic tensor bulk-protocol parity lane.

use allocation_counter as _;
use emel_io::{read, staged_read};
use emel_model::tensor::Store;
use emel_model::tensor::dependency::Actors;
use emel_model::tensor::event::{
    ApplyBoundEffectResults, ApplyEffectError, ApplyOwnedEffectResults, BindStorage,
    CaptureTensorState, EffectBuffer, EffectError, EffectRequest, Error, OwnedEffectResult,
    PlanLoad, ReadLoad, StagedLoad, StorageBatch, StorageEntry, StrategyKind, TensorMetadata,
    WithTensor,
};
use sml as _;

#[allow(
    clippy::too_many_lines,
    reason = "the deterministic trace stays linear for direct comparison with the C++ lane"
)]
fn main() {
    let mut store = Store::new(2).expect("fixture capacity");
    println!("model-tensor-parity-snapshot/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit=843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("source_tree=06306d4ffad3455fcf5df71dc692df52514b9865");
    println!(
        "source_files=src/emel/model/tensor/events.hpp,errors.hpp,context.hpp,detail.hpp,guards.hpp,actions.hpp,sm.hpp"
    );
    println!("source_tests=tests/model/tensor/lifecycle_tests.cpp");
    println!(
        "fixture_config=public_actor_typed_events,owned_batches,two_tensors,supported_strategy_families_plus_rust_unknown_extension"
    );
    println!(
        "contract_delta=rust_owned_batches_replace_raw_spans_and_pointers;rust_busy_is_typed;rust_mmap_requires_a_caller_stability_capability"
    );

    let bind = store
        .process_event(BindStorage::new(storage(&[
            (4_096, 2, 1, Some(&[1, 2, 3])),
            (8_192, 3, 2, Some(&[4, 5, 6])),
        ])))
        .expect("bulk bind");
    println!(
        "case=bind outcome=done active_extent={}",
        bind.active_extent()
    );

    let plan = store
        .process_event(PlanLoad::new(StrategyKind::None, effects(2)))
        .expect("none plan");
    let planned = plan.into_effects();
    println!(
        "case=plan_none outcome=done effect_count=2 first={} second={}",
        effect(&planned.effects()[0]),
        effect(&planned.effects()[1])
    );
    let busy = store
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effects(2)))
        .expect_err("second plan is busy");
    println!(
        "extension=typed_busy reference_behavior=unexpected_recovery_to_ready rust_error={} phase=awaiting_bound",
        error(busy.error()),
    );
    store
        .process_event(ApplyBoundEffectResults::new(vec![0, 1].into_boxed_slice()))
        .expect("bound results");
    let first = store
        .process_event(CaptureTensorState::new(0))
        .expect("capture first");
    let second = store
        .process_event(CaptureTensorState::new(1))
        .expect("capture second");
    println!(
        "case=apply_bound outcome=done first=resident:{} second=resident:{}",
        first.buffer_bytes(),
        second.buffer_bytes()
    );

    store
        .process_event(BindStorage::new(storage(&[(16_384, 4, 5, None)])))
        .expect("read storage");
    let plan = store
        .process_event(PlanLoad::new(StrategyKind::ReadCopy, effects(1)))
        .expect("read plan");
    println!(
        "case=plan_read outcome=done effect_count=1 first={}",
        effect(&plan.into_effects().effects()[0])
    );
    store
        .process_event(ApplyOwnedEffectResults::new(
            vec![OwnedEffectResult::new(
                0,
                vec![9, 8, 7, 6, 5].into_boxed_slice(),
            )]
            .into_boxed_slice(),
        ))
        .expect("owned results");
    let state = store
        .process_event(CaptureTensorState::new(0))
        .expect("capture owned");
    println!(
        "case=apply_owned outcome=done lifecycle=resident buffer_bytes={}",
        state.buffer_bytes()
    );

    for (name, strategy) in [
        ("external", StrategyKind::ExternalBuffer),
        ("staged", StrategyKind::StagedRead),
    ] {
        store
            .process_event(BindStorage::new(storage(&[(16_384, 4, 5, None)])))
            .expect("strategy storage");
        let plan = store
            .process_event(PlanLoad::new(strategy, effects(1)))
            .expect("strategy plan");
        println!(
            "case=plan_{name} outcome=done effect_count=1 first={}",
            effect(&plan.into_effects().effects()[0])
        );
        let backend = store
            .process_event(ApplyEffectError::new(0, EffectError::Backend))
            .expect_err("strategy backend failure");
        println!(
            "case={name}_error outcome=error error={} phase=ready",
            error(backend)
        );
    }

    let invalid = store
        .process_event(BindStorage::new(StorageBatch::new(
            Vec::new().into_boxed_slice(),
        )))
        .expect_err("empty bind");
    let preserved = store
        .process_event(CaptureTensorState::new(0))
        .expect("preserved capture");
    println!(
        "case=invalid_bind outcome=error error={} preserved_offset={}",
        error(invalid.error()),
        preserved.metadata().file_offset()
    );

    let unknown = store
        .process_event(PlanLoad::new(StrategyKind::Unknown(255), effects(1)))
        .expect_err("unknown strategy");
    println!(
        "extension=typed_unknown reference_behavior=planned_as_io_load rust_error={}",
        error(unknown.error())
    );

    let mapped = store
        .process_event(PlanLoad::new(StrategyKind::MappedFile, effects(1)))
        .expect("mapped plan");
    println!(
        "case=plan_mapped outcome=done effect_count=1 first={}",
        effect(&mapped.into_effects().effects()[0])
    );
    let backend = store
        .process_event(ApplyEffectError::new(0, EffectError::Backend))
        .expect_err("mapped backend failure");
    println!(
        "case=mapped_error outcome=error error={} phase=ready",
        error(backend)
    );

    let no_plan = store
        .process_event(ApplyBoundEffectResults::new(vec![0].into_boxed_slice()))
        .expect_err("no outstanding plan");
    println!(
        "case=result_without_plan outcome=error error={}",
        error(no_plan.error())
    );

    let source = [11_u8, 22, 33, 44, 55, 66];
    let dependencies = Actors::new((), read::Reader::new(), staged_read::Stager::new());
    let mut io_store = Store::with_dependencies(1, dependencies).expect("I/O fixture capacity");
    io_store
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 4]))])))
        .expect("read target storage");
    let read = io_store
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&source), 1, 4))
        .expect("direct read route");
    let mut checksum = |bytes: &[u8]| bytes.iter().copied().map(u64::from).sum::<u64>();
    let read_checksum = io_store
        .process_event(WithTensor::new(0, &mut checksum))
        .expect("read access");
    println!(
        "case=direct_read outcome=done bytes={} checksum={read_checksum}",
        read.bytes_copied()
    );
    let duplicate = io_store
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&source), 1, 4))
        .expect_err("resident read is rejected");
    println!(
        "case=direct_read_already_resident outcome=error error={}",
        error(duplicate)
    );

    io_store
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 8]))])))
        .expect("read validation storage");
    let invalid = io_store
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&source), 0, 5))
        .expect_err("logical size is enforced");
    println!(
        "case=direct_read_invalid outcome=error error={}",
        error(invalid)
    );

    io_store
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 8]))])))
        .expect("read error storage");
    let failed = io_store
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&source[..3]), 0, 4))
        .expect_err("short source reaches the read child");
    println!(
        "case=direct_read_error outcome=error error={}",
        error(failed)
    );

    io_store
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 4]))])))
        .expect("staged target storage");
    let staged = io_store
        .process_event(StagedLoad::new(0, Some(&source), 1, 4, 3))
        .expect("direct staged route");
    let mut checksum = |bytes: &[u8]| bytes.iter().copied().map(u64::from).sum::<u64>();
    let staged_checksum = io_store
        .process_event(WithTensor::new(0, &mut checksum))
        .expect("staged access");
    println!(
        "case=direct_staged outcome=done bytes={} checksum={staged_checksum}",
        staged.bytes_copied()
    );

    io_store
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 8]))])))
        .expect("staged validation storage");
    let invalid = io_store
        .process_event(StagedLoad::new(0, Some(&source), 1, 5, 3))
        .expect_err("staged logical size is enforced");
    println!(
        "case=direct_staged_invalid outcome=error error={}",
        error(invalid)
    );

    let mut absent = Store::new(1).expect("absent fixture capacity");
    absent
        .process_event(BindStorage::new(storage(&[(0, 4, 0, Some(&[0; 4]))])))
        .expect("absent target storage");
    let missing = absent
        .process_event(ReadLoad::new(0, "fixture.bin", Some(&source), 0, 4))
        .expect_err("missing reader is explicit");
    println!("case=missing_reader outcome=error error={}", error(missing));
    let missing = absent
        .process_event(StagedLoad::new(0, Some(&source[..4]), 0, 4, 2))
        .expect_err("missing stager is explicit");
    println!("case=missing_stager outcome=error error={}", error(missing));
}

fn storage(entries: &[(u64, u64, u16, Option<&[u8]>)]) -> StorageBatch {
    StorageBatch::new(
        entries
            .iter()
            .map(|(offset, size, file_index, bytes)| {
                StorageEntry::new(
                    TensorMetadata::new(*offset, *size, *file_index, 7),
                    bytes.map(|value| value.to_vec().into_boxed_slice()),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    )
}

fn effects(count: usize) -> EffectBuffer {
    EffectBuffer::new(vec![EffectRequest::Empty; count].into_boxed_slice())
}

fn effect(value: &EffectRequest) -> String {
    match value {
        EffectRequest::Empty => "empty".into(),
        EffectRequest::None {
            tensor_id,
            file_index,
            offset,
            size,
        } => format!("none:{tensor_id}:{file_index}:{offset}:{size}"),
        EffectRequest::MappedFile {
            tensor_id,
            file_index,
            offset,
            size,
        } => format!("mapped_file:{tensor_id}:{file_index}:{offset}:{size}"),
        EffectRequest::ReadCopy {
            tensor_id,
            file_index,
            offset,
            size,
        } => format!("read_copy:{tensor_id}:{file_index}:{offset}:{size}"),
        EffectRequest::ExternalBuffer {
            tensor_id,
            file_index,
            offset,
            size,
        } => format!("external_buffer:{tensor_id}:{file_index}:{offset}:{size}"),
        EffectRequest::StagedRead {
            tensor_id,
            file_index,
            offset,
            size,
        } => format!("staged_read:{tensor_id}:{file_index}:{offset}:{size}"),
    }
}

const fn error(value: Error) -> &'static str {
    match value {
        Error::InvalidRequest => "invalid_request",
        Error::Capacity => "capacity",
        Error::Busy => "busy",
        Error::UnsupportedStrategy => "unsupported_strategy",
        Error::BackendError => "backend_error",
        Error::ReadUnavailable => "read_unavailable",
        Error::StagerUnavailable => "stager_unavailable",
        Error::TensorAlreadyResident => "tensor_already_resident",
        Error::Read(_) => "read_failed",
        _ => "other",
    }
}
use emel_gguf as _;
use emel_token as _;
